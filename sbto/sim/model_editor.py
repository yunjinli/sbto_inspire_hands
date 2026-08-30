import mujoco
import numpy as np
from typing import List, Dict, Optional, Callable, Any
from functools import wraps
import os

class ModelEditor():
    DEFAULT_NAME = "static"
    AVAILABLE_GEOM = ["box", "cylinder", "sphere", "urdf", "mesh"]

    def __init__(self, xml_path, callback_fn: Optional[Callable[[mujoco.MjModel], Any]] = None):
        self.xml_path = xml_path
        self.callback_fn = callback_fn
        self.reset()

    @staticmethod
    def with_callback():
        """
        Call callback_fn to update variables depending on mj_model.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self: 'ModelEditor', *args, **kwargs):
                res = func(self, *args, **kwargs)
                mj_model = self.get_model()
                if not self.callback_fn is None:
                    self.callback_fn(mj_model)
                return res
            return wrapper
        return decorator
    
    @staticmethod
    def _to_quat(euler) -> np.array:
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, euler, "xyz")
        return quat
        
    @staticmethod
    def _set_attr_from_kwargs(obj: Any, **kwargs) -> None:
        for k, v in kwargs.items():
            try:
                setattr(obj, k, v)
            except:
                pass

    @staticmethod
    def _copy_attr_obj(obj_src: Any, obj_dst: Any) -> None:
            for attr in obj_src.__dir__():
                try:
                    setattr(obj_dst, attr, getattr(obj_src, attr))
                except Exception as e:
                    continue

    def _body_id(self, name) -> int:
        for i, body in enumerate(self.mj_spec.bodies):
            if body.name == name:
                return i
        return -1
    
    @with_callback()
    def add_body(
        self,
        pos : np.ndarray,
        euler : np.ndarray,
        name : str = "",
        **kwargs,
        ) -> int:
        # cast to array
        pos = np.asarray(pos)
        euler = np.asarray(euler)

        if name:
            if name in self.name2id:
                name = f"{name}_{self.id}"
        else:
            name = f"{ModelEditor.DEFAULT_NAME}_{self.id}"
        
        body = self.mj_spec.worldbody.add_body(
            name=name,
            pos=pos.copy(),
            quat=self._to_quat(euler),
            )
        self._set_attr_from_kwargs(body, **kwargs)

        # Update maps
        self.id2name[self.id] = name
        self.name2id[name] = self.id
        self.id += 1
        i = self.id - 1

        return body, i

    @with_callback()
    def _add_body_and_geom(
        self,
        geom_type,
        pos : np.ndarray,
        size : np.ndarray,
        euler : np.ndarray,
        rgba : List[float],
        name : str = "",
        freejoint: bool = False,
        **kwargs,
        ) -> int:
        # cast to array
        pos = np.asarray(pos)
        size = np.asarray(size)
        euler = np.asarray(euler)
        
        body, id = self.add_body(
            pos,
            euler,
            name,
        )

        # Body already at the desired pos
        pos_geom = np.zeros_like(pos)
        euler_geom = np.zeros_like(euler)
        self._add_geom_to_body(
            body,
            geom_type,
            pos_geom,
            size,
            euler_geom,
            rgba,
            name,
            **kwargs,
        )
        
        if freejoint:
            body.add_freejoint()

        return id
    
    @with_callback()
    def _add_geom_to_body(
        self,
        body,
        geom_type,
        pos : np.ndarray,
        size : np.ndarray,
        euler : np.ndarray,
        rgba : List[float],
        name : str = "",
        **kwargs,
        ) -> int:
        geom = body.add_geom()
        if rgba is None:
            rgba = np.ones(4)

        geom.pos = pos
        geom.quat = self._to_quat(euler)
        geom.type = geom_type
        geom.size = size.copy()
        geom.rgba = rgba
        geom.name = name
        geom.mass = 1.
        self._set_attr_from_kwargs(geom, **kwargs)
        
    @with_callback()
    def add_urdf(
        self,
        urdf_path: str,
        pos: np.ndarray = np.zeros(3),
        euler: np.ndarray = np.zeros(3),
        rgba: Optional[List[float]] = None,
        name: str = "",
        bodyname: str = "",
        freejoint: bool = False,
        **kwargs,
    ) -> None:
        """
        Add an object defined by a URDF file as a subtree in this model.
        """
        # Load the URDF into a temporary Spec
        urdf_spec = mujoco.MjSpec.from_file(urdf_path)

        # Find the target parent in the main spec
        if bodyname:
            # Use find_body to look up the parent body by name
            parent = self.mj_spec.body(bodyname)
            if parent is None:
                raise ValueError(f"Parent body '{bodyname}' not found")
        else:
            if name:
                parent = self.add_body(name=name)
            else:
                parent = self.mj_spec.worldbody

        # Change color:
        if rgba is not None:
            for geom in urdf_spec.geoms:
                geom.rgba = rgba

        # Add a freejoint if requested (usually for the root body)
        if freejoint:
            parent.add_freejoint()

        # Create a frame to handle the position and orientation offset
        frame = parent.add_frame()
        frame.pos = pos
        frame.quat = self._to_quat(euler)
        self.mj_spec.attach(urdf_spec, frame=frame)

    @with_callback()
    def add_mesh(
        self,
        mesh_path: str,
        pos: np.ndarray = np.zeros(3),
        euler: np.ndarray = np.zeros(3),
        rgba: Optional[List[float]] = None,
        name: str = "",
        bodyname: str = "",
        freejoint: bool = False,
        **kwargs,
    ) -> None:
        """
        Add an object defined by a URDF file as a subtree in this model.
        """
        if not name:
            filename = os.path.split(mesh_path)[-1]
            name = os.path.splitext(filename)[0]

        mesh_full_path = os.path.join(self.mj_spec.modelfiledir, self.mj_spec.meshdir, mesh_path)
        with open(mesh_full_path, "r") as f:
            mesh_asset = f.readlines()

        # Add mesh to the spec
        mesh_name = f"{name}_mesh"
        self.mj_spec.add_mesh(file = mesh_path, name = mesh_name)
        self.mj_spec.assets.update({
            mesh_path : "".join(mesh_asset)
        })
        # mesh.name = name

        # Find the target parent in the main spec
        if bodyname:
            # Use find_body to look up the parent body by name
            parent = self.mj_spec.body(bodyname)
            if parent is None:
                raise ValueError(f"Parent body '{bodyname}' not found")
            if freejoint:
                parent.add_freejoint()
        else:
            if name:
                parent = self.add_body(name=name)
            else:
                parent = self.mj_spec.worldbody

        geom = parent.add_geom()
        geom.pos = pos
        geom.quat = self._to_quat(euler)
        geom.pos = pos
        geom.quat = self._to_quat(euler)
        geom.meshname = mesh_name
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        if rgba is not None:
            geom.rgba = rgba
        geom.name = name
        geom.mass = 1.
        self._set_attr_from_kwargs(geom, **kwargs)

    def add_box(
        self,
        pos : np.ndarray,
        size : np.ndarray,
        euler : np.ndarray,
        rgba : Optional[List[float]] = None,
        name : str = "",
        freejoint : bool = False,
        bodyname :str = "",
        **kwargs,
        ) -> int:

        body_id = self._body_id(bodyname)
        if bodyname and body_id >= 0:
            body = self.mj_spec.body(bodyname)
            i = body_id
            self._add_geom_to_body(
                body,
                geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=size,
                euler=euler,
                rgba=rgba,
                name=name if name else "box",
                **kwargs,
            )
            if freejoint:
                body.add_freejoint()
        else:
            i = self._add_body_and_geom(
                geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=size,
                euler=euler,
                rgba=rgba,
                name=name if name else "box",
                freejoint=freejoint,
                **kwargs,
            )
        return i

    def add_sphere(
        self,
        pos : np.ndarray,
        radius : float,
        rgba : Optional[List[float]] = None,
        name : str = "",
        freejoint: bool = False,
        bodyname :str = "",
        **kwargs,
        ) -> int:

        size = np.array([radius, 0, 0])
        euler = np.zeros(3)
        body_id = self._body_id(bodyname)
        if bodyname and body_id >= 0:
            body = self.mj_spec.body(bodyname)
            i = body_id
            self._add_geom_to_body(
                body,
                geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=pos,
                size=size,
                euler=euler,
                rgba=rgba,
                name=name if name else "sphere",
                **kwargs,
            )
            if freejoint:
                body.add_freejoint()
        else:
            i = self._add_body_and_geom(
                geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=pos,
                size=size,
                euler=euler,
                rgba=rgba,
                name=name if name else "sphere",
                freejoint=freejoint,
                **kwargs,
            )
        return i
    
    def add_cylinder(
        self,
        pos : np.ndarray,
        radius : float,
        height : float,
        euler : np.ndarray,
        rgba : Optional[List[float]] = None,
        name : str = "",
        freejoint: bool = False,
        bodyname :str = "",
        **kwargs,
        ) -> int:
        
        size = np.array([radius, height / 2., 0])
        body_id = self._body_id(bodyname)
        if bodyname and body_id >= 0:
            body = self.mj_spec.body(bodyname)
            i = body_id
            self._add_geom_to_body(
                body,
                geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=pos,
                size=size,
                euler=euler,
                rgba=rgba,
                name=name if name else "cylinder",
                **kwargs,
            )
            if freejoint:
                body.add_freejoint()

        else:
            i = self._add_body_and_geom(
                geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=pos,
                size=size,
                euler=euler,
                rgba=rgba,
                name=name if name else "cylinder",
                freejoint=freejoint,
                **kwargs,
            )
        return i

    def get_body(
        self, 
        name: Optional[str] = None, 
        id: Optional[int] = None
        ) -> None:

        if name is None and id is None:
            raise ValueError("get_body: provide a name or an id.")

        if id is not None:
            if id in self.id2name:
                name = self.id2name[id]

        if name is not None:
            if name in self.name2id:
                id = self.name2id[name]
                body = self.mj_spec.find_body(name)
                return body
        
        return None
    
    @with_callback()
    def remove(
        self, 
        name: Optional[str] = None, 
        id: Optional[int] = None
        ) -> None:
        body = self.get_body(name, id)
        if body:
            self.mj_spec.detach_body(body)

            if id is not None:
                name = self.id2name[id]
            if name is not None:
                id = self.name2id[name]
            
            del self.id2name[id]
            del self.name2id[name]
        
        self.callback_fn()

    @with_callback()
    def move(
        self, 
        new_pos: np.ndarray, 
        new_euler: Optional[np.ndarray] = None,
        name : Optional[str] = None, 
        id : Optional[int] = None, 
        ) -> None:
        body = self.get_body(name, id)
        if body:
            geom = body.first_geom()
            if geom:
                geom.pos = new_pos.copy()
                if new_euler is not None:
                    geom.quat = self._to_quat(new_euler)

    @with_callback()
    def set_color(self,
                  rgba : np.ndarray, 
                  name : Optional[str] = None, 
                  id : Optional[int] = None) -> None:
        body = self.get_body(name, id)
        if body:
            geom = body.first_geom()
            geom.rgba = rgba

    @with_callback()
    def add_contact_pair(
        self,
        geom1: str,
        geom2: str,
        condim: int = 1,
        **kwargs
        ) -> None:

        pair = self.mj_spec.add_pair()
        name = f"{geom1}_{geom2}"
        pair.name = name
        pair.geomname1 = geom1
        pair.geomname2 = geom2
        pair.condim = condim
        self._set_attr_from_kwargs(pair, **kwargs)

        return name

    @with_callback()
    def add_contact_sensor(
        self,
        geom1: str,
        geom2: str,
        data: str = "found",
        **kwargs
        ) -> None:
        
        sensor = self.mj_spec.add_sensor()
        name = f"{geom1}_{geom2}"
        sensor.name = name
        sensor.type = mujoco.mjtSensor.mjSENS_CONTACT
        sensor.objtype = mujoco.mjtObj.mjOBJ_GEOM
        sensor.objname = geom1
        sensor.reftype = mujoco.mjtObj.mjOBJ_GEOM
        sensor.refname = geom2

        if data == "found force":
            sensor.intprm = [3, 2, 1]
        elif data == "found":
            sensor.intprm = [1, 0, 1]
        else:
            print(f"Invalid data {data}. Setting to 'found'")
            sensor.intprm = [1, 0, 1]

        return name
    
    @with_callback()
    def delete_sensors(
        self,
        sensor_name: str | list[str],
        ) -> None:
        if isinstance(sensor_name, str):
            sensor_name = [sensor_name]

        for sns_name in sensor_name:
            if self.mj_spec.sensor(sns_name):
                self.mj_spec.delete(self.mj_spec.sensor(sns_name))

    @with_callback()
    def add_sensors_from_file(
        self,
        file_path: str,
        ) -> None:
        mj_spec_src = mujoco.MjSpec.from_file(file_path)
        for sensor_src in mj_spec_src.sensors:
            sensor_dst = self.mj_spec.add_sensor()
            self._copy_attr_obj(sensor_src, sensor_dst)

    @with_callback()
    def add_keyframes_from_file(
        self,
        file_path: str,
        ) -> None:
        mj_spec_src = mujoco.MjSpec.from_file(file_path)
        for key_src in mj_spec_src.keys:
            key_dst = self.mj_spec.add_key()
            self._copy_attr_obj(key_src, key_dst)

    @with_callback()
    def add_cnt_pairs_from_file(
        self,
        file_path: str,
        ) -> None:
        mj_spec_src = mujoco.MjSpec.from_file(file_path)
        for pair_src in mj_spec_src.pairs:
            pair_dst = self.mj_spec.add_pair()
            self._copy_attr_obj(pair_src, pair_dst)

    @with_callback()
    def reset(self):
        self.mj_spec = mujoco.MjSpec.from_file(os.path.abspath(self.xml_path))
        # Disable BVH-activity visualization bookkeeping. With bvactive on
        # (the default up to MuJoCo 3.11), mj_collision memsets
        # mjData.bvh_active -- one byte per BVH node, and every <mesh> asset
        # gets a full-face BVH regardless of contype/conaffinity -- on every
        # step, purely so the viewer can highlight visited bounding boxes.
        # For dense visual STLs (G1 body ~0.8M nodes, RH56E2 hands ~1.6M)
        # that memset dominated mj_collision and saturated memory bandwidth
        # in multithreaded rollouts. It has no effect on physics (bit-exact)
        # or on rendering meshes. MuJoCo flips this default to "false" in the
        # release after 3.11 for the same reason (changelog: "bottleneck for
        # models with large meshes").
        self.mj_spec.visual.global_.bvactive = 0
        self.id : int = len(self.mj_spec.bodies)
        self.id2name : Dict[int, str] = {}
        self.name2id : Dict[str, int] = {}

    def get_model(self):
        return self.mj_spec.compile()

if __name__ == "__main__":
    import mujoco
    def callback(mj_model):
        print("Callback") 
    xml = "sbto/models/unitree_g1/scene_mjx_25dof_no_hands.xml"
    edit = ModelEditor(xml, callback)

    # Add custom geometries
    print("--- Adding custom geometries...")
    pos = np.array([1., 1., 1.])
    size = np.array([0.1, 0.1, 0.1])
    euler = np.array([0., 0., .5])

    # contype="0" conaffinity="1" rgba="0.3 0.3 0.3 1" priority="0" solref="0.008 1." friction="0.6 0.003 0.001"
    box_id = edit.add_box(
        pos,
        size,
        euler,
        rgba=(0.3, 0.3, 0.3, 1),
        name="obj",
        freejoint=True,
        # bodyname="static",
        priority=0,
        contype=0,
        conaffinity=1,
        solref=(0.008, 1.),
        friction=(0.6, 0.003, 0.001),
    )
    xml_sensor = "sbto/models/unitree_g1/utils/obj_floor/sensors.xml"
    edit.add_sensors_from_file(xml_sensor)

    with open("test/edit_model_.xml", "w") as f:
        f.write(edit.mj_spec.to_xml())
    
    mj_model = edit.get_model()
    mj_data = mujoco.MjData(mj_model)

    N = 5
    t = np.arange(N) / N
    x_0 = np.concatenate((mj_data.qpos, mj_data.qvel))
    x_traj = np.tile(x_0[None, :], (N, 1))

    print(edit.mj_spec.__dir__())

    from sbto.utils.viewer import visualize_trajectory
    visualize_trajectory(
        mj_model,
        mj_data,
        t,
        x_traj
    )

        