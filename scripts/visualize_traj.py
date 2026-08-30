import argparse
import mujoco
from hydra.utils import instantiate

from sbto.main import instantiate_from_cfg
from sbto.data.utils import get_config_from_rundir, load_best_trajectory_from_rundir
from sbto.utils.viewer import visualize_trajectory_with_reference, visualize_trajectory
from sbto.data.constants import *

def main(rundir: str, with_ref: bool = True):

    cfg = get_config_from_rundir(rundir)

    if with_ref:
        # Needs the full task (task.ref), which -- unlike sim/mj_scene
        # construction -- looks up cost-specific sensor names (e.g.
        # left_hand_cnt/right_hand_cnt) by name. Those names can move to a
        # different sensor xml file as the model evolves (see
        # rh56e2_hand_obj_contact.xml's history), so replaying an old rundir
        # with --with-ref can break even though the run itself is fine --
        # re-run SBTO against the current config, or use --no-ref, if so.
        sim, task, _, _ = instantiate_from_cfg(cfg)
        mj_scene = sim.mj_scene
    else:
        # --no-ref only needs mj_model/mj_scene for FK-based playback, so
        # skip building the task entirely -- sim/mj_scene construction just
        # compiles whatever sensors the xml files list, with no by-name
        # lookups, so it stays immune to the sensor-file drift above.
        sim = instantiate(cfg.task.sim)
        mj_scene = sim.mj_scene

    data = load_best_trajectory_from_rundir(rundir, mj_scene=mj_scene)
    mj_model = mj_scene.mj_model
    mj_data = mujoco.MjData(mj_model)

    if with_ref:
        visualize_trajectory_with_reference(
            mj_model, mj_data, task.ref.time, data[KEY_FULL_STATE], task.ref.x
        )
    else:
        visualize_trajectory(
            mj_model, mj_data, data[KEY_TIME], data[KEY_FULL_STATE]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize best trajectory from a run directory."
    )

    parser.add_argument(
        "rundir",
        type=str,
        help="Path to run directory containing config and trajectory data.",
    )

    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="Disable reference trajectory visualization.",
    )

    args = parser.parse_args()

    main(
        rundir=args.rundir,
        with_ref=not args.no_ref,
    )
