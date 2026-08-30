"""Re-lay out a reference motion's qpos for a model with extra joints.

OmniRetarget clips store one qpos row per frame for the stock 29-dof G1 plus
the object: [root free joint (7) | 29 joints | object free joint (7)] = 43.
A model with hands (e.g. RH56E2: 12 extra finger joints per side, interleaved
right after each wrist in MuJoCo's qpos order) has a different, wider layout,
and the reference has nothing to say about the new joints. This script builds
the target layout by *joint name*: every joint the target model shares with
the source model gets its columns copied over, every joint the source does
not have is filled with the target model's own qpos0 (zero for the hand
hinges). Nothing else in the npz is touched.

Both layouts are read from the compiled scenes (via the same Hydra configs
SBTO uses), so this stays correct if the models change -- no hard-coded
column indices.

.. code-block:: bash

    python scripts/pad_motion_for_model.py \
        datasets/robot-object/sub3_largebox_005_original.npz \
        sbto/tasks/g1/motion/sub3_largebox_005_original_rh56e2.npz \
        --src-scene box --dst-scene rh56e2_box
"""

import argparse
import os

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def build_model(scene: str) -> mujoco.MjModel:
    with initialize_config_dir(config_dir=os.path.abspath("sbto/conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"task/g1/sim/mj_scene@task.sim.mj_scene={scene}"])
    return instantiate(cfg.task.sim.mj_scene).mj_model


def joint_layout(m: mujoco.MjModel) -> dict[str, tuple[int, int]]:
    """{joint key: (qpos address, qpos width)}. Unnamed joints (e.g. the
    object's free joint added by the scene editor) are keyed by their body."""
    widths = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
              mujoco.mjtJoint.mjJNT_HINGE: 1, mujoco.mjtJoint.mjJNT_SLIDE: 1}
    layout = {}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name:
            name = f"<body:{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j])}>"
        layout[name] = (int(m.jnt_qposadr[j]), widths[mujoco.mjtJoint(m.jnt_type[j])])
    return layout


def pad_qpos(qpos: np.ndarray, m_src: mujoco.MjModel, m_dst: mujoco.MjModel) -> np.ndarray:
    src, dst = joint_layout(m_src), joint_layout(m_dst)
    if qpos.shape[1] != m_src.nq:
        raise ValueError(f"motion has {qpos.shape[1]} qpos columns but the source scene has nq={m_src.nq}")
    out = np.tile(m_dst.qpos0, (qpos.shape[0], 1))
    copied, filled = [], []
    for key, (adr_d, w) in dst.items():
        if key in src:
            adr_s, w_s = src[key]
            if w_s != w:
                raise ValueError(f"joint {key}: width {w_s} in source vs {w} in target")
            out[:, adr_d:adr_d + w] = qpos[:, adr_s:adr_s + w]
            copied.append(key)
        else:
            filled.append(key)
    missing = [k for k in src if k not in dst]
    if missing:
        raise ValueError(f"source joints absent from the target model (would be dropped): {missing}")
    print(f"copied {len(copied)} joints, filled {len(filled)} from target qpos0: {filled}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="source motion npz (qpos in the --src-scene layout)")
    p.add_argument("output", help="where to write the re-laid-out npz")
    p.add_argument("--src-scene", default="box", help="task/g1/sim/mj_scene config matching the input layout")
    p.add_argument("--dst-scene", default="rh56e2_box", help="task/g1/sim/mj_scene config of the target model")
    args = p.parse_args()

    data = dict(np.load(args.input, allow_pickle=True))
    m_src, m_dst = build_model(args.src_scene), build_model(args.dst_scene)
    data["qpos"] = pad_qpos(np.asarray(data["qpos"], dtype=np.float64), m_src, m_dst)
    np.savez(args.output, **data)
    print(f"wrote {args.output}: qpos {data['qpos'].shape}  (source nq={m_src.nq}, target nq={m_dst.nq})")


if __name__ == "__main__":
    main()
