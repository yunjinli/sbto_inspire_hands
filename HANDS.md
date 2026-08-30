# Running SBTO with hands (RH56E2 / dummy hands)

This fork adds two G1 body variants on top of upstream SBTO's stock 29-dof
"no hands" model. The stock task (`python sbto/main.py` with no overrides) is
untouched: same configs, same sensors, same cost values as upstream.

- **Inspire RH56E2** — G1 with the RH56E2 dexterous hand attached at each
  wrist: 41 actuators (29 body + 6 per hand), plus 12 passive coupled finger
  joints. The hand MJCF and meshes are private and live in the
  `third_party/g1-rh56e2-assets` submodule (branch `rl_tracking`); you need
  access to that repository.
- **Dummy hands** — the stock skeleton with the moulded rubber-hand visual
  mesh and two box collision geoms per hand. Same DOF layout as the stock
  model, so it tracks unmodified OmniRetarget references.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/yunjinli/sbto_inspire_hands.git
cd sbto_inspire_hands
# or, in an existing clone:  git submodule update --init

pixi run python sbto/main.py +experiment=rh56e2_box        # RH56E2 hands
pixi run python sbto/main.py +experiment=dummy_hands_box   # dummy box hands
pixi run python sbto/main.py                               # stock task, unchanged
```

(`pixi.toml`/`pixi.lock` pin the whole environment; plain `python3` works too
if you set up the conda environment from the main README.)

Each experiment (`sbto/conf/experiment/*.yaml`) selects the matching rollout
and reference scenes and turns on two hand-specific cost terms that default
to 0 in the task config (see "Why the source changed" below).

### Real cardboard box

Each variant has a `*_real_box` experiment that swaps the rollout object for
our physical 280 x 205 x 125 mm box, stood on its 205 x 125 footprint, while
the reference scene keeps the OmniRetarget box the demo was recorded with:

```bash
pixi run python sbto/main.py +experiment=real_box               # stock model
pixi run python sbto/main.py +experiment=dummy_hands_real_box
pixi run python sbto/main.py +experiment=rh56e2_real_box
```

These use a reference with 1.5 s of lead-in and 1 s of tail frames
(`*_leadin_tail*.npz`) so the box settles and the robot reaches a steady pose
before the grasp.

## Reference motions and padding

The RH56E2 model has more `qpos` entries than the stock model, and MuJoCo
orders them by kinematic tree -- the 12 finger joints per side sit right
*after* each wrist, not at the end. A stock-layout clip fed to the RH56E2
model therefore misaligns every column past the left wrist. Convert clips
with:

```bash
pixi run python scripts/pad_motion_for_model.py \
    datasets/robot-object/sub10_largebox_000_original.npz \
    sbto/tasks/g1/motion/sub10_largebox_000_original_rh56e2.npz \
    --src-scene box --dst-scene rh56e2_box

pixi run python sbto/main.py +experiment=rh56e2_box \
    task.cfg_ref.motion_path=sbto/tasks/g1/motion/sub10_largebox_000_original_rh56e2.npz
```

The script maps joints *by name* between the two compiled scenes (which is
why it takes scene configs, not model files -- the object's free joint only
exists after scene assembly), copies every shared joint's columns, and fills
joints the source doesn't have with the target model's rest pose. It refuses
to drop source joints silently.

### Lead-in / lead-out (time padding)

OmniRetarget clips are cut from the middle of a human capture: frame 0 is
mid-stride (peak joint velocity of the whole clip, one foot off the floor,
torso leaning over the box) and the last frame is still moving. A real robot
cannot be placed into that state, and a locomotion policy hands over an
upright, standing robot. `scripts/prepend_motion_leadin.py` eases from a
standing keyframe (default `loco`, the pose the deployed locomotion policy
actually settles into) into the demo's first frame, and from the demo's last
frame back to standing while decelerating -- smoothstep in the joints, slerp
on the base orientation, so there is no velocity step at the joins. The
lead-in/out only needs to be a tracking *cost*, not dynamically feasible;
SBTO finds the feasible way to perform it.

The shipped motions were generated exactly by:

```bash
# stock layout, 1.5 s lead-in + 1 s lead-out:
pixi run python scripts/prepend_motion_leadin.py \
    sbto/tasks/g1/motion/sub3_largebox_005_original.npz \
    -o sbto/tasks/g1/motion/sub3_largebox_005_leadin_tail.npz --tail-seconds 1.0

# then re-lay-out for the RH56E2 model:
pixi run python scripts/pad_motion_for_model.py \
    sbto/tasks/g1/motion/sub3_largebox_005_leadin_tail.npz \
    sbto/tasks/g1/motion/sub3_largebox_005_leadin_tail_rh56e2.npz \
    --src-scene box --dst-scene rh56e2_box
```

(Order matters: add the lead-in/out on the stock 43-column layout first --
`prepend_motion_leadin.py` intentionally refuses other layouts -- then convert
joints.) The horizon grows by the padded frames (193 -> 268 here), so check
the solver's `T`/knot budget on long clips.

Naming convention in `sbto/tasks/g1/motion/`:

| suffix | meaning |
|---|---|
| `_rh56e2` | joint layout converted for the RH56E2 model (`pad_motion_for_model.py`) |
| `leadin_tail` | 1.5 s lead-in + 1 s lead-out frames added (`prepend_motion_leadin.py`) |

## Why the source changed

The diff against upstream is deliberately small; every change is one of
these:

1. **Full-state layout for models with passive joints**
   (`sbto/utils/extract_ref.py`, `sbto/data/postprocess.py`). Upstream builds
   the full `[qpos, qvel]` state by concatenating root / actuated-joints /
   object blocks. That happens to match MuJoCo's layout only when every
   non-root joint is actuated. The RH56E2's passive coupled finger joints are
   interleaved after each wrist, so concatenation silently shifts every
   downstream entry -- reference states, sensors computed from them, and
   saved-trajectory reconstruction were all wrong for the hand model. Both
   paths now scatter each block to its raw MuJoCo `qpos`/`qvel` address;
   untracked passive joints get the model's rest pose.

2. **Hand-object contact sensing is model-specific**
   (`sensors/obj.xml` -> `sensors/{hand,rh56e2_hand,dummy_hand}_obj_contact.xml`).
   Upstream declared `left_hand_cnt`/`right_hand_cnt` inline as a subtree
   match from the wrist-pitch link. On a model with real hands that counts
   wrist and palm contact alike, so the contact cost cannot tell a genuine
   grip from resting the box on the wrist. The sensor now lives in one file
   per body model; the stock file keeps upstream's exact definition, so stock
   scenes are sensor-identical to upstream.

3. **Two new cost terms, off by default** (`tasks/g1/robot_object_ref.py`,
   `sensors/obj_collision.xml`): `wrist_obj_collision` penalises wrist-object
   contact (with the sensor split above, hand contact is no longer
   double-counted) and `hand_floor_collision` penalises dragging the hands on
   the floor. Both default to 0 -- the stock task's cost is bit-for-bit
   upstream's -- and the hand experiments set them to 0.25.

4. **BVH-activity bookkeeping disabled** (`sbto/sim/model_editor.py` sets
   `<visual><global bvactive="false"/>`). With it on (MuJoCo's default up to
   3.11), `mj_collision` clears a one-byte-per-BVH-node *visualization* array
   on every step. Every `<mesh>` asset gets a full-face BVH regardless of
   `contype`, so with the RH56E2's dense visual meshes (~2.4 M nodes) this
   memset dominated `mj_collision` and made multithreaded rollouts ~5x
   slower. It is pure viewer bookkeeping: physics is bit-exact with it off,
   and MuJoCo flips the default to "false" in the release after 3.11 for the
   same reason.

5. **Diverged samples must not corrupt a run** (`sbto/solvers/cem.py`,
   `sbto/data/{save,aggregate}.py`, `sbto/utils/plotting.py`). Contact-rich
   hand rollouts diverge (NaN/Inf cost) more often than the stock model.
   Non-finite samples are now excluded from elite selection and from the
   best-so-far, `save_results` falls back to the solver's best instead of
   saving a NaN "best" trajectory, and the cost plot tolerates NaN.

6. **Small fixes**: `sbto/data/load.py` joined the run directory onto an
   already-joined path, breaking `warm_start.rundir`;
   `scripts/visualize_traj.py --no-ref` now builds only the sim, so runs
   whose saved config predates a sensor file rename still replay.
