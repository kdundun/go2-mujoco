# Go2 Goal Reaching with a Depth Camera

## What was changed
Built on [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).
All changes are for the Go2. The main file is `mujoco_playground/_src/locomotion/go2/goal.py` (the goal-reaching environment).
Terrains and the depth camera are in `mujoco_playground/_src/locomotion/go2/xmls/`.

## Task

- **Observation:** target position (dx, dy) in the robot's yaw frame and the fraction of time remaining (replaces the velocity command).
- **Camera settings:** A = blind; B = depth image showing terrain only (`enabled_geom_groups=[0]`); C = depth image showing terrain and feet (`[0, 3]`).
- **Reward:** adapted from Rudin et al., IROS 2022 (task reward, exploration bias, stall penalty), plus a heading penalty. All other Playground Go2 regularization terms are kept.

## Terrains

| Terrain | Layout | Time |
|---|---|---|
| Flat | target 1–5 m away, random direction | 6 s |
| Bridge | two 5×8 m platforms, 1.5 m gap (1 m deep), 1.0 m wide bridge; target 3.0–4.9 m away | 8 s |
| Beams | spawn at center, target on outer ring (3.8–4.4 m), 12 beams 0.8 m wide | 8 s |

The heightfield PNGs are in `mujoco_playground/_src/locomotion/go2/xmls/assets/`.

## Reproduce

Training uses a curriculum: Flat (100M steps) → Bridge (50M) → Beams (50M). Each stage starts from the previous stage's final checkpoint.
All settings (steps, learning rate, γ, overrides) are in the YAML files.
```bash
python run_curriculum.py curriculum/a_flat_bridge_beams.yaml   # A: blind
python run_curriculum.py curriculum/b_flat_bridge_beams.yaml   # B: depth, terrain only
python run_curriculum.py curriculum/c_flat_bridge_beams.yaml   # C: depth, terrain + feet
```

## Playback

**notebook.** Open `play_go2.ipynb` from the repository root and run the cells. Change `NAME` (e.g. `"bridge_C"`) to choose a run. It opens an interactive MuJoCo viewer (N = new target, K = restart) and can record videos with the depth image shown on the right.

**command line.** Render a video with the training script (no training):
```bash
python learning/train_jax_ppo.py --env_name=Go2GoalBeams --impl=warp --vision \
  --playground_config_overrides='{"vision_config.enabled_geom_groups": [0, 3]}' \
  --load_checkpoint_path=$(ls -dt logs/*-beams_c_heading | head -1)/checkpoints \
  --play_only --num_videos=1
```
The video is saved as `rollout0.mp4` in a new folder under `logs/`.
For A, remove `--vision` and the override. For B, use `[0]`.