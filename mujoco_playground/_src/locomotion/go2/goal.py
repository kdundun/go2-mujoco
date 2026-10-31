
"""Goal reaching task for Go2"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import base as go2_base
from mujoco_playground._src.locomotion.go2 import go2_constants as consts



def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      # Go2: same PD gains as Go1. Unitree's own Go2 task uses hip/thigh
      # Kp=20, Kd=1 and calf Kp=40, Kd=2 (with action_scale=0.25).
    #   Kp=[20.0, 20.0, 40.0] * 4,   
    #   Kd=[1.0, 1.0, 2.0] * 4,
    #   action_repeat=1,
    #   action_scale=0.25,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              # tracking_lin_vel=1.0,
              # tracking_ang_vel=0.5,
              # Goal.
              reach=10.0,
              bias=1.0,
              stall=-1.0,
              heading=-0.5,
              heading_track=0.0, 
              # Base reward.
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              # Other.
              dof_pos_limits=-1.0,
              pose=0.5,
              # Other.
              termination=-50.0,
              stand_still=-1.0,
              # Regularization.
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.001,
              # Feet.
              feet_clearance=-2.0,
              feet_height=-0.2,
              feet_slip=-0.1,
              feet_air_time=0.1,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.1,  # Go2: swing-foot apex target (m).
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude: [vx, vy, yaw rate].
          a=[1.5, 0.8, 1.2],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      goal_config=config_dict.create(
       dist_range=[1.0, 3.0],  # target distance range
       segment_time=5.0,       # time to reach the target
       reward_window=2.0,      # reward window for the target (T_r)
       success_radius=0.5,     # success radius for the target (R_s)
       spawn_box=[[-2.25, -1.0], [-1.5, 1.0]],    # [[xmin,ymin],[xmax,ymax]] 
       target_box=[[1.5, -1.0], [2.25, 1.0]],  # target box [[xmin,ymin],[xmax,ymax]]
       bias_off_frac=0.5,
       boxes=False,          # True: spawn and target boxes; False: spawn and target circles
       fall_height=-0.05,     # height below which the robot is considered fallen
       ring=False, 
       ring_radius=[3.8, 4.4], 
       heading_goal=False,       # True: goal includes heading; False: goal is only position
       init_yaw_range=3.14,       # initial yaw range for the robot (radians)
       cam_offset_x=0.0,         # camera offset in x direction (m)
       heading_success_tol=0.5,  # heading success tolerance (radians)
       yaw_mode="random",        # "random": target yaw ~ U(-pi, pi); "travel": along spawn->target direction
       yaw_jitter=0.5,           # +-jitter (rad) added to the travel direction in "travel" mode
       time_mode="window",       # "window": reward in the last T_r s (current); "gauss": Gaussian around a deadline
       peak_range=[4.0, 4.0],# deadline t_dl (s after segment start), sampled U(lo, hi) per segment
       time_sigma=0.5,           # Gaussian width (s) for time_mode="gauss"

      ),
      collision="model",
      naccdmax=None,   
      illegal_contact=False,
      impl="warp",
      naconmax=4 * 8192,
      njmax=40,
    #   naconmax=16 * 8192,
    #   njmax=128,
    # vision 
      vision=False,
      vision_config=config_dict.create(
          nworld=1024,
          # cam_res=(64, 64),
          cam_res=(64, 48),
          render_rgb=(False,),
          render_depth=(True,),
          use_textures=False,
          use_shadows=False,
          enabled_geom_groups=[0],
      ),
      vision_camera="depth_down",
      depth_max=1.2,

  )

def full_collision_config() -> config_dict.ConfigDict:

  cfg = default_config()
  cfg.collision = "no_self"
  cfg.illegal_contact = True
  cfg.naconmax = 16 * 8192
  cfg.njmax = 128
  return cfg

class Goal(go2_base.Go2Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if task.startswith("rough") or task.startswith("bridge") or task == "beams":
      config.naconmax = 8 * 8192
      config.njmax = 128
      config.naccdmax = 8 * 8192 
    if task.startswith("bridge"):
      config.goal_config.boxes = True
      config.goal_config.segment_time = 8.0
      config.episode_length = 400

    if task == "beams":
      config.goal_config.ring = True
      config.goal_config.segment_time = 8.0
      config.episode_length = 400
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
     # Shift the depth camera forward/backward along the body x axis.
    off = self._config.goal_config.cam_offset_x
    if off != 0.0:
      cam_id = self._mj_model.camera(self._config.vision_camera).id
      self._mj_model.cam_pos[cam_id, 0] += off
      self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)

    self._post_init()

    # rendering
    self._skip_render = False
    if self._config.vision:
      self._init_renderer()

  def _init_renderer(self) -> None:
    cam_id = self._mj_model.camera(self._config.vision_camera).id
    kwargs = self._config.vision_config.to_dict()
    kwargs["cam_active"] = [i == cam_id for i in range(self._mj_model.ncam)]
    self._rc = mjx.create_render_context(mjm=self._mj_model, **kwargs)
    self._rc_pytree = self._rc.pytree()

  def _render_depth(self, data):
    render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
    # rgb, depth_raw = mjx.render(self.mjx_model, render_data, self._rc_pytree)
    rgb, depth_raw, _ = mjx.render(self.mjx_model, render_data, self._rc_pytree)
    return mjx.get_depth(self._rc_pytree, 0, depth_raw, self._config.depth_max)

  @property
  def observation_size(self):
    if not self._config.vision:
      return super().observation_size
    self._skip_render = True
    try:
      sizes = dict(super().observation_size)
    finally:
      self._skip_render = False
    width, height = self._config.vision_config.cam_res
    sizes["pixels/depth"] = (height, width, 1)
    return sizes

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)
    goal = self._config.goal_config
    self._segment_steps = int(round(goal.segment_time / self._config.ctrl_dt))   # 5s -> 250 steps
    self._window_steps = int(round(goal.reward_window / self._config.ctrl_dt))   # 1s -> 50 steps
    self._peak_lo = int(round(goal.peak_range[0] / self._config.ctrl_dt))
    self._peak_hi = int(round(goal.peak_range[1] / self._config.ctrl_dt))

    # self._illegal_contact_sensor = [
    # self._mj_model.sensor(f"{b}_floor_found").id
    #     for b in ["base", "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]
    # ]
    self._illegal_contact_sensor = []
    if self._config.get("illegal_contact", False):
      self._illegal_contact_sensor = [
          self._mj_model.sensor(f"{b}_floor_found").id
          for b in ["base", "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]
      ]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    # dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    # qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    g = self._config.goal_config
    if g.ring:
      xy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
    elif g.boxes:
      lo, hi = jp.array(g.spawn_box[0]), jp.array(g.spawn_box[1])
      xy = jax.random.uniform(key, (2,), minval=lo, maxval=hi)
    else:
      xy = qpos[0:2] + jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(xy)

    rng, key = jax.random.split(rng)
    # yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    yaw = jax.random.uniform(key, (1,), minval=-g.init_yaw_range, maxval=g.init_yaw_range)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
        naccdmax=self._config.get("naccdmax", None),
    )
    data = mjx.forward(self.mjx_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    # rng, key1, key2 = jax.random.split(rng, 3)
    # time_until_next_cmd = jax.random.exponential(key1) * 5.0
    # steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
    #     jp.int32
    # )
    # cmd = jax.random.uniform(
    #     key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    # )
    rng, key = jax.random.split(rng)
    spawn_xy = qpos[0:2]                    
    target_xy = self._sample_target(key, spawn_xy)
    # rng, key_yaw = jax.random.split(rng)
    # target_yaw = jax.random.uniform(key_yaw, (), minval=-jp.pi, maxval=jp.pi)
    rng, key_yaw = jax.random.split(rng)
    target_yaw = self._sample_target_yaw(key_yaw, spawn_xy, target_xy)
    rng, key_peak = jax.random.split(rng)
    peak_left = self._sample_peak_left(key_peak)

    info = {
        "rng": rng,
        # "command": cmd,
        # "steps_until_next_cmd": steps_until_next_cmd,
        "spawn_xy": spawn_xy,
        "target_xy": target_xy,
        "target_yaw": target_yaw,
        "steps_left": jp.array(self._segment_steps, dtype=jp.int32),
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        "bias_on": jp.array(1.0),     
        "reach_ema": jp.array(0.0),   
        "ep_reach": jp.array(0.0),   
        "peak_left": peak_left,
        "arrived_left": jp.array(-1, dtype=jp.int32)
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    metrics["success"] = jp.zeros(())
    metrics["success_heading"] = jp.zeros(())
    metrics["final_heading_err"] = jp.zeros(())
    metrics["final_dist"] = jp.zeros(())
    metrics["bias_on"] = jp.zeros(())
    metrics["fell_off"] = jp.zeros(())
    metrics["success_peak"] = jp.zeros(())
    metrics["arrival_time"] = jp.zeros(())

    obs = self._get_obs(data, info)
    if self._config.vision and not self._skip_render:
      obs["pixels/depth"] = self._render_depth(data)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

    

  # def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
  #   qpos = state.data.qpos
  #   new_x = jp.where(jp.abs(qpos[0]) > 9.5, 0.0, qpos[0])
  #   new_y = jp.where(jp.abs(qpos[1]) > 9.5, 0.0, qpos[1])
  #   qpos = qpos.at[0:2].set(jp.array([new_x, new_y]))
  #   state = state.replace(data=state.data.replace(qpos=qpos))
  #   return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    # state = self._reset_if_outside_bounds(state)

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    # obs = self._get_obs(data, state.info)
    state.info["steps_left"] -= 1
    fell = self._get_termination(data)            
    timeout = state.info["steps_left"] <= 0        
    done = fell | timeout

    obs = self._get_obs(data, state.info)
    if self._config.vision and not self._skip_render:
      obs["pixels/depth"] = self._render_depth(data)
    # done = self._get_termination(data)

    # rewards = self._get_reward(
    #     data, action, state.info, state.metrics, done, first_contact, contact
    # )
    rewards = self._get_reward(
        data, action, state.info, state.metrics, fell, first_contact, contact
    )
    reach_raw = rewards["reach"]
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    # reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    reward = jp.clip(sum(rewards.values()) * self.dt, -10000.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    # state.info["steps_until_next_cmd"] -= 1
    # state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    # state.info["command"] = jp.where(
    #     state.info["steps_until_next_cmd"] <= 0,
    #     self.sample_command(key1, state.info["command"]),
    #     state.info["command"],
    # )
    # state.info["steps_until_next_cmd"] = jp.where(
    #     done | (state.info["steps_until_next_cmd"] <= 0),
    #     jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
    #     state.info["steps_until_next_cmd"],
    # )

    d = jp.linalg.norm(state.info["target_xy"] - data.qpos[0:2]) # distance to target
    fwd = data.xmat[self._torso_body_id][:2, 0]
    yaw = jp.arctan2(fwd[1], fwd[0])
    dpsi = jp.arctan2(jp.sin(state.info["target_yaw"] - yaw), jp.cos(state.info["target_yaw"] - yaw))

    r = self._config.goal_config.success_radius
    at_peak = state.info["steps_left"] == state.info["peak_left"]
    state.metrics["success_peak"] = (at_peak & (d < r)).astype(jp.float32)
    first = (d < r) & (state.info["arrived_left"] < 0)
    state.info["arrived_left"] = jp.where(first, state.info["steps_left"], state.info["arrived_left"])
    arr_t = (self._segment_steps - state.info["arrived_left"]) * self._config.ctrl_dt
    state.metrics["arrival_time"] = jp.where(state.info["arrived_left"] >= 0, arr_t,
                                             self._config.goal_config.segment_time) * timeout
    state.info["arrived_left"] = jp.where(done, -1, state.info["arrived_left"])


    state.info["rng"], key = jax.random.split(state.info["rng"])
    new_target = self._sample_target(key, state.info["spawn_xy"])
    state.info["target_xy"] = jp.where(done, new_target, state.info["target_xy"])
    state.info["rng"], key_yaw = jax.random.split(state.info["rng"])
    # new_yaw = jax.random.uniform(key_yaw, (), minval=-jp.pi, maxval=jp.pi)
    # state.info["target_yaw"] = jp.where(done, new_yaw, state.info["target_yaw"])
    new_yaw = self._sample_target_yaw(key_yaw, state.info["spawn_xy"], new_target)
    state.info["target_yaw"] = jp.where(done, new_yaw, state.info["target_yaw"])
    state.info["rng"], key_peak = jax.random.split(state.info["rng"])
    state.info["peak_left"] = jp.where(done, self._sample_peak_left(key_peak),          state.info["peak_left"])

    state.info["steps_left"] = jp.where(
        done, self._segment_steps, state.info["steps_left"]
    )
    
    state.info["ep_reach"] = state.info["ep_reach"] + reach_raw * self.dt
    ema = 0.95 * state.info["reach_ema"] + 0.05 * state.info["ep_reach"]
    state.info["reach_ema"] = jp.where(done, ema, state.info["reach_ema"])
    state.info["ep_reach"] = jp.where(done, 0.0, state.info["ep_reach"])
    state.info["bias_on"] = jp.where(
        state.info["reach_ema"] > self._config.goal_config.bias_off_frac,
        0.0, state.info["bias_on"])
    state.metrics["bias_on"] = state.info["bias_on"]


    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    success = timeout & ~fell & (d < self._config.goal_config.success_radius)
    state.metrics["success"] = success.astype(jp.float32)
    state.metrics["fell_off"] = (data.qpos[2] < self._config.goal_config.fall_height).astype(jp.float32)
    state.metrics["final_dist"] = d * timeout

    heading_ok = jp.abs(dpsi) < self._config.goal_config.heading_success_tol
    state.metrics["success_heading"] = (success & heading_ok).astype(jp.float32)
    state.metrics["final_heading_err"] = jp.abs(dpsi) * timeout

    done = done.astype(reward.dtype)
    state = state.replace(  # pyrefly: ignore[missing-attribute]
        data=data, obs=obs, reward=reward, done=done
    )
    return state

#   def _get_termination(self, data: mjx.Data) -> jax.Array:
#     fall_termination = self.get_upvector(data)[-1] < 0.0
#     return fall_termination

#   def _get_termination(self, data: mjx.Data) -> jax.Array:
#     fall_termination = self.get_upvector(data)[-1] < 0.0
#     illegal = jp.array([
#         data.sensordata[self._mj_model.sensor_adr[s]] > 0
#         for s in self._illegal_contact_sensor
#     ])
#     return fall_termination | jp.any(illegal)
  def _get_termination(self, data: mjx.Data) -> jax.Array:
    done = self.get_upvector(data)[-1] < 0.0
    done = done | (data.qpos[2] < self._config.goal_config.fall_height)
    if self._illegal_contact_sensor:
      illegal = jp.array([
          data.sensordata[self._mj_model.sensor_adr[s]] > 0
          for s in self._illegal_contact_sensor
      ])
      done = done | jp.any(illegal)
    return done

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    delta = info["target_xy"] - data.qpos[0:2]      
    fwd = data.xmat[self._torso_body_id][:, 0]         
    yaw = jp.arctan2(fwd[1], fwd[0])
    c, s = jp.cos(yaw), jp.sin(yaw)
    target_b = jp.array([c * delta[0] + s * delta[1],  
                         -s * delta[0] + c * delta[1]])
    # time_left = jp.array([info["steps_left"] / self._segment_steps])   
    if self._config.goal_config.time_mode == "gauss":
      time_left = jp.array([(info["steps_left"] - info["peak_left"]) / self._segment_steps])
    else:
      time_left = jp.array([info["steps_left"] / self._segment_steps])

    dpsi = info["target_yaw"] - yaw
    dpsi = jp.arctan2(jp.sin(dpsi), jp.cos(dpsi))      # normalize to [-pi, pi]
    head_obs = jp.where(self._config.goal_config.heading_goal,
                        jp.array([jp.sin(dpsi), jp.cos(dpsi)]),
                        jp.zeros(2))


    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        # info["command"],  # 3
        target_b,  # 2
        time_left,  # 1
        head_obs,  # 2
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        angvel,  # 3
        joint_angles - self._default_pose,  # 12
        joint_vel,  # 12
        data.actuator_force,  # 12
        info["last_contact"],  # 4
        feet_vel,  # 4*3
        info["feet_air_time"],  # 4
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    delta = info["target_xy"] - data.qpos[0:2]
    d = jp.linalg.norm(delta)                            
    to_target = delta / (d + 1e-6)                     
    v_xy = self.get_global_linvel(data)[:2]             
    moving = d > self._config.goal_config.success_radius
    fwd = data.xmat[self._torso_body_id][:2, 0]       
    fwd_xy = fwd / (jp.linalg.norm(fwd) + 1e-6)
    yaw = jp.arctan2(fwd[1], fwd[0])
    dpsi = info["target_yaw"] - yaw
    dpsi = jp.arctan2(jp.sin(dpsi), jp.cos(dpsi))   
    return {
        # "tracking_lin_vel": self._reward_tracking_lin_vel(
        #     info["command"], self.get_local_linvel(data)
        # ),
        # "tracking_ang_vel": self._reward_tracking_ang_vel(
        #     info["command"], self.get_gyro(data)
        # ),
        # "reach": self._reward_reach(d, info["steps_left"]),
        "reach": self._reward_reach(d, info["steps_left"], info["peak_left"]),
        # "bias": self._reward_bias(v_xy, to_target),
        "bias": self._reward_bias(v_xy, to_target) * info["bias_on"],
        "stall": self._cost_stall(moving, v_xy),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        # "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "stand_still": self._cost_stand_still(moving, data.qpos[7:]),
        # "heading": self._reward_heading(fwd_xy, to_target, moving),
        "heading": self._cost_heading(fwd_xy, to_target, moving),
        # "heading_track": self._reward_heading_track(dpsi, info["steps_left"])
        #                  * self._config.goal_config.heading_goal,
        "heading_track": self._reward_heading_track(dpsi, info["steps_left"], info["peak_left"])
                         * self._config.goal_config.heading_goal,
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        # "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_slip": self._cost_feet_slip(data, contact, moving),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, moving
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, moving
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
    }

  # Tracking rewards.

  # def _reward_tracking_lin_vel(
  #     self,
  #     commands: jax.Array,
  #     local_vel: jax.Array,
  # ) -> jax.Array:
  #   # Tracking of linear velocity commands (xy axes).
  #   lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
  #   return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  # def _reward_tracking_ang_vel(
  #     self,
  #     commands: jax.Array,
  #     ang_vel: jax.Array,
  # ) -> jax.Array:
  #   # Tracking of angular velocity commands (yaw).
  #   ang_vel_error = jp.square(commands[2] - ang_vel[2])
  #   return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Goal rewards 
  def _time_factor(self, steps_left, peak_left):
    g = self._config.goal_config
    if g.time_mode == "gauss":
      dt_s = (steps_left - peak_left) * self._config.ctrl_dt     # >0 before the deadline, <0 after
      return jp.exp(-0.5 * jp.square(dt_s / g.time_sigma)) / (g.time_sigma * jp.sqrt(2 * jp.pi))
    return (steps_left < self._window_steps) * (1.0 / g.reward_window)


  # def _reward_reach(self, d: jax.Array, steps_left: jax.Array) -> jax.Array:
  #   # Reward for reaching the target.
  #   t_r = self._config.goal_config.reward_window
  #   active = steps_left < self._window_steps
  #   return active * (1.0 / t_r) / (1.0 + jp.square(d))

  def _reward_reach(self, d, steps_left, peak_left):
    return self._time_factor(steps_left, peak_left) / (1.0 + jp.square(d))

  def _reward_heading_track(self, dpsi, steps_left, peak_left):
    return self._time_factor(steps_left, peak_left) * (1.0 - 0.5 * jp.abs(dpsi))

  def _sample_peak_left(self, rng):
    dl = jax.random.randint(rng, (), self._peak_lo, self._peak_hi + 1)   # steps after segment start
    return (self._segment_steps - dl).astype(jp.int32)               # steps_left at the deadline

  # def _reward_heading_track(self, dpsi: jax.Array, steps_left: jax.Array) -> jax.Array:
  #   # Reward for facing the target heading in the last T_r seconds
  #   t_r = self._config.goal_config.reward_window
  #   active = steps_left < self._window_steps
  #   return active * (1.0 / t_r) * (1.0 - 0.5 * jp.abs(dpsi))

  def _reward_bias(self, v_xy: jax.Array, to_target: jax.Array) -> jax.Array:
    # Reward for moving in the direction of the target.
    # return jp.dot(v_xy, to_target) / (jp.linalg.norm(v_xy) + 1e-6)
    return jp.dot(v_xy, to_target) / jp.maximum(jp.linalg.norm(v_xy), 0.1)

  def _cost_stall(self, moving: jax.Array, v_xy: jax.Array) -> jax.Array:
    # Penalize stalling when moving towards the target.
    return (moving & (jp.linalg.norm(v_xy) <  0.1)).astype(jp.float32)  # 0.1

  # def _reward_heading(self, fwd_xy, to_target, moving):
  #      return jp.dot(fwd_xy, to_target) * moving
  def _cost_heading(self, fwd_xy, to_target, moving):
       return 0.5 * (1.0 - jp.dot(fwd_xy, to_target)) * moving

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  # Other rewards.

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    # Stay close to the default pose.
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_stand_still(
      self,
      # commands: jax.Array,
      moving: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    # cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (~moving)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, moving: jax.Array
  ) -> jax.Array:
    # cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * moving

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      moving: jax.Array,
  ) -> jax.Array:
    # cmd_norm = jp.linalg.norm(info["command"])
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * moving

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, moving: jax.Array
  ) -> jax.Array:
    # Reward air time.
    # cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    # rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
    rew_air_time *= moving
    return rew_air_time

  # Perturbation and command sampling.

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
      force = (
          u_t  # (unitless)
          * self._torso_mass  # kg
          * state.info["pert_mag"]  # m/s
          / state.info["pert_duration_seconds"]  # 1/s
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)  # pyrefly: ignore[missing-attribute]
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)  # pyrefly: ignore[missing-attribute]

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )
  def _sample_target(self, rng: jax.Array, center_xy: jax.Array) -> jax.Array:
    # Sample a target position in a circular annulus around the center_xy.
    g = self._config.goal_config
    if g.ring:
      rng_a, rng_r = jax.random.split(rng)
      angle = jax.random.uniform(rng_a, minval=-jp.pi, maxval=jp.pi)
      r = jax.random.uniform(rng_r, minval=g.ring_radius[0], maxval=g.ring_radius[1])
      return r * jp.array([jp.cos(angle), jp.sin(angle)])     
    
    if g.boxes:                                 
      lo, hi = jp.array(g.target_box[0]), jp.array(g.target_box[1])
      return jax.random.uniform(rng, (2,), minval=lo, maxval=hi)
    rng_a, rng_r = jax.random.split(rng)
    angle = jax.random.uniform(rng_a, minval=-jp.pi, maxval=jp.pi)
    lo, hi = self._config.goal_config.dist_range
    r = jax.random.uniform(rng_r, minval=lo, maxval=hi)
    return center_xy + r * jp.array([jp.cos(angle), jp.sin(angle)])
  
  def _sample_target_yaw(self, rng: jax.Array, from_xy: jax.Array, target_xy: jax.Array) -> jax.Array:
    g = self._config.goal_config
    if g.yaw_mode == "travel":
      d = target_xy - from_xy
      travel = jp.arctan2(d[1], d[0])
      return travel + jax.random.uniform(rng, (), minval=-g.yaw_jitter, maxval=g.yaw_jitter)
    return jax.random.uniform(rng, (), minval=-jp.pi, maxval=jp.pi)

  
  
  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    return x_kp1
