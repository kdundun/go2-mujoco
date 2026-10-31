# Copyright 2026 Google LLC (derived from the Go1 environment in MuJoCo Playground)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines Unitree Go2 quadruped constants.

Robot model: MuJoCo Menagerie unitree_go2/go2_mjx.xml (unmodified), loaded by
xmls/scene_mjx_flat_terrain.xml. Fixes applied in base.py: knee torque limit,
actuator damping removed (Kd = config.Kd).

Joint / actuator order is FL, FR, RL, RR (Menagerie order).
The Unitree SDK and Kine2Go use FR, FL, RR, RL, so reorder when deploying
or comparing. Foot sites / geoms below are looked up by name, so their
order (FR, FL, RR, RL) is independent of the joint order.
"""


from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go2"
# FEET_ONLY_FLAT_TERRAIN_XML = (
#     ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
# )
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
FEET_ONLY_BRIDGE_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_bridge.xml"
FEET_ONLY_BRIDGE_FILLED_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_bridge_filled.xml"
FEET_ONLY_BEAMS_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_beams.xml"



def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,      
      "flat_terrain_full": FULL_FLAT_TERRAIN_XML,      
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
      "bridge": FEET_ONLY_BRIDGE_XML,
      "bridge_filled": FEET_ONLY_BRIDGE_FILLED_XML,
      "beams": FEET_ONLY_BEAMS_XML,
  }[task_name]


FEET_SITES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]   

# FEET_SITES = [
#     "FR",
#     "FL",
#     "RR",
#     "RL",
# ]

FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base"  # Go1 calls this "trunk".

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
