# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def _goal_distance_xy(
    env: ManagerBasedRLEnv, goal_offset: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    goal_offset_t = torch.tensor(goal_offset, device=env.device, dtype=asset.data.root_pos_w.dtype).unsqueeze(0)
    goal_pos_w = env.scene.env_origins + goal_offset_t
    return torch.norm(goal_pos_w[:, :2] - asset.data.root_pos_w[:, :2], dim=-1)

def goal_position_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    goal_offset: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for getting closer to the goal on the far side of the gaps."""
    distance_xy = _goal_distance_xy(env, goal_offset, asset_cfg)
    return 1.0 - torch.tanh(distance_xy / std)

    
def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    goal_offset: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse success reward when the robot reaches the goal."""
    distance_xy = _goal_distance_xy(env, goal_offset, asset_cfg)
    return (distance_xy < threshold).float()
