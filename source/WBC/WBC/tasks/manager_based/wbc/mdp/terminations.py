from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

def base_height_below(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if robot base height drops below threshold (fall condition)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < threshold

def goal_reached(
    env: ManagerBasedRLEnv,
    threshold: float,
    goal_offset: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successful episodes when robot reaches the goal in XY."""
    asset: Articulation = env.scene[asset_cfg.name]
    goal_offset_t = torch.tensor(goal_offset, device=env.device, dtype=asset.data.root_pos_w.dtype).unsqueeze(0)
    goal_pos_w = env.scene.env_origins + goal_offset_t
    distance_xy = torch.norm(goal_pos_w[:, :2] - asset.data.root_pos_w[:, :2], dim=-1)
    return distance_xy < threshold