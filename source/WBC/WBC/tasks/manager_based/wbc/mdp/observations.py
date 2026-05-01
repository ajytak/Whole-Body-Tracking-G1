from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def terrain_scan_points_b(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    grid_shape: tuple[int, int],
    no_hit_value: float = 10.0,
    flatten: bool = False,
) -> torch.Tensor:
    """Ray-cast hit points in the sensor frame as a (HxWx3) scan matrix.

    Args:
        env: The environment.
        sensor_cfg: Scene entity config for the ray-caster sensor.
        grid_shape: Expected grid shape as (num_x, num_y).
        no_hit_value: Fallback value for rays with no valid hit.
        flatten: If True, return a flattened vector [B, H*W*3].

    Returns:
        Tensor with shape [B, H, W, 3] if ``flatten=False`` else [B, H*W*3].
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w

    num_x, num_y = int(grid_shape[0]), int(grid_shape[1])
    expected_num_rays = num_x * num_y
    if ray_hits_w.shape[1] != expected_num_rays:
        raise RuntimeError(
            "Scan grid shape mismatch with ray-caster pattern: "
            f"expected {expected_num_rays} rays for grid_shape={grid_shape}, got {ray_hits_w.shape[1]} rays."
        )

    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)
    relative_hits_w = ray_hits_w - sensor_pos_w

    # Some rays may miss the terrain and produce non-finite values.
    valid_mask = torch.isfinite(relative_hits_w).all(dim=-1, keepdim=True)
    safe_relative_hits_w = torch.where(valid_mask, relative_hits_w, torch.zeros_like(relative_hits_w))

    num_rays = ray_hits_w.shape[1]
    sensor_quat_w = sensor.data.quat_w.unsqueeze(1).expand(-1, num_rays, -1)
    relative_hits_b = quat_apply_inverse(
        sensor_quat_w.reshape(-1, 4), safe_relative_hits_w.reshape(-1, 3)
    ).reshape(env.num_envs, num_rays, 3)

    if torch.any(~valid_mask):
        fallback_hits_b = torch.full_like(relative_hits_b, fill_value=no_hit_value)
        fallback_hits_b[..., 2] = 0.0
        relative_hits_b = torch.where(valid_mask.expand_as(relative_hits_b), relative_hits_b, fallback_hits_b)

    scan_matrix = relative_hits_b.view(env.num_envs, num_x, num_y, 3)
    if flatten:
        return scan_matrix.reshape(env.num_envs, -1)
    return scan_matrix


def terrain_scan_points_b_flat(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    grid_shape: tuple[int, int],
    no_hit_value: float = 10.0,
) -> torch.Tensor:
    """Flattened version of :func:`terrain_scan_points_b` for MLP-style policies."""
    return terrain_scan_points_b(
        env=env,
        sensor_cfg=sensor_cfg,
        grid_shape=grid_shape,
        no_hit_value=no_hit_value,
        flatten=True,
    )

def goal_position_b(
    env: ManagerBasedEnv,
    goal_offset: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Goal position expressed in the robot base frame.

    The goal is specified as an offset from each environment origin.
    """
    asset = env.scene[asset_cfg.name]
    goal_offset_t = torch.tensor(goal_offset, device=env.device, dtype=asset.data.root_pos_w.dtype).unsqueeze(0)
    goal_pos_w = env.scene.env_origins + goal_offset_t
    goal_vec_w = goal_pos_w - asset.data.root_pos_w
    return quat_apply_inverse(asset.data.root_quat_w, goal_vec_w)

def robot_body_pos_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body positions in base frame. Shape: (num_envs, num_bodies * 3) or raw (..., 3) if caller flattens later."""
    asset: Articulation = env.scene[asset_cfg.name]

    root_pos_w = asset.data.root_pos_w[:, None, :]              # (N, 1, 3)
    root_quat_w = asset.data.root_quat_w[:, None, :]            # (N, 1, 4)

    body_pos_w = asset.data.body_pos_w                          # (N, B, 3)

    body_pos_b = quat_apply(
        quat_inv(root_quat_w).expand(-1, body_pos_w.shape[1], -1),
        body_pos_w - root_pos_w,
    )
    return body_pos_b.reshape(body_pos_b.shape[0], -1)


def robot_body_ori_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body orientations relative to base frame. Returns quaternions. Shape: (num_envs, num_bodies * 4)."""
    asset: Articulation = env.scene[asset_cfg.name]

    root_quat_w = asset.data.root_quat_w[:, None, :]            # (N, 1, 4)
    body_quat_w = asset.data.body_quat_w                        # (N, B, 4)

    body_quat_b = quat_mul(
        quat_inv(root_quat_w).expand(-1, body_quat_w.shape[1], -1),
        body_quat_w,
    )
    return body_quat_b.reshape(body_quat_b.shape[0], -1)


def robot_body_lin_vel_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body linear velocities in base frame. Shape: (num_envs, num_bodies * 3)."""
    asset: Articulation = env.scene[asset_cfg.name]

    root_quat_w = asset.data.root_quat_w[:, None, :]            # (N, 1, 4)
    body_lin_vel_w = asset.data.body_lin_vel_w                  # (N, B, 3)

    body_lin_vel_b = quat_apply(
        quat_inv(root_quat_w).expand(-1, body_lin_vel_w.shape[1], -1),
        body_lin_vel_w,
    )
    return body_lin_vel_b.reshape(body_lin_vel_b.shape[0], -1)


def robot_body_ang_vel_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body angular velocities in base frame. Shape: (num_envs, num_bodies * 3)."""
    asset: Articulation = env.scene[asset_cfg.name]

    root_quat_w = asset.data.root_quat_w[:, None, :]            # (N, 1, 4)
    body_ang_vel_w = asset.data.body_ang_vel_w                  # (N, B, 3)

    body_ang_vel_b = quat_apply(
        quat_inv(root_quat_w).expand(-1, body_ang_vel_w.shape[1], -1),
        body_ang_vel_w,
    )
    return body_ang_vel_b.reshape(body_ang_vel_b.shape[0], -1)


def _resolve_body_indices(
    asset: Articulation,
    body_names: Sequence[str] | None,
) -> torch.Tensor:
    if body_names is None:
        return torch.arange(asset.data.body_pos_w.shape[1], device=asset.data.body_pos_w.device, dtype=torch.long)

    body_ids, resolved_names = asset.find_bodies(body_names)
    if len(body_ids) != len(body_names):
        raise ValueError(f"Failed to resolve all body names. Requested={body_names}, resolved={resolved_names}")
    return torch.as_tensor(body_ids, device=asset.data.body_pos_w.device, dtype=torch.long)


def _resolve_single_body_index(asset: Articulation, body_name: str) -> int:
    body_ids, resolved_names = asset.find_bodies([body_name])
    if len(body_ids) != 1:
        raise ValueError(f"Failed to resolve body `{body_name}`. Resolved={resolved_names}")
    return int(body_ids[0])


def _reorder_bodies(x: torch.Tensor, body_ids: torch.Tensor) -> torch.Tensor:
    return x.index_select(dim=1, index=body_ids)


def _canonicalize_amp_state_from_pelvis(
    pelvis_pos_w: torch.Tensor,
    pelvis_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
):
    B = body_pos_w.shape[1]

    pelvis_pos_w = pelvis_pos_w[:, None, :]                      # (N, 1, 3)
    pelvis_yaw_quat_w = yaw_quat(pelvis_quat_w).unsqueeze(1)    # (N, 1, 4)

    pelvis_pos_w = pelvis_pos_w.expand(-1, B, -1)
    pelvis_yaw_quat_w = pelvis_yaw_quat_w.expand(-1, B, -1)
    inv_pelvis_yaw_quat_w = quat_inv(pelvis_yaw_quat_w)

    body_pos_amp = quat_apply(inv_pelvis_yaw_quat_w, body_pos_w - pelvis_pos_w)
    body_quat_amp = quat_mul(inv_pelvis_yaw_quat_w, body_quat_w)
    body_lin_vel_amp = quat_apply(inv_pelvis_yaw_quat_w, body_lin_vel_w)
    body_ang_vel_amp = quat_apply(inv_pelvis_yaw_quat_w, body_ang_vel_w)

    return (
        joint_pos,
        joint_vel,
        body_pos_amp.reshape(body_pos_amp.shape[0], -1),
        body_quat_amp.reshape(body_quat_amp.shape[0], -1),
        body_lin_vel_amp.reshape(body_lin_vel_amp.shape[0], -1),
        body_ang_vel_amp.reshape(body_ang_vel_amp.shape[0], -1),
    )

def robot_amp_joint_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def robot_amp_joint_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def robot_amp_body_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolve_body_indices(asset, asset_cfg.body_names)
    pelvis_id = _resolve_single_body_index(asset, "pelvis")

    _, _, body_pos_amp, _, _, _ = _canonicalize_amp_state_from_pelvis(
        pelvis_pos_w=asset.data.body_pos_w[:, pelvis_id, :],
        pelvis_quat_w=asset.data.body_quat_w[:, pelvis_id, :],
        body_pos_w=_reorder_bodies(asset.data.body_pos_w, body_ids),
        body_quat_w=_reorder_bodies(asset.data.body_quat_w, body_ids),
        body_lin_vel_w=_reorder_bodies(asset.data.body_lin_vel_w, body_ids),
        body_ang_vel_w=_reorder_bodies(asset.data.body_ang_vel_w, body_ids),
        joint_pos=asset.data.joint_pos,
        joint_vel=asset.data.joint_vel,
    )
    return body_pos_amp


def robot_amp_body_ori(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolve_body_indices(asset, asset_cfg.body_names)
    pelvis_id = _resolve_single_body_index(asset, "pelvis")

    _, _, _, body_quat_amp, _, _ = _canonicalize_amp_state_from_pelvis(
        pelvis_pos_w=asset.data.body_pos_w[:, pelvis_id, :],
        pelvis_quat_w=asset.data.body_quat_w[:, pelvis_id, :],
        body_pos_w=_reorder_bodies(asset.data.body_pos_w, body_ids),
        body_quat_w=_reorder_bodies(asset.data.body_quat_w, body_ids),
        body_lin_vel_w=_reorder_bodies(asset.data.body_lin_vel_w, body_ids),
        body_ang_vel_w=_reorder_bodies(asset.data.body_ang_vel_w, body_ids),
        joint_pos=asset.data.joint_pos,
        joint_vel=asset.data.joint_vel,
    )
    return body_quat_amp


def robot_amp_body_lin_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolve_body_indices(asset, asset_cfg.body_names)
    pelvis_id = _resolve_single_body_index(asset, "pelvis")

    _, _, _, _, body_lin_vel_amp, _ = _canonicalize_amp_state_from_pelvis(
        pelvis_pos_w=asset.data.body_pos_w[:, pelvis_id, :],
        pelvis_quat_w=asset.data.body_quat_w[:, pelvis_id, :],
        body_pos_w=_reorder_bodies(asset.data.body_pos_w, body_ids),
        body_quat_w=_reorder_bodies(asset.data.body_quat_w, body_ids),
        body_lin_vel_w=_reorder_bodies(asset.data.body_lin_vel_w, body_ids),
        body_ang_vel_w=_reorder_bodies(asset.data.body_ang_vel_w, body_ids),
        joint_pos=asset.data.joint_pos,
        joint_vel=asset.data.joint_vel,
    )
    return body_lin_vel_amp


def robot_amp_body_ang_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolve_body_indices(asset, asset_cfg.body_names)
    pelvis_id = _resolve_single_body_index(asset, "pelvis")

    _, _, _, _, _, body_ang_vel_amp = _canonicalize_amp_state_from_pelvis(
        pelvis_pos_w=asset.data.body_pos_w[:, pelvis_id, :],
        pelvis_quat_w=asset.data.body_quat_w[:, pelvis_id, :],
        body_pos_w=_reorder_bodies(asset.data.body_pos_w, body_ids),
        body_quat_w=_reorder_bodies(asset.data.body_quat_w, body_ids),
        body_lin_vel_w=_reorder_bodies(asset.data.body_lin_vel_w, body_ids),
        body_ang_vel_w=_reorder_bodies(asset.data.body_ang_vel_w, body_ids),
        joint_pos=asset.data.joint_pos,
        joint_vel=asset.data.joint_vel,
    )
    return body_ang_vel_amp


def robot_amp_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolve_body_indices(asset, asset_cfg.body_names)
    pelvis_id = _resolve_single_body_index(asset, "pelvis")

    joint_pos, joint_vel, body_pos_amp, body_quat_amp, body_lin_vel_amp, body_ang_vel_amp = (
        _canonicalize_amp_state_from_pelvis(
            pelvis_pos_w=asset.data.body_pos_w[:, pelvis_id, :],
            pelvis_quat_w=asset.data.body_quat_w[:, pelvis_id, :],
            body_pos_w=_reorder_bodies(asset.data.body_pos_w, body_ids),
            body_quat_w=_reorder_bodies(asset.data.body_quat_w, body_ids),
            body_lin_vel_w=_reorder_bodies(asset.data.body_lin_vel_w, body_ids),
            body_ang_vel_w=_reorder_bodies(asset.data.body_ang_vel_w, body_ids),
            joint_pos=asset.data.joint_pos,
            joint_vel=asset.data.joint_vel,
        )
    )

    return torch.cat(
        [joint_pos, joint_vel, body_pos_amp, body_quat_amp, body_lin_vel_amp, body_ang_vel_amp],
        dim=-1,
    )