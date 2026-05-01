from __future__ import annotations

from collections.abc import Sequence

import torch
import os
import numpy as np

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

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

class AmpMotionLoader:
    """Minimal AMP motion loader.

    Assumptions:
    - accepts one or more npz files
    - each file is stacked format only
    - all files have same fps
    - if body_names exist in file, reorder to requested body_names
    - if body_names do not exist in file, assumes file is already in requested order
    """

    def __init__(self, motion_files: str | Sequence[str], body_indices: Sequence[int], device: str = "cpu"):
        if isinstance(motion_files, (str, os.PathLike)):
            motion_files = [str(motion_files)]
        else:
            motion_files = [str(f) for f in motion_files]

        if len(motion_files) == 0:
            raise ValueError("No motion files provided.")

        for motion_file in motion_files:
            if not os.path.isfile(motion_file):
                raise ValueError(f"Invalid file path: {motion_file}")

        self.device = torch.device(device)
        self._body_indices = list(body_indices)
        self._pelvis_idx = 0

        joint_pos_list = []
        joint_vel_list = []
        body_pos_list = []
        body_quat_list = []
        body_lin_vel_list = []
        body_ang_vel_list = []
        traj_lens = []
        fps_list = []

        joint_dim = None

        for motion_file in motion_files:
            with np.load(motion_file, allow_pickle=True) as data:
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                fps_list.append(fps)

                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
                body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
                body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
                body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
                body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

                # allow single trajectory file without leading trajectory dim
                if joint_pos.ndim == 2:
                    joint_pos = joint_pos[None]
                    joint_vel = joint_vel[None]
                    body_pos_w = body_pos_w[None]
                    body_quat_w = body_quat_w[None]
                    body_lin_vel_w = body_lin_vel_w[None]
                    body_ang_vel_w = body_ang_vel_w[None]

                num_traj = joint_pos.shape[0]

                if joint_dim is None:
                    joint_dim = joint_pos.shape[-1]
                elif joint_pos.shape[-1] != joint_dim:
                    raise ValueError("Joint dim mismatch across files.")

                # if body_pos_w.shape[2] != len(self._body_names):
                #     raise ValueError(
                #         f"Body count mismatch. File has {body_pos_w.shape[2]}, expected {len(self._body_names)}."
                #     )

                order = np.asarray(self._body_indices, dtype=np.int64)

                if np.any(order < 0) or np.any(order >= body_pos_w.shape[2]):
                    raise ValueError(
                        f"Invalid body_indices {self._body_indices} for motion file with {body_pos_w.shape[2]} bodies."
                    )

                body_pos_w = body_pos_w[:, :, order, :]
                body_quat_w = body_quat_w[:, :, order, :]
                body_lin_vel_w = body_lin_vel_w[:, :, order, :]
                body_ang_vel_w = body_ang_vel_w[:, :, order, :]

                for i in range(num_traj):
                    joint_pos_list.append(joint_pos[i])
                    joint_vel_list.append(joint_vel[i])
                    body_pos_list.append(body_pos_w[i])
                    body_quat_list.append(body_quat_w[i])
                    body_lin_vel_list.append(body_lin_vel_w[i])
                    body_ang_vel_list.append(body_ang_vel_w[i])
                    traj_lens.append(joint_pos[i].shape[0])

        if not all(abs(f - fps_list[0]) < 1.0e-6 for f in fps_list):
            raise ValueError(f"All motion files must have same fps. Got {fps_list}")

        self.fps = fps_list[0]
        self.dt = 1.0 / self.fps

        self.trajectory_lengths = torch.tensor(traj_lens, dtype=torch.long, device=self.device)
        self.num_trajectories = int(self.trajectory_lengths.numel())

        self._traj_starts = torch.zeros(self.num_trajectories, dtype=torch.long, device=self.device)
        if self.num_trajectories > 1:
            self._traj_starts[1:] = torch.cumsum(self.trajectory_lengths[:-1], dim=0)

        self.joint_pos = self._concat(joint_pos_list)
        self.joint_vel = self._concat(joint_vel_list)
        self.body_pos_w = self._concat(body_pos_list)
        self.body_quat_w = self._concat(body_quat_list)
        self.body_lin_vel_w = self._concat(body_lin_vel_list)
        self.body_ang_vel_w = self._concat(body_ang_vel_list)

    @property
    def body_indices(self) -> list[int]:
        return self._body_indices

    @property
    def num_bodies(self) -> int:
        return len(self._body_indices)

    def _concat(self, arrays: list[np.ndarray]) -> torch.Tensor:
        return torch.cat([torch.tensor(a, dtype=torch.float32, device=self.device) for a in arrays], dim=0)

    def _flat_index(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        traj = torch.as_tensor(trajectory_ids, dtype=torch.long, device=self.device)
        step = torch.as_tensor(time_steps, dtype=torch.long, device=self.device)
        step = torch.minimum(torch.clamp(step, min=0), self.trajectory_lengths[traj] - 1)
        return self._traj_starts[traj] + step

    def sample_frames(
        self,
        trajectory_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self._flat_index(trajectory_ids, time_steps)
        return (
            self.joint_pos[idx],
            self.joint_vel[idx],
            self.body_pos_w[idx],
            self.body_quat_w[idx],
            self.body_lin_vel_w[idx],
            self.body_ang_vel_w[idx],
        )

    def sample_amp_frames(
        self,
        trajectory_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = self.sample_frames(
            trajectory_ids, time_steps
        )

        return _canonicalize_amp_state_from_pelvis(
            pelvis_pos_w=body_pos_w[:, self._pelvis_idx, :],
            pelvis_quat_w=body_quat_w[:, self._pelvis_idx, :],
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

    def sample_random_time_indices(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        traj_ids = torch.randint(0, self.num_trajectories, (num_samples,), device=self.device)
        lengths = self.trajectory_lengths[traj_ids]
        time_steps = torch.floor(torch.rand(num_samples, device=self.device) * lengths.float()).long()
        time_steps = torch.minimum(time_steps, lengths - 1)
        return traj_ids, time_steps

    def sample_random_frames(
        self,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj_ids, time_steps = self.sample_random_time_indices(num_samples)
        return self.sample_frames(traj_ids, time_steps)

    def sample_random_amp_frames(
        self,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj_ids, time_steps = self.sample_random_time_indices(num_samples)
        return self.sample_amp_frames(traj_ids, time_steps)

    def sample_random_amp_state(self, num_samples: int) -> torch.Tensor:
        joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel = self.sample_random_amp_frames(
            num_samples
        )
        return torch.cat([joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel], dim=-1)