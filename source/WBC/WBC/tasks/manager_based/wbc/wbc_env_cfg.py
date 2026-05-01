from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.terrains as terrain_gen
from isaaclab.sensors.ray_caster import patterns

import WBC.tasks.manager_based.wbc.mdp as mdp
from WBC.tasks.manager_based.wbc.terrains.gap_course_terrain import GapCourseTerrainCfg
from WBC.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from WBC.tasks.manager_based.wbc.motions.motion_loader import AmpMotionLoader, _resolve_body_indices


# -----------------------------------------------------------------------------
# Constants / helpers
# -----------------------------------------------------------------------------

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class HighLevelScanCfg:
    """Configuration for a local terrain scan around the robot base."""

    front: float = 0.5
    back: float = 0.5
    left: float = 0.5
    right: float = 0.5
    resolution: float = 0.05
    include_endpoints: bool = False

    sensor_height: float = 1.0
    max_distance: float = 5.0
    no_hit_value: float = 5.0


def _resolve_scan_axis_bins(
    span: float,
    resolution: float,
    include_endpoints: bool,
    axis_name: str,
) -> tuple[int, float]:
    if span <= 0.0:
        raise ValueError(f"Scan span along {axis_name}-axis must be positive. Received: {span}.")
    if resolution <= 0.0:
        raise ValueError(f"Scan resolution must be positive. Received: {resolution}.")

    ratio = span / resolution
    bins = int(round(ratio))
    if bins <= 0 or abs(ratio - bins) > 1.0e-6:
        raise ValueError(
            f"Scan span/resolution mismatch on {axis_name}-axis: span={span}, resolution={resolution}. "
            "Expected span to be an integer multiple of resolution."
        )

    if include_endpoints:
        return bins + 1, span
    return bins, span - resolution


def resolve_scan_grid_config(
    scan_cfg: HighLevelScanCfg,
) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float]]:
    """Resolve scan-grid shape, ray-pattern size, and sensor XY offset from scan config."""
    span_x = scan_cfg.front + scan_cfg.back
    span_y = scan_cfg.left + scan_cfg.right

    num_x, pattern_size_x = _resolve_scan_axis_bins(
        span=span_x,
        resolution=scan_cfg.resolution,
        include_endpoints=scan_cfg.include_endpoints,
        axis_name="x",
    )
    num_y, pattern_size_y = _resolve_scan_axis_bins(
        span=span_y,
        resolution=scan_cfg.resolution,
        include_endpoints=scan_cfg.include_endpoints,
        axis_name="y",
    )

    offset_x = 0.5 * (scan_cfg.front - scan_cfg.back)
    offset_y = 0.5 * (scan_cfg.left - scan_cfg.right)

    return (num_x, num_y), (pattern_size_x, pattern_size_y), (offset_x, offset_y)

# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

@configclass
class G1AmpGapSceneCfg(InteractiveSceneCfg):
    """Scene with robot, gap terrain, contacts, and scan sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    robot: ArticulationCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.13, 0.13, 0.13),
            intensity=1000.0,
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )

    terrain_scan: RayCasterCfg | None = None


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Single policy directly outputs robot joint targets."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
    )


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """Policy gets scan + goal + proprio + body states."""

    @configclass
    class PolicyCfg(ObsGroup):
        # local terrain perception
        scan_points_flat = ObsTerm(
            func=mdp.terrain_scan_points_b_flat,
            params={
                "sensor_cfg": SceneEntityCfg("terrain_scan"),
                "grid_shape": (20, 20),   # overwritten in env __post_init__
                "no_hit_value": 5.0,      # overwritten in env __post_init__
            },
        )

        # task command
        goal_pos_b = ObsTerm(
            func=mdp.goal_position_b,
            params={
                "goal_offset": (2.0, 0.0, 0.0),  # overwritten in env __post_init__
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # base proprioception
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # body states
        # body_pos = ObsTerm(func=mdp.robot_body_pos_b)
        # body_ori = ObsTerm(func=mdp.robot_body_ori_b)
        # body_lin_vel = ObsTerm(func=mdp.robot_body_lin_vel_b)
        # body_ang_vel = ObsTerm(func=mdp.robot_body_ang_vel_b)

        # autoregressive context
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        scan_points_flat = ObsTerm(
            func=mdp.terrain_scan_points_b_flat,
            params={
                "sensor_cfg": SceneEntityCfg("terrain_scan"),
                "grid_shape": (20, 20),
                "no_hit_value": 5.0,
            },
        )
        goal_pos_b = ObsTerm(
            func=mdp.goal_position_b,
            params={
                "goal_offset": (2.0, 0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b)
        body_ori = ObsTerm(func=mdp.robot_body_ori_b)
        body_lin_vel = ObsTerm(func=mdp.robot_body_lin_vel_b)
        body_ang_vel = ObsTerm(func=mdp.robot_body_ang_vel_b)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Optional: separate AMP discriminator obs group if your runner expects one.
    # If your AMP implementation already builds discriminator observations elsewhere,
    # you can remove this whole group.
    @configclass
    class AmpCfg(ObsGroup):
        amp_state = ObsTerm(
            func=mdp.robot_amp_state,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "pelvis",
                        "left_hip_roll_link",
                        "left_knee_link",
                        "left_ankle_roll_link",
                        "right_hip_roll_link",
                        "right_knee_link",
                        "right_ankle_roll_link",
                        "torso_link",
                        "left_shoulder_roll_link",
                        "left_elbow_link",
                        "left_wrist_yaw_link",
                        "right_shoulder_roll_link",
                        "right_elbow_link",
                        "right_wrist_yaw_link",
                    ],
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 2

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpCfg = AmpCfg()


@configclass
class AMPCfg:
    motion_files: tuple[str, ...] = (
        "/workspace/isaac_projects/WBC/source/WBC/WBC/tasks/manager_based/wbc/motions/run/motion.npz",
        "/workspace/isaac_projects/WBC/source/WBC/WBC/tasks/manager_based/wbc/motions/jump/motion.npz",
    )
    body_names: tuple[str, ...] = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    num_amp_observations: int = 2
    amp_observation_space: int = (
        29 + 29 + 14 * 3 + 14 * 4 + 14 * 3 + 14 * 3
    )  # joint_pos + joint_vel + body_pos + body_quat + body_lin_vel + body_ang_vel

# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------

@configclass
class EventCfg:
    """Domain randomization + reset events."""

    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.3, 1.6),
    #         "dynamic_friction_range": (0.3, 1.2),
    #         "restitution_range": (0.0, 0.5),
    #         "num_buckets": 64,
    #     },
    # )

    # add_joint_default_pos = EventTerm(
    #     func=mdp.randomize_joint_default_pos,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "pos_distribution_params": (-0.01, 0.01),
    #         "operation": "add",
    #     },
    # )

    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
    #     },
    # )

    # reset events for stable task initialization
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Goal-task rewards from your hierarchical env, plus optional AMP rewards."""

    goal_distance = RewTerm(
        func=mdp.goal_position_error_tanh,
        weight=4.0,
        params={
            "std": 1.0,                     # overwritten in env __post_init__
            "goal_offset": (2.0, 0.0, 0.0), # overwritten in env __post_init__
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    goal_reached_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=6.0,
        params={
            "threshold": 0.35,              # overwritten in env __post_init__
            "goal_offset": (2.0, 0.0, 0.0), # overwritten in env __post_init__
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={
            "threshold": 0.35,               # overwritten in env __post_init__
            "goal_offset": (2.0, 0.0, 0.0),  # overwritten in env __post_init__
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    base_fall = DoneTerm(
        func=mdp.base_height_below,
        params={
            "threshold": 0.2,                # overwritten in env __post_init__
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class CurriculumCfg:
    pass


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

@configclass
class G1AmpGapGoalEnvCfg(ManagerBasedRLEnvCfg):
    """Single-policy G1 env for AMP on gap terrain with goal navigation."""

    scene: G1AmpGapSceneCfg = G1AmpGapSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    amp: AMPCfg = AMPCfg()
    actions: ActionsCfg = ActionsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # task-specific configs
    high_level_scan: HighLevelScanCfg = HighLevelScanCfg()

    gap_course: GapCourseTerrainCfg = GapCourseTerrainCfg(
        proportion=1.0,
        size=(8.0, 4.0),
        num_gaps=1,
        gap_width_range=(0.4, 0.4),
        gap_depth=1.0,
        first_gap_center_x=1.0,
        gap_center_spacing=1.5,
        gap_centers_x=None,
        gap_y_span=None,
        gap_y_center_offset=0.0,
        surface_thickness=1.0,
        floor_thickness=1.0,
    )

    terrain_num_rows: int = 8
    terrain_num_cols: int = 8
    terrain_curriculum: bool = False
    terrain_max_init_level: int | None = 0

    goal_margin_after_last_gap: float = 1.0
    goal_lateral_offset: float = 0.0
    goal_reward_std: float = 1.0
    goal_reached_threshold: float = 0.35
    base_height_termination_threshold: float = 0.2

    def _resolve_goal_offset(self) -> tuple[float, float, float]:
        if self.gap_course.gap_centers_x is not None and len(self.gap_course.gap_centers_x) > 0:
            last_gap_center_x = float(max(self.gap_course.gap_centers_x[: self.gap_course.num_gaps]))
        else:
            last_gap_center_x = float(
                self.gap_course.first_gap_center_x
                + (self.gap_course.num_gaps - 1) * self.gap_course.gap_center_spacing
            )

        max_gap_width = float(self.gap_course.gap_width_range[1])
        goal_x = last_gap_center_x + 0.5 * max_gap_width + self.goal_margin_after_last_gap
        max_goal_x = 0.5 * float(self.gap_course.size[0]) - 0.5
        goal_x = float(min(max(goal_x, 0.0), max_goal_x))

        goal_y = float(self.gap_course.gap_y_center_offset + self.goal_lateral_offset)
        max_abs_goal_y = 0.5 * float(self.gap_course.size[1]) - 0.25
        goal_y = float(max(min(goal_y, max_abs_goal_y), -max_abs_goal_y))

        return (goal_x, goal_y, 0.0)

    def __post_init__(self):
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # viewer
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        # terrain replacement: plane -> generator
        terrain_physics_material = self.scene.terrain.physics_material
        terrain_visual_material = self.scene.terrain.visual_material
        terrain_size = self.gap_course.size

        self.scene.terrain = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_gen.TerrainGeneratorCfg(
                curriculum=self.terrain_curriculum,
                size=terrain_size,
                border_width=0.0,
                num_rows=self.terrain_num_rows,
                num_cols=self.terrain_num_cols,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={"gap_course": self.gap_course},
            ),
            max_init_terrain_level=self.terrain_max_init_level,
            collision_group=-1,
            physics_material=terrain_physics_material,
            visual_material=terrain_visual_material,
            debug_vis=False,
        )

        self.sim.physics_material = self.scene.terrain.physics_material

        # terrain scan
        grid_shape, pattern_size, offset_xy = resolve_scan_grid_config(self.high_level_scan)

        self.scene.terrain_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(
                pos=(offset_xy[0], offset_xy[1], self.high_level_scan.sensor_height),
            ),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=self.high_level_scan.resolution,
                size=pattern_size,
                ordering="xy",
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            max_distance=self.high_level_scan.max_distance,
        )

        if self.scene.contact_forces is not None:
            self.scene.terrain_scan.update_period = self.scene.contact_forces.update_period
        else:
            self.scene.terrain_scan.update_period = self.sim.dt

        # goal
        goal_offset = self._resolve_goal_offset()

        # observation patching
        self.observations.policy.scan_points_flat.params["grid_shape"] = grid_shape
        self.observations.policy.scan_points_flat.params["no_hit_value"] = self.high_level_scan.no_hit_value
        self.observations.policy.goal_pos_b.params["goal_offset"] = goal_offset

        self.observations.critic.scan_points_flat.params["grid_shape"] = grid_shape
        self.observations.critic.scan_points_flat.params["no_hit_value"] = self.high_level_scan.no_hit_value
        self.observations.critic.goal_pos_b.params["goal_offset"] = goal_offset

        # reward patching
        self.rewards.goal_distance.params["std"] = self.goal_reward_std
        self.rewards.goal_distance.params["goal_offset"] = goal_offset

        self.rewards.goal_reached_bonus.params["threshold"] = self.goal_reached_threshold
        self.rewards.goal_reached_bonus.params["goal_offset"] = goal_offset

        # termination patching
        self.terminations.goal_reached.params["threshold"] = self.goal_reached_threshold
        self.terminations.goal_reached.params["goal_offset"] = goal_offset

        self.terminations.base_fall.params["threshold"] = self.base_height_termination_threshold

class G1AmpGapGoalEnv(ManagerBasedRLEnv):
    cfg: G1AmpGapGoalEnvCfg

    def __init__(self, cfg: G1AmpGapGoalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        robot_asset = self.scene["robot"]
        motion_body_indices = _resolve_body_indices(robot_asset, self.cfg.amp.body_names).tolist()

        self._motion_loader = AmpMotionLoader(
            motion_files=self.cfg.amp.motion_files,
            body_indices=motion_body_indices,
            device=self.device,
        )

        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.cfg.amp.num_amp_observations * self.cfg.amp.amp_observation_space,
            ),
            dtype=np.float32,
        )

    def step(self, action: torch.Tensor):
        obs, rew, terminated, truncated, extras = super().step(action)

        if extras is None:
            extras = {}
        extras["amp_obs"] = self.observation_manager.compute_group("amp")

        return obs, rew, terminated, truncated, extras

    def collect_reference_motions(
        self,
        num_samples: int,
        trajectory_ids: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Collect reference AMP history using policy-rate spacing."""
        if trajectory_ids is None or time_steps is None:
            trajectory_ids, time_steps = self._motion_loader.sample_random_time_indices(num_samples)
        else:
            trajectory_ids = torch.as_tensor(trajectory_ids, dtype=torch.long, device=self.device)
            time_steps = torch.as_tensor(time_steps, dtype=torch.long, device=self.device)

        step_offsets = torch.arange(
            self.cfg.amp.num_amp_observations - 1,
            -1,
            -1,
            device=self.device,
            dtype=torch.long,
        )

        hist_time_steps = (time_steps[:, None] - step_offsets[None, :]).reshape(-1)
        hist_traj_ids = trajectory_ids[:, None].expand(-1, self.cfg.amp.num_amp_observations).reshape(-1)

        joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel = self._motion_loader.sample_amp_frames(
            hist_traj_ids,
            hist_time_steps,
        )

        amp_frames = torch.cat([joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel], dim=-1)

        return amp_frames.view(
            num_samples,
            self.cfg.amp.num_amp_observations * amp_frames.shape[-1],
        )

    def sample_reference_motions(self, num_samples: int) -> torch.Tensor:
        """Convenience wrapper for random reference AMP samples."""
        return self.collect_reference_motions(num_samples=num_samples)