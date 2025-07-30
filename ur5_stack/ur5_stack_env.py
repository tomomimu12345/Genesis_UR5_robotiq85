import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import inv_quat


def rand_uniform(low, high, shape):
    return torch.rand(shape, device=gs.device) * (high - low) + low


class UR5StackEnv:
    def __init__(self, num_envs, num_blocks=5, show_viewer=True):
        self.num_envs = num_envs
        self.num_blocks = num_blocks
        self.device = gs.device
        self.dt = 0.05
        self.max_episode_length = 300

        self.action_scale_arm = 0.05
        self.action_scale_grip = 0.01
        self.arm_dof_idx = list(range(2, 8))
        self.grip_dof_idx = [8]

        self.reset_height_threshold = 0.05

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt),
                camera_pos=(1.0, 1.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=60,
            ),
            show_viewer=show_viewer,
            rigid_options=gs.options.RigidOptions(  
                box_box_detection=True,  
            ),  
        )

        self.scene.add_entity(
            morph=gs.morphs.Plane(),
        )

        self.ur5 = self.scene.add_entity(
            gs.morphs.URDF(file="./ur5/ur5_robotiq85.urdf", fixed=True),
        )

        self.left_tip = self.ur5.get_link("robotiq_85_left_finger_tip_link")
        self.right_tip = self.ur5.get_link("robotiq_85_right_finger_tip_link")

        self.block_shapes = [(0.04, 0.04, 0.04), (0.06, 0.03, 0.03), (0.05, 0.05, 0.02)]
        self.calc_block_pos()

        self.blocks = [
            self.scene.add_entity(
                morph=gs.morphs.Box(size=self.block_shapes[i % len(self.block_shapes)],
                              pos=self._block_init_positions[i].cpu().numpy())
            ) for i in range(self.num_blocks)
        ]

        self.scene.build(n_envs=num_envs)

        self.first_pos=np.array([-0, -0.9,  -0.5,  -1.4,  -1.3,  -0.3, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])

        # アームの6自由度（idx: 2~7）＋ グリッパの主制御1自由度（idx: 8）
        self.dof_idx = list(range(2, 9))

        init_qpos = torch.Tensor(self.first_pos[2:9]).unsqueeze(0).repeat(num_envs, 1)
        init_qpos += torch.randn_like(init_qpos) * 0.05

        self.ur5.set_dofs_position(
            position=init_qpos,
            dofs_idx_local=self.dof_idx,
            zero_velocity=True,
            envs_idx=np.arange(num_envs),
        )


        kp = np.array([4500, 3500, 2000, 100, 100,  100, 100, ])  
        kv = np.array([450,  350,  200,  10, 10, 10, 10])  
        self.ur5.set_dofs_kp(kp, self.dof_idx)
        self.ur5.set_dofs_kv(kv, self.dof_idx)


        self.actions = torch.zeros((num_envs, len(self.dof_idx)), device=gs.device)
        self.obs_dim = 2 * len(self.dof_idx) + 3 + 10 * self.num_blocks
        self.obs_buf = torch.zeros((num_envs, self.obs_dim), device=gs.device)

        self.num_actions = len(self.dof_idx) # need

        self.rew_buf = torch.zeros((num_envs,), device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=torch.bool, device=gs.device)

        self.episode_length_buf = torch.zeros((num_envs,), device=gs.device)

        self.extras = dict()
        self.extras["observations"] = dict()

        self.max_block_height = torch.zeros((num_envs,), device=gs.device)

    def calc_block_pos(self):
        self._block_init_positions = rand_uniform(
            low=torch.tensor([0.3, -0.2, 0.05], device=gs.device),
            high=torch.tensor([0.7,  0.2, 0.05], device=gs.device),
            shape=(self.num_blocks, 3)
        )

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, env_ids):
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.actions[env_ids] = 0

        init_qpos = torch.Tensor(self.first_pos[2:9]).unsqueeze(0).repeat(len(env_ids), 1)
        init_qpos += torch.randn_like(init_qpos) * 0.05

        self.ur5.set_dofs_position(
            position=init_qpos,
            dofs_idx_local=self.dof_idx,
            zero_velocity=True,
            envs_idx=env_ids,
        )

        self.calc_block_pos()
        self.max_block_height[env_ids] = 0

        for i, block in enumerate(self.blocks):
            pos = self._block_init_positions[i].repeat(len(env_ids), 1)
            
            block.set_pos(pos, zero_velocity=True, envs_idx=env_ids)
            block.set_quat(torch.tensor([1, 0, 0, 0], device=gs.device).repeat(len(env_ids), 1), envs_idx=env_ids)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf

    def _reward_stack_height(self, block_positions):
        """
        持ち上げてないブロックで一番高さが高いブロックが、これまでの最高高さを超えた時に報酬を出す。
        """
        left_pos = self.left_tip.get_pos()
        right_pos = self.right_tip.get_pos()
        finger_gap = torch.norm(left_pos - right_pos, dim=1)
        center_pos = (left_pos + right_pos) / 2

        reward = torch.zeros(self.num_envs, device=gs.device)
        current_max_height = torch.zeros(self.num_envs, device=gs.device)

        for block_pos in block_positions:
            block_dist_to_center = torch.norm(block_pos - center_pos, dim=1)  
            grasped = (block_dist_to_center < 0.02) & (finger_gap < 0.05)
            block_height = block_pos[:, 2]
            current_max_height = torch.where(grasped, torch.maximum(current_max_height, block_height), current_max_height)

        height_diff = current_max_height - self.max_block_height
        reward = torch.where(height_diff > 0.01, height_diff, torch.zeros_like(height_diff))

        self.max_block_height = torch.maximum(self.max_block_height, current_max_height)

        return reward * 150
    
    def _reward_block_z_velocity(self):
        """
        ブロックの z方向の速度に対して、
        +Z 方向なら報酬、-Z 方向ならペナルティを与える。
        """

        block_vels = torch.stack([b.get_vel() for b in self.blocks], dim=1)
        block_z_vel = block_vels[:, :, 2]  # [num_envs, num_blocks]


        mean_z_vel = block_z_vel.mean(dim=1)  # [num_envs]

        reward_weight = 10

        return reward_weight * mean_z_vel 


    def _reward_action_penalty(self):
        """
        行動の大きさに対するペナルティ
        """
        arm_penalty = torch.sum(self.actions[:, :6] ** 2, dim=1)
        grip_penalty = torch.sum(self.actions[:, 6:] ** 2, dim=1)

        self.arm_weight = 0.01
        self.grip_weight = 0.01

        return -self.arm_weight * arm_penalty - self.grip_weight * grip_penalty

    def _reward_joint_deviation_penalty(self):
        """
        現在の関節角が初期姿勢から離れているほどペナルティを与える。
        """
        current_q = self.ur5.get_dofs_position(self.dof_idx)  # [num_envs, 7]

        target_q = torch.tensor(self.first_pos[2:9], device=gs.device).unsqueeze(0)  # [1, 7]
        delta_q = current_q - target_q  # [num_envs, 7]

        penalty = torch.sum(delta_q ** 2, dim=1) 

        penalty_weight = 0.01

        return -penalty_weight * penalty


    def _reward_ee_below_plane_penalty(self):
        """
        エンドエフェクタのz座標が Plane のz座標より小さくなった場合にペナルティ
        """
        plane_z = 0.0

        left_pos = self.left_tip.get_pos()    # Tensor[N,3]
        right_pos = self.right_tip.get_pos()  # Tensor[N,3]

        ee_pos = (left_pos + right_pos) * 0.5  # Tensor[N,3]
        ee_z = ee_pos[:, 2]                   # Tensor[N]

        penalty = torch.where(
            ee_z < plane_z,
            torch.full_like(ee_z, -5.0),
            torch.zeros_like(ee_z)
        )

        return penalty

    def _reward_block_xy_concentration(self):
        """
        すべてのブロックの x, y 座標がなるべく一か所に集中するように、
        重心からの平均距離にペナルティを与える。
        """

        block_pos_tensor = torch.stack([b.get_pos() for b in self.blocks], dim=1)  # [num_envs, num_blocks, 3]

        block_xy = block_pos_tensor[:, :, :2]  # [num_envs, num_blocks, 2]

        centroid = block_xy.mean(dim=1, keepdim=True)  # [num_envs, 1, 2]

        dist_to_centroid = torch.norm(block_xy - centroid, dim=2)  # [num_envs, num_blocks]
        mean_dist = dist_to_centroid.mean(dim=1)  # [num_envs]

        penalty_weight = 5.0

        return -penalty_weight * mean_dist

    def _reward_ee_block_distance_penalty(self):
        """
        EEとブロックの距離の平均に対してペナルティを与える。
        """
        ee_pos = (self.left_tip.get_pos() + self.right_tip.get_pos()) / 2  # [num_envs, 3]
        block_pos_tensor = torch.stack([b.get_pos() for b in self.blocks], dim=1)  # [num_envs, num_blocks, 3]

        dists = torch.norm(block_pos_tensor - ee_pos.unsqueeze(1), dim=2)  # [num_envs, num_blocks]
        mean_dist = dists.mean(dim=1)  # [num_envs]

        penalty_weight = 5.0 
        return -penalty_weight * mean_dist




    def step(self, actions):
        self.actions = torch.clip(actions, -1.0, 1.0)

        arm_target = self.ur5.get_dofs_position(self.arm_dof_idx) + actions[:, :6] * self.action_scale_arm
        grip_target = self.ur5.get_dofs_position(self.grip_dof_idx) + actions[:, 6:7] * self.action_scale_grip

        target = torch.zeros_like(self.ur5.get_dofs_position(self.dof_idx))
        target[:, :6] = arm_target
        target[:, 6:7] = grip_target
        self.ur5.control_dofs_position(target, self.dof_idx)


        self.scene.step()

        self.episode_length_buf += 1
        self.reset_buf |= self.episode_length_buf >= self.max_episode_length

        ee_pos = (self.left_tip.get_pos()+self.right_tip.get_pos())/2.0

        block_positions = [b.get_pos() for b in self.blocks]
        block_velocity = [b.get_vel() for b in self.blocks]
        block_quats = [b.get_quat() for b in self.blocks]

        dof_pos = self.ur5.get_dofs_position(self.dof_idx)  # [num_envs, 7]
        dof_vel = self.ur5.get_dofs_velocity(self.dof_idx)  # [num_envs, 7]


        self.obs_buf = torch.cat([dof_pos, dof_vel, ee_pos] + block_positions + block_velocity + block_quats, dim=1)

        self.rew_buf = self._reward_stack_height(block_positions) + self._reward_action_penalty()
        self.rew_buf += self._reward_ee_below_plane_penalty() + self._reward_block_xy_concentration()
        self.rew_buf += self._reward_joint_deviation_penalty() + self._reward_block_z_velocity()
        self.rew_buf += self._reward_ee_block_distance_penalty()

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).squeeze(-1))

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
