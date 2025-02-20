"""
Environment file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gpugym.utils.math import *
from gpugym.envs import LeggedRobot
from gpu_rl.rsl_rl.datasets.motion_loader import AMPLoader

from isaacgym import gymtorch, gymapi, gymutil


class LegAMP(LeggedRobot):

    def _custom_init(self, cfg):
        self.dt_step = self.cfg.sim.dt * self.cfg.control.decimation
        self.pbrs_gamma = 0.99
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.eps = 0.2
        self.phase_freq = 1.

        # Load the motion files into the MotionLib
        self._motion_lib = AMPLoader(
            motion_files=self.cfg.env.amp_motion_files, 
            device=self.device, 
            time_between_frames=self.dt
        )

        self.include_history_steps = cfg.env.include_history_steps

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs        

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        if self.cfg.asset.disable_actions:
            self.actions[:] = 0.
        else:
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.pre_physics_step()
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):

            if self.cfg.control.exp_avg_decay:
                self.action_avg = exp_avg_filter(self.actions, self.action_avg,
                                                self.cfg.control.exp_avg_decay)
                self.torques = self._compute_torques(self.action_avg).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            if self.cfg.asset.disable_motors:
                self.torques[:] = 0.

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        # Need to override the post_physics_step method to return terminal states
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf,
                                                 -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def post_physics_step(self):
        super().post_physics_step()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        return env_ids, terminal_amp_states

    def compute_observations(self):
  
        if self.cfg.terrain.measure_heights:
            base_z = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)*self.obs_scales.base_z
        else:
            base_z = self.root_states[:, 2].unsqueeze(1)*self.obs_scales.base_z

        in_contact = torch.gt(
            self.contact_forces[:, self.end_eff_ids, 2], 0).int()
        in_contact = torch.cat(
            (in_contact[:, 0].unsqueeze(1), in_contact[:, 1].unsqueeze(1)),
            dim=1)
        self.commands[:, 0:2] = torch.where(
            torch.norm(self.commands[:, 0:2], dim=-1, keepdim=True) < 0.5,
            0., self.commands[:, 0:2].double()).float()
        self.commands[:, 2:3] = torch.where(
            torch.abs(self.commands[:, 2:3]) < 0.5,
            0., self.commands[:, 2:3].double()).float()
        self.obs_buf = torch.cat((
            base_z,                                 # [1] Base height
            self.base_lin_vel,                      # [3] Base linear velocity
            self.base_ang_vel,                      # [3] Base angular velocity
            self.projected_gravity,                 # [3] Projected gravity
            self.commands[:, 0:3],                  # [3] Velocity commands
            self.smooth_sqr_wave(self.phase),       # [1] Contact schedule
            torch.sin(2*torch.pi*self.phase),       # [1] Phase variable
            torch.cos(2*torch.pi*self.phase),       # [1] Phase variable
            self.dof_pos,                           # [10] Joint states
            self.dof_vel,                           # [10] Joint velocities
            in_contact,                             # [2] Contact states
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                * self.noise_scale_vec
            
    def foot_positions_in_base_frame(self):
        right_foot_pos = self._rigid_body_pos[:, 5, :]; left_foot_pos = self._rigid_body_pos[:, 10, :]
        right_foot_pos_from_base = right_foot_pos - self.base_pos; left_foot_pos_from_base = left_foot_pos - self.base_pos
    
        base_frame_right_foot_pos_from_base = quat_rotate_inverse(self.base_quat, right_foot_pos_from_base)
        base_frame_left_foot_pos_from_base = quat_rotate_inverse(self.base_quat, left_foot_pos_from_base)
        return torch.cat((base_frame_right_foot_pos_from_base, base_frame_left_foot_pos_from_base), dim=-1)
    
    def get_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = self.foot_positions_in_base_frame()
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        z_pos = self.root_states[:, 2:3]
        return torch.cat((self.base_quat, base_lin_vel, base_ang_vel, joint_pos, joint_vel, z_pos, foot_pos), dim=-1)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0] = noise_scales.base_z * self.obs_scales.base_z
        noise_vec[1:4] = noise_scales.lin_vel
        noise_vec[4:7] = noise_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity
        noise_vec[10:16] = 0.   # commands
        noise_vec[16:26] = noise_scales.dof_pos
        noise_vec[26:36] = noise_scales.dof_vel
        noise_vec[36:38] = noise_scales.in_contact  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements \
                * noise_level \
                * self.obs_scales.height_measurements
        noise_vec = noise_vec * noise_level
        return noise_vec

    def _custom_reset(self, env_ids):
        if self.cfg.commands.resampling_time == -1:
            self.commands[env_ids, :] = 0.
        self.phase[env_ids, 0] = torch.rand(
            (torch.numel(env_ids),), requires_grad=False, device=self.device)

    def _post_physics_step_callback(self):
        self.phase = torch.fmod(self.phase + self.dt, 1.0)
        env_ids = (
            self.episode_length_buf
            % int(self.cfg.commands.resampling_time / self.dt) == 0) \
            .nonzero(as_tuple=False).flatten()
        if self.cfg.commands.resampling_time == -1 :
            # print(self.commands)
            pass  # when the joystick is used, the self.commands variables are overridden
        else:
            self._resample_commands(env_ids)

            if (self.cfg.domain_rand.push_robots and
                (self.common_step_counter
                % self.cfg.domain_rand.push_interval == 0)):
                self._push_robots()

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def _draw_debug_vis(self):
        # draws height measurement points
        super()._draw_debug_vis()

        # draws base frame
        axes_geom = gymutil.AxesGeometry(1, None)
        base_pos = (self.root_states[0, :3]).cpu().numpy()
        base_quat = self.base_quat[0].cpu().numpy()
        axes_pose = gymapi.Transform(gymapi.Vec3(*base_pos), gymapi.Quat(*base_quat))
        gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], axes_pose)

        # draws contact force
        contact_forces = self.contact_forces.cpu().numpy()
        for body_index, contact_force in enumerate(contact_forces[0]):
            if np.linalg.norm(contact_force) < 1e-5: continue
            
            body_pos = self._rigid_body_pos[0, body_index].cpu().numpy()
            end_pos = body_pos + 0.5 * contact_force / np.linalg.norm(contact_force)
            verts = np.empty((1, 2), gymapi.Vec3.dtype)
            verts[0][0] = (body_pos[0], body_pos[1], body_pos[2])
            verts[0][1] = (end_pos[0], end_pos[1], end_pos[2])
            self.gym.add_lines(self.viewer, self.envs[0], 1, verts, (1.0, 0.0, 0.0))





    def _push_robots(self):
        # Randomly pushes the robots.
        # Emulates an impulse by setting a randomized base velocity.
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:8] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 1), device=self.device)
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # Termination for contact
        term_contact = torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1)
        self.reset_buf = torch.any((term_contact > 1.), dim=1)

        # Termination for velocities, orientation, and low height
        self.reset_buf |= torch.any(
          torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)
        self.reset_buf |= torch.any(
          torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 5., dim=1)
        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)
        # self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)

        # # no terminal reward for time-outs
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf


# ########################## REWARDS ######################## #

    # * "True" rewards * #

    def _reward_tracking_lin_vel(self):
        # Reward tracking specified linear velocity command
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Reward tracking yaw angular velocity command
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2])*2/torch.pi)
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # * Shaping rewards * #

    def _reward_base_height(self):
        # Reward tracking specified base height
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = (base_height-self.cfg.rewards.base_height_target)
        error *= self.obs_scales.base_z
        error = error.flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)

    def _reward_orientation(self):
        # Reward tracking upright orientation
        error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_dof_vel(self):
        # Reward zero dof velocities
        dof_vel_scaled = self.dof_vel/self.cfg.normalization.obs_scales.dof_vel
        return torch.sum(self.sqrdexp(dof_vel_scaled), dim=-1)

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.
        # Yaw joints regularization around 0
        error += self.sqrdexp(
            (self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint symmetry
        error += self.sqrdexp(
            (self.dof_pos[:, 1] - self.dof_pos[:, 6])
            / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint symmetry
        error += self.sqrdexp(
            (self.dof_pos[:, 2] + self.dof_pos[:, 7])
            / self.cfg.normalization.obs_scales.dof_pos)
        return error/4

    def _reward_ankle_regularization(self):
        # Ankle joint regularization around 0
        error = 0
        error += self.sqrdexp(
            (self.dof_pos[:, 4]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 9]) / self.cfg.normalization.obs_scales.dof_pos)
        return error

    # * Potential-based rewards * #

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()

    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt_step

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt_step

    def _reward_baseHeight_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt_step

# ##################### HELPER FUNCTIONS ################################## #

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)

    def smooth_sqr_wave(self, phase):
        p = 2.*torch.pi*phase * self.phase_freq
        return torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + self.eps**2.)) + 1./2.
