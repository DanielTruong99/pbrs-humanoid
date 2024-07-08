# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gpugym import LEGGED_GYM_ROOT_DIR
import os
import pandas as pd

import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy, export_critic, task_registry, Logger

import numpy as np
import torch

DOF_NAMES = [
    'R_hip_joint',
    'R_hip2_joint',
    'R_thigh_joint',
    'R_calf_joint',  
    'R_toe_joint',
    'L_hip_joint',
    'L_hip2_joint',
    'L_thigh_joint',
    'L_calf_joint',  
    'L_toe_joint'
]

def play(args):
    args.task = 'leg'
    # args.headless = True
    is_debug_visualize = True
    if is_debug_visualize:
        pass
        # args.use_gpu_pipeline = False
        # args.rl_device = 'cpu'
        # args.sim_device = 'cpu'

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)

    # set domain randomization
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False #True
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 1.0

    # set initial state for robot
    env_cfg.init_state.reset_ratio = 0.8

    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    # env_cfg.terrain.horizontal_scale = 0.05
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.terrain_proportions = [0.0, 1.0, 0.0, 0, 0] 
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.measure_heights = True if is_debug_visualize else False
    # env_cfg.terrain.terrain_length = 20
    # env_cfg.terrain.terrain_width = 20
    env_cfg.terrain.selected = True
    env_cfg.terrain.user_custom_terrain = True
    env_cfg.terrain.custom_terrain_kwargs = [
        {
            'type': 'terrain_utils.random_uniform_terrain',
            'options': {
                'min_height': -0.05,
                'max_height': 0.05, 
                'step' : 0.005, 
                'downsampled_scale': 0.2
            }
        },
        {
            'type': 'terrain_utils.discrete_obstacles_terrain',
            'options': {
                'max_height': 0.05,
                'min_size': 1,
                'max_size': 2,
                'num_rects': 20,
                'platform_size': 3
            }            
        }
    ]
    env_cfg.terrain.terrain_kwargs = {
        'type': 'terrain_utils.random_uniform_terrain',
        'options': {
            'min_height': -0.05,
            'max_height': 0.05, 
            'step' : 0.005, 
            'downsampled_scale': 0.2
        }
    }

    # viewer written in root frame without follow the base orientation
    env_cfg.viewer.pos = [0, -3.5, 1.5]
    env_cfg.viewer.lookat = [0, 0, 0]

    # set simulation time period (second)
    simulation_time = 20.0

    # choose model checkpoint
    # train_cfg.runner.load_run = "Jun11_11-52-33_Demo"
    # train_cfg.runner.checkpoint = "6350"


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    env.debug_viz = True if is_debug_visualize else False

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy(ppo_runner.alg.actor_critic, path)
        print('Exported policy model to: ', path)

    # export critic as a jit module (used to run it from C++)
    if EXPORT_CRITIC:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'critics')
        export_critic(ppo_runner.alg.actor_critic, path)
        print('Exported critic model to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_relative_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    play_log = []
    env.max_episode_length = simulation_time/env.dt
    stop_state_log = env.max_episode_length  # number of steps before plotting states

    for i in range(1*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        env.commands[:, 0] = 1.0   # 1.0
        env.commands[:, 1] = 0.
        env.commands[:, 2] = 0.
        env.commands[:, 3] = 0.

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            # camera_position += camera_vel * env.dt
            base_pos = env.root_states[robot_index, 0:3].detach().cpu().numpy()
            base_orientation = env.root_states[robot_index, 3:7].detach().cpu().numpy()
            camera_position = base_pos + camera_relative_position
            env.set_camera(camera_position, base_pos)

        if i < stop_state_log:
            ### Humanoid PBRS Logging ###
            # [ 1]  Timestep
            # [38]  Agent observations
            # [10]  Agent actions (joint setpoints)
            # [13]  Floating base states in world frame
            # [ 6]  Contact forces for feet
            # [10]  Joint torques
            play_log.append(
                [i*env.dt]
                + obs[robot_index, :].cpu().numpy().tolist()
                + actions[robot_index, :].detach().cpu().numpy().tolist()
                + env.root_states[0, :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[0], :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[1], :].detach().cpu().numpy().tolist()
                + env.torques[robot_index, :].detach().cpu().numpy().tolist()
            )

            data_frame = {
                'time_step': i*env.dt,
                'base_x': env.root_states[robot_index, 0].item(),
                'base_y': env.root_states[robot_index, 1].item(),
                'base_z': env.root_states[robot_index, 2].item(),
                'base_vx': env.base_lin_vel[robot_index, 0].item(),
                'base_vy': env.base_lin_vel[robot_index, 1].item(),
                'base_vz': env.base_lin_vel[robot_index, 2].item(),
                'base_wx': env.base_ang_vel[robot_index, 0].item(),
                'base_wy': env.base_ang_vel[robot_index, 1].item(),
                'base_wz': env.base_ang_vel[robot_index, 2].item(),
                **{'pos_' + key : env.dof_pos[robot_index, index].item() for index, key in enumerate(DOF_NAMES)},
                **{'vel_' + key : env.dof_vel[robot_index, index].item() for index, key in enumerate(DOF_NAMES)},
                **{'torque_' + key : env.torques[robot_index, index].item() for index, key in enumerate(DOF_NAMES)},
            }

            logger.log_states(data_frame)

        elif i==stop_state_log:
            # np.savetxt('gpugym/analysis/data/play_log_leg_2.csv', play_log, delimiter=',')
            # logger.plot_states()
            pass

        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                # if num_episodes>0:
                    # logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

        pd.DataFrame(logger.state_log).to_csv('gpugym/analysis/data/play_log_leg_2.csv', index=False)

if __name__ == '__main__':
    EXPORT_POLICY = False
    EXPORT_CRITIC = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
