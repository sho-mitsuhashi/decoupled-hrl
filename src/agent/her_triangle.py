import copy
import numpy as np
import time
import torch

from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.sampler import Sampler
from src.agent.ddpg import DDPG

import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter
import sys

import gymnasium as gym
import gymnasium_robotics
from gymnasium.vector import AsyncVectorEnv 



class HER_TRIANGLE(DDPG):
    """
    Hindsight Experience Replay agentを改良
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.sample_func = self.sampler.sample_her_transitions
        self.triangle_sample_func=self.sampler.sample_triangle_transitions
        self.buffer = ReplayBuffer(args, self.sample_func,triangle_sample_func=self.triangle_sample_func)

        self.envs=self.env
        
        all_params = [
            self.critic.parameters(),
            self.critic_target.parameters(),
            self.actor.parameters(),
            self.actor_target.parameters()
        ]

        num_param = sum(p.numel() for p_generator in all_params for p in p_generator)
        #print(f"[info] num parameters: {num_param}")



    def _update(self):
        transition , triangle_transition= self.buffer.sample(self.args.batch_size)
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']
        NG = transition['NG']
        AG = transition['AG']   #現状使用していない.sをgの形式にしたもの.encoder sgのcをgとするなら、encoder saのcはAGと言える
        _, AG = self._preproc_inputs(None, AG)

        #deprecated param     
        tri_S=triangle_transition['S']
        tri_A=triangle_transition['A']
        tri_via_AG=triangle_transition['via_AG']
        tri_via_S=triangle_transition['via_S']
        tri_via_A=triangle_transition['via_A']
        tri_end_AG=triangle_transition['end_AG']
        tri_gamma_t=triangle_transition['gamma_t']
        q_total_step=triangle_transition['q_total_step']

        # S/NS: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)

        tri_A = numpy2torch(tri_A, unsqueeze=False, cuda=self.args.cuda)
        tri_via_A = numpy2torch(tri_via_A, unsqueeze=False, cuda=self.args.cuda)
        tri_gamma_t = numpy2torch(np.array(tri_gamma_t), unsqueeze=False, cuda=self.args.cuda)
        q_total_step = numpy2torch(np.array(q_total_step), unsqueeze=False, cuda=self.args.cuda)

        #torchにする処理とclipをやってる
        S, G = self._preproc_inputs(S, G)
        NS, NG = self._preproc_inputs(NS, NG)

        tri_S, tri_via_AG = self._preproc_inputs(tri_S, tri_via_AG)
        tri_via_S, tri_end_AG = self._preproc_inputs(tri_via_S, tri_end_AG)




        with torch.no_grad():
            NA = self.actor_target(NS, G)
            NQ = self.critic_target(NS, NA, G).detach()
            
            clip_return = 1 / (1 - self.args.gamma)

            target = (R + self.args.gamma * NQ).detach().clamp_(-clip_return, 0)



        Q = self.critic.forward(S, A, G)


        critic_loss =  (Q - target).pow(2).mean()    

        self.critic_optim.zero_grad()



        (critic_loss*self.args.loss_scale).backward()
        critic_grad_norm = sync_grads(self.critic)
        self.critic_optim.step()
            
        A_ = self.actor(S, G)
        actor_loss = - self.critic(S, A_, G).mean()
        actor_loss += self.args.action_l2 * (A_ / self.args.max_action).pow(2).mean()

        self.actor_optim.zero_grad()
        (actor_loss*self.args.loss_scale).backward()
        actor_grad_norm = sync_grads(self.actor)
        self.actor_optim.step()


        return actor_loss.item(), critic_loss.item(), actor_grad_norm, critic_grad_norm



    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            log_dir =  f"./tensorboard/{self.env_name}/{self.args.experiment_name}"

            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
                'actor_grad_norms': [],
                'critic_grad_norms': [],
            }

        # put something to the buffer first
        self.prefill_buffer()

        for epoch in range(self.args.n_epochs):

            AL, CL, AGN, CGN = [], [], [], []

            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)
                for _ in range(self.args.n_batches):
                    a_loss, c_loss, a_gn, c_gn = self._update()
                    AL.append(a_loss); CL.append(c_loss)
                    AGN.append(a_gn); CGN.append(c_gn)

                self._soft_update(self.actor_target, self.actor)
                self._soft_update(self.critic_target, self.critic)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL)
                AGN = np.array(AGN); CGN = np.array(CGN)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                stats['actor_grad_norms'].append(AGN.mean())
                stats['critic_grad_norms'].append(CGN.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f" actor loss {AL.mean():6.4f} | critic loss {CL.mean():6.4f} | "+\
                        f" actor gradnorm {AGN.mean():6.4f} | critic gradnorm {CGN.mean():6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")

                # TensorBoard へのログ出力
                writer.add_scalar('Success Rate', global_success_rate, epoch)
                writer.add_scalar('Hitting Time', global_hitting_time, epoch)
                writer.add_scalar('Actor Loss', AL.mean(), epoch)
                writer.add_scalar('Critic Loss', CL.mean(), epoch)
                writer.add_scalar('Actor Grad Norm', AGN.mean(), epoch)
                writer.add_scalar('Critic Grad Norm', CGN.mean(), epoch)
                writer.flush()

                # ここで損失がNaNになっていないかをチェック
                if np.isnan(AL.mean()) or np.isnan(CL.mean()):
                    print("[error] NaN loss encountered. Exiting.")
                    sys.exit(1)
                # エポック20以降に過去3エポックの成功率の平均が0.1以下ならプログラム終了
                if epoch >= self.eval_epoch_for_this_env:
                    recent_success = stats['successes'][-3:]
                    if np.mean(recent_success) < 0.05:
                        print("[warning] Past 3 epochs' average success rate is below 0.1. Terminating training.")
                        sys.exit(1)
                    
                    if np.mean(recent_success) > 0.99:
                        print("[warning] Past 3 epochs' average success rate is above 0.99. Terminating training.")
                        sys.exit(1)

                    
        if MPI.COMM_WORLD.Get_rank() == 0:
            writer.close()




    def collect_rollout(self, uniform_random_action=False, stochastic=True):
        n_episodes = self.args.rollout_n_episodes
        dim_state  = self.args.dim_state
        dim_action = self.args.dim_action
        dim_goal   = self.args.dim_goal
        T          = self.args.max_episode_steps
        max_action = self.args.max_action

        S       = np.zeros((n_episodes, T+1, dim_state),  np.float32)
        A       = np.zeros((n_episodes, T,   dim_action), np.float32)
        AG      = np.zeros((n_episodes, T+1, dim_goal),   np.float32)
        G       = np.zeros((n_episodes, T,   dim_goal),   np.float32)
        success = np.zeros((n_episodes), np.float32)


        obs_full, _ = self.envs.reset()
        obs_dict   = {k: v[:n_episodes] for k, v in obs_full.items()}

        for t in range(T):
            if uniform_random_action:
                actions = np.random.uniform(low=-max_action,
                                    high=max_action,
                                    size=(self.args.rollout_n_episodes,dim_action))
            else:
                # 観測・ゴールをバッチ化
                obs_batch = obs_dict['observation']
                ag_batch  = obs_dict['achieved_goal']
                if "AntMaze" in self.args.env_name:
                    s_batch = np.concatenate([obs_batch, ag_batch], axis=-1)
                else:
                    s_batch = obs_batch
                g_batch = obs_dict['desired_goal']

                # Tensor へ
                s_tensor, g_tensor = self._preproc_inputs(s_batch, g_batch, unsqueeze=False)

                

                with torch.inference_mode():
                    a_tensor = self.actor(s_tensor, g_tensor)



                actions = a_tensor.cpu().numpy()
                if stochastic:
                    actions = self._add_noise_and_epsgreedy_batch(actions)
                actions = np.clip(actions, -max_action, max_action)

            full_actions = np.zeros((self.envs.num_envs, self.args.dim_action), np.float32)
            full_actions[:n_episodes] =actions
            # 環境ステップ & バッファ記録
            next_obs, rewards, terms, truncs, infos = self.envs.step(full_actions)

            next_obs  = {k: v[:n_episodes] for k, v in next_obs.items()}


            if "AntMaze" in self.args.env_name:
                S[:, t] = np.concatenate(
                              [obs_dict['observation'],
                               obs_dict['achieved_goal']],
                              axis=-1
                          )
            else:
                S[:, t] = obs_dict['observation']
            

            AG[:, t] = obs_dict['achieved_goal']
            G[:, t]  = obs_dict['desired_goal']
            A[:, t]  = actions

            obs_dict = next_obs



        # ===== エピソード終了後の success =====

        infos_dict = {k: v[:n_episodes] for k, v in infos.items()}

        # 成功履歴を記録
        if 'is_success' in infos_dict:
            success[:] = infos_dict['is_success'].astype(np.float32)
        elif 'success' in infos_dict:
            success[:] = infos_dict['success'].astype(np.float32)
        else:
            print('error Hierarchical 565')
            sys.exit(1)


        # 最終状態も格納
        if "AntMaze" in self.args.env_name:
            S[:, T] = np.concatenate(
                        [obs_dict['observation'],
                        obs_dict['achieved_goal']],
                        axis=-1
                    )
        else:
            S[:, T] = obs_dict['observation']
        AG[:, T] = obs_dict['achieved_goal']



            
        return (S, A, AG, G), success
 
    def eval_agent(self):
        n_envs = self.args.eval_rollout_n_episodes
        T      = self.args.max_episode_steps
        max_action  = self.args.max_action
        obs_full, _ = self.envs.reset()

        obs_dict   = {k: v[:n_envs] for k, v in obs_full.items()}

        # success_hist[i, t] に i番目の env が時刻 t に成功しているか (1.0/0.0) を記録
        success_hist = np.zeros((n_envs, T), dtype=np.float32)
        
        for t in range(T):

            # 観測・ゴールをバッチ化
            obs_batch = obs_dict['observation']
            ag_batch  = obs_dict['achieved_goal']
            if "AntMaze" in self.args.env_name:
                s_batch = np.concatenate([obs_batch, ag_batch], axis=-1)
            else:
                s_batch = obs_batch
            g_batch = obs_dict['desired_goal']

            # Tensor へ
            s_tensor, g_tensor = self._preproc_inputs(s_batch, g_batch, unsqueeze=False)

            

            with torch.inference_mode():

                a_tensor = self.actor(s_tensor, g_tensor)

                    

            actions = a_tensor.cpu().numpy()
            actions = np.clip(actions, -max_action, max_action)

            full_actions = np.zeros((self.envs.num_envs, self.args.dim_action), np.float32)
            full_actions[:n_envs] =actions
            # 環境ステップ & バッファ記録
            next_obs, _, _, _, infos_full = self.envs.step(full_actions)

            obs_dict   = {k: v[:n_envs] for k, v in next_obs.items()}#現状態
            infos_dict = {k: v[:n_envs] for k, v in infos_full.items()}#次状態に対する成功フラグなど

            # 成功履歴を記録
            if 'is_success' in infos_dict:
                success_hist[:, t] = infos_dict['is_success'].astype(np.float32)
            elif 'success' in infos_dict:
                success_hist[:, t] = infos_dict['success'].astype(np.float32)
            else:
                print('error Hierarchical 565')
                sys.exit(1)




        local_success_rate = success_hist[:, -1].mean()

        local_hitting_time = first_nonzero(success_hist, axis=1, invalid_val=T+1).mean()
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_hitting_time = MPI.COMM_WORLD.allreduce(local_hitting_time, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_hitting_time / MPI.COMM_WORLD.Get_size()




    
    def _add_noise_and_epsgreedy_batch(self, actions: np.ndarray) -> np.ndarray:
        """
        actions: shape (batch_size, dim_action)
        returns:  shape (batch_size, dim_action)
        """
        max_a = self.args.max_action
        # Gaussian noise
        noise = np.random.randn(*actions.shape) \
                * (self.args.noise_eps * max_a)
        a_noisy = actions + noise
        a_noisy = np.clip(a_noisy, -max_a, max_a)

        # eps-greedy
        rand_a = np.random.uniform(
            low=-max_a, high=max_a, size=actions.shape
        )
        # mask: shape (batch_size, 1)
        mask = np.random.binomial(1, self.args.random_eps,
                                size=(actions.shape[0], 1))
        # ブロードキャストで (batch_size, dim_action) に適用
        a_final = a_noisy + mask * (rand_a - a_noisy)

        return a_final
