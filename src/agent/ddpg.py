import copy
import numpy as np
import time
import torch
import sys

from mpi4py import MPI
from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.sampler import Sampler
from src.agent.base import Agent

from torch.utils.tensorboard import SummaryWriter



class DDPG(Agent):
    """
    Deep Deterministic Policy Gradient agent
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        
        critic_map = {

            'CriticHierarchical':CriticHierarchical,

        }
        self.seed=args.seed
        self.env_name=args.env_name
        self.critic_name = args.critic
        self.agent_name=args.agent
        self.triangle_scale=args.triangle_scale
        self.eval_epoch_for_this_env=args.eval_epoch_for_this_env
        self.triangle_eq_step=args.triangle_eq_step


        self.critic = critic_map[args.critic](args)

        sync_networks(self.critic)
        #hierarchicalでないなら実行
        if not getattr(self, 'disable_default_actor', False):
            self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if self.args.cuda:
            self.critic.cuda()
            self.critic_target.cuda()

        self.critic_optim  = torch.optim.Adam(self.critic.parameters(),
                                              lr=self.args.lr_critic)
        self.buffer = ReplayBuffer(args, self.sampler.sample_ddpg_transitions)

    def _update(self):
        transition = self.buffer.sample(self.args.batch_size)
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']
        NG = transition['NG']
        # S/NS: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)

        S, G = self._preproc_inputs(S, G)
        NS, NG = self._preproc_inputs(NS, NG)

        with torch.no_grad():
            NA = self.actor_target(NS, G)
            NQ = self.critic_target(NS, NA, G).detach()
            clip_return = 1 / (1 - self.args.gamma)
            if self.args.negative_reward:
                target = (R + self.args.gamma * NQ).detach().clamp_(-clip_return, 0)
                if self.args.terminate:
                    target = target * (-R)
            else:
                target = (R + self.args.gamma * NQ).detach().clamp_(0, clip_return)
                if self.args.terminate:
                    target = (1-R) * target + R

        if self.critic_name == "asym-new":
            Q, r = self.critic.sep_forward(S, A, G)
            critic_loss = (Q - target).pow(2).mean() + (r - R).pow(2).mean()
        else:
            Q = self.critic.forward(S, A, G)
            critic_loss = (Q - target).pow(2).mean()

        A_ = self.actor(S, G)
        actor_loss = - self.critic(S, A_, G).mean()
        actor_loss += self.args.action_l2 * (A_ / self.args.max_action).pow(2).mean()

        self.actor_optim.zero_grad()
        (actor_loss*self.args.loss_scale).backward()
        actor_grad_norm = sync_grads(self.actor)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        (critic_loss*self.args.loss_scale).backward()
        critic_grad_norm = sync_grads(self.critic)
        self.critic_optim.step()
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
