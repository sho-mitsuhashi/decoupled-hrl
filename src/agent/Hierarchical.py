#最新のgymnasium1.3.1はfetch環境に誤りがあるので注意
#https://github.com/Farama-Foundation/Gymnasium-Robotics/issues/269

import copy
import numpy as np
import time
import torch

from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.sampler import Sampler
from src.agent.ddpg import DDPG

from torch.utils.tensorboard import SummaryWriter#MS
import sys

import torch.nn as nn
from torch.autograd import grad

from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import gymnasium as gym
import gymnasium_robotics
from gymnasium.vector import AsyncVectorEnv 


class Project_Latent_To_Goal(nn.Module):
    def __init__(self, input_dim, dim_goal):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, dim_goal)
        )

        
    def forward(self, x):
        h = self.decoder(x)

        return  h

class Hierarchical(DDPG):
    """
    Hindsight Experience Replay agentを改良
    """
    def __init__(self, args, env):
        self.disable_default_actor = True
        super().__init__(args, env)

        self.use_multi_ll = args.multi_ll

        # --multi_ll フラグに応じてLL Policyの数を決定
        if self.use_multi_ll:
            self.num_ll_policies = 4
            self.prm_actors = nn.ModuleList([PrimitiveActor(args) for _ in range(self.num_ll_policies)])
            for actor in self.prm_actors:
                sync_networks(actor)
            
            if self.args.cuda:
                self.prm_actors.cuda()
            

            self.prm_actor_targets = nn.ModuleList([copy.deepcopy(actor) for actor in self.prm_actors])
            self.prm_actor_optims = [torch.optim.Adam(actor.parameters(), lr=args.lr_actor) for actor in self.prm_actors]
        else:
            self.num_ll_policies = 1
            self.prm_actor = PrimitiveActor(args)
            sync_networks(self.prm_actor)
            if self.args.cuda:
                self.prm_actor.cuda()

            self.prm_actor_target = copy.deepcopy(self.prm_actor)
            self.prm_actor_optim = torch.optim.Adam(self.prm_actor.parameters(), lr=self.args.lr_actor)


        self.hl_actor=HighLevelActor(args)
        sync_networks(self.hl_actor)

        if self.args.cuda:
            self.hl_actor.cuda()





        self.hl_actor_optim  = torch.optim.Adam(self.hl_actor.parameters(),
                                             lr=self.args.lr_actor)        


        self.sample_func = self.sampler.sample_her_transitions
        self.triangle_sample_func=self.sampler.sample_triangle_transitions
        self.buffer = ReplayBuffer(args, self.sample_func,triangle_sample_func=self.triangle_sample_func)
       
        

        
        
        self.high_level_actor_num=args.high_level_actor_num
        self.eval_hl_layers = getattr(args, 'eval_hl_layers', [self.high_level_actor_num])
        self.dim_hierarchical_latent=args.dim_hierarchical_latent

        self.envs=self.env



        
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        
        

        dim_latent=args.dim_hierarchical_latent
        self.dim_goal=args.dim_goal
        self.dim_state=args.dim_state
        self.dim_action=args.dim_action

        
        self.project_latent_to_goal=Project_Latent_To_Goal(input_dim=dim_latent,dim_goal=self.dim_goal)
        if self.args.cuda:

            self.project_latent_to_goal.cuda()

        self.project_latent_to_goal_optim = torch.optim.Adam(self.project_latent_to_goal.parameters(),lr=self.args.lr_actor)


        #calculate params
        if self.use_multi_ll:
            all_params = [
                self.critic.parameters(),
                self.critic_target.parameters(),
                self.hl_actor.parameters(),
                self.prm_actors.parameters(),
                self.prm_actor_targets.parameters(),
                self.project_latent_to_goal.parameters(),
            ]

            num_param = sum(p.numel() for p_generator in all_params for p in p_generator)
            #print(f"[info] num parameters: {num_param}")


        else:
            all_params = [
                self.critic.parameters(),
                self.critic_target.parameters(),
                self.hl_actor.parameters(),
                self.prm_actor.parameters(),
                self.prm_actor_target.parameters(),
                self.project_latent_to_goal.parameters(),
            ]

            num_param = sum(p.numel() for p_generator in all_params for p in p_generator)
            #print(f"[info] num parameters: {num_param}")



    def get_area_id(self, s: torch.Tensor) -> torch.Tensor:
        """状態s (Tensor) のx, y座標に基づいてエリアIDのTensorを返す"""
        if not self.use_multi_ll:
            # sがバッチでも単一でも、同じデバイス上にlong型の0のTensorを返す
            return torch.zeros(s.shape[0], dtype=torch.long, device=s.device)

        # x, y 座標をバッチで取得
        x = s[:, 0]
        y = s[:, 1]
        
        # torch.zerosで結果を初期化 (デフォルトはエリア0)
        area_ids = torch.zeros_like(x, dtype=torch.long)
        
        # PyTorchの論理インデキシングで条件に合う要素を一括で更新
        area_ids[(x < 0) & (y >= 0)] = 1  # 第2象限
        area_ids[(x < 0) & (y < 0)]  = 2  # 第3象限
        area_ids[(x >= 0) & (y < 0)] = 3  # 第4象限
        
        return area_ids

    def _update(self):
        transition , triangle_transition= self.buffer.sample(self.args.batch_size)
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']
        NG = transition['NG']
        AG = transition['AG']
        _, AG = self._preproc_inputs(None, AG)


        #for triangle loss        
        tri_S=triangle_transition['S']
        tri_A=triangle_transition['A']
        tri_via_AG=triangle_transition['via_AG']
        tri_via_S=triangle_transition['via_S']
        tri_via_A=triangle_transition['via_A']
        tri_end_AG=triangle_transition['end_AG']
        tri_gamma_t=triangle_transition['gamma_t']
        q_total_step=triangle_transition['q_total_step']


        tri_A = numpy2torch(tri_A, unsqueeze=False, cuda=self.args.cuda)
        tri_via_A = numpy2torch(tri_via_A, unsqueeze=False, cuda=self.args.cuda)
        tri_gamma_t = numpy2torch(np.array(tri_gamma_t), unsqueeze=True, cuda=self.args.cuda)
        q_total_step = numpy2torch(np.array(q_total_step), unsqueeze=False, cuda=self.args.cuda)

        tri_S, tri_via_AG = self._preproc_inputs(tri_S, tri_via_AG)
        tri_via_S, tri_end_AG = self._preproc_inputs(tri_via_S, tri_end_AG)
        # S/NS: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)

        #torchにする処理とclipをやってる
        S, G = self._preproc_inputs(S, G)
        NS, NG = self._preproc_inputs(NS, NG)





        with torch.no_grad():
            
            phih_nograd=self.critic_target.forward_encoder_sg(NS, G)         


            if self.use_multi_ll:
                # 次状態NSに応じて適切なターゲットLL Actorを選択
                area_ids_ns = self.get_area_id(NS)
                NA = torch.zeros_like(A)
                for i in range(self.num_ll_policies):
                    idx_mask = (area_ids_ns == i) # 論理マスクを作成
                    if idx_mask.any(): # マスクにTrueが1つでもあれば処理
                        NA[idx_mask] = self.prm_actor_targets[i](NS[idx_mask], G[idx_mask])
                        
            else:
                NA = self.prm_actor_target(NS, G)


            fh_nograd=self.critic_target.forward_encoder_sa(NS, NA)
            NQ = self.critic_target.forward_latent(fh_nograd,phih_nograd).detach()
            
            Q13=self.critic_target(tri_S, tri_A, tri_end_AG).detach()


            clip_return = 1 / (1 - self.args.gamma)

            target = (R + self.args.gamma * NQ).detach().clamp_(-clip_return, 0)
            Q13target = Q13.clamp_(-clip_return, 0)




        # ここでは、メイン損失と triangle loss の2つに分ける
        fh=self.critic.forward_encoder_sa(S, A)
        phih=self.critic.forward_encoder_sg(S, G)#
        Q = self.critic.forward_latent(fh , phih)
        main_loss = (Q - target).pow(2).mean()


        # triangle loss
        tri_Q12 = self.critic.forward(tri_S, tri_A, tri_via_AG)
        tri_Q23 = self.critic.forward(tri_via_S, tri_via_A, tri_end_AG)
        triangle_loss = ((tri_Q12 + tri_Q23 - Q13target).pow(2)/(q_total_step.pow(2))).mean()
        

        critic_loss = main_loss  +  triangle_loss * self.args.triangle_scale     

        

        self.critic_optim.zero_grad()



        (critic_loss*self.args.loss_scale).backward()
        critic_grad_norm = sync_grads(self.critic)
        self.critic_optim.step()



        pred_g=self.project_latent_to_goal(phih.detach())
        loss_vis=F.mse_loss(pred_g,G)
        self.project_latent_to_goal_optim.zero_grad()
        (loss_vis*self.args.loss_scale).backward()
        self.project_latent_to_goal_optim.step()



        
        # 初期の埋め込みとゴール潜在を得る
        for p in self.critic.parameters():
            p.requires_grad_(False)
        for p in self.project_latent_to_goal.parameters():
            p.requires_grad_(False)
        
        if self.use_multi_ll:
            for i in range(self.num_ll_policies):
                for p in self.prm_actors[i].parameters():
                    p.requires_grad_(False)
        else:
            for p in self.prm_actor.parameters():
                p.requires_grad_(False)

        area_ids = self.get_area_id(S)
        

        with torch.no_grad():
            goal_latent= self.critic.forward_encoder_sg(S,G).detach()
            if self.use_multi_ll:
                a_next = torch.zeros((S.shape[0], self.dim_action), device=S.device)
                for i in range(self.num_ll_policies):
                    idx_mask = (area_ids == i)
                    if idx_mask.any():
                        a_next[idx_mask] = self.prm_actors[i](S[idx_mask], G[idx_mask])

            else:
                a_next = self.prm_actor(S, G).detach()
            fh_k = self.critic.forward_encoder_sa(S, a_next).detach()


        q_sum    = 0.0
        

        plus_munus_n_layer_latent_num=0
        num_hops = max(1, self.high_level_actor_num+plus_munus_n_layer_latent_num)
        subgoals = []
        subgoals.append(goal_latent.detach())#0層
        #  multi-hop で z_sub を連鎖生成
        for ith_hop in range(num_hops):#self.high_level_actor_num):



            z_sub=self.critic.forward_encoder_head(self.hl_actor(fh_k, goal_latent))
            subgoals.append(z_sub.detach())





            Q23  = self.critic.forward_latent(z_sub, goal_latent)
            q_sum+=Q23

            goal_latent = z_sub

            # 次ホップ用に状態埋め込み fh_k を更新

            if self.use_multi_ll:
                
                a_next = torch.zeros_like(A)
                for i in range(self.num_ll_policies):
                    idx_mask = (area_ids == i)
                    if idx_mask.any():
                        a_next[idx_mask] = self.prm_actors[i](S[idx_mask], self.project_latent_to_goal(goal_latent[idx_mask]))

            else:
                a_next = self.prm_actor(S, self.project_latent_to_goal(goal_latent))
     
            fh_k = self.critic.forward_encoder_sa(S, a_next)




        Q12  = self.critic.forward_latent(fh_k,z_sub)
        q_sum+=Q12



        # ホップ数で平均化してスケール調整
        q_avg = q_sum


        # actor loss を計算
        hl_actor_loss = - q_avg.mean()

        # high-level actor の更新

        self.hl_actor_optim.zero_grad()
        (hl_actor_loss * self.args.loss_scale).backward()
        hl_actor_grad_norm = sync_grads(self.hl_actor)
        self.hl_actor_optim.step()

        

        if self.use_multi_ll:
            for i in range(self.num_ll_policies):
                for p in self.prm_actors[i].parameters():
                    p.requires_grad_(True)
        else:
            for p in self.prm_actor.parameters():
                p.requires_grad_(True)




        # --- Low-Level Policy Loss ---

        if self.use_multi_ll:
            total_prm_actor_loss = 0.0
            total_prm_actor_grad_norm = 0.0
            for i in range(self.num_ll_policies):
                idx_mask = (area_ids == i)
                if not idx_mask.any(): continue
                A_ = self.prm_actors[i](S[idx_mask], G[idx_mask])

                prm_actor_loss = -self.critic(S[idx_mask], A_, G[idx_mask]).mean() + self.args.action_l2 * (A_ / self.args.max_action).pow(2).mean()
                self.prm_actor_optims[i].zero_grad()
                (prm_actor_loss * self.args.loss_scale).backward()
                prm_grad_norm = sync_grads(self.prm_actors[i])
                self.prm_actor_optims[i].step()
                total_prm_actor_loss += prm_actor_loss
                total_prm_actor_grad_norm += prm_grad_norm

            #単一LLとそろえる
            prm_actor_loss=total_prm_actor_loss
            prm_actor_grad_norm=total_prm_actor_grad_norm


        else: # 単一LL-Policyの学習
            prm_actor_loss=0
            A_ = self.prm_actor(S, G)
            prm_actor_loss -=  self.critic(S, A_,G).mean()
            prm_actor_loss += self.args.action_l2 * (A_ / self.args.max_action).pow(2).mean()


            self.prm_actor_optim.zero_grad()
            (prm_actor_loss*self.args.loss_scale).backward()
            prm_actor_grad_norm = sync_grads(self.prm_actor)
            self.prm_actor_optim.step()



        for p in self.critic.parameters():
            p.requires_grad_(True)

        for p in self.project_latent_to_goal.parameters():
            p.requires_grad_(True)


        
        return hl_actor_loss.item(),prm_actor_loss.item(), critic_loss.item(), hl_actor_grad_norm,prm_actor_grad_norm,  critic_grad_norm



    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            log_dir =  f"./tensorboard/{self.env_name}/{self.args.experiment_name}"
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'hl_actor_losses': [],
                'prm_actor_losses': [],                
                'critic_losses': [],
                'hl_actor_grad_norms': [],
                'prm_actor_grad_norms': [],
                'critic_grad_norms': [],
            }

        # put something to the buffer first
        self.prefill_buffer()

        last_best_L    = None
        same_best_cnt  = 0

        for epoch in range(self.args.n_epochs):



            HL_AL,PRM_AL, CL, HL_AGN,PRM_AGN, CGN = [], [], [], [],[],[]
            regen_rates = []
            for _ in range(self.args.n_cycles):


                (S, A, AG, G,latent_G), success,regen_ratio = self.collect_rollout()
                regen_rates.append(regen_ratio)
                self.buffer.store_episode(S, A, AG, G,latent_G)
                self._update_normalizer(S, A, AG, G)
                for _ in range(self.args.n_batches):
                    hl_a_loss,prm_a_loss, c_loss, hl_a_gn,prm_a_gn, c_gn = self._update()
                    HL_AL.append(hl_a_loss); PRM_AL.append(prm_a_loss); CL.append(c_loss)
                    HL_AGN.append(hl_a_gn);PRM_AGN.append(prm_a_gn); CGN.append(c_gn)

                                
                if self.use_multi_ll:
                    for i in range(self.num_ll_policies):
                        self._soft_update(self.prm_actor_targets[i], self.prm_actors[i])
                else:
                    self._soft_update(self.prm_actor_target, self.prm_actor)


                self._soft_update(self.critic_target, self.critic)
                



            if MPI.COMM_WORLD.Get_rank() == 0:
                avg_regen = np.mean(regen_rates)
                writer.add_scalar('Epoch/AvgRegenRate', avg_regen, epoch)


                # 複数階層での評価を回す
                eval_results = {}
                for L in self.eval_hl_layers:
                    succ_L, time_L = self.eval_agent(L)
                    eval_results[L] = (succ_L, time_L)
                    writer.add_scalar(f'Eval/Success/L{L}', succ_L, epoch)
                    writer.add_scalar(f'Eval/Time/L{L}',     time_L, epoch)
                
                # select among current layer ±1
                curr = self.high_level_actor_num
                neighbors = [ curr]
                # ensure non-negative indices
                candidates = neighbors#[L for L in neighbors if L >= 0 and L<=4]

                # for any candidate not yet evaluated, run eval
                for L in candidates:
                    if L not in eval_results:
                        succ_L, time_L = self.eval_agent(L)
                        eval_results[L] = (succ_L, time_L)
                
                # choose best candidate by success rate
                best_L = max(candidates, key=lambda L: eval_results[L][0])




                writer.add_scalar('Eval/Used_L', self.high_level_actor_num, epoch)

                if self.args.visualize_flag and epoch%5==0:
                    self.visualize_agent(self.high_level_actor_num, f"./tensorboard/{self.env_name}/{self.args.experiment_name}_{epoch}.mp4",30)

                # record chosen metrics

                global_success_rate, global_hitting_time = eval_results[self.high_level_actor_num]

                t1 = time.time()
                HL_AL = np.array(HL_AL);PRM_AL = np.array(PRM_AL); CL = np.array(CL)
                HL_AGN = np.array(HL_AGN); PRM_AGN = np.array(PRM_AGN); CGN = np.array(CGN)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['hl_actor_losses'].append(HL_AL.mean())
                stats['prm_actor_losses'].append(PRM_AL.mean())
                stats['critic_losses'].append(CL.mean())
                stats['hl_actor_grad_norms'].append(HL_AGN.mean())
                stats['prm_actor_grad_norms'].append(PRM_AGN.mean())
                stats['critic_grad_norms'].append(CGN.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f"hl actor loss {HL_AL.mean():6.4f} |prm actor loss {PRM_AL.mean():6.4f} | critic loss {CL.mean():6.4f} | "+\
                        f"hl actor gradnorm {HL_AGN.mean():6.4f} |prm actor gradnorm {PRM_AGN.mean():6.4f} | critic gradnorm {CGN.mean():6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")

                # TensorBoard へのログ出力
                writer.add_scalar('Success Rate', global_success_rate, epoch)
                writer.add_scalar('Hitting Time', global_hitting_time, epoch)
                writer.add_scalar('HL Actor Loss', HL_AL.mean(), epoch)
                writer.add_scalar('PRM Actor Loss', PRM_AL.mean(), epoch)
                writer.add_scalar('Critic Loss', CL.mean(), epoch)
                writer.add_scalar('HL Actor Grad Norm', HL_AGN.mean(), epoch)
                writer.add_scalar('PRM Actor Grad Norm', PRM_AGN.mean(), epoch)
                writer.add_scalar('Critic Grad Norm', CGN.mean(), epoch)
                writer.flush()

                # ここで損失がNaNになっていないかをチェック
                if np.isnan(HL_AL.mean()) or np.isnan(PRM_AL.mean()) or np.isnan(CL.mean()):
                    print("[error] NaN loss encountered. Exiting.")
                    sys.exit(1)
                # エポック20以降に過去3エポックの成功率の平均が0.1以下ならプログラム終了
                if epoch >= self.eval_epoch_for_this_env:
                    recent_success = stats['successes'][-3:]
                    if np.mean(recent_success) < 0.02:#修正
                        print("[warning] Past 3 epochs' average success rate is below 0.1. Terminating training.")
                        sys.exit(1)

                    
        if MPI.COMM_WORLD.Get_rank() == 0:
            writer.close()
                
    def eval_agent(self, num_hops):
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

                # 最終ゴール潜在
                goal_latent = self.critic.forward_encoder_sg(s_tensor, g_tensor)

                s_id = self.get_area_id(s_tensor)
                for lvl in range(num_hops):
                    if self.use_multi_ll:
                        a_est = torch.zeros((s_tensor.shape[0], self.dim_action), device=s_tensor.device)
                        for i in range(self.num_ll_policies):
                            idx_mask = (s_id == i)
                            if idx_mask.any():
                                a_est[idx_mask] = self.prm_actors[i](s_tensor[idx_mask], self.project_latent_to_goal(goal_latent[idx_mask]))
                    else:
                        a_est = self.prm_actor(s_tensor, self.project_latent_to_goal(goal_latent))#最も遠いゴールから推定したA
                    
                    fh_cur = self.critic.forward_encoder_sa(s_tensor, a_est)
                    goal_latent = self.critic.forward_encoder_head(self.hl_actor(fh_cur, goal_latent))

                if num_hops == 0:
                    final_g = g_tensor
                else:
                    final_g = self.project_latent_to_goal(goal_latent)
                
                if self.use_multi_ll:
                    a_tensor = torch.zeros((s_tensor.shape[0], self.dim_action), device=s_tensor.device)
                    for i in range(self.num_ll_policies):
                        idx_mask = (s_id == i)
                        if idx_mask.any():
                            # num_hops=0 の場合とそれ以外で final_g の参照方法を統一
                            a_tensor[idx_mask] = self.prm_actors[i](s_tensor[idx_mask], final_g[idx_mask])
                else: # single policy
                    a_tensor = self.prm_actor(s_tensor, final_g)



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



    def visualize_agent(
            self,
            num_hops: int,
            save_path: str = "agent_playback.mp4",
            fps: int = 30,
            img_size: tuple[int, int] = (640, 480),
    ):

        import imageio, numpy as np, torch, mujoco
        from mujoco import mjv_initGeom, mjtGeom, Renderer

        env   = self.args.visualize_env
        model = env.unwrapped.model       # MjModel
        data  = env.unwrapped.data        # MjData
        H, W  = img_size

        # --- Renderer を作る前に offscreen バッファを広げる -----------------
        # MuJoCo 3.x では vis.global.offwidth / offheight を書き換えられる
        if model.vis.global_.offwidth  < W:
            model.vis.global_.offwidth  = W
        if model.vis.global_.offheight < H:
            model.vis.global_.offheight = H
        # オフスクリーンレンダラ
        renderer = Renderer(model, H, W)

        # 球体の色 (0–1 正規化 RGBA)
        base_colors = [
            (1, 1, 1, 0.3), (0, 0, 0, 0.3), (1, 0, 0, 0.3),
            (1, 0, 1, 0.3), (0, 0, 1, 0.3), (0, 1, 1, 0.3), (0, 1, 0, 0.3), (1, 1, 0, 0.3),
        ]

        frames = []
        o, _   = env.reset()
        T      = self.args.max_episode_steps

        for _ in range(T):
            # ------------------------------------------------
            # ポリシー推論：subgoals_world を計算
            # ------------------------------------------------
            obs = o["observation"];  ag = o["achieved_goal"]
            s   = np.concatenate((obs, ag), axis=0).astype(np.float32) \
                if "AntMaze" in env.spec.id else obs.astype(np.float32)
            g   = o["desired_goal"].astype(np.float32)

            s_tensor, g_tensor = self._preproc_inputs(s, g, unsqueeze=True)

            subgoals_norm = []

            with torch.inference_mode():

                # 最終ゴール潜在
                goal_latent = self.critic.forward_encoder_sg(s_tensor, g_tensor)

                subgoals_norm.append(
                    self.project_latent_to_goal(goal_latent).cpu().numpy().squeeze()
                )

                s_id = self.get_area_id(s_tensor)
                ll_id_display = int(s_id.item()) if self.use_multi_ll else 0
                
                for _ in range(num_hops):
                    if self.use_multi_ll:
                        a_est = torch.zeros((s_tensor.shape[0], self.dim_action), device=s_tensor.device)
                        for i in range(self.num_ll_policies):
                            idx_mask = (s_id == i)
                            if idx_mask.any():
                                a_est[idx_mask] = self.prm_actors[i](s_tensor[idx_mask], self.project_latent_to_goal(goal_latent[idx_mask]))
                    else:
                        a_est = self.prm_actor(s_tensor, self.project_latent_to_goal(goal_latent))#最も遠いゴールから推定したA
                    
                    fh_cur = self.critic.forward_encoder_sa(s_tensor, a_est)
                    goal_latent = self.critic.forward_encoder_head(self.hl_actor(fh_cur, goal_latent))
                    subgoals_norm.append(
                        self.project_latent_to_goal(goal_latent).cpu().numpy().squeeze()
                    )
                    
                if num_hops == 0:
                    final_g = g_tensor
                else:
                    final_g = self.project_latent_to_goal(goal_latent)
                
                if self.use_multi_ll:
                    a_tensor = torch.zeros((s_tensor.shape[0], self.dim_action), device=s_tensor.device)
                    for i in range(self.num_ll_policies):
                        idx_mask = (s_id == i)
                        if idx_mask.any():
                            # num_hops=0 の場合とそれ以外で final_g の参照方法を統一
                            a_tensor[idx_mask] = self.prm_actors[i](s_tensor[idx_mask], final_g[idx_mask])
                else: # single policy
                    a_tensor = self.prm_actor(s_tensor, final_g)




            actions = np.clip(
                a_tensor.cpu().numpy().squeeze(0),
                -self.args.max_action,
                self.args.max_action,
            )

            # 正規化 → ワールド座標
            subgoals_world = [g] + [
                self.g_norm.unnormalize(gn) for gn in subgoals_norm
            ]

            #spheresize=0.10 if "PointMaze" in env.spec.id else 0.03
            spheresize = 10.0 if "AntMaze" in env.spec.id else 0.10 if "PointMaze" in env.spec.id else 0.03


            # ------------------------------------------------
            #  シーン更新：既存 geoms + subgoal 球体
            # ------------------------------------------------
            renderer.update_scene(data)         # MuJoCo 内蔵の geoms をまず更新
            scn = renderer.scene 

            base = scn.ngeom                     #  ここで環境ジオメトリ数を保存
            need = len(subgoals_world)
            for i, pos in enumerate(subgoals_world):
                idx = base + i
                if pos.shape[0] == 2:
                    pos3 = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
                else:
                    pos3 = np.asarray(pos, dtype=np.float64)

                mjv_initGeom(
                    scn.geoms[idx],
                    mjtGeom.mjGEOM_SPHERE,
                    size=np.array([spheresize, 0.0, 0.0], dtype=np.float64),
                    pos=pos3,
                    mat=np.eye(3, dtype=np.float64).ravel(),
                    rgba=np.asarray(base_colors[i % len(base_colors)], dtype=np.float32),
                )

            scn.ngeom = base + need              # ← 総数を「環境+追加」に更新
            # ------------------------------------------------
            # 描画 → フレーム収集 & シミュレーション 
            # ------------------------------------------------

            frame = renderer.render()            # H×W×3 uint8 (numpy array)

            _color_map = {
                0: (160, 160, 160),  # single-LL もここ（灰）
                1: (255,  64,  64),
                2: ( 64, 255,  64),
                3: ( 64,  64, 255),
            }
            _c = _color_map.get(ll_id_display, (160,160,160))
            _bar_h = max(4, H // 50)  # フレーム高さの約2%をバー高さに
            frame[:_bar_h, :, 0] = _c[0]
            frame[:_bar_h, :, 1] = _c[1]
            frame[:_bar_h, :, 2] = _c[2]
            frames.append(frame)
            o, _, _, _, _ = env.step(actions)

        # ------------------------------------------------
        # 動画書き出し
        # ------------------------------------------------
        imageio.mimwrite(save_path, frames, fps=fps, quality=8)
        print(f"動画を保存しました: {save_path}")




    def collect_rollout(self, uniform_random_action=False, stochastic=True):
        n_episodes = self.args.rollout_n_episodes
        dim_state  = self.args.dim_state
        dim_action = self.args.dim_action
        dim_goal   = self.args.dim_goal
        dim_latent = self.dim_hierarchical_latent
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

                    # 最終ゴール潜在
                    goal_latent = self.critic.forward_encoder_sg(s_tensor, g_tensor)

                    s_id = self.get_area_id(s_tensor)
                    for lvl in range(self.high_level_actor_num):
                        if self.use_multi_ll:
                            a_est = torch.zeros((s_tensor.shape[0], self.dim_action), device=s_tensor.device)
                            for i in range(self.num_ll_policies):
                                idx_mask = (s_id == i)
                                if idx_mask.any():
                                    a_est[idx_mask] = self.prm_actors[i](s_tensor[idx_mask], self.project_latent_to_goal(goal_latent[idx_mask]))
                        else:
                            a_est = self.prm_actor(s_tensor, self.project_latent_to_goal(goal_latent))#最も遠いゴールから推定したA
                        
                        fh_cur = self.critic.forward_encoder_sa(s_tensor, a_est)
                        goal_latent = self.critic.forward_encoder_head(self.hl_actor(fh_cur, goal_latent))

                    if self.high_level_actor_num==0:
                        final_g = g_tensor
                    else:
                        final_g = self.project_latent_to_goal(goal_latent)
                    
                    if self.use_multi_ll:
                        a_tensor = torch.zeros((s_tensor.shape[0], self.dim_action), device=s_tensor.device)
                        for i in range(self.num_ll_policies):
                            idx_mask = (s_id == i)
                            if idx_mask.any():
                                # num_hops=0 の場合とそれ以外で final_g の参照方法を統一
                                a_tensor[idx_mask] = self.prm_actors[i](s_tensor[idx_mask], final_g[idx_mask])
                    else: # single policy
                        a_tensor = self.prm_actor(s_tensor, final_g)

    

                        

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


        latent_G=None#今回はNoneを返す



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

        regen_ratio = -1.0  # 今回は未計算


            
        return (S, A, AG, G,latent_G), success,regen_ratio





    
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

