import numpy as np


class Sampler(object):
    """
    Helper class to sample transitions for learning.
    Methods like sample_her_transitions will relabel part of trajectories.
    """
    def __init__(self, args, env_reward_func):
        self.relabel_rate = args.relabel_rate
        plus = 1.0 if not args.negative_reward else 0.0
        # make reward to {-1, 0} instead of {0, 1} if negative reward
        #mazeenv has 0,1 reward
        maze_reward_minus=0
        if 'Maze' in args.env_name:
            maze_reward_minus=-1.0

        self.reward_func = lambda ag, g, c: env_reward_func(ag, g, c) + plus + maze_reward_minus



        self.global_threshold = 80
        self.triangle_eq_step =args.triangle_eq_step

        self.Subgoal_est_horizon=args.Subgoal_est_horizon



    def sample_ddpg_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
        }
        return transition

    def sample_her_transitions(self, S, A, AG, G, size, latent_G=None):
        B, T = A.shape[:2]
        epi_idx = np.random.randint(0, B, size)
        t       = np.random.randint(T,   size=size)

        # future_t_all は目標リラベリング用
        future_offset  = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t_all   = t + 1 + future_offset

        her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)[0]
        mask    = np.zeros(size, dtype=np.float32)
        mask[her_idx] = 1.0


        horizon = self.Subgoal_est_horizon

        low_i    = np.maximum(0, t - horizon)

        widths   = t - low_i + 1
        # [0, widths) のオフセットを一様にサンプリング
        offsets  = (np.random.uniform(size=size) * widths).astype(int)
        # 最終的な i
        i        = low_i + offsets


        S_mid = S[epi_idx, i].copy()



        S_   = S[epi_idx, t].copy()
        A_   = A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   = G[epi_idx, t].copy()
        NS_  = S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        her_AG      = AG[epi_idx[her_idx], future_t_all[her_idx]]
        G_[her_idx] = her_AG

        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1)


        result = {
            'S':     S_, 'NS': NS_,
            'A':     A_, 'G':  G_,
            'R':     R_, 'NG': NAG_,
            'mask':  mask, 'AG': AG_,
            'S_mid': S_mid,
        }
        # latent_G が渡された場合のみ追加
        if latent_G is not None:
            result['latent_G'] = latent_G[epi_idx, t].copy()

        return result



    def sample_triangle_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch

        epi_idx = np.random.randint(0, B, size)


        max_step = T 
        triangle_eq_step = np.random.randint(2, max_step, size=size)  # shape=(size,)


        t_max = T - triangle_eq_step


        t = (np.random.rand(size) * t_max).astype(int)  # shape=(size,)
        tviaoffset = np.random.randint(1, triangle_eq_step, size=size)


        t_via = t + tviaoffset
        t_end = t + triangle_eq_step


        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        via_AG_   =  AG[epi_idx, t_via].copy()
        via_S_   =  S[epi_idx, t_via].copy() # (size, dim_state)
        via_A_   =  A[epi_idx, t_via].copy()
        end_AG_   =  AG[epi_idx,t_end].copy()

        gamma_t =tviaoffset






        triangle_transition = {
            'S' : S_,
            'A' : A_,
            'via_AG' : via_AG_,
            'via_S':via_S_,
            'via_A':via_A_,
            'end_AG' : end_AG_,
            'gamma_t':gamma_t,
            'q_total_step':triangle_eq_step,

            'S_hl':S_,
            'A_hl':A_,
            'mid_AG_hl':via_AG_,
            'end_AG_hl':end_AG_,
        }
        return triangle_transition

    