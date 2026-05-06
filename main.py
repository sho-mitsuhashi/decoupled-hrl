

import gymnasium as gym
import gymnasium_robotics
from gymnasium.vector import AsyncVectorEnv 
###
import numpy as np
import os
import random
import torch

from mpi4py import MPI
from src.args import get_args
from src.agent import DDPG, HER, HER_TRIANGLE, Hierarchical 


def make_env(args):
    dic = {
        'FetchPush' : 'FetchPush-v4',
        'FetchSlide': 'FetchSlide-v4',
        'FetchPick' : 'FetchPickAndPlace-v4',
        'HandManipulateEggRotate'          : 'HandManipulateEggRotate-v1',
        'PointMaze_Medium_Diverse_GR':'PointMaze_Medium_Diverse_GR-v3',
        
    }


    max_envs = max(args.rollout_n_episodes,args.eval_rollout_n_episodes)
    env_id = args.env_name

    def make_single_env(visualize=False):
        kwargs = {}
        if "AntMaze" in env_id:
            kwargs["include_cfrc_ext_in_observation"] = False

        # 可視化フラグが立っていれば RGB 配列取得モードに
        if visualize:
            kwargs["render_mode"] = "rgb_array"

        single_env=gym.make(dic[env_id], **kwargs)
        return single_env
    try:

        env_fns = [make_single_env for _ in range(max_envs)]
        env = AsyncVectorEnv(env_fns)
        raw = make_single_env(visualize=True)#tmp env for attribute
        args.bandit_eval_envs=AsyncVectorEnv([make_single_env])
    except:
        raise Exception(
                f"[error] unknown environment name {args.env_name}")



    # let argument know max episode length
    args.max_episode_steps = raw._max_episode_steps
    args.dim_action = raw.action_space.shape[0]
    args.max_action = raw.action_space.high[0]
    args.reward_func=raw.unwrapped.compute_reward
    if args.visualize_flag:
        args.visualize_env=raw
    else:
        raw.close()

    eval_epoch_dic = {#このepoch以上のときに過去３stepの成功率が0.1以下なら見込み無しとして停止
    #0.99以上でも終了
        'FetchPush' :10000,
        'FetchSlide': 10000,
        'FetchPick' : 10000,
        'HandManipulateEggRotate'          : 10000,
        'PointMaze_Medium_Diverse_GR':10000,
    }
    args.eval_epoch_for_this_env=eval_epoch_dic[env_id]

    return env


def setup(args, env):
    obs,info = env.reset()#M
    o, ag, g = obs['observation'][0], obs['achieved_goal'][0], obs['desired_goal'][0]
    if 'AntMaze' in args.env_name:#agを追加するための次元
        args.dim_state  = o.shape[0]+2
    else:
        args.dim_state  = o.shape[0]
    args.dim_goal   = g.shape[0]



    suffix = "(+)rew" if not args.negative_reward else "(-)rew"################
    if args.agent in ["her",  "ddpg", "her_triangle",'Hierarchical']:
        suffix += f"_{args.critic}"
        if args.critic != "monolithic":
            suffix += f"_emb{args.dim_embed}"
        if args.terminate:
            suffix += "_terminate"
        if args.adversarial_flag:
            suffix += "_adv"


    args.experiment_name = f"{args.env_name}_{args.agent}_{suffix}_{args.triangle_scale}_{args.triangle_eq_step}_lr{args.lr_critic}_{args.high_level_actor_num}_{args.Subgoal_est_horizon}_{args.gradnorm_alpha}_{args.tmp_weight}_{args.lambda_mi}_{args.dim_prm_hidden}_{args.dim_hierarchical_latent}_{args.dim_hidden}_{args.dim_critic_hidden}_sd{args.seed}"

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"[info] start experiment {args.experiment_name}")


def main(args):
    # create environment
    env = make_env(args)

    # control seed
    env.reset(seed=args.seed + MPI.COMM_WORLD.Get_rank())#M
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # update arguments based on environment
    setup(args, env)
    ##########################
    agent_map = {
        'ddpg'    : DDPG,
        'her'     : HER,
        'her_triangle': HER_TRIANGLE,
        'Hierarchical' : Hierarchical,
    }
    agent = agent_map[args.agent](args, env)
    agent.learn()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    args = get_args()
    main(args)
