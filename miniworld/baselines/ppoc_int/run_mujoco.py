# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
# import gym_miniworld
import pdb
from baselines import logger
import sys
# from gym_miniworld.wrappers import GreyscaleWrapper

def train(env_id, num_timesteps, seed, num_options,app, saves ,wsaves, epoch,dc,plots,w_intfc,switch,mainlr,intlr,fewshot):
    from baselines.ppoc_int import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    
    if env_id=="TMaze":
        from twod_tmaze import TMaze
        env=TMaze()
        env.seed(seed) 
    elif env_id=="TMaze2":
        from twod_tmaze2 import TMaze2
        env=TMaze2()
        env.seed(seed)         
    elif env_id=="AntWalls":      
        from antwalls import AntWallsEnv
        env=AntWallsEnv()
        env.seed(seed)
    elif env_id=="AntMaze":
        from ant_maze_env import AntMazeEnv
        mazeid = 'Maze'
        env = AntMazeEnv(mazeid)
        env.seed(seed)
    else:
        env = gym.make(env_id)
        env._seed(seed)


    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, dc=dc, w_intfc=w_intfc)

    gym.logger.setLevel(logging.WARN)

    optimsize=int(64/num_options)

    # pdb.set_trace()
    num_timesteps = num_timesteps if env_id!="TMaze" else 5e5
    tperbatch = 2048 if not epoch else int(1e4)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=tperbatch,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=mainlr, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,dc=dc,plots=plots,
            w_intfc=w_intfc,switch=switch,intlr=intlr,fewshot=fewshot
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='TMaze')
    parser.add_argument('--timesteps', help='number of timesteps', type=int, default=1000000)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--plots', dest='plots', action='store_true', default=False)    
    parser.add_argument('--switch', dest='switch', action='store_true', default=False)    
    parser.add_argument('--fewshot', dest='fewshot', action='store_true', default=False)
    parser.add_argument('--nointfc', dest='w_intfc', action='store_false', default=True)
    parser.add_argument('--epoch', help='Epoch', type=int, default=0) 
    parser.add_argument('--dc', type=float, default=0.)
    parser.add_argument('--mainlr', type=float, default=3e-4)
    parser.add_argument('--intlr', type=float, default=1e-4)

    # pdb.set_trace()
    args = parser.parse_args()

    train(args.env, num_timesteps=args.timesteps, seed=args.seed, num_options=args.opt, app=args.app,
     saves=args.saves, wsaves=args.wsaves, epoch=args.epoch,dc=args.dc,plots=args.plots,
     w_intfc=args.w_intfc,switch=args.switch,mainlr=args.mainlr,intlr=args.intlr,fewshot=args.fewshot)


if __name__ == '__main__':
    main()