# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
import gym, logging
from baselines import logger
from half_cheetah import *


def train(env_id, num_timesteps, seed, num_options,app, saves ,wsaves, epoch,dc,plots,w_intfc,switch,mainlr,intlr,piolr,fewshot,k):
    from baselines.ppoc_int import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    
    if env_id=="TMaze":
        from twod_tmaze import TMaze
        env=TMaze()
        env.seed(seed)             
    else:
        env = gym.make(env_id)
        env._seed(seed)


    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, dc=dc, w_intfc=w_intfc,k=k)

    gym.logger.setLevel(logging.WARN)

    if num_options ==1:
        optimsize=64
    elif num_options ==2:
        optimsize=32
    else:
        optimsize=int(64/num_options)


    num_timesteps = num_timesteps #if env_id!="TMaze" else 5e5
    tperbatch = 2048 if not epoch else int(1e4)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=tperbatch,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=mainlr, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,dc=dc,plots=plots,
            w_intfc=w_intfc,switch=switch,intlr=intlr,piolr=piolr,fewshot=fewshot,k=k
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='TMaze')
    parser.add_argument('--timesteps', help='number of timesteps', type=int, default=1e6) 
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', help='Save the returns at each iteration', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', help='Save the weights',dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--plots',  help='Plot some visualization', dest='plots', action='store_true', default=False)    
    parser.add_argument('--switch', help='Switch task after 150 iterations', dest='switch', action='store_true', default=False)    
    parser.add_argument('--fewshot', help='Value learning after 150 iterations', dest='fewshot', action='store_true', default=False)    
    parser.add_argument('--nointfc', help='Disables interet functions', dest='w_intfc', action='store_false', default=True)    
    parser.add_argument('--epoch', help='Load weights from a certain epoch', type=int, default=0) 
    parser.add_argument('--dc', help='Deliberation cost  (not used)', type=float, default=0.)
    parser.add_argument('--mainlr', type=float, default=3e-4)
    parser.add_argument('--intlr', type=float, default=1e-4)
    parser.add_argument('--piolr', type=float, default=1e-4)
    parser.add_argument('--k', type=float, default=0., help='threshold for interest function')




    args = parser.parse_args()

    train(args.env, num_timesteps=args.timesteps, seed=args.seed, num_options=args.opt, app=args.app,
     saves=args.saves, wsaves=args.wsaves, epoch=args.epoch,dc=args.dc,plots=args.plots,
     w_intfc=args.w_intfc,switch=args.switch,mainlr=args.mainlr,intlr=args.intlr,piolr=args.piolr,fewshot=args.fewshot,k=args.k)


if __name__ == '__main__':
    main()
