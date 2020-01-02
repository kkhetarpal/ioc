# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from mpi4py import MPI
from baselines import bench
import os.path as osp
import gym, logging
import gym_miniworld
from baselines import logger



def train(env_id, num_timesteps, seed, num_options, app, saves, wsaves, epoch, dc, plots, w_intfc, switch, mainlr, intlr, piolr, fewshot):
    from baselines.ppoc_int import cnn_policy, pposgd_simple
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)

    env = gym.make(env_id)
    env.seed(workerseed)


    def policy_fn(name, ob_space, ac_space):
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, num_options=num_options, dc=dc, w_intfc=w_intfc)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)))

    optimsize = int(64 / num_options)


    num_timesteps = num_timesteps
    tperbatch = 2048 if not epoch else int(1e4)
    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_batch=tperbatch,
                        clip_param=0.2, entcoeff=0.01,
                        optim_epochs=4, optim_stepsize=mainlr, optim_batchsize=optimsize,
                        gamma=0.99, lam=0.95, schedule='linear', num_options=num_options,
                        app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed, dc=dc, plots=plots,
                        w_intfc=w_intfc, switch=switch, intlr=intlr, piolr=piolr, fewshot=fewshot
                        )
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='MiniWorld-OneRoom-v0')
    parser.add_argument('--timesteps', help='number of timesteps', type=int, default=1000000)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2)
    parser.add_argument('--app', help='Append to folder name', type=str, default='')
    parser.add_argument('--saves', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)
    parser.add_argument('--plots', dest='plots', action='store_true', default=False)
    parser.add_argument('--switch', dest='switch', help='switch task after 150 iterations', action='store_true', default=False)
    parser.add_argument('--fewshot', dest='fewshot', help='value learning after 150 iterations', action='store_true', default=False)
    parser.add_argument('--nointfc', dest='w_intfc', help='disables interet functions', action='store_false', default=True)
    parser.add_argument('--epoch', help='Epoch', type=int, default=0)
    parser.add_argument('--dc', type=float, default=0.)
    parser.add_argument('--mainlr', type=float, default=3e-4)
    parser.add_argument('--intlr', type=float, default=1e-4)
    parser.add_argument('--piolr', type=float, default=1e-4)


    args = parser.parse_args()

    train(args.env, num_timesteps=args.timesteps, seed=args.seed, num_options=args.opt, app=args.app, saves=args.saves,
          wsaves=args.wsaves, epoch=args.epoch, dc=args.dc, plots=args.plots, w_intfc=args.w_intfc, switch=args.switch,
          mainlr=args.mainlr, intlr=args.intlr, piolr=args.piolr, fewshot=args.fewshot)


if __name__ == '__main__':
    main()