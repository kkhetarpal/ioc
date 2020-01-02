
# from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
# from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
# from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv

# from rllab.envs.mujoco.hill.half_cheetah_hill_env import HalfCheetahHillEnv
# from rllab.envs.mujoco.hill.swimmer3d_hill_env import Swimmer3DHillEnv
import pdb
import time
import gym
import numpy as np
# import my_gym;
#from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
# from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
# from rllab.envs.mujoco.gather.point_gather_env import PointGatherEnv
# from rllab.envs.box2d.mountain_car_env import MountainCarEnv
#from twod_tmaze2 import TMaze2
#from antwalls import AntWallsEnv

import time
import gym_miniworld
from gym_miniworld.entity import Box as miniBox
from gym_miniworld.envs.oneroom import OneRoom


#from antmaze import AntMazeEnv

# from wheeled import WheeledEnv
# from wheeled_maze import WheeledMazeEnv
# from blockplaypen import BlockPlayPen
# from twod_multi import TwoDMultiEnv
# env = BlockPlayPen()
# env = TwoDMaze()
# env = TwoDMultiEnv()


#env = SwimmerGatherEnv()
#env = AntMazeEnv()

#env = gym.make('MiniWorld-Hallway-v0')
#env = gym.make('MiniWorld-OneRoom-v0')
#env = gym.make('MiniWorld-PutNext-v0')
env = gym.make('MiniWorld-PickupObjs-v0')


#env=AntWallsEnv()
#env= TMaze2()
# env= gym.make("Reacher-v1")
# env.seed(0)
# pdb.set_trace()
# env= PointMazeEnv()
# env = gym.make("Acrobot-v1")
#env.reset()
# env.render()
# state,reward, done, _ = env.step(np.array([0.,10.]))
# env.render()
# state,reward, done, _ = env.step(np.array([0.,10.]))
# env.render()
# state,reward, done, _ = env.step(np.array([0.,10.]))

episodes = 0

for step in range(500):
	env.render()
	time.sleep(1)
	# pdb.set_trace()
	# print(t)
	# if True:
	# 	continue
	# 	print("aaa")
	# state,reward, done, _ = env.step(np.array([0.,0.]))
	# pdb.set_trace()
	state,reward, done, _ = env.step(env.action_space.sample())
	#print(env.box.pos)

	done = True
	if done:
		#pdb.set_trace()
		env.reset()
		episodes += 1

		# if episodes == 10:
		# 	env = OneRoom(change_goal=True)
		#
		# time.sleep(0.1)