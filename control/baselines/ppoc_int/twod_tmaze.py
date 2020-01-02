import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import seeding


class TwoDEnv(mujoco_env.MujocoEnv):
    def __init__(self, model_path, frame_skip, xbounds, ybounds):
        super(TwoDEnv, self).__init__(model_path=model_path, frame_skip=frame_skip)
        assert isinstance(self.observation_space, Box)
        assert self.observation_space.shape == (2,)
        
    def get_viewer(self):
        return self._get_viewer()

import numpy as np
from gym import utils
import os




def get_asset_xml(xml_name):
    return os.path.join(os.path.join(os.path.dirname(__file__), 'assets'), xml_name)
    
class TMaze(TwoDEnv, utils.EzPickle):
    NAME='TMaze'
    def __init__(self, verbose=False,change_goal=None):
        self.verbose = verbose
        self.steps = 0
        self.change_goal = change_goal
        utils.EzPickle.__init__(self)
        TwoDEnv.__init__(self, get_asset_xml('twod_tmaze.xml'), 2, xbounds=[-0.3,0.3], ybounds=[-0.3,0.3])
        

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:2]
        
        if not self.change_goal:
            target = self.model.body_pos.copy()[-1][:2]
        else: 
            target = self.change_goal
        dist_thresh = 0.1



        if pos[0]>target[0]-dist_thresh and pos[0]<target[0]+dist_thresh\
         and pos[1]<target[1]+dist_thresh and pos[1]>target[1]-dist_thresh:
            reward = 1.
        else:
            reward = 0.

        self.steps += 1
        if self.verbose:
            print(pos, reward)
        done = self.steps >= 500 or int(reward)
        return ob, reward, done, np.concatenate([self.model.data.qvel]).ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        init_pos = self.model.body_pos.copy()[1][:2]
        return np.concatenate([self.model.data.qpos]).ravel() + init_pos

    def viewer_setup(self):
        v = self.viewer

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
