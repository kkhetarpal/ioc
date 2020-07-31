#Environment File for Classic Fourrooms Grid World
import numpy as np
import gym
from gym import core, spaces
from gym.envs.registration import register
from random import uniform

#class Fourrooms(gym.Env):
class Fourrooms():
    def __init__(self,initstate_seed):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""


        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # Action Space: from any state the agent can perform one of the four actions; Up, Down, Left and Right
        self.action_space = spaces.Discrete(4)

        # Observation Space
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

        self.rng = np.random.RandomState(1234)

        self.initstate_seed = initstate_seed
        self.rng_init_state = np.random.RandomState(self.initstate_seed)

        self.tostate = {}

        self.occ_dict = dict(zip(range(self.observation_space.n),
                                 np.argwhere(self.occupancy.flatten() == 0).squeeze()))


        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1

        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)


    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    # def reset(self):
    #     state = self.rng.choice(self.init_states)
    #     self.currentcell = self.tocell[state]
    #     return state


    def reset(self):
        state = self.rng_init_state.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. 
        We consider a case in which rewards are zero on all state transitions 
        except the goal state which has a reward of +50.
        """

        reward = 0
        if self.rng.uniform() < 1/3:
            empty_cells = self.empty_around(self.currentcell)
            nextcell = empty_cells[self.rng.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + self.directions[action])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]

        if state == self.goal:
            reward = 50

        done = state == self.goal
        return state, reward, float(done), None

    register(
        id='Fourrooms-v0',
        entry_point='fourrooms:Fourrooms',
        timestep_limit=20000,
        reward_threshold=1,
    )
