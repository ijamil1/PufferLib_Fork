'''A custom environment for the sharks and minnows game.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.sharks_and_minnows import binding

class SharksAndMinnows(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=256, height=256, num_agents=8, render_mode=None, log_interval=256 * 4, size=11, buf=None, seed=0):

        num_minnows = num_agents
        num_sharks = num_agents
        
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(2*(num_sharks) + 2,), dtype=np.float32)
        
        self.single_action_space = gymnasium.spaces.Discrete(5)

        self.render_mode = render_mode
        self.num_minnows = num_envs*num_minnows
        self.log_interval = log_interval
        self.num_agents = num_envs*num_minnows
   
        assert buf == None
        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_minnows:(i+1)*num_minnows],
                self.actions[i*num_minnows:(i+1)*num_minnows],
                self.rewards[i*num_minnows:(i+1)*num_minnows],
                self.terminals[i*num_minnows:(i+1)*num_minnows],
                self.truncations[i*num_minnows:(i+1)*num_minnows],
                seed, width=width, height=height,
                num_minnows=num_minnows, num_sharks=num_sharks)
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 512

    env = SharksAndMinnows(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(5, size=(CACHE, 1))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_minnows
        i += 1

    print('Sharks and Minnows SPS:', int(steps / (time.time() - start)))
