import logging

import numpy as np
import torch

from .agent import Agent


class MCAgent(Agent):

    def get_return(self, memory, q_tensor) -> torch.Tensor:
        r_list = memory.r_list()
        g_list = list(r_list)
        for i in range(len(g_list) - 2, -1, -1):
            g_list[i] += self._gamma * g_list[i+1]
        g_tensor = torch.tensor(g_list)
        return g_tensor

    def learn(self, env, epochs=100, render=False):
        for epoch in range(epochs):
            state = np.zeros(self._ob_space)
            done = False
            steps = 0
            while not done:
                action = self.get_one_action(state)
                ob, reward, done, info = env.step(action)
                self.record(state, action, reward)
                if render:
                    env.render()
                state = ob
                steps += 1
            loss = self.train()

            logging.info('Epoch {}: step={}, loss={}'.format(epoch, steps, loss))
            env.reset()
