import logging

import torch

from .agent import Agent


class TDAgent(Agent):

    def get_return(self, memory, q_tensor):
        s_prime_tensor = memory.s_prime_tensor
        a_prime_tensor = self.get_action(s_prime_tensor, explore=False)
        q_prime_tensor = self._model(s_prime_tensor, a_prime_tensor)

        r_list = memory.r_list()
        g_list = list(r_list)
        for i in range(len(g_list)):
            g_list[i] += self._gamma * q_prime_tensor[i]
        return torch.tensor(g_list)

    def learn(self, env, epochs=100, render=False):
        for epoch in range(epochs):
            state = [.0, .0, .0, .0]
            done = False
            steps = 0
            loss_sum = 0.0
            while not done:
                action = self.get_one_action(state)
                ob, reward, done, info = env.step(action)
                if done and steps < 199:
                    reward = 0
                self.record(state, action, reward, s_prime=ob)
                if render:
                    env.render()
                state = ob
                steps += 1
                loss = self.train()
                loss_sum += loss

            logging.info('Epoch {}: step={}, loss={}'.format(epoch, steps, loss_sum / steps))
            env.reset()
