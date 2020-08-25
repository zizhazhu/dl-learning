import os
import logging
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from memory import Memory


class Agent(metaclass=ABCMeta):

    def __init__(self, model: torch.nn.Module, action_space, epsilon=0.05, gamma=0.1,
                 optimizer: torch.optim.Optimizer = None,
                 model_file=None):
        self._model = model
        self._action_space = action_space
        self._epsilon = epsilon
        self._gamma = gamma

        self._memory = Memory()
        if optimizer is None:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        else:
            self._optimizer = optimizer
        self._criterion = nn.MSELoss()

        self._model_file = model_file
        if model_file is not None and os.path.exists(model_file):
            self._model.load_state_dict(torch.load(model_file))

    def get_action(self, s):
        # s.shape: m * n
        # a.shape: m
        max_q, max_a = 1, -1

        import random
        if random.uniform(0, 1) < self._epsilon:
            max_a = random.randint(0, self._action_space - 1)
            logging.debug('explore')
        else:
            for a in range(self._action_space):
                q = self._model(s, a).item()
                logging.debug('{}, {} has value: {}'.format(s, a, q))
                if max_a == -1 or max_q < q:
                    max_q, max_a = q, a
        logging.debug('choose {}'.format(max_a))
        return max_a

    def record(self, s, a, r):
        self._memory.add(s, a, r)

    def train(self):
        self._optimizer.zero_grad()

        memory = self._memory
        q_tensor = self._model(memory.s_tensor, memory.a_tensor)
        g_tensor = self.get_return(memory.r_list(), q_tensor)

        loss = self._criterion(q_tensor, g_tensor)
        loss.backward()
        self._optimizer.step()

        self._memory.forget()

        return loss.item()

    @abstractmethod
    def get_return(self, r_list, q_tensor):
        raise NotImplemented()

    def dump(self, file=None):
        if file is None:
            file = self._model_file
        if file is None:
            return
        torch.save(self._model.state_dict(), file)
