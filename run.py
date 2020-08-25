import os
import logging
import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--save', '-S', action='store_true')
    parser.add_argument('--verbose', '-V', action='store_true')
    return parser


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.action_embed = torch.nn.Embedding(2, 64)
        self.state_trans = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(128, 128)
        self.output = torch.nn.Linear(128, 1)

    def forward(self, s, a):
        if not isinstance(s, torch.Tensor):
            s_tensor = torch.tensor(s, dtype=torch.float)
        else:
            s_tensor = s
        if not isinstance(a, torch.Tensor):
            a_tensor = torch.tensor(a, dtype=torch.long)
        else:
            a_tensor = a
        a_embed = self.action_embed(a_tensor)
        s_embed = F.relu(self.state_trans(s_tensor))
        sa = torch.cat((s_embed, a_embed), dim=-1)
        sa = F.relu(self.fc2(sa))
        q = torch.squeeze(self.output(sa))
        return q


class Memory:

    def __init__(self):
        self._s = []
        self._a = []
        self._r = []

    def add(self, s, a, r):
        self._s.append(s)
        self._a.append(a)
        self._r.append(r)

    @property
    def s_tensor(self):
        return torch.tensor(self._s, dtype=torch.float)

    @property
    def a_tensor(self):
        return torch.tensor(self._a, dtype=torch.long)

    @property
    def r_tensor(self):
        return torch.tensor(self._r, dtype=torch.float)

    def r_list(self):
        return list(self._r)

    def forget(self):
        self._s.clear()
        self._a.clear()
        self._r.clear()


class MCAgent:

    def __init__(self, model: torch.nn.Module, action_space, epsilon=0.05, gamma=0.1, model_file=None):
        self._model = model
        self._action_space = action_space
        self._epsilon = epsilon
        self._gamma = gamma

        self._memory = Memory()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
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

        g_list = self._memory.r_list()
        for i in range(len(g_list) - 2, -1, -1):
            g_list[i] += self._gamma * g_list[i+1]
        g_tensor = torch.tensor(g_list)

        outputs = self._model(self._memory.s_tensor, self._memory.a_tensor)

        loss = self._criterion(outputs, g_tensor)
        loss.backward()
        self._optimizer.step()

        self._memory.forget()

        return loss.item()

    def dump(self, file=None):
        if file is None:
            file = self._model_file
        if file is None:
            return
        torch.save(self._model.state_dict(), file)


def main(**kwargs):
    env = gym.make('CartPole-v0')
    env.reset()
    net = Net()

    agent = MCAgent(net, env.action_space.n, gamma=0.2, model_file='./model/mc.model')

    for epoch in range(300):
        state = np.array([0, 0, 0, 0], dtype=np.float32)
        done = False
        steps = 0
        while not done:
            action = agent.get_action(state, verbose=kwargs['verbose'])
            ob, reward, done, info = env.step(action)
            if done and steps != 199:
                reward = 0
            agent.record(state, action, reward)
            if kwargs['render']:
                env.render()
            state = ob
            steps += 1

        loss = agent.train()
        logging.info('Epoch {}: step={}, loss={}'.format(epoch, steps, loss))
        env.reset()
    env.close()
    if kwargs['save']:
        agent.dump()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(**vars(args))
