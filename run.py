import logging
import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import TDAgent


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
        self.a_fc_layer = torch.nn.Linear(64, 64)

        self.state_trans = torch.nn.Linear(4, 64)
        self.s_fc_layer = torch.nn.Linear(64, 64)

        self.fc = torch.nn.Linear(128, 128)
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

        a_embed = F.relu(self.action_embed(a_tensor))
        a_embed = F.relu(self.a_fc_layer(a_embed))

        s_embed = F.relu(self.state_trans(s_tensor))
        s_embed = F.relu(self.s_fc_layer(s_embed))

        sa = torch.cat((s_embed, a_embed), dim=-1)
        sa = F.relu(self.fc(sa))
        sa = F.relu(self.fc2(sa))

        q = torch.squeeze(self.output(sa))
        return q


def main(**kwargs):
    env = gym.make('CartPole-v0')
    env.reset()
    net = Net()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
    agent = TDAgent(net, env.action_space.n, gamma=0.5, epsilon=0.1,
                    optimizer=optimizer, model_file='./model/mc.model')

    for epoch in range(300):
        state = np.array([0, 0, 0, 0], dtype=np.float32)
        done = False
        steps = 0
        while not done:
            action = agent.get_action(state)
            ob, reward, done, info = env.step(action)
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
