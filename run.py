import logging
import argparse

import gym
import torch
import torch.nn.functional as F

from agent import MCAgent


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--save', '-S', action='store_true')
    parser.add_argument('--verbose', '-V', action='store_true')
    return parser


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.action_embed = torch.nn.Embedding(2, 16)

        self.state_trans = torch.nn.Linear(4, 16)

        self.fc = torch.nn.Linear(32, 32)
        self.output = torch.nn.Linear(32, 1)

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
        s_embed = F.relu(self.state_trans(s_tensor))
        sa = torch.cat((s_embed, a_embed), dim=-1)

        q = torch.squeeze(self.output(sa), dim=-1)
        return q


def main(**kwargs):
    env = gym.make('CartPole-v0')
    env.reset()
    net = Net()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    agent = MCAgent(net, env.action_space.n, gamma=0.1, epsilon=0.05,
                    optimizer=optimizer, model_file='./model/mc.model')
    agent.learn(env, 300, render=kwargs['render'])

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
