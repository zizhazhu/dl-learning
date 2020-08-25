import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.action_embed = torch.nn.Embedding(2, 64)
        self.state_trans = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(128, 128)
        self.output = torch.nn.Linear(128, 1)

    def forward(self, s, a):
        s_tensor = torch.tensor(s, dtype=torch.float)
        a_tensor = torch.tensor(a, dtype=torch.long)
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
        return torch.tensor(self._s)

    @property
    def a_tensor(self):
        return torch.tensor(self._a)

    @property
    def r_tensor(self):
        return torch.tensor(self._r)

    def r_list(self):
        return list(self._r)


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

    def get_action(self, s, verbose=False):
        # s.shape: m * n
        # a.shape: m
        max_q, max_a = 1, -1

        import random
        if random.uniform(0, 1) < self._epsilon:
            max_a = random.randint(0, self._action_space - 1)
            if verbose:
                print('explore {}'.format(max_a))
        else:
            for a in range(self._action_space):
                q = self._model(s, a).item()
                if verbose:
                    print('{}, {} has value: {}'.format(s, a, q))
                if max_a == -1 or max_q < q:
                    max_q, max_a = q, a

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

        return loss.item()

    def dump(self, file=None):
        if file is None:
            file = self._model_file
        if file is None:
            return
        torch.save(self._model.state_dict(), file)


def main():
    env = gym.make('CartPole-v0')
    env.reset()
    net = Net()

    agent = MCAgent(net, env.action_space.n, gamma=0.2, model_file='./model/mc.model')

    for epoch in range(300):
        state = np.array([0, 0, 0, 0], dtype=np.float32)
        done = False
        steps = 0
        while not done:
            action = agent.get_action(state, verbose=False)
            ob, reward, done, info = env.step(action)
            agent.record(state, action, reward)
            # env.render()
            state = ob
            steps += 1

        loss = agent.train()
        print('Epoch {}: step={}, loss={}'.format(epoch, steps, loss))
        env.reset()
    env.close()
    agent.dump()


if __name__ == '__main__':
    main()

