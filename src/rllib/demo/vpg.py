import os

import torch
import torch.nn as nn
import gym
import logging

from torch.distributions.categorical import Categorical
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self, action_n=2, ob_n=2):
        super(Net, self).__init__()
        self.action_embed = torch.nn.Embedding(action_n, 16)

        self.state_embed = torch.nn.Embedding(ob_n, 16)

        self.fc = torch.nn.Linear(32, 32)
        self.output = torch.nn.Linear(32, 1)

    def forward(self, s, a):
        if not isinstance(s, torch.Tensor):
            s_tensor = torch.tensor(s, dtype=torch.long)
        else:
            s_tensor = s
        if not isinstance(a, torch.Tensor):
            a_tensor = torch.tensor(a, dtype=torch.long)
        else:
            a_tensor = a
        s_tensor = torch.squeeze(s_tensor)

        a_embed = F.relu(self.action_embed(a_tensor))
        s_embed = F.relu(self.state_embed(s_tensor))
        sa = torch.cat((s_embed, a_embed), dim=-1)

        q = torch.squeeze(self.output(sa), dim=-1)
        return q


def get_model(observation_space, mlp_layers=(32,), output_space=1):
    layers = []
    layers.append(nn.Embedding(observation_space, mlp_layers[0]))
    for i in range(1, len(mlp_layers)):
        layers.append(nn.Linear(mlp_layers[i-1], mlp_layers[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(mlp_layers[-1], output_space))
    return nn.Sequential(*layers)


class VanillaPolicyGradient:

    def __init__(self, model):
        self._model = model

    def get_policy(self, obs):
        logits = self._model(obs)
        cate = Categorical(logits=logits)
        return cate

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs, act, weights) -> torch.Tensor:
        log_p = self.get_policy(obs).log_prob(act)
        # 因为优化器是梯度下降，所以取负
        weighted_logp = - log_p * weights
        return weighted_logp.mean()


def train(epoch=1, gamma=1.0, render=False, model_path='./model/FrozenLake-v0/pg.model'):
    env = gym.make('FrozenLake-v0')

    # iter common
    model = get_model(
        env.observation_space.n,
        [32, 32],
        env.action_space.n,
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    agent = VanillaPolicyGradient(model)
    optimizer = torch.optim.Adam(model.parameters())

    def one_epoch():
        batch_obs = []
        batch_act = []
        batch_weights = []
        tra_length = 0
        g_return = 0

        obs, done = env.reset(), False

        while not done:

            if render:
                env.render()

            batch_obs.append(obs)

            action = agent.get_action(torch.as_tensor(obs, dtype=torch.long))
            obs, reward, done, _ = env.step(action)
            tra_length += 1
            g_return = reward + gamma * g_return

            batch_act.append(action)
        batch_weights.extend([g_return] * tra_length)

        optimizer.zero_grad()
        loss = agent.compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.long),
            act=torch.as_tensor(batch_act, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        loss.backward()
        optimizer.step()
        logging.info(f"Loss:{loss:.3} Return:{g_return} Length:{tra_length}")

    for epoch_n in range(epoch):
        logging.info(f"Epoch {epoch_n}:")
        one_epoch()

    env.close()
    directory, file = os.path.split(model_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train(gamma=0.99, epoch=5000, render=False)
