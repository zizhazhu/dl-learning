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


def get_policy_model(observation_space, mlp_layers=(32,), output_space=1):
    layers = []
    layers.append(nn.Embedding(observation_space, mlp_layers[0]))
    for i in range(1, len(mlp_layers)):
        layers.append(nn.Linear(mlp_layers[i-1], mlp_layers[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(mlp_layers[-1], output_space))
    return nn.Sequential(*layers)


def get_value_model(observation_space, mlp_layers=(32,)):
    layers = []
    layers.append(nn.Embedding(observation_space, mlp_layers[0]))
    for i in range(1, len(mlp_layers)):
        layers.append(nn.Linear(mlp_layers[i-1], mlp_layers[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(mlp_layers[-1], 1))
    return nn.Sequential(*layers)


class VanillaPolicyGradient:

    def __init__(self, a_model, v_model):
        self._a_model = a_model
        self._v_model = v_model

    def get_policy(self, obs):
        logits = self._a_model(obs)
        cate = Categorical(logits=logits)
        return cate

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def get_value(self, obs):
        value = self._v_model(obs)
        return value

    def compute_policy_loss(self, obs, act, weights) -> torch.Tensor:
        log_p = self.get_policy(obs).log_prob(act)
        # 因为优化器是梯度下降，所以取负
        weighted_logp = - log_p * weights
        return weighted_logp.mean()

    def compute_value_loss(self, obs, returns):
        values = self.get_value(obs)
        loss = torch.nn.functional.mse_loss(values, returns)
        return loss


def train(epoch=1, gamma=1.0, render=False, model_path='./model/FrozenLake-v0/vpg.model'):
    env = gym.make('FrozenLake-v0')

    # iter common
    a_model = get_policy_model(
        env.observation_space.n,
        [32, 32],
        env.action_space.n,
    )
    v_model = get_value_model(
        env.observation_space.n,
        [32, 32],
    )
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        a_model.load_state_dict(state_dict['a'])
        v_model.load_state_dict(state_dict['v'])
    agent = VanillaPolicyGradient(a_model, v_model)
    a_optimizer = torch.optim.Adam(a_model.parameters())
    v_optimizer = torch.optim.Adam(v_model.parameters())

    def one_epoch():
        # step 1: collect trajectories
        # step 2: calculate return
        # step 3: calculate Advantage estimate
        # step 4: backward policy gradient
        # step 5: backward value function
        batch_obs = []
        batch_acts = []
        batch_rewards = []
        batch_values = []
        batch_weights = []

        obs, done = env.reset(), False

        while not done:

            if render:
                env.render()

            batch_obs.append(obs)
            value = agent.get_value(torch.as_tensor(obs, dtype=torch.long))
            batch_values.append(value)

            action = agent.get_action(torch.as_tensor(obs, dtype=torch.long))
            obs, reward, done, _ = env.step(action)

            batch_rewards.append(reward)
            batch_acts.append(action)

        batch_values.append(0)
        for i in range(len(batch_rewards)):
            batch_weights.append(batch_values[i+1] + batch_rewards[i] - batch_values[i])
        batch_returns = [0 for _ in range(len(batch_rewards))]
        batch_returns[-1] = batch_rewards[-1]
        for i in reversed(range(len(batch_rewards) - 1)):
            batch_returns[i] = batch_rewards[i] + gamma * batch_returns[i + 1]

        a_optimizer.zero_grad()
        v_optimizer.zero_grad()
        policy_loss = agent.compute_policy_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.long),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        policy_loss.backward()
        value_loss = agent.compute_value_loss(
            torch.as_tensor(batch_obs, dtype=torch.long),
            torch.as_tensor(batch_returns, dtype=torch.float32),
        )
        value_loss.backward()
        a_optimizer.step()
        v_optimizer.step()
        logging.info(f"Policy_loss:{policy_loss:.3} Value_loss:{value_loss:.3} Return:{batch_returns[0]} Length:{len(batch_returns)}")
        logging.info("")

    for epoch_n in range(epoch):
        logging.info(f"Epoch {epoch_n}:")
        one_epoch()

    env.close()
    directory, file = os.path.split(model_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save({'a': a_model.state_dict(), 'v': v_model.state_dict()}, model_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train(gamma=0.99, epoch=3, render=True)
