import torch
import torch.nn as nn


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    # 这个只能处理一维的输入和输出
    def __init__(self, observation_dim, action_dim, act_limit, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        sizes = [observation_dim] + list(hidden_sizes) + [action_dim]
        self.layers = mlp(sizes, activation, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.layers(obs)


class Critic(nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        sizes = [observation_dim + action_dim] + list(hidden_sizes) + [1]
        self.layers = mlp(sizes, activation)

    def forward(self, obs, act):
        all_input = torch.cat([obs, act], dim=-1)
        return self.layers(all_input)


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.act_limit = observation_space.high[0]
        self.pi = Actor(self.observation_dim, self.action_dim, self.act_limit, hidden_sizes)
        self.q = Critic(self.observation_dim, self.action_dim, hidden_sizes)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs)
