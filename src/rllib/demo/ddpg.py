import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import gym
import logging

from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from rllib.memory.replay_buffer import ReplayBuffer
from rllib.model.actor_critic import ActorCritic


class DDPG:

    def __init__(self, ac_model: ActorCritic):
        self._ac_model = ac_model
        self._ac_target_model = deepcopy(ac_model)

    def get_action(self, obs, noise_scale=0.0):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        action = self._ac_model.act(obs).numpy()
        action += noise_scale * np.random.rand(self._ac_model.action_dim)
        return np.clip(action, -self._ac_model.act_limit, self._ac_model.act_limit)

    def compute_pi_loss(self, data) -> torch.Tensor:
        obs = data['obs']
        q = self._ac_model.q(obs, self._ac_model.pi(obs))
        # 因为优化器是梯度下降，所以取负
        return -q.mean()

    def compute_q_loss(self, data):
        obs, action, reward, obs2, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self._ac_model.q(obs, action)

        with torch.no_grad():
            target = self._ac_target_model.q(obs2, self._ac_target_model.pi(obs2))
            target = reward + target

        loss = torch.nn.functional.mse_loss(q, target)
        return loss


def train(epoch=1, gamma=1.0, polyak=0.995, batch_size=100, render=False, model_path='./model/Pendulum-v0/vpg.model'):
    env = gym.make('Pendulum-v0')
    observation_space, action_space = env.observation_space, env.action_space
    steps = 200

    ac = ActorCritic(env.observation_space, env.action_space)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        ac.load_state_dict(state_dict)

    agent = DDPG(ac)
    pi_optimizer = torch.optim.Adam(ac.pi.parameters())
    q_optimizer = torch.optim.Adam(ac.q.parameters())

    buffer = ReplayBuffer(observation_space.shape[0], action_space.shape[0], size=int(1e6))

    def update(data):
        q_optimizer.zero_grad()
        loss_q = agent.compute_q_loss(data)
        loss_q.backward()
        q_optimizer.step()

        for para in ac.q.parameters():
            para.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi = agent.compute_pi_loss(data)
        loss_pi.backward()
        pi_optimizer.step()

        for para in ac.q.parameters():
            para.requires_grad = True

        with torch.no_grad():
            for para, para_target in zip(agent._ac_model.parameters(), agent._ac_target_model.parameters()):
                para_target.data.mul_(polyak)
                para_target.data.add_((1 - polyak) * para.data)

        return loss_q, loss_pi

    def one_epoch():

        obs, done = env.reset(), False
        rsum, rlen = 0.0, 0

        for t in range(steps):
            if render:
                env.render()

            # TODO: start step
            # 逆时针是正的
            action = agent.get_action(obs)
            obs_next, reward, done, _ = env.step(action)
            rsum += reward
            rlen += 1
            buffer.store(obs, action, reward, obs_next, done)
            obs = obs_next

            if done:
                obs, rsum, rlen = env.reset(), 0.0, 0

            # TODO: update after & update_every
            loss_pi, loss_q = update(buffer.sample_batch(batch_size))
            logging.info(f"Policy_loss:{loss_pi:.3} Value_loss:{loss_q:.3} Reward:{reward} Return:{rsum} Length:{rlen}")
            logging.info("")

    for epoch_n in range(epoch):
        logging.info(f"Epoch {epoch_n}:")
        one_epoch()

    env.close()
    directory, file = os.path.split(model_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(ac.state_dict(), model_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train(gamma=0.99, epoch=3, render=True)
