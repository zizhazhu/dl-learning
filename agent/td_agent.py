import torch

from .agent import Agent


class TDAgent(Agent):

    def get_return(self, r_list, q_tensor):
        g_list = list(r_list)
        for i in range(len(r_list) - 2, -1, -1):
            g_list[i] += self._gamma * q_tensor[i]
        return torch.tensor(g_list)
