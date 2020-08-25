import torch

from .agent import Agent


class MCAgent(Agent):

    def get_return(self, r_list, q_tensor) -> torch.Tensor:
        g_list = list(r_list)
        for i in range(len(g_list) - 2, -1, -1):
            g_list[i] += self._gamma * g_list[i+1]
        g_tensor = torch.tensor(g_list)
        return g_tensor

