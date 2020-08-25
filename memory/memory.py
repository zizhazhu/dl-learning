import torch

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
