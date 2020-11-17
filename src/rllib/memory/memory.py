import torch

class Memory:

    def __init__(self):
        self._s = []
        self._a = []
        self._r = []
        self._s_prime = []

    def add(self, s, a, r, s_prime=None):
        self._s.append(s)
        self._a.append(a)
        self._r.append(r)
        if s_prime is not None:
            self._s_prime.append(s_prime)

    @property
    def s_tensor(self):
        return torch.tensor(self._s, dtype=torch.float)

    @property
    def a_tensor(self):
        return torch.tensor(self._a, dtype=torch.long)

    @property
    def r_tensor(self):
        return torch.tensor(self._r, dtype=torch.float)

    @property
    def s_prime_tensor(self):
        return torch.tensor(self._s_prime, dtype=torch.float)

    def r_list(self):
        return list(self._r)

    def forget(self):
        self._s.clear()
        self._a.clear()
        self._r.clear()
        self._s_prime.clear()
