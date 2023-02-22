import torch
import numpy as np


class LangevinSampler:
    def __init__(self, f, eps, init_sample, device=None, clip_norm=False):
        self.init_sample = init_sample
        self.f = f
        self.eps = eps
        self.device = device
        self.clip_norm = clip_norm

    def _grad(self, z,  t, n_steps, max_norm=10.):
        grad = torch.autograd.grad(self.f(z).sum(),
                                   z,
                                   create_graph=False)[0]
        if self.clip_norm:
            with torch.no_grad():
                total_norm = torch.max(grad.detach().abs())
                clip_coef = max_norm / (total_norm + 1e-6)
                if clip_coef < 1:
                    grad = grad * clip_coef.detach()
        return grad

    def initialize(self):
        x = self.init_sample()
        return x

    def _proposal(self, x, eps, t, n_steps):
        xnew = x
        g = self._grad(xnew.requires_grad_(), t, n_steps)
        xnew = xnew + eps * g + np.sqrt(2.0 * eps) * torch.randn_like(xnew)
        return xnew.detach()

    def step(self, x, t, n_steps, eps):
        x = self._proposal(x, eps, t, n_steps)
        return x.detach()

    def sample(self, n_steps, x0=None):
        if x0 is not None:
            x = x0.clone()
        else:
            x = self.init_sample.clone().detach().to(self.device)

        for t in range(n_steps):
            x = self.step(x, t, n_steps, self.eps)
        return x
