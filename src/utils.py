import torch
import math


def kernel_system(gram):
    device = gram.device
    n = gram.shape[0]
    l, v = torch.linalg.eigh(gram.cpu())
    sorted_ind = torch.argsort(l, descending=True)
    v = v[:, sorted_ind]
    l = l[sorted_ind]

    g = gram
    for _ in range(100):
        if not torch.all(l > 0):
            # logging.info("adding noise to gram")
            g = g + torch.eye(gram.shape[0]).to(gram.device) * 1e-5

            l, v = torch.linalg.eigh(g.cpu())

            sorted_ind = torch.argsort(l, descending=True)
            v = v[:, sorted_ind]
            l = l[sorted_ind]

    if not torch.all(l > 0):
        print(torch.sum(l > 0))
        l[l< 0] = 0
    l = l / n
    v = v * n ** 0.5
    return l.to(device), v.to(device)


class DomainEncoder:
    # from generative models as functions paper
    def __init__(self, d_x, d_e, device='cuda'):
        super().__init__()
        d_e = d_e // 2
        self.B = torch.randn(d_x, d_e)
        self._out_dim = d_e

    @property
    def out_dim(self):
        return self._out_dim

    def __call__(self, x):
        '''
        x : n x d_x
        '''
        bx = 2 * math.pi * torch.einsum("...d, do->...o", x, self.B)
        cos_feat = torch.cos(bx)
        sin_feat = torch.sin(bx)
        return torch.cat((cos_feat, sin_feat), dim=-1)
