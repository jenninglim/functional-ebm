import torch
import re
import torch.nn as nn
from gpytorch import kernels as kernel_lib
from src.utils import kernel_system
import gpytorch
from functools import partial
import pytorch_lightning as pl
from src.utils import DomainEncoder
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Net(pl.LightningModule):
    def __init__(self, in_dim, out_dim, n_hidden=512, encode=128):
        super(Net, self).__init__()
        # branch
        self.encoder = DomainEncoder(in_dim, encode)
        self.fc1 = nn.Linear(encode, n_hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(n_hidden, out_dim)

    def forward(self, weights):
        weights = self.encoder(weights)
        weights = self.act1(self.fc1(weights))
        weights = self.act2(self.fc2(weights))
        weights = self.fc3(weights)
        return weights

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self(x)
        mse = ((y - y_out) ** 2).sum(-1).mean()
        self.log("train_loss", mse)
        return mse

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class Kernel(gpytorch.kernels.Kernel):
    def set_system(self,
                   x_lin,
                   n_weights,
                   y_dim,
                   device='cpu',
                   interpolate_method=True):
        self.x_lin = x_lin.to(device)
        self.l, self.kernel_v = self.get_system(self.x_lin)
        self.kernel_v = self.kernel_v[:, :n_weights] * torch.sqrt(self.l[:n_weights])
        self.l = torch.ones(self.kernel_v.shape[1])
        self.n_weights = self.kernel_v.shape[1]
        self.kernel_v = self.kernel_v.to(device=device)
        self.interpolate_method = interpolate_method

        if interpolate_method == 'nn':
            self.nn = Net(x_lin.shape[-1], self.n_weights).to(device)
            trainer = pl.Trainer(max_epochs=1000, callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=30)])
            dataset = torch.utils.data.TensorDataset(self.x_lin, self.kernel_v)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=x_lin.shape[0])
            trainer.fit(model=self.nn, train_dataloaders=dataloader)

    def get_basis(self, x_in):
        """
        Return the basis evaluated at x_in
        """
        if self.interpolate_method == 'nn':
            basis = self.nn(x_in.to(self.nn.device)).to(x_in.device)
        else:
            Kxy = self(x_in, self.x_lin)
            Kyy = self(self.x_lin, self.x_lin)
            if type(Kxy) == gpytorch.lazy.LazyEvaluatedKernelTensor:
                Kxy = Kxy.evaluate()
            if type(Kyy) == gpytorch.lazy.LazyEvaluatedKernelTensor:
                Kyy = Kyy.evaluate()
            weighting = torch.einsum("...ij,jk-> ...ik",
                                     Kxy,
                                     torch.inverse(Kyy
                                                   + 1e-3 * torch.eye(self.x_lin.shape[0], device=self.x_lin.device)))
            basis = torch.einsum("...ij, jk -> ...ik",
                                 weighting,
                                 self.kernel_v)
        return basis

    def get_system(self, X):
        g = self.__call__(X, X)
        if type(g) == gpytorch.lazy.LazyEvaluatedKernelTensor:
            g = g.evaluate()
        l, v = kernel_system(g)
        return l, v


class StringToKernel(nn.Module):
    def __init__(self, kernel_str, device='cpu', n_weights=10):
        super().__init__()
        self.device = device
        self.kernel = []
        if type(kernel_str) == str:
            kernel_str = [kernel_str]
        for k in kernel_str:
            self.kernel.append(string_to_kernel(k, device=device, n_weights=n_weights).to(device=device))

    def __call__(self, X, Y):
        out = 0
        for k in self.kernel:
            k = k(X, Y)
            if type(k) == gpytorch.lazy.LazyEvaluatedKernelTensor:
                k = k.evaluate()
            out = out + k
        return out

    def set_system(self,
                   x_lin,
                   n_weights,
                   y_dim,
                   device='cpu',
                   interpolate_method='nn'):
        for k in self.kernel:
            k.set_system(x_lin,
                         n_weights,
                         y_dim,
                         device=device,
                         interpolate_method=interpolate_method)

    def save(self, kernel_path, nn_path=None):
        torch.save(self, kernel_path)

    def load(self, kernel_path, nn_path=None):
        self.__dict__.update(torch.load(kernel_path).__dict__)

    def cpu(self,):
        self.kernel.to(device='cpu')
        self.device = 'cpu'

    def get_basis(self, x_in):
        out = None
        for i, k in enumerate(self.kernel):
            k_basis = k.get_basis(x_in)
            if out is None:
                out = k_basis
            else:
                out = torch.stack([out, k_basis], dim=-1)
        return out


class RandomNN(Kernel):
    def __init__(self, x_in=1, n_weights=128, n_hidden=512, device='cpu'):
        super().__init__()
        self.n_features = n_weights
        self.nn = Net(x_in, n_hidden).to(device)
        self.kernel = FourierKernel(n_hidden,
                                    n_weights,
                                    device=device)

    def forward(self, x1, x2, **params):
        b1 = self.get_basis(x1)
        b2 = self.get_basis(x1)
        return self.kernel(b1, b2)

    def get_basis(self, x_in):
        out = self.nn(x_in)
        return self.kernel.get_basis(out)

    def set_system(self, *args, **kwargs):
        pass


class FourierKernel(Kernel):
    def __init__(self,
                 x_in=1,
                 n_weights=128,
                 type='g',
                 device='cpu'):
        super().__init__()
        self.n_features = n_weights // 2
        self.w = torch.empty(x_in, n_weights // 2)
        if type == 'g':  # Gaussian kernel
            self.w = torch.randn(x_in, n_weights // 2)
        elif type == 'l':  # Laplace
            self.w.cauchy_()
        elif type == 'c':
            self.w = torch.distributions.Laplace(0, 1).sample(self.w.shape)
        else:
            assert 1 == 0
        self.w = self.w.to(device)
        # nn.init.normal_(self.fc.weight)

    def forward(self, x1, x2, **params):
        proj_x1 = self.fourier_features(x1)
        proj_x2 = self.fourier_features(x2)
        return torch.einsum("...i,...i->...", proj_x1, proj_x2) / self.n_features # ** 0.5

    def cos_features(self, x):
        proj_x = torch.einsum("ba,...b->...a", self.w, x)
        return torch.cos(proj_x)

    def sin_features(self, x):
        proj_x = torch.einsum("ba,...b->...a", self.w, x)
        return torch.sin(proj_x)

    def fourier_features(self, x):
        cos_feats = self.cos_features(x)
        sin_feats = self.sin_features(x)
        return torch.cat([cos_feats, sin_feats], dim=-1)

    def get_basis(self, x_in):
        out = self.fourier_features(x_in)
        return out

    def set_system(self, *args, **kwargs):
        pass


class Matern(Kernel, kernel_lib.MaternKernel):
    def __init__(self, x_in=1, **kwargs):
        super().__init__()


class Gaussian(Kernel, kernel_lib.RBFKernel):
    pass


class RQ(Kernel, gpytorch.kernels.RQKernel):
    pass


def string_to_kernel(kernel_str: str, **kwargs):
    kernels_str = re.split(r"[+]", kernel_str)
    kernels = []
    for prodkernels_str in kernels_str:
        prod_kernels = []
        for kernel_str in prodkernels_str[::2]:
            if kernel_str in kernel_lookup.keys():
                kernel = kernel_lookup[kernel_str](**kwargs)
                prod_kernels.append(kernel)
        if len(prod_kernels) > 1:
            prod_kernel = kernel_lib.ProductKernel(*prod_kernels)
        else:
            prod_kernel = prod_kernels[0]
        kernels.append(prod_kernel)
    if len(kernels) > 1:
        final_kernel = kernel_lib.AdditiveKernel(*kernels)
    else:
        final_kernel = kernels[0]
    return final_kernel


kernel_lookup = {'m': Matern,
                 'g': Gaussian,
                 'n': RandomNN,
                 'q': RQ,
                 'f': partial(FourierKernel, type='g'),
                 'l': partial(FourierKernel, type='l'),
                 'c': partial(FourierKernel, type='c')}
