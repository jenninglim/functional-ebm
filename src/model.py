import torch
from src.nn import Generator, Energy
from src.sampler import LangevinSampler
from torchtyping import TensorType
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
from contextlib import nullcontext
import torch.nn.functional as F


def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)


class TorchSeedContext:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.rstate = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        return self

    def __exit__(self, *args):
        torch.random.set_rng_state(self.rstate)


class FEBM(torch.nn.Module):
    def __init__(self,
                 n_weights,
                 x_lin,
                 y_dim,
                 kernel,
                 interpolate_method='nn',
                 transform=None,
                 sigma=0.5,
                 prior_type='tilt',
                 encoder=None,
                 latent_dim=10,
                 device='cpu',
                 likelihood='normal'):
        super(FEBM, self).__init__()
        self.likelihood = likelihood
        self.kernel = kernel
        self.y_dim = y_dim
        self.n_weights = n_weights
        self.encoder = encoder
        self.transform = transform
        self.device = device
        self.prior_type = prior_type
        self.energy = Energy(latent_dim, n_hidden=512).to(device)
        self.sigma = torch.nn.Parameter(inv_softplus(torch.tensor(sigma)))
        self.prior_scale = torch.tensor(1.)
        self.latent_dim = latent_dim
        self.param_model = Generator(latent_dim,
                                     self.n_weights,
                                     y_dim,
                                     512).to(device)

        with torch.no_grad():
            self.kernel.set_system(x_lin,
                                   self.n_weights,
                                   y_dim,
                                   self.device,
                                   interpolate_method=interpolate_method)

    def cpu(self,):
        super(FEBM, self).cpu()
        self.param_model = self.param_model.cpu()
        self.device = 'cpu'

    def sample_function(self,
                        x_in,
                        n_functions=20,
                        return_weights=False,
                        seed=None,
                        steps=100,
                        clip_norm=False,
                        sampler_step_size=1e-3):
        obs, latent = self.sample(n_functions,
                                  x_in,
                                  seed=seed,
                                  steps=steps,
                                  sampler_step_size=sampler_step_size,
                                  clip_norm=clip_norm)
        return self.weight_to_function(latent,
                                       x_in=x_in,
                                       return_weights=return_weights)

    def weight_to_function(self,
                           latent: TensorType["...", "n_functions", "latent_dim"],
                           x_in: TensorType["n_functions", "n_evals", "x_dim"],
                           return_weights: bool = False) -> TensorType["...", "n_functions", "n_evals", "y_dim"]:
        # weight: TensorType["...", "n_functions", "n_weights", "y_dim"]
        weights = self.param_model(latent)
        # basis: TensorType["n_functions", "n_evals", "n_weights"]
        basis = self.kernel.get_basis(x_in)
        out = torch.einsum("...fwd, few-> ...fed", weights, basis)
        if self.transform is not None:
            out = self.transform(out)
        if return_weights:
            return out, weights
        else:
            return out

    def _initialize(self,
                    n_samples: int,
                    x_in: TensorType["n_functions", "n_evals", "x_dim"],
                    seed: int = None):
        ctx_manager = nullcontext() if seed is None else TorchSeedContext(seed)
        with ctx_manager:
            latent = torch.randn(n_samples, x_in.shape[0], self.latent_dim)
            latent = latent.to(device=self.device)
            fx = self.weight_to_function(latent, x_in=x_in)
            if self.likelihood == 'normal':
                obs = torch.randn_like(fx) * F.softplus(self.sigma) ** 0.5 + fx
            else:
                obs = ContinuousBernoulli(logits=fx).rsample(torch.Size([1]))[0]
        return obs, latent

    def find_weights(self,
                     y_in: TensorType["n_in", "y_dim"],
                     x_in: TensorType["n_in", "x_dim"],
                     x_out: TensorType["...", "x_dim"],
                     steps: int = 300,
                     n_functions: int = 3,
                     seed=None,
                     sampler_step_size=1.e-3) -> TensorType["n_functions", "...", "y_dim"]:
        latent = self.sample(n_functions,
                             ys=y_in,
                             x_in=x_in,
                             steps=steps,
                             seed=seed,
                             sampler_step_size=sampler_step_size,
                             clip_norm=True)
        return self.weight_to_function(latent, x_in=x_out)

    def sample_likelihood(self, fx):
        if self.likelihood == 'normal':
            obs = torch.randn_like(fx) * F.softplus(self.sigma) ** 0.5 + fx
        else:
            obs = ContinuousBernoulli(logits=fx).rsample(torch.Size([1]))[0]
        return obs

    def sample(self,
               n_functions: int,
               x_in,
               steps: int = 100,
               ys=None,
               init=None,
               seed=None,
               sampler_step_size=1.0e-3,
               clip_norm=False):
        if init is not None:
            init = init.to(self.device)
        if init is not None:
            assert n_functions == init.shape[0],\
                f"n_samples {n_functions} should be the same as init n_samples {init.shape[0]}"
        ctx_manager = nullcontext() if seed is None else TorchSeedContext(seed)
        with ctx_manager:
            if ys is None:  # Samples from joint distribution
                if init is not None:
                    latent = init.to(self.device)
                else:
                    _, latent = self._initialize(n_functions, x_in=x_in)
                prior_sampler = LangevinSampler(self.log_prior_prob,
                                                sampler_step_size,
                                                latent,
                                                clip_norm=clip_norm)
                latent = prior_sampler.sample(steps)
                obs = self.sample_likelihood(self.weight_to_function(latent,
                                                                     x_in))
            else:  # Sample from posterior conditional on ys.
                if init is None:
                    _, latent = self._initialize(n_samples=n_functions,
                                                 x_in=x_in,
                                                 seed=seed)

                log_density = lambda x: self.log_density(x, ys, x_in=x_in)
                sampler = LangevinSampler(log_density,
                                          sampler_step_size,
                                          latent,
                                          clip_norm=clip_norm)
                latent = sampler.sample(steps)
        if ys is None:
            assert(latent.ndim == 3)
            assert(latent.shape[0] == n_functions)
            assert (latent.shape[1] == x_in.shape[0])
            assert(latent.shape[2] == self.latent_dim)
            assert(obs.ndim == 4)
            assert(obs.shape[0] == n_functions)
            assert (obs.shape[1] == x_in.shape[0])
            assert(obs.shape[2] == x_in.shape[1])
            assert(obs.shape[3] == self.y_dim)
            return obs, latent
        else:
            assert(latent.ndim == 3)
            assert(latent.shape[0] == n_functions)
            assert(latent.shape[1] == ys.shape[0])
            assert(latent.shape[2] == self.latent_dim)
            return latent

    def log_density(self, latent, obs, x_in):
        log_prob = self.log_likelihood_prob(obs, latent, x_in=x_in)
        log_prior = self.log_prior_prob(latent)
        if latent.ndim == 3:
            log_prior = log_prior.sum(-1)
            log_prob = log_prob.sum(-1)
        assert(log_prior.ndim == 1)
        assert log_prior.shape[0] == latent.shape[0], f"Log prior dimension 0 should equal n_functions {x.shape[0]} instead is f{log_prior.shape[0]}"
        assert(log_prob.ndim == 1)
        assert(log_prob.shape == log_prior.shape)
        return log_prob + log_prior

    def log_likelihood_prob(self, obs, latent, x_in):
        y_model, transformed_w = self.weight_to_function(latent,
                                                         x_in=x_in,
                                                         return_weights=True)
        if self.likelihood == 'normal':
            sigma2 = F.softplus(self.sigma)
            L = torch.sum((obs - y_model) ** 2 / (2 * sigma2),
                          axis=[-1, -2]) + 0.5 * torch.log(sigma2)
        else:
            obs = torch.clip(obs, min=0., max=1.)
            L = - ContinuousBernoulli(logits=y_model).log_prob(obs).sum([-1, -2])
        return - L

    def log_prior_prob(self, latent):
        if self.prior_type == 'tilt':
            prior_scale2 = F.softplus(self.prior_scale)
            return - self.energy(latent).sum(-1) - (latent ** 2).sum(-1) / (2. * prior_scale2) - 0.5 * torch.log(prior_scale2)
        if self.prior_type == 'gaussian':
            return - (latent ** 2).sum(-1) / (2. * 0.1)
        if self.prior_type == 'ebm':
            return - self.energy(latent).sum(-1)
        assert 1 == 0, f"{self.prior_type} not accepted"

    def eval(self):
        self.kernel.requires_grad_(False)
        self.param_model.requires_grad_(False)
        self.energy.requires_grad_(False)
        self.kernel.eval()
        self.param_model.eval()
        self.energy.eval()

    def train(self):
        self.kernel.requires_grad_(True)
        self.param_model.requires_grad_(True)
        self.energy.requires_grad_(True)
        self.kernel.train()
        self.param_model.train()
        self.energy.train()

    def __repr__(self,):
        return "FEBM"
