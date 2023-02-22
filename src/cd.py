import torch


class CD:
    def __init__(self,
                 model,
                 j_n_steps=300,
                 j_step_size=1e-3,
                 c_n_steps=300,
                 c_step_size=1e-3):
        self.model = model
        self.replay = None

        self.c_n_steps = c_n_steps
        self.c_step_size = c_step_size
        self.j_n_steps = j_n_steps
        self.j_step_size = j_step_size

    def __call__(self, x, y, o=None):
        x_o = x if o is None else x[:, o]
        y_o = y if o is None else y[:, o]
        self.model.eval()
        weights_post = self.model.sample(1,
                                         ys=y_o,
                                         steps=self.c_n_steps,
                                         x_in=x_o,
                                         sampler_step_size=self.c_step_size,
                                         clip_norm=True,
                                         init=None).squeeze(0)
        assert weights_post.shape == torch.Size([y.shape[0],
                                                 self.model.latent_dim])
        self.model.train()
        log_likeli = self.model.log_likelihood_prob(y, weights_post.detach().clone(), x_in=x)
        self.model.eval()
        model_y, weights_joint = self.model.sample(1,
                                                   steps=self.j_n_steps,
                                                   x_in=x,
                                                   sampler_step_size=self.j_step_size,
                                                   clip_norm=True,
                                                   init=None)
        weights_joint = weights_joint.squeeze(0)
        self.model.train()
        self.model.param_model.requires_grad_(False)
        logp_prior = self.model.log_prior_prob(weights_joint.detach().clone())
        logp_post = self.model.log_prior_prob(weights_post.detach().clone())
        self.model.param_model.requires_grad_(True)
        assert logp_prior.shape == torch.Size([y.shape[0]])
        assert logp_post.shape == torch.Size([y.shape[0]])
        assert log_likeli.shape == torch.Size([y.shape[0]])
        joint_nenergy = (logp_prior - logp_post).mean()
        loss = - log_likeli.mean() + joint_nenergy

        return loss,  log_likeli.mean().item(),  joint_nenergy.item()
