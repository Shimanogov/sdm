import torch
from torch.nn import Module, L1Loss, MSELoss

from diffusion.diffusions.utils import cosine_beta_schedule, make_timesteps


class GaussianDiffusion(Module):

    def __init__(
            self,
            model,
            n_timesteps=1000,
            loss_type='MSE',
            device='cuda',
    ):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.model = model
        self.device = device
        self.model.to(device)
        if loss_type == 'MSE':
            self.loss_fn = MSELoss()
        else:
            raise NotImplementedError('Only L2Loss is implemented')
        rep = (1, self.model.horizon, self.model.transition_dim)
        betas = cosine_beta_schedule(n_timesteps)
        betas = betas.to(self.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to('cuda'), alphas_cumprod[:-1]])
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        unsq = lambda x: x.unsqueeze(-1).unsqueeze(-1).repeat(rep)
        self.sqrt_alphas_cumprod = unsq(self.sqrt_alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = unsq(self.sqrt_one_minus_alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = unsq(self.sqrt_recip_alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = unsq(self.sqrt_recipm1_alphas_cumprod)
        self.posterior_variance = unsq(self.posterior_variance)
        self.posterior_log_variance_clipped = unsq(self.posterior_log_variance_clipped)
        self.posterior_mean_coef1 = unsq(self.posterior_mean_coef1)
        self.posterior_mean_coef2 = unsq(self.posterior_mean_coef2)

    def q_sample(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)

        a = self.sqrt_alphas_cumprod[t]
        oma = self.sqrt_one_minus_alphas_cumprod[t]
        sample = a * x + oma * noise

        return sample

    def p_losses(self, x, t):
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x=x, t=t, noise=noise)
        noise_pred = self.model(x_noisy, t)

        loss = self.loss_fn(noise_pred, noise)

        return loss

    def loss(self, x, return_t=False):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,)).to(self.device)
        p_losses = self.p_losses(x, t)
        if return_t:
            return p_losses, t
        return p_losses

    # sampling part

    def predict_start_from_noise(self, x_t, t, noise):
        t = t
        a = self.sqrt_recip_alphas_cumprod[t]
        oma = self.sqrt_recipm1_alphas_cumprod[t]
        return a * x_t - oma * noise

    def q_posterior(self, x_start, x_t, t):
        a = self.posterior_mean_coef1[t]
        b = self.posterior_mean_coef2[t]
        posterior_mean = a * x_start + b * x_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t))
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def sample_fn(self, x, t):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        model_std = torch.exp(0.5 * model_log_variance)
        # no noise when t == 0
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        return model_mean + model_std * noise

    def wm_sample(self, conditions, actions):
        # TODO: use EMA
        # conditions should be tensor (hist, slots, slotdim)
        # actions should be a dict of (episode, actiondim)
        conditions = torch.swapaxes(conditions, 0, 1)
        shape = (conditions.shape[0], self.model.horizon, self.model.transition_dim)
        noise = torch.randn(shape, device=self.device)
        # create condition tensor and mask tensor
        condition_tensor = torch.zeros(shape, device=self.device)
        condition_mask = torch.zeros(shape, device=self.device)
        cond_shape = conditions.shape
        condition_tensor[:, :cond_shape[1], :cond_shape[2]] = conditions
        condition_mask[:, :cond_shape[1], :cond_shape[2]] = 1
        condition_tensor[:, :, cond_shape[2]:] = actions
        condition_mask[:, :, cond_shape[2]:] = 1
        x = noise
        for i in reversed(range(self.n_timesteps)):
            t = make_timesteps(cond_shape[0], i, self.device)
            x = self.sample_fn(x, t)
            if i != 0:
                condition_noised = self.q_sample(condition_tensor, t, noise)
            else:
                condition_noised = condition_tensor
            x = (1 - condition_mask) * x + condition_mask * condition_noised
        return torch.swapaxes(x, 0, 1)

class GuidedGaussianDiffusion(GaussianDiffusion):
    def __init__(
            self,
            value_model,
            n_timesteps=1000,
            loss_type='MSE',
            device='cuda',
    ):
        super(GuidedGaussianDiffusion, self).__init__(value_model, n_timesteps, loss_type, device)

    def p_losses(self, x, t, rew):
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x=x, t=t, noise=noise)
        noise_pred = self.model(x_noisy, t).flo

        loss = self.loss_fn(noise_pred, rew)

        return loss

    def loss(self, x, rew, return_t=False):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,)).to(self.device)
        p_losses = self.p_losses(x, t, rew)
        if return_t:
            return p_losses, t
        return p_losses