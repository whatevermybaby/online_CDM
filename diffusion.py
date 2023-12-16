import math
import torch
from torch import nn
import torch.nn.functional as F


# gaussian diffusion trainer class
def extract(a, t, x_shape):
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start, beta_end = scale * 0.0001, scale * 0.02

    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, timesteps=1000, beta_schedule='cosine'):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # \bar{\alpha}_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.) # \bar{\alpha}_t-1

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod)) # \sqrt{\bar{\alpha}_t}
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod)) # \sqrt{1 - \bar{\alpha}_t}
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod)) # \log(1 - \bar{\alpha}_t)
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod)) # \sqrt{1 / \bar{\alpha}_t}
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1)) # \sqrt{1 / \bar{\alpha}_t - 1}

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance) # \sigma^2_{t-1}
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20))) # \log(\sigma^2_{t-1})
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # \beta_t * \sqrt{\bar{\alpha}_t-1} / (1 - \bar{\alpha}_t)
        
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        # (1 - \bar{\alpha}_t-1) * \sqrt{\bar{\alpha}_t} / (1 - \bar{\alpha}_t)

    def p_mean_variance(self, x, t, classes):
        pred_noise = self.model(x, t, classes)
        x_start = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - \
            extract(self.sqrt_recipm1_alphas_cumprod,t, x.shape) * pred_noise
        posterior_mean = extract(self.posterior_mean_coef1, t, x.shape) * x_start + \
            extract(self.posterior_mean_coef2, t, x.shape) * x
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x.shape)

        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x, t, classes):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, model_log_variance = self.p_mean_variance(x=x, t=batched_times, classes=classes)
        noise = torch.randn_like(x) if t > 0 else 0.

        return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, classes):
        img = torch.randn((classes.shape[0], self.image_size), device=self.betas.device)
        for t in range(self.num_timesteps - 1, -1, -1):
            img = self.p_sample(img, t, classes)

        return img

    def p_losses(self, x_start, t, classes):
        noise = torch.randn_like(x_start)
        x = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        return F.mse_loss(self.model(x, t, classes), noise)

    def forward(self, img, classes):
        t = torch.randint(0, self.num_timesteps, (img.shape[0],), device=img.device).long()

        return self.p_losses(img, t, classes)
