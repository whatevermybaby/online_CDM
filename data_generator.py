import torch
from collections import defaultdict
from scipy.stats import ortho_group
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)



class GaussianLatentSampler(object):
    def __init__(self, d_inner, d_outer):
        self.d_inner, self.d_outer = d_inner, d_outer
        self.A = ortho_group.rvs(dim=d_outer)[:d_inner, :] # (16,64)
        beta = np.random.randn(d_inner, 1)
        beta /= np.linalg.norm(beta)  # normalize beta,(16,1)
        self.theta = self.A.T.dot(beta)

    def generate_data(self, N, theta=None, torch_tensor=False):
        x = np.random.randn(N, self.d_inner).dot(self.A)
        label = x.dot(theta) if theta is not None else x.dot(self.theta)
        if torch_tensor:
            x, label = torch.from_numpy(x).float(), torch.from_numpy(label).float()

        return x, label

    def generate_conditional_data(self, N, theta_estimate, v, torch_tensor=False):
        w = self.A.dot(theta_estimate)
        conditional_mean = (w * v / (w.T.dot(w))).reshape(-1)
        conditional_var = (1 + 1e-8) * np.eye(self.d_inner) - w.dot(w.T) / (w.T.dot(w))
        x = np.random.multivariate_normal(conditional_mean, conditional_var, size=N).dot(self.A)

        return torch.from_numpy(x).float() if torch_tensor else x




if __name__ == '__main__':
    from diffusion import GaussianDiffusion
    from unet_1d import Unet1D
    from torch.utils.data import DataLoader

    # random seed
    set_seed(seed=1234)

    # hyperparameters
    N_pred, N_diff, N_eval, N_loss = 8192, 65536, 2048, 8192
    d_inner, d_outer = 16, 64
    lam = 5.0

    # generate weights
    generator = GaussianLatentSampler(d_inner, d_outer)

    # train predictor
    data_pred, label_pred = generator.generate_data(N_pred)
    theta_estimate = np.linalg.pinv(data_pred.T.dot(data_pred) + np.eye(d_outer)).dot(data_pred.T.dot(label_pred))

    # train diffusion model
    diffusion = GaussianDiffusion(model=Unet1D(dim=64), image_size=64, timesteps=200)
    optimizer_diff = torch.optim.Adam(diffusion.model.parameters(), lr=8e-5, betas=(0.9, 0.99))

    data_diff, _ = generator.generate_data(N_diff, torch_tensor=True)
    dataset_diff = DataLoader(data_diff, batch_size=32)
    print(dataset_diff)
