import torch
from collections import defaultdict
from scipy.stats import ortho_group
from exploration import D_exp
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
        z = np.random.randn(N, self.d_inner)
        
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        
        x = z.dot(self.A)
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
    
    def evaluate(self, x, penalty=5.0):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        parallel = x.dot(self.A.T).dot(self.A)
        vertical = x - parallel
        penalty_term = penalty * np.sum(np.square(vertical), axis=-1).reshape(-1, 1)
        scores_raw = x.dot(self.theta)
        scores = scores_raw - penalty_term
        return torch.from_numpy(scores).float()




if __name__ == '__main__':
    from diffusion import GaussianDiffusion
    from unet_1d import Unet1D
    from torch.utils.data import DataLoader

    # random seed
    set_seed(seed=42)

    # hyperparameters
    N_pred, N_diff, N_eval, N_loss = 8192, 65536, 2048, 8192
    d_inner, d_outer = 16, 64
    lam = 5.0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # generate weights
    generator = GaussianLatentSampler(d_inner, d_outer)

    # exploration dataset
    X_initial, Y_initial = generator.generate_data(50, torch_tensor=True)
    D_explored = D_exp(X_initial , Y_initial)
    print(D_explored.x.shape)
    print(D_explored.y.shape)
    
    theta_TS = D_explored.TS_estimator(beta_0=1, t_count=1).to(device)
    print('TS', theta_TS.shape)
    
    theta_hat = D_explored.RLS_estimate().to(device)
    print('RLS', theta_hat.shape)
    
    y_max = D_explored.acquisition()
    print('y_max', y_max)
