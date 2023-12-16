import torch
from tqdm import tqdm
import torch
import numpy as np
from unet_1d import Unet1D
from torch.utils.data import DataLoader

class D_exp:
    def __init__(self, tensor1=None, tensor2=None, device='cuda'):
        self.x = tensor1.to(device)
        self.y = tensor2.to(device)
        self.device = device
        self.x_dim = self.x.shape[1]

    def update(self, new_x, new_y):
        # Update the lists with new lists of tensors
        self.x = torch.cat((self.x, new_x), dim=0) if new_x.ndim == 2 else torch.cat((self.x, new_x.unsqueeze(0)), dim=0)
        self.y = torch.cat((self.y, new_y), dim=0) if new_y.ndim == 2 else torch.cat((self.y, new_y.unsqueeze(0)), dim=0)
    
    def design_matrix(self, reg_lambda=1): 
        return torch.mm(self.x.t(), self.x) + reg_lambda*torch.eye(self.x_dim).to(self.device)
    
    def RLS_estimate(self):
        V_t = self.design_matrix()
        theta_hat = torch.mm(torch.linalg.pinv(V_t), torch.mm(self.x.t(), self.y))
        
        return theta_hat
    
    def TS_estimator(self, beta_0=1, t_count=1):
        theta_hat = self.RLS_estimate()
        V_t = self.design_matrix()
        V_t_pinv = torch.linalg.pinv(V_t)
        U, S, V = torch.linalg.svd(V_t_pinv)
        S_sqrt_inv = torch.sqrt(S)
        V_t_inv_sqrt = torch.mm(U, torch.mm(torch.diag(S_sqrt_inv), V.t()))
        
        beta_t = beta_0*torch.sqrt(self.x.shape[1]* torch.log(torch.tensor(t_count)))
        eta_t = torch.randn(self.x_dim, 1).to(self.device)
        
        theta_TS = theta_hat + beta_t*torch.mm(V_t_inv_sqrt, eta_t)
        
        return theta_TS
    
    def acquisition(self):   
        return self.y.max().view(-1)