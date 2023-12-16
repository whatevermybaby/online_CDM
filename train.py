import torch
from tqdm import tqdm
import torch
import numpy as np
from unet_1d import Unet1D
from torch.utils.data import DataLoader
from diffusion import GaussianDiffusion
from collections import defaultdict
from scipy.stats import ortho_group
import numpy as np
from data_generator import GaussianLatentSampler, set_seed
from exploration import D_exp
from diffusion import GaussianDiffusion
import wandb
import datetime
import argparse

if __name__   == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs for Bayesian optimization')
    parser.add_argument('--num_inner_epochs', type=int, default=10, help='number of epochs to train diffusion')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--num_unlabelled', type=int, default=65536)
    parser.add_argument('--num_eval', type=int, default=2048)
    parser.add_argument('--penalty', type=float, default=5.0)
    
    parser.add_argument('--dim_latent', type=int, default=16)
    parser.add_argument('--dim_input', type=int, default=64)
    
    parser.add_argument('--num_initial_data', type=int, default=10)
    parser.add_argument('--beta_0', type=float, default=1.0)
    parser.add_argument('--run_name', type=str, default='test')

    args = parser.parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = unique_id
    else:
        args.run_name += "_" + unique_id

    wandb.init(project="online_CDM", name=args.run_name,
        config=args)
    

    # random seed
    set_seed(seed=42)

    # unlabelled/labelled data generator
    generator = GaussianLatentSampler(args.dim_latent, args.dim_input)

    # train diffusion model
    diffusion = GaussianDiffusion(model=Unet1D(dim=64), image_size=64, timesteps=200)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    # exploration dataset
    X_initial, Y_initial = generator.generate_data(args.num_initial_data, torch_tensor=True)
    D_explored = D_exp(X_initial , Y_initial, device=device)
    
    optimizer = torch.optim.Adam(diffusion.model.parameters(),\
                        lr=args.learning_rate, betas=(0.9, 0.99))

    unlabelled_data, _ = generator.generate_data(args.num_unlabelled, torch_tensor=True)
    dataloader = DataLoader(unlabelled_data, batch_size=args.batch_size, shuffle=True)
     
    print("***** Running training *****")

    for t in tqdm(range(1,args.num_epochs+1), desc='Bayesian optimization'):
        theta_TS = D_explored.TS_estimator(beta_0=args.beta_0, t_count=t)
        
        diffusion.model.train()
        for inner_epoch in range(args.num_inner_epochs):
                epoch_loss = list()
                for idx, sample_batch in tqdm(enumerate(dataloader), total=len(dataloader),\
                            desc=f"Diffusion Training (epoch {inner_epoch+1}/{args.num_inner_epochs})"):
                    sample_batch = sample_batch.to(device)

                    labeled_samples = sample_batch.matmul(theta_TS)
                    noisy_score_batch = labeled_samples + 0.01 * torch.randn_like(labeled_samples)
                    
                    loss = diffusion(sample_batch, noisy_score_batch.detach().view(-1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss.append(loss.item())
                print(f'Epoch [{inner_epoch+1}/{args.num_inner_epochs}], Loss: {np.mean(epoch_loss):.4f}')
                # wandb.log({"inner_epoch": inner_epoch, "mean_batch_loss": np.mean(epoch_loss)})
                
                # if (inner_epoch+1) % args.num_inner_epochs == 0:
                #     torch.save(diffusion.model.state_dict(), f'./checkpoints/{args.run_name}_{inner_epoch+1}.pth')
                #     print(f'Model saved at epoch {inner_epoch+1}')
        new_sample = diffusion.sample(D_explored.acquisition()).view(-1, args.dim_input)
        print('maximum y', D_explored.acquisition().item())
        wandb.log({"epoch": t, "maximum y": D_explored.acquisition().item()})
        
        D_explored.update(new_sample, generator.evaluate(new_sample).to(device))
        

