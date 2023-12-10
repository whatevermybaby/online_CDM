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
from diffusion import GaussianDiffusion
import wandb
import datetime
import argparse

if __name__   == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--num_labelled', type=int, default=8192)
    parser.add_argument('--num_unlabelled', type=int, default=65536)
    parser.add_argument('--num_eval', type=int, default=2048)
    parser.add_argument('--num_loss', type=int, default=8192)
    parser.add_argument('--penalty', type=float, default=5.0)
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--dim_latent', type=int, default=16)
    parser.add_argument('--dim_input', type=int, default=64)

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

    # train predictor
    data_pred, label_pred = generator.generate_data(args.num_labelled)
    theta_estimate = np.linalg.pinv(data_pred.T.dot(data_pred) + np.eye(args.dim_input)).dot(data_pred.T.dot(label_pred))

    # train diffusion model
    diffusion = GaussianDiffusion(model=Unet1D(dim=64), image_size=64, timesteps=200)
    optimizer_diff = torch.optim.Adam(diffusion.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))

    data_diff, _ = generator.generate_data(args.num_unlabelled, torch_tensor=True)
    dataloader = DataLoader(data_diff, batch_size=args.batch_size, shuffle=True)
    
    # move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    theta_estimate = torch.from_numpy(theta_estimate).float().to(device)
    
    # Start training
    print("***** Running training *****")
    for epoch in tqdm(range(args.num_epochs), desc='diffusion training'):
            epoch_loss = list()
            for sample_batch in tqdm(dataloader, desc="Batches"):
                sample_batch = sample_batch.to(device)

                predicted_score_batch = sample_batch.matmul(theta_estimate)
                predicted_score_batch += 0.01 * torch.randn_like(predicted_score_batch)
                
                loss_diffusion = diffusion(sample_batch, predicted_score_batch.detach().view(-1))

                optimizer_diff.zero_grad()
                loss_diffusion.backward()
                optimizer_diff.step()

                wandb.log({"mean_sample_loss": loss_diffusion.item() / args.batch_size})
                epoch_loss.append(loss_diffusion.item())
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {np.mean(epoch_loss):.4f}')
            wandb.log({"epoch": epoch, "mean_batch_loss": np.mean(epoch_loss)})
            
            if (epoch+1) % 10 == 0:
                torch.save(diffusion.model.state_dict(), f'./logs/checkpoints/{args.run_name}_{epoch+1}.pth')
                print(f'Model saved at epoch {epoch+1}')
    
    # Start evaluation
    print("***** Running evaluation *****")
    x_sample, label_sample = generator.generate_data(args.num_loss, theta=theta_estimate, torch_tensor=True)
    with torch.no_grad():
        # noise prediction error of the un-trained UNet
        loss_diff = diffusion(x_sample.to(device), label_sample.view(-1).to(device))
        
    for target in 0.1 * np.arange(100):
        target_scores = target * torch.ones(args.num_eval).to(device)
        samples = diffusion.sample(target_scores).view(-1, args.dim_input).cpu().numpy()

        parallel = samples.dot(generator.A.T).dot(generator.A) # (args.num_eval,64)
        vertical = samples - parallel  # (args.num_eval,64)
        ratio = np.linalg.norm(vertical, axis=-1) / (np.linalg.norm(parallel, axis=-1) + 1e-8)
        penalty = args.penalty * np.sum(np.square(vertical), axis=-1).reshape(-1, 1)
        scores_raw = samples.dot(generator.theta)
        scores = scores_raw - penalty

        # compute conditional diffusion loss
        x_sample = generator.generate_conditional_data(args.num_loss, theta_estimate, target, torch_tensor=True)
        with torch.no_grad():
            loss_diff_cond = diffusion(x_sample.to(device), target * torch.ones(args.num_loss).to(device))

        # log the data
        wandb.log({
            "norm_mean":np.mean(np.linalg.norm(samples, axis=-1)),
            "ratio":np.mean(ratio),
            "dis_mismatch":loss_diff_cond / loss_diff,
            "penalty":np.mean(penalty),
            "ave_raw_score":np.mean(scores_raw),
            "ave_score":np.mean(scores)
        })
        print("dis_mismatch: %f"%(loss_diff_cond / loss_diff))
