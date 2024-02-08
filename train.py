import torch
from tqdm import tqdm
import torch
import numpy as np
from unet_1d import Unet1D
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusion import GaussianDiffusion
from collections import defaultdict
from scipy.stats import ortho_group
import numpy as np
from plot_umap import umap_plot,plot_distribution
from data_generator import GaussianLatentSampler, set_seed
from exploration import D_exp
from diffusion import GaussianDiffusion
import wandb
import datetime
import argparse
import time
import matplotlib.pyplot as plt
from torch.optim import LBFGS
import torch.nn.functional as F
import csv

def optimize_vector_for_max_inner_product(theta_TS):
    # 初始化一个变量，该变量将在优化过程中被更新
    optimized_vector = torch.nn.Parameter(torch.randn_like(theta_TS))

    # # 定义优化目标函数，即与 theta_TS 的负内积
    # def objective_function():
    #     return -torch.matmul(optimized_vector.T, theta_TS).view(-1)

    # # 使用LBFGS优化器进行优化
    # optimizer = LBFGS([optimized_vector], lr=0.01)

    # # 优化过程
    # def closure():
    #     optimizer.zero_grad()
    #     loss = objective_function()
    #     loss.backward()
    #     return loss

    # # 执行优化
    # for _ in range(100):  # 可以根据需要更改迭代次数
    #     optimizer.step(closure)

    # 得到优化后的向量
    optimized_vector = F.normalize(theta_TS.data, p=2, dim=0)  # 对向量进行2-范数归一化
    # optimized_vector = theta_TS.data
    
    # 计算最终内积值
    max_inner_product = torch.matmul(optimized_vector.T, theta_TS)

    return optimized_vector, max_inner_product

if __name__   == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs for Bayesian optimization')
    parser.add_argument('--num_inner_epochs', type=int, default=10, help='number of epochs to train diffusion')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.00008, help='learning rate')
    parser.add_argument('--num_unlabelled', type=int, default=65536)
    parser.add_argument('--num_eval', type=int, default=2048)
    parser.add_argument('--penalty', type=float, default=5.0)
    parser.add_argument('--initial',type=str,default='no')
    parser.add_argument('--dim_latent', type=int, default=8)
    parser.add_argument('--dim_input', type=int, default=32)
    # parser.add_argument('--sample_batch', type=int, default=2,help='how many samples we generate at each iteration')
    
    parser.add_argument('--num_data', type=int, default=10,help='how many samples we generate at each iteration')
    parser.add_argument('--beta_0', type=float, default=1e-3)
    parser.add_argument('--w',type=float,default=1,help='how much weight you put on the guidance')
    parser.add_argument('--seed',type=int,default=42,help='random seed')
    parser.add_argument('--run_name', type=str, default='test')

    args = parser.parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = unique_id
    else:
        args.run_name += "_" + unique_id

    # wandb.init(project="online_CDM", name=args.run_name,
    #     config=args)
    writer = SummaryWriter(log_dir=f"test/{args.run_name}")

    # random seed
    set_seed(seed=args.seed)

    # unlabelled/labelled data generator
    generator = GaussianLatentSampler(args.dim_latent, args.dim_input)
    csv_sample_path = f"/nfsshare/home/xiechenghan/online_CDM/test/data/True_theta_{args.dim_input}_{args.dim_latent}_{args.initial}_0.3.csv"
    with open(csv_sample_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([f"True_theta_dim_{i+1}" for i in range(args.dim_input)])
        # for i in range(args.num_epochs):
            # theta_TS_doc = [np.squeeze(item.cpu().numpy())if isinstance(item, torch.Tensor) else item for item in theta_TS_set[i]]
            # theta_RLS_doc = [np.squeeze(item.cpu().numpy()) if isinstance(item, torch.Tensor) else item for item in theta_RLS_set[i]]
        csv_writer.writerow(list(generator.theta.reshape(-1)))

    # # train diffusion model
    # diffusion = GaussianDiffusion(model=Unet1D(dim=args.dim_input), image_size=args.dim_input, timesteps=200)
    # print(diffusion.model)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # diffusion = diffusion.to(device)
    
    # # exploration dataset
    # if args.initial=='yes':
    #     X_initial, Y_initial = generator.generate_data(args.num_data, torch_tensor=True)
    #     D_explored_BOCD = D_exp(X_initial , Y_initial, device=device)
    #     D_explored_TS=D_exp(X_initial , Y_initial, device=device)
    # elif args.initial=='no':
    #     D_explored_BOCD = D_exp(device=device)
    #     D_explored_TS=D_exp(device=device)
    
    # optimizer = torch.optim.Adam(diffusion.model.parameters(),\
    #                     lr=args.learning_rate, betas=(0.9, 0.99))

    # unlabelled_data, _ = generator.generate_data(args.num_unlabelled, torch_tensor=True)
    # dataloader = DataLoader(unlabelled_data, batch_size=args.batch_size, shuffle=True)
    
    
    # print("***** Running BOCD training *****")
    # start_time_BOCD = time.time()
    # regret=0;theta_RLS_set=[];theta_TS_set=[]
    # for t in tqdm(range(1,args.num_epochs+1), desc='Bayesian optimization'):
    #     if args.initial=='yes':
    #         print('mean y', D_explored_BOCD._mean().item())
    #         print('max y', D_explored_BOCD.acquisition().item())
    #         theta_RLS,theta_TS = D_explored_BOCD.TS_estimator(beta_0=args.beta_0, t_count=t)
    #     if args.initial=='no':
    #         theta_RLS,theta_TS = D_explored_BOCD.TS_estimator_revise(beta_0=args.beta_0, t_count=t,shape=args.dim_input)
    #     diffusion.model.train()
    #     if t <4:
    #         num_inner_epochs=args.num_inner_epochs
    #     else:
    #         num_inner_epochs=args.num_inner_epochs-5
    #     for inner_epoch in range(num_inner_epochs):
    #             epoch_loss = list()
    #             for idx, sample_batch in tqdm(enumerate(dataloader), total=len(dataloader),\
    #                         desc=f"Diffusion Training (epoch {inner_epoch+1}/{num_inner_epochs})"):
    #                 sample_batch = sample_batch.to(device)

    #                 labeled_samples = sample_batch.matmul(theta_TS)
    #                 noisy_score_batch = labeled_samples + 0.01 * torch.randn_like(labeled_samples)
    #                 w=args.w+t*0.08
    #                 loss = diffusion(sample_batch, noisy_score_batch.detach().view(-1))

    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()

    #                 epoch_loss.append(loss.item())
    #             print(f'Epoch [{inner_epoch+1}/{args.num_inner_epochs}], Loss: {np.mean(epoch_loss):.4f}')
    # #     #         # wandb.log({"inner_epoch": inner_epoch, "mean_batch_loss": np.mean(epoch_loss)})
                
    # #     #         # if (inner_epoch+1) % args.num_inner_epochs == 0:
    # #     #         #     torch.save(diffusion.model.state_dict(), f'./checkpoints/{args.run_name}_{inner_epoch+1}.pth')
    # #     #         #     print(f'Model saved at epoch {inner_epoch+1}')
    #     for sample_batch in range(args.num_data):
    #         regret_iter=0
    #         if args.initial=='yes':
    #             if t <6:
    #                 new_sample = diffusion.sample(torch.tensor(0.5+t*0.1).view(-1).to('cuda:0')).view(-1, args.dim_input)
    #             else:
    #                 new_sample = diffusion.sample(D_explored_BOCD.acquisition()).view(-1, args.dim_input)
    #         elif args.initial=='no':
    #             new_sample = diffusion.sample(torch.tensor(1).view(-1).to(device),w).view(-1, args.dim_input)
    #         # wandb.log({"epoch": t, "maximum y": D_explored_BOCD.acquisition().item()})
    #         value=generator.evaluate(new_sample).to(device)
    #         D_explored_BOCD.update(new_sample, value)
    #         regret_iter+=torch.from_numpy(np.array([np.linalg.norm(generator.theta)**2]))-value.to('cpu')
    #         print('sample value',value)
    #     theta_TS_set.append(theta_TS),theta_RLS_set.append(theta_RLS)
    #     regret+=(regret_iter/args.num_data)
    #     print('mean y', D_explored_BOCD._mean().item())
    #     print('max y', D_explored_BOCD.acquisition().item())
    #     writer.add_scalar("Metrics/BOCD_regret", regret.item(), global_step=t)
    #     writer.add_scalar("Metrics/BOCD_single_regret", (regret_iter/args.num_data).item(), global_step=t)
    #     writer.add_scalar("Metrics/BOCD_mean_y", D_explored_BOCD._mean().item(), global_step=t)
    #     writer.add_scalar("Metrics/BOCD_maximum_y", D_explored_BOCD.acquisition().item(), global_step=t)
        
    # end_time_BOCD = time.time()
    # execution_time_BOCD = end_time_BOCD - start_time_BOCD
    # print(f"Time taken for BOCD: {execution_time_BOCD} seconds")
    # writer.add_scalar("Metrics/execution_time_BOCD", execution_time_BOCD)
    
    
    # ## note samples
    # csv_sample_path = f"/nfsshare/home/xiechenghan/online_CDM/test/data/BOCD_sample_{args.dim_input}_{args.dim_latent}_{args.initial}_{args.run_name}.csv"
    # with open(csv_sample_path, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow([f"dim_{i+1}" for i in range(args.dim_input)]+["Value"])
    #     for i in range(D_explored_BOCD.x.shape[0]):
    #         sample_values = [item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in D_explored_BOCD.x[i]]
    #         csv_writer.writerow(list(sample_values)+[D_explored_BOCD.y[i].item()])
    # csv_theta_path = f"/nfsshare/home/xiechenghan/online_CDM/test/data/BOCD_theta_{args.dim_input}_{args.dim_latent}_{args.initial}_{args.run_name}.csv"
    # ## note theta
    # with open(csv_theta_path, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow([f"TS_dim_{i+1}" for i in range(args.dim_input)]+[f"RLS_dim_{i+1}" for i in range(args.dim_input)])
    #     for i in range(args.num_epochs):
    #         theta_TS_doc = [np.squeeze(item.cpu().numpy())if isinstance(item, torch.Tensor) else item for item in theta_TS_set[i]]
    #         theta_RLS_doc = [np.squeeze(item.cpu().numpy()) if isinstance(item, torch.Tensor) else item for item in theta_RLS_set[i]]
    #         csv_writer.writerow(list(theta_TS_doc) + list(theta_RLS_doc))
            
    # ### Plot
    # umap_plot(csv_sample_path,csv_theta_path,args.dim_input,generator.theta,args.num_data,f"_{args.dim_input}_{args.dim_latent}_{args.run_name}")
    # plot_distribution(csv_sample_path,args.num_data,f"_{args.dim_input}_{args.dim_latent}_{args.run_name}")
    # writer.add_tensor('true_theta', generator.theta)
    
    # #### Original TS
    # print("***** Running TS training *****")
    # start_time_TS = time.time()
    # regret=0
    # for t in tqdm(range(1,10*args.num_epochs+1), desc='TS optimization'):
    #     if args.initial=='yes':
    #         print('mean y', D_explored_TS._mean().item())
    #         print('max y', D_explored_TS.acquisition().item())
    #         theta_RLS,theta_TS = D_explored_TS.TS_estimator(beta_0=args.beta_0, t_count=t)
    #     if args.initial=='no':
    #         theta_RLS,theta_TS = D_explored_TS.TS_estimator_revise(beta_0=args.beta_0, t_count=t,shape=args.dim_input)
    #     # theta_RLS,theta_TS = D_explored_TS.TS_estimator(beta_0=args.beta_0, t_count=t)
    #     new_sample,_=optimize_vector_for_max_inner_product(theta_TS)
    #     new_sample=new_sample.T
    #     value=generator.evaluate(new_sample).to(device)
    #     regret+=torch.from_numpy(np.array([np.linalg.norm(generator.theta)**2]))-value
    #     D_explored_TS.update(new_sample, value)
    #     # diffusion.model.train()
    #     # for inner_epoch in range(args.num_inner_epochs):
    #     #         epoch_loss = list()
    #     #         for idx, sample_batch in tqdm(enumerate(dataloader), total=len(dataloader),\
    #     #                     desc=f"Diffusion Training (epoch {inner_epoch+1}/{args.num_inner_epochs})"):
    #     #             sample_batch = sample_batch.to(device)

    #     #             labeled_samples = sample_batch.matmul(theta_TS)
    #     #             noisy_score_batch = labeled_samples + 0.01 * torch.randn_like(labeled_samples)
                    
    #     #             loss = diffusion(sample_batch, noisy_score_batch.detach().view(-1))

    #     #             optimizer.zero_grad()
    #     #             loss.backward()
    #     #             optimizer.step()

    #     #             epoch_loss.append(loss.item())
    #     #         print(f'Epoch [{inner_epoch+1}/{args.num_inner_epochs}], Loss: {np.mean(epoch_loss):.4f}')
    #     #         # wandb.log({"inner_epoch": inner_epoch, "mean_batch_loss": np.mean(epoch_loss)})
                
    #             # if (inner_epoch+1) % args.num_inner_epochs == 0:
    #             #     torch.save(diffusion.model.state_dict(), f'./checkpoints/{args.run_name}_{inner_epoch+1}.pth')
    #             #     print(f'Model saved at epoch {inner_epoch+1}')
    #     # new_sample = diffusion.sample(D_explored_TS.acquisition()).view(-1, args.dim_input)
    #     print('mean y', D_explored_TS._mean().item())
    #     print('max y', D_explored_TS.acquisition().item())
    #     # wandb.log({"epoch": t, "maximum y": D_explored_BOCD.acquisition().item()})
    #     writer.add_scalar("Metrics/TS_single_regret", (torch.from_numpy(np.array([np.linalg.norm(generator.theta)**2]))-value).item(), global_step=t)
    #     writer.add_scalar("Metrics/TS_regret", regret.item(), global_step=t)
    #     writer.add_scalar("Metrics/TS_mean_y", D_explored_TS._mean().item(), global_step=t)
    #     writer.add_scalar("Metrics/TS_maximum_y", D_explored_TS.acquisition().item(), global_step=t)
    #     D_explored_TS.update(new_sample, generator.evaluate(new_sample).to(device))
    # end_time_TS = time.time()
    # execution_time_TS = end_time_TS - start_time_TS
    # writer.add_scalar("Metrics/execution_time_TS", execution_time_TS)