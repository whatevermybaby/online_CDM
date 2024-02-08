import torch
import pandas as pd
import numpy as np
# Set random seed for reproducibility
from data_generator import GaussianLatentSampler, set_seed
# import umap
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
# import umap.plot

# set_seed(seed=42)

# # Generate a random tensor using torch.randn
# random_tensor = torch.randn(3, 3)

# # Display the random tensor
# print(random_tensor)


def umap_plot(sample_url,theta_url,dim_input,x_star,num_data,name):
# set the style
    sns.set()
    sns.set_palette('muted')
    sns.set_context("paper")
    custom_style = {
        'axes.facecolor': 'lightgrey',  # Custom grey background
        'axes.edgecolor': 'black',      # Black edges
        'axes.grid': True,             # No grid (you can turn this on if you want grid)
        'figure.facecolor': 'white'     # White figure background
    }
    # Set the style
    sns.set_style('dark', rc=custom_style)
    
    df_sample=pd.read_csv(sample_url)

    # 提取元素
    selected_data = df_sample.iloc[:, :dim_input].values
    # 将数据转换为 NumPy 数组
    data = np.array(selected_data)
    df_theta=pd.read_csv(theta_url)
    theta_TS=np.array(df_theta.iloc[:, :dim_input].values);theta_RLS=np.array(df_theta.iloc[:, dim_input:].values)
    mapper = umap.UMAP(random_state=42).fit(data)
    # x_star=theta_RLS[-1]
    # x_star_trans = mapper.transform([x_star.reshape(dim_input)])
    x_star_trans = mapper.transform([x_star.reshape(dim_input)])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    # plt.tick_params(axis='both', which='major', labelsize=16)
    handles = []  # List to store legend handles
    labels = []   # List to store legend labels

    i = 0
    for t in [1, 2, 4, 9, 19, 29]:
        row = i % 3  # Calculate the current row
        col = i // 3   # Calculate the current column
        if t ==1:
            trans = mapper.transform(data[num_data * t:num_data * t + num_data])
            scatter=axes[row, col].scatter(trans[:, 0], trans[:, 1], cmap='Sequential', label=r'$t={{{0}}}$'.format(t), alpha=0.3)
            contour=sns.kdeplot(x=trans[:, 0], y=trans[:, 1], ax=axes[row, col], fill=True, cmap='Blues', levels=5)
            # handles.append(contour)  # Add scatter handle to the list
            # Access color from the scatter plot
            contour_color = scatter.get_facecolor()[0, :]
            # handles.append(Line2D([0], [0], color=contour_color, linewidth=2))  # Customize the line properties
            # labels.append(r'$t={{{0}}}$'.format(t))  # Add scatter label to the list
            # axes[row, col].scatter(x_star_trans[0, 0], x_star_trans[0, 1], marker='*', label=r'$\theta^*$', s=150)
            x_star_scatter=axes[row, col].scatter(x_star_trans[0, 0], x_star_trans[0, 1], marker='*', s=150)
            theta_ts = mapper.transform([theta_TS[t]])
            theta_rls = mapper.transform([theta_RLS[t]])
            # axes[row, col].scatter(theta_ts[0, 0], theta_ts[0, 1], marker='^', label=r'$\tilde\theta_{{{0}}}$'.format(t+1))
            # axes[row, col].scatter(theta_rls[0, 0], theta_rls[0, 1], marker='^', label=r'$\hat\theta_{{{0}}}$'.format(t+1))
            theta_TS_scatter=axes[row, col].scatter(theta_ts[0, 0], theta_ts[0, 1], marker='^')
            theta_RLS_scatter=axes[row, col].scatter(theta_rls[0, 0], theta_rls[0, 1], marker='^')
            handles.append(theta_RLS_scatter)
            labels.append(r'$\hat\theta_t$')
            handles.append(theta_TS_scatter)
            labels.append(r'$\tilde\theta_t$')
            handles.append(x_star_scatter)
            labels.append(r'$\theta^*$')
        else:
            trans = mapper.transform(data[num_data * t:num_data * t + num_data])
            scatter=axes[row, col].scatter(trans[:, 0], trans[:, 1], cmap='Sequential', label=r'$t={{{0}}}$'.format(t+1), alpha=0.3)
            contour=sns.kdeplot(x=trans[:, 0], y=trans[:, 1], ax=axes[row, col], fill=True, cmap='Blues', levels=5)
            contour_color = scatter.get_facecolor()[0, :]
            # handles.append(contour)  # Add scatter handle to the list
            # handles.append(Line2D([0], [0], color=contour_color, linewidth=2))  # Customize the line properties
            # labels.append(r'$t={{{0}}}$'.format(t+1))  # Add scatter label to the list
            # axes[row, col].scatter(x_star_trans[0, 0], x_star_trans[0, 1], marker='*', label=r'$\theta^*$', s=150)
            axes[row, col].scatter(x_star_trans[0, 0], x_star_trans[0, 1], marker='*', s=150)
            theta_ts = mapper.transform([theta_TS[t]])
            theta_rls = mapper.transform([theta_RLS[t]])
            # axes[row, col].scatter(theta_ts[0, 0], theta_ts[0, 1], marker='^', label=r'$\tilde\theta_{{{0}}}$'.format(t+1))
            # axes[row, col].scatter(theta_rls[0, 0], theta_rls[0, 1], marker='^', label=r'$\hat\theta_{{{0}}}$'.format(t+1))
            axes[row, col].scatter(theta_ts[0, 0], theta_ts[0, 1], marker='^')
            axes[row, col].scatter(theta_rls[0, 0], theta_rls[0, 1], marker='^')
     
        i += 1
    # Set x and y axis limits to be the maximum range across all subplots
    # max_xlim = max(ax.get_xlim()[1] for ax in axes)
    # min_xlim = min(ax.get_xlim()[0] for ax in axes)
    # max_ylim = max(ax.get_ylim()[1] for ax in axes)
    # min_ylim = min(ax.get_ylim()[0] for ax in axes)
     # Set x and y axis limits to be the maximum range across all subplots
    max_xlim = max(ax.get_xlim()[1] for row_axes in axes for ax in row_axes)
    min_xlim = min(ax.get_xlim()[0] for row_axes in axes for ax in row_axes)
    max_ylim = max(ax.get_ylim()[1] for row_axes in axes for ax in row_axes)
    min_ylim = min(ax.get_ylim()[0] for row_axes in axes for ax in row_axes)
    i=0
    for t in [1, 2, 4, 9, 19, 29]:
        row = i % 3  # Calculate the current row
        col = i // 3   # Calculate the current column
        axes[row, col].set_xlim(min_xlim, max_xlim)
        axes[row, col].set_ylim(min_ylim, max_ylim)
        axes[row, col].legend(loc='lower right',fontsize=20)
        axes[row, col].tick_params(axis='both', which='major', labelsize=22)
        i+=1
    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5, -0.000001), ncol=3, 
           frameon=False, fontsize=30)
    fig.subplots_adjust(wspace=0)  # Adjust the width of the spacing
    plt.tight_layout(rect=[0, 0.10, 1, 0.98])
    plt.savefig(f'/nfsshare/home/xiechenghan/online_CDM/test/data/figures/umap_plot_{name}.pdf',dpi=512, bbox_inches='tight', pad_inches=0)
    plt.show()
    #plot each map
#     fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(24, 4))

#     i = 0
#     # trans = mapper.transform(data[:num_data])
#     # axes[row, col].scatter(trans[:, 0], trans[:, 1], cmap='Sequential', label='t=0', alpha = 0.3)
#     # i+=1
#     for t in [1,2,5,7,10,14]:
#         trans = mapper.transform(data[num_data*t:num_data*t+num_data])
#         axes[row, col].scatter(trans[:, 0], trans[:, 1], cmap='Sequential', label=r'$S_{{{0}}}$'.format(t), alpha = 0.3)
#         # axes[row, col].scatter(theta_star_trans[0, 0],theta_star_trans[0, 1], marker='^', label=r'$\theta^*$')
#         axes[row, col].scatter(x_star_trans[0, 0],x_star_trans[0, 1], marker='*', label=r'$\theta^*$', s=150)
#         theta_ts = mapper.transform([theta_TS[t]])
#         theta_rls = mapper.transform([theta_RLS[t]])
#         axes[row, col].scatter(theta_ts[0, 0],theta_ts[0, 1], marker='^', label=r'$\tilde\theta_{{{0}}}$'.format(t))
#         axes[row, col].scatter(theta_rls[0, 0],theta_rls[0, 1], marker='^', label=r'$\hat\theta_{{{0}}}$'.format(t))
#         axes[row, col].legend(loc='lower right')
#         i += 1

#     fig.tight_layout()
#     plt.savefig(f'/nfsshare/home/xiechenghan/online_CDM/test/data3/figures/umap_plot_{name}.png')
#     plt.show()



df=pd.read_csv('/nfsshare/home/xiechenghan/online_CDM/test/data/True_theta_32_8_no_0.3.csv')
true_theta_1 = df.iloc[:, :].values
    # 将数据转换为 NumPy 数组
true_theta = np.array(true_theta_1 )
umap_plot('/nfsshare/home/xiechenghan/online_CDM/test/data/BOCD_sample_32_8_no_diminput_32_no_w0.3_30epo_2024.02.02_01.06.57.csv','/nfsshare/home/xiechenghan/online_CDM/test/data/BOCD_theta_32_8_no_diminput_32_no_w0.3_30epo_2024.02.02_01.06.57.csv',
          32,true_theta,10,f"_32_8_no_w0.3")
# umap_plot('/nfsshare/home/xiechenghan/online_CDM/test/data3/BOCD_sample_16_8_no_diminput_16_no_w1_15epo_2024.02.01_16.58.47.csv','/nfsshare/home/xiechenghan/online_CDM/test/data3/BOCD_theta_16_8_no_diminput_16_no_w1_15epo_2024.02.01_16.58.47.csv',
#           16,None,10,f"_16_8_no_w1")
# umap_plot('/nfsshare/home/xiechenghan/online_CDM/test/data3/BOCD_sample_16_8_no_diminput_16_no_w2_15epo_2024.02.01_17.00.24.csv','/nfsshare/home/xiechenghan/online_CDM/test/data3/BOCD_theta_16_8_no_diminput_16_no_w1_15epo_2024.02.01_16.58.47.csv',
#           16,None,10,f"_16_8_no_w2")
# # 创建一个维度为25的向量，模拟5轮迭代，每轮生成5个分量
# data = np.random.rand(25).reshape(5, 5)

# # 绘制柱状分布图
# fig, axs = plt.subplots(5, 1, figsize=(8, 12))

# # 遍历每一轮迭代
# for i in range(5):
#     axs[i].bar(range(1, 6), data[i, :], label=f'Iteration {i+1}')
#     axs[i].set_xlabel('Component')
#     axs[i].set_ylabel('Value')
#     axs[i].set_title(f'Distribution - Iteration {i+1}')
#     axs[i].legend()

# plt.tight_layout()

# # 保存柱状分布图为图像文件（可根据需要修改文件格式）
# plt.savefig('/nfsshare/home/xiechenghan/online_CDM/test/bar_chart.png')


def plot_distribution(data_url,num_data,name):
    df=pd.read_csv(data_url)
    data = df.iloc[:,-1].values.reshape(-1,num_data)
    overall_min = data.min()
    overall_max = data.max()
    # 绘制柱状分布图
    fig, axs = plt.subplots(6, 1, figsize=(8, 12))

    # 遍历每一轮迭代
    i=0
    for t in [0,2,4,9,19,29]:
        axs[i].hist(data[t, :], bins=100, density=True, range=(overall_min, overall_max),alpha=0.7, label=f'Iteration {t+1}')
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Density')
        axs[i].set_title(f'Distribution - Iteration {t+1}')
        axs[i].legend()
        i+=1

    plt.tight_layout()

    # 保存柱状分布图为图像文件（可根据需要修改文件格式）
    plt.savefig(f'/nfsshare/home/xiechenghan/online_CDM/test/data/figures/histogram_{name}.png')

    # 显示图形
    plt.show()

# data = (np.random.rand(200)*10).reshape(10, 20)
# plot_distribution('/nfsshare/home/xiechenghan/online_CDM/test/data2/BOCD_sample_32_16_no.csv',10,f"_32_16_no")
# fig, axs = plt.subplots(5, 1, figsize=(8, 12))

#     # 遍历每一轮迭代


# plt.tight_layout()

# # 保存柱状分布图为图像文件（可根据需要修改文件格式）
# plt.savefig(f'/nfsshare/home/xiechenghan/online_CDM/test/histogram.png')

# # 显示图形
# plt.show()