import torch
import pandas as pd
import numpy as np
# Set random seed for reproducibility
from data_generator import GaussianLatentSampler, set_seed
# import umap
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns


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
df_1=pd.read_csv("/nfsshare/home/xiechenghan/online_CDM/results/data2/BOCD_sample_32_8_yes.csv")
df_2=pd.read_csv("/nfsshare/home/xiechenghan/online_CDM/test/data/BOCD_sample_32_8_no_diminput_32_no_w0.5_30epo_2024.02.02_01.07.06.csv")
data_yes=df_1.iloc[:,-1].values.reshape(-1,10)
data = df_2.iloc[:,-1].values.reshape(-1,10)
overall_min = data.min()
overall_max = data.max()
# # 绘制柱状分布图
# fig, axs = plt.subplots(6, 1, figsize=(8, 12))

# # 遍历每一轮迭代
# i=0
# axs[i].hist(data_yes[0, :], bins=100, density=True, range=(overall_min, overall_max),alpha=0.7, label=f'Iteration {1}')
# axs[i].set_xlabel('Value')
# axs[i].set_ylabel('Density')
# axs[i].set_title(f'Distribution - Iteration {1}')
# axs[i].legend()
# i+=1
# for t in [1,4,9,19,29]:
#     axs[i].hist(data[t, :], bins=100, density=True, range=(overall_min, overall_max),alpha=0.7, label=f'Iteration {t+1}')
#     axs[i].set_xlabel('Value')
#     axs[i].set_ylabel('Density')
#     axs[i].set_title(f'Distribution - Iteration {t+1}')
#     axs[i].legend()
#     i+=1

# plt.tight_layout()

# # 保存柱状分布图为图像文件（可根据需要修改文件格式）
# plt.savefig(f'/nfsshare/home/xiechenghan/online_CDM/histogram_32_8_no.png')

# # 显示图形
# plt.show()

# 绘制柱状分布图
fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # 将子图的布局改为3行2列

# 遍历每一轮迭代
i = 0
axs[0, 0].hist(data_yes[0, :], bins=100, density=True, range=(overall_min, overall_max), alpha=0.7, label=f'Iter{1}',color='orange')
# axs[0, 0].set_xlabel('Value',fontsize=22)
# axs[0, 0].set_ylabel('Density',fontsize=22)
# axs[0, 0].set_title(f'Reward Distribution - Iteration {1}',fontsize=22)
axs[0,0].tick_params(axis='both', which='major', labelsize=22)
axs[0, 0].legend(fontsize=20)
i += 1

for t in [1, 4, 9, 19, 29]:
    row, col = divmod(i, 3)  # 计算当前迭代的行和列
    axs[row, col].hist(data[t, :], bins=100, density=True, range=(overall_min, overall_max), alpha=0.7, label=f'Iter {t+1}',color='orange')
    # axs[row, col].set_xlabel('Value',fontsize=22)
    # axs[row, col].set_ylabel('Density',fontsize=22)
    # axs[row, col].set_title(f'Reward Distribution - Iter {t+1}',fontsize=22)
    axs[row, col].tick_params(axis='both', which='major', labelsize=22)
    axs[row, col].legend(fontsize=20)
    i += 1

plt.tight_layout()
fig.subplots_adjust(wspace=0)  # Adjust the width of the spacing
plt.tight_layout(rect=[0, 0.10, 1, 0.98])

# 保存柱状分布图为图像文件（可根据需要修改文件格式）
plt.savefig(f'/nfsshare/home/xiechenghan/online_CDM/histogram_32_8_no_change.pdf',dpi=512, bbox_inches='tight', pad_inches=0)

# 显示图形
plt.show()