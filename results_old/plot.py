import torch
import pandas as pd
import numpy as np
# Set random seed for reproducibility
from ..online_CDM.data_generator import GaussianLatentSampler, set_seed
import umap
import matplotlib.pyplot as plt
import umap.plot

set_seed(seed=42)

# # Generate a random tensor using torch.randn
# random_tensor = torch.randn(3, 3)

# # Display the random tensor
# print(random_tensor)



# 读取 CSV 文件
df = pd.read_csv('./results/data/recordedBOCD_32.csv')

# 提取除第一行外的前 64 列数据
selected_data = df.iloc[1:, :32].values

# 将数据转换为 NumPy 数组
data = np.array(selected_data)
mapper = umap.UMAP(random_state=42).fit(data)

# x_star = Config.theta_star.copy()
x_star=np.rand(32)
theta_star=np.rand(32)+0.01
x_star[x_star>0] = 1
x_star[x_star<0] = 0


for t in range(3):
    trans = mapper.transform(data[t:t+2])
    plt.scatter(trans[:, 0], trans[:, 1], cmap='Spectral', label=r'$S_{{{0}}}$'.format(t), alpha = 0.3)
    
theta_star_trans = mapper.transform([theta_star])
# plt.scatter(theta_star_trans[0, 0],theta_star_trans[0, 1], marker='*', label=r'$\theta^*$')
x_star_trans = mapper.transform([x_star])
plt.scatter(x_star_trans[0, 0],x_star_trans[0, 1], marker='*', label=r'$x^*$', s=150)

plt.legend(loc='lower right')
plt.show()