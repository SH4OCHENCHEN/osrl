import numpy as np
import matplotlib.pyplot as plt

# 加载NPY文件
data = np.load('artifacts/fisor_2024/results/FLOWCHUNK_OfflineCarGoal1_123_costs.npy', allow_pickle=True)  # 替换为你的文件路径

# 确保数据是二维数组形式（多个列表的列表）
if data.ndim == 1:
    # 如果是一维数组，可能需要重新组织数据
    data = np.array([np.array(x) for x in data])

# 计算平均值、最大值和最小值
mean_values = np.mean(data, axis=0)
max_values = np.max(data, axis=0)
min_values = np.min(data, axis=0)

# 创建x轴数据点
x = np.arange(len(mean_values))

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, mean_values, 'b-', linewidth=2, label='mean')
plt.fill_between(x, min_values, max_values, alpha=0.3, color='gray')

# 添加标签和标题
plt.xlabel('Gradient Steps')
plt.ylabel('Value')
plt.title('Reward')
plt.legend()
plt.grid(True, alpha=0.3)

# 显示图形
plt.tight_layout()
plt.show()
