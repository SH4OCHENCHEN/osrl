import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制曲线
ax.plot(x, y1, label=r'$\sin(x)$', linewidth=2)
ax.plot(x, y2, label=r'$\cos(x)$', linewidth=2)

# 设置标题和坐标轴标签
ax.set_title(r'Trigonometric Functions: Relationship with $\bar{V}_r$', fontsize=16)
ax.set_xlabel(r'Angle ($\theta$, radians)', fontsize=14)
ax.set_ylabel(r'Function Value', fontsize=14)

# 添加图例
ax.legend(loc='upper right', fontsize=12)

# 添加文本注释
ax.text(2, 0.5, r'Important formula: $\bar{V}_r = \frac{1}{N}\sum_{i=1}^N V_{r,i}$',
        fontsize=14, bbox=dict(facecolor='white', alpha=0.7))

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()