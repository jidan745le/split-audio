import numpy as np
import matplotlib.pyplot as plt
import math

def normal(x, mu, sigma):
    """
    计算高斯分布概率密度
    x: 输入值
    mu: 均值
    sigma: 标准差
    """
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 创建 x 轴数据
x = np.arange(-7, 7, 0.01)

# 定义不同的均值和标准差组合
params = [(0, 1), (0, 2), (3, 1)]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制每个高斯分布
for mu, sigma in params:
    y = normal(x, mu, sigma)
    plt.plot(x, y, label=f'mean {mu}, std {sigma}')

# 设置图形属性
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Gaussian Distribution')
plt.legend()
plt.grid(True)

# 显示图形
plt.show() 