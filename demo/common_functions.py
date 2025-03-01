import numpy as np
import matplotlib.pyplot as plt

# 创建子图布局
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Common Mathematical Functions', fontsize=16)

# 1. 线性函数 y = mx + b
x1 = np.linspace(-5, 5, 100)
axs[0, 0].plot(x1, 2*x1 + 1, label='y = 2x + 1')
axs[0, 0].plot(x1, -x1 - 2, label='y = -x - 2')
axs[0, 0].set_title('Linear Functions')
axs[0, 0].grid(True)
axs[0, 0].legend()

# 2. 二次函数 y = ax² + bx + c
x2 = np.linspace(-4, 4, 100)
axs[0, 1].plot(x2, x2**2, label='y = x²')
axs[0, 1].plot(x2, -(x2**2) + 4, label='y = -x² + 4')
axs[0, 1].set_title('Quadratic Functions')
axs[0, 1].grid(True)
axs[0, 1].legend()

# 3. 指数函数 y = aˣ
x3 = np.linspace(-2, 4, 100)
axs[1, 0].plot(x3, np.exp(x3), label='y = e^x')
axs[1, 0].plot(x3, 2**x3, label='y = 2^x')
axs[1, 0].set_title('Exponential Functions')
axs[1, 0].grid(True)
axs[1, 0].legend()

# 4. 对数函数 y = log(x)
x4 = np.linspace(0.1, 5, 100)
axs[1, 1].plot(x4, np.log(x4), label='y = ln(x)')
axs[1, 1].plot(x4, np.log2(x4), label='y = log₂(x)')
axs[1, 1].set_title('Logarithmic Functions')
axs[1, 1].grid(True)
axs[1, 1].legend()

# 5. 三角函数
x5 = np.linspace(-2*np.pi, 2*np.pi, 100)
axs[2, 0].plot(x5, np.sin(x5), label='y = sin(x)')
axs[2, 0].plot(x5, np.cos(x5), label='y = cos(x)')
axs[2, 0].set_title('Trigonometric Functions')
axs[2, 0].grid(True)
axs[2, 0].legend()

# 6. 幂函数 y = xⁿ
x6 = np.linspace(0, 3, 100)
axs[2, 1].plot(x6, x6**0.5, label='y = √x')
axs[2, 1].plot(x6, x6**2, label='y = x²')
axs[2, 1].plot(x6, x6**3, label='y = x³')
axs[2, 1].set_title('Power Functions')
axs[2, 1].grid(True)
axs[2, 1].legend()

# 调整布局
plt.tight_layout()
plt.show() 