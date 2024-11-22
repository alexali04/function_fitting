import matplotlib.pyplot as plt
import torch

# Seed for reproducibility
torch.manual_seed(1)


# Sample uniformly from (min, max)
def uniform_dist(min, max, size):
    return torch.rand(size) * (max - min) + min

# Noisy labels
noise = torch.randn(10) * 2.0
x = uniform_dist(0, 1, 10)
y_noisy = 3.0 * x + noise
y = 3.0 * x



# Standard Linear Regression - beta = x^T y / x^T x = cov(x, y) / var(x)
cov_xy = torch.matmul(x, y_noisy)
cov_xx = torch.matmul(x, x)
beta = cov_xy / cov_xx

f_x = beta * x

# Draw Lines 
plt.plot(x, y_noisy, 'o', label='Noisy samples', c='red')
plt.plot(x, y, '-o', label='True labels', c='green')
plt.plot(x, f_x, '-o', label="Standard Linear Regression", c='blue')

plt.legend()
plt.show()







