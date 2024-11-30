import matplotlib.pyplot as plt
import torch
import imageio
import imageio
import logging
import os
from typing import Union

FOLDER = "IMAGES"

def extract_num(
    filename: str
) -> Union[int, float]:
    """
    Extracts number from filename
    """
    base = os.path.basename(filename)
    number = ''.join(filter(str.isdigit, base))
    return int(number) if number.isdigit() else float('inf')

def make_gif(
    folder: str,
    name: str,
    fps: int
) -> None:
    """
    Construct gif 'name' from the images in 'folder'

    Args:
        folder: path to gif folder
        name: name of gif to make
        fps: frames per second
    """
    logging.info(f"Making {name}.gif from images in {folder}")

    with imageio.get_writer(f'{name}.gif', mode = 'I', fps = fps, loop=0) as writer:
        for filename in sorted(os.listdir(folder), key=extract_num):
            if filename.endswith('png'):
                image = imageio.imread(folder+"/"+filename)
                writer.append_data(image)


# Seed for reproducibility
torch.manual_seed(42)

# W/ 20 samples,
# torch seed 1, 6, 7, 9 - MLE overestimates variance
# torch seed 2, 3, 5 - MLE underestimates variance
# torch seed 4, 8, 10 - MLE unbiased underestimates variance but MLE biased overestimates variance

# W/ 40 samples,
# torch seed 1, 2, 4, 6, 10, 13 - MLE overestimates variance
# torch seed 3, 5, 8, 9, 11, 12 - MLE underestimates variance
# torch seed 7 - MLE unbiased underestimates variance but MLE biased overestimates variance



# W/ 100 samples, 
# torch seed 1, 3, 6 - MLE overestimates variance
# torch seed 2, 4, 5, 7, 8, 9, 10, 42 - MLE underestimates variance

# a = torch.stack([torch.ones(100), 2.0 * torch.ones(100)], dim=1)    # this is how we include bias
# b = torch.tensor([1.0, 2.0])
# l = a @ b

def get_mle_slope(X, y):
    cov_xx = X.T @ X
    cov_xy = X.T @ y
    if cov_xx.numel() == 1:
        return cov_xy / cov_xx
    else:
        return torch.linalg.inv(cov_xx) @ cov_xy

def linear_regression_mle(x, y, noise_errors, num_samples, true_noise_var, id):
    """
    Linear Regression using MLE
    - produces point estimates for w and sigma^2
    """

    # we don't want to modify the original y so we define y_noise
    # Also, we need to multiply by sqrt of true_noise_var - multiplying a RV by a scalar a scales the variance by a^2

    y_noise = y + noise_errors * torch.sqrt(true_noise_var)       

    X = torch.stack([torch.ones(num_samples), x], dim=1)
    w_MLE = get_mle_slope(X, y_noise)
    if w_MLE.numel() == 1:
        y_hat = w_MLE * x
    else:
        y_hat = X @ w_MLE
    
    err = y_noise - y_hat
    sigma_sq_mle = torch.mean(err**2)   # biased estimator
    sigma_sq_mle_unbiased = torch.sum(err**2) / (num_samples - X.shape[1])  # unbiased estimator

    plt.plot(x, y_hat, '-*', label="Predictions", c="red")
    plt.plot(x, y_noise, 'o', label="Noisy samples", c='blue')
    plt.ylim(0, 225)
    slope_str = f"biased: {sigma_sq_mle:.2f}, unbiased: {sigma_sq_mle_unbiased:.2f}, true variance: {true_noise_var:.2f}"
    if w_MLE.numel() > 1:   
        slope_str = f"{slope_str}, Predicted slope: {w_MLE[1].item():.2f}"
    else:
        slope_str = f"{slope_str}, Predicted slope: {w_MLE.item():.2f}"

    print(slope_str)
    plt.title(slope_str, fontsize=10, color="black")
    plt.legend(loc="upper left")
    plt.savefig(f"{FOLDER}/{str(id)}.png")
    plt.close()

variances = torch.linspace(0, 100, 21)
true_slope, bias = 3.0, 40.0
max_x, min_x = 50.0, 0.0
num_samples = 100

# for better visualization, we'll fix the noise and input samples
x = (max_x - min_x) * torch.rand(num_samples) + min_x
y = bias + true_slope * x
noise_errors = torch.randn(num_samples)
for i, var in enumerate(variances):
    linear_regression_mle(x=x, y=y, noise_errors=noise_errors, num_samples=num_samples, true_noise_var=var, id=i)

make_gif(FOLDER, "lin_reg_mle", 1.5)



# What if we don't have a bias?

# cov_x_no_bias = x.T @ x
# cov_x_y_no_bias = x.T @ y
# w_mle_no_bias = cov_x_y_no_bias / cov_x_no_bias         # Since its a scalar, the "inverse" is just division
# y_hat_no_bias = x * w_mle_no_bias

# plt.plot(x, y_hat_no_bias, '-*', label="Predictions without bias", c="green")
# plt.plot(x, y_hat, '-*', label="Predictions", c="red")
# plt.plot(x, y, 'o', label="Noisy samples", c='blue')
# plt.legend()
# plt.show()


# MAP Estimation













# # Noisy labels
# sigma_2 = torch.randn(20)
# x = uniform_dist(0, 3, 20)
# y_noisy = 3.0 * x + 1.0 * sigma_2
# y = 3.0 * x

# def polynomial_basis(x, degree):
#     return torch.stack([x**i for i in range(degree)])

# def get_beta_scalar(x, y):
#     """
#     Standard Linear Regression - beta = x^T y / x^T x = cov(x, y) / var(x)
#     """
#     cov_xy = torch.matmul(x, y)
#     cov_xx = torch.matmul(x, x)
#     return cov_xy / cov_xx

# beta_scalar = get_beta_scalar(x, y_noisy)

# f_x = beta_scalar * x

# # Draw Lines 
# plt.plot(x, y_noisy, 'o', label='Noisy samples', c='red')
# plt.plot(x, y, '-o', label='True labels', c='green')
# # plt.plot(x, f_x, '-o', label="Standard Linear Regression", c='blue')

# plt.legend()
# plt.show()


# Variance
# var_y = torch.mean((y_noisy - f_x)**2)
# print(var_y)
# print(torch.var(y_noisy))
# print(f"True Variance: {0.3}")




# # MAP Estimation
# x = torch.linspace(0, 3, 100)
# y_noise = 3.0 * x + 1.0 * torch.randn(100)

# alpha_2 = 1e-6
# sigma_2 = 1.0
# lambda_2 = sigma_2 / alpha_2

# beta_map = (x.T @ y_noise) / (x.T @ x + lambda_2)
# beta_mle = (x.T @ y_noise) / (x.T @ x)

# y_map = beta_map * x
# y_mle = beta_mle * x
# plt.plot(x, y_map, '-o', label="MAP Estimation", c='blue')
# plt.plot(x, y_noise, 'o', label="Noisy samples", c='red')
# plt.plot(x, y_mle, '-o', label="MLE Estimation", c='green')
# plt.legend()
# plt.show()


