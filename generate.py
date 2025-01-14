# generate.py
import torch
import matplotlib.pyplot as plt
from VAE import model, input_size, mnist # 从 VAE.py 中导入模型、输入大小和 MNIST 数据集

# 加载已训练好的模型
model.load_state_dict(torch.load('vae.pth'))

# 选择mnist的样本图像
sample_image = mnist[18][0] # mnist[1][0]是数字5的数据集

# 使用 VAE 的编码器将样本图像编码为 latent variables
mu, log_var = model.encoder(sample_image.view(-1, input_size))
std = torch.exp(0.5 * log_var)  # 从 log_var 计算标准差
z = mu + std * torch.randn_like(std)  # 添加随机扰动

# 将生成的 latent variables 作为输入传递给 VAE 的解码器，生成数字图像
generated_image = model.decoder(mu).view(28, 28)

# 添加随机扰动后的生成图像
diverse_generated_image = model.decoder(z).view(28, 28)

# 可视化对比
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(sample_image.view(28, 28), cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Noise-Free Generated Image")
plt.imshow(generated_image.detach().numpy(), cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Noise-Injected Generated Image")
plt.imshow(diverse_generated_image.detach().numpy(), cmap='gray')
plt.show()