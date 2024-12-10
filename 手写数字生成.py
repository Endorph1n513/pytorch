import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# 创建用于保存生成图片的文件夹
os.makedirs("generated_images", exist_ok=True)

# 数据转换（将图像转为Tensor并进行归一化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载训练数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.fc(z).view(-1, 1, 28, 28)

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# 初始化模型
generator = Generator()
discriminator = Discriminator()

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 固定噪声，用于生成图像
fixed_noise = torch.randn(64, 100)

# 训练循环
num_epochs = 200
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(trainloader):
        # ---------------------
        # 训练判别器
        # ---------------------
        optimizer_D.zero_grad()
        
        real_labels = torch.ones(images.size(0), 1)
        fake_labels = torch.zeros(images.size(0), 1)
        
        # 训练判别器：真实图像
        real_images = images
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        # 训练判别器：生成图像
        z = torch.randn(images.size(0), 100)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        
        optimizer_D.step()

        # ---------------------
        # 训练生成器
        # ---------------------
        optimizer_G.zero_grad()
        
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        
        optimizer_G.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss_real.item()+d_loss_fake.item()}, G Loss: {g_loss.item()}')

    # 生成并保存图像
    if (epoch+1) % 4 == 0:
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
            grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(f"generated_images/epoch_{epoch+1}.png")  # 保存图片到文件夹
            plt.close()
