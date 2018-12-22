import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='Model and Parameters for GAN Training')
parser.add_argument('--cnn', action='store_true')
parser.add_argument('--img_dir', type=int, required=True)
args = parser.parse_args()


# cuda ---------------------------------------------------------
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# model_mlp ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 100)
        self.fc3 = nn.Linear(100, 784)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = x.view(-1, 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 10)
        self.fc2_bn = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x).view(-1, 1, 1, 1)
        return x

# model_cnn ---------------------------------------------------------
class Generator_CNN(nn.Module):
    def __init__(self, c=128):
        super(Generator_CNN, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, c*8, 2, 1, 0),
            nn.BatchNorm2d(c*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(c*8, c*4, 4, 2, 1),
            nn.BatchNorm2d(c*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(c*4, c*2, 3, 2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(c*2, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            nn.ConvTranspose2d(c, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        out = self.model(input)
        return out

class Discriminator_CNN(nn.Module):
    def __init__(self, c=128):
        super(Discriminator_CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(True),
            nn.Conv2d(c, c*2, 4, 2, 1),
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(True),
            nn.Conv2d(c*2, c*4, 3, 2, 1),
            nn.BatchNorm2d(c*4),
            nn.LeakyReLU(True),
            nn.Conv2d(c*4, c*8, 4, 2, 1),
            nn.BatchNorm2d(c*8),
            nn.LeakyReLU(True),
            nn.Conv2d(c*8, 1, 2, 1, 0),
        )

    def forward(self, input):
        out = self.model(input)
        return out


# train ---------------------------------------------------------
batch_size = 64
epochs = 100
lam = 10
lr = 1e-4
beta_1 = 0.0
beta_2 = 0.9


train_data = MNIST('~/Project/data/mnist_data',
        train=True, download=True,
        transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size,
        shuffle=True)
cycle = torch.floor(torch.FloatTensor([len(train_loader)], device=device) / 10).item()

if args.cnn:
    G = Generator_CNN()
    D = Discriminator_CNN()
else:
    G = Generator()
    D = Discriminator()

G.to(device)
D.to(device)

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta_1, beta_2))

for epoch in range(epochs):
    Tloss_D = 0.0
    Tloss_G = 0.0
    for i, data in enumerate(train_loader, 0):
        for j in range(5):
            optimizer_D.zero_grad()
            # real
            real = data[0].view(-1, 1, 28, 28)
            real = real.to(device)
            mini_batch = real.shape[0]
            
            # fake
            if args.cnn:
                z = torch.randn(mini_batch, 100, device=device).view(-1, 100, 1, 1)
            else:
                z = torch.randn(mini_batch, 1, device=device).view(-1, 1, 1, 1)
            fake = G(z)
            
            # loss
            eps = torch.rand(mini_batch, 1, device=device).view(-1, 1, 1, 1)
            mid = eps * real + (1 - eps) * fake
            grad_outputs = torch.ones(D(mid).shape, device=device)
            grad_mid = torch.autograd.grad(D(mid), mid, grad_outputs=grad_outputs, retain_graph=True)[0]
            grad_p = grad_mid ** 2
            grad_p = torch.sum(grad_p, 2, True)
            grad_p = torch.sum(grad_p, 3, True)
            grad_p = torch.sqrt(grad_p)
            grad_p = lam * (grad_p - 1) ** 2
            loss_D = torch.mean(- D(real) + D(fake) + grad_p)
            loss_D.backward()
            optimizer_D.step()

            Tloss_D += loss_D

        if epoch >= 1:
            optimizer_G.zero_grad()
            if args.cnn:
                z = torch.randn(mini_batch, 100, device=device).view(-1, 100, 1, 1)
            else:
                z = torch.randn(mini_batch, 1, device=device).view(-1, 1, 1, 1)
            fake = G(z)
            loss_G = - torch.mean(D(fake))
            loss_G.backward()
            optimizer_G.step()

            Tloss_G += loss_G

        if i % cycle == 0:
            print('[%d, %6d] loss_G: %f loss_D: %f' %
                    (epoch, i, Tloss_G / cycle, Tloss_D / (cycle * 5)))
            Tloss_D = 0.0
            Tloss_G = 0.0

    if args.cnn:
        z = torch.randn(mini_batch, 100, device=device).view(-1, 100, 1, 1)
    else:
        z = torch.randn(mini_batch, 1, device=device).view(-1, 1, 1, 1)
    gen_imgs = G(z)
    save_image(gen_imgs.data[:25], 'images/images{}/{}.png'.format(args.img_dir, epoch), nrow=5, normalize=True)
