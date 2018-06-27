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
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--img_dir', type=int, required=True)
parser.add_argument('--epoch', type=int, default=100)
args = parser.parse_args()

# cuda ---------------------------------------------------------
device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

# new model ----------------------------------------------------
class Generator2(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (self.img_size[0] / 16, self.img_size[1] / 16)

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator2(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * dim * (img_size[0] / 16) * (img_size[1] / 16)
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)

# ==================Definition Start======================
DIM =  64
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.view(-1, 128)
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

# model ---------------------------------------------------------
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
                nn.Sigmoid(),
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
epochs = args.epoch

train_data = MNIST('~/Project/data/mnist_data',
        train=True, download=True,
        transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size,
        shuffle=True)
cycle = torch.floor(torch.FloatTensor([len(train_loader)], device=device) / 10).item()

G = Generator()
D = Discriminator()

G.to(device)
D.to(device)

optimizer_G = optim.RMSprop(G.parameters(), lr=1e-5)
optimizer_D = optim.RMSprop(D.parameters(), lr=1e-5)

for epoch in range(epochs):
    Tloss_D = 0.0
    Tloss_G = 0.0
    for i, data in enumerate(train_loader, 0):
        for j in range(5):
            optimizer_D.zero_grad()
            # real
            real = data[0].view(-1, 1, 28, 28)
            real = real.to(device)
            
            # fake
            z = torch.randn(batch_size, 128, device=device).view(-1, 128, 1, 1)
            fake = G(z)
            
            # loss
            loss_D = - torch.mean(D(real)) + torch.mean(D(fake))
            loss_D.backward()
            optimizer_D.step()

            clip_value = float(0.01)
            for p in D.parameters():
                p.grad.data.clamp_(min=-clip_value, max=clip_value)

            Tloss_D += loss_D

        optimizer_G.zero_grad()
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

        z = torch.randn(batch_size, 128, device=device).view(-1, 128, 1, 1)
        gen_imgs = G(z)
        save_image(gen_imgs.data[:25], 'images/images{}/{}.png'.format(args.img_dir, epoch), nrow=5, normalize=True)
