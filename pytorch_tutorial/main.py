import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from model import Generator, Discriminator
from utils import weights_init
from dataset import get_loader
from train import train, visualize
from train_d import train, visualize


# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

config = {
    # Root directory for dataset
    'dataroot': "./data/celeba",
    
    # Number of workers for dataloader
    'workers': 2,
    
    # Batch size during training
    'batch_size': 128,
    
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    'image_size': 64,
    
    # Number of channels in the training images. For color images this is 3
    'nc': 3,
    
    # Size of z latent vector (i.e. size of generator input)
    'nz': 100,
    
    # Size of feature maps in generator
    'ngf': 64,
    
    # Size of feature maps in discriminator
    'ndf': 64,
    
    # Number of training epochs
    'num_epochs': 5,
    
    # Learning rate for optimizers
    'lr': 0.0002,
    
    # Beta1 hyperparam for Adam optimizers
    'beta1': 0.5,
    
    # Number of GPUs available. Use 0 for CPU mode.
    'ngpu': 1

    }
    
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
netG = Generator(ngpu, nz, ngf, nc).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
# netG.load_state_dict(torch.load('./models/netG.pth'))
# netG.eval()

# Create the Discriminator
netD = Discriminator(ngpu, nz, ndf, nc).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


dataloader = get_loader(dataroot, image_size, batch_size, workers)
G_losses, D_losses, img_list = train(num_epochs, dataloader, netG, netD, real_label, fake_label,
    optimizerG, optimizerD, criterion, device, fixed_noise, nz)
visualize(G_losses, D_losses, img_list, dataloader, device)
