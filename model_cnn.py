import torchimport torch.nn as nnimport torch.nn.functional as F# model ---------------------------------------------------------class Generator_CNN(nn.Module):    def __init__(self, c=128):        super(Generator_CNN, self).__init__()        self.deconv1 = nn.ConvTranspose2d(100, c*8, 2, 1, 0)        self.deconv1_bn = nn.BatchNorm2d(c*8)        self.deconv2 = nn.ConvTranspose2d(c*8, c*4, 4, 2, 1)        self.deconv2_bn = nn.BatchNorm2d(c*4)        self.deconv3 = nn.ConvTranspose2d(c*4, c*2, 3, 2, 1)        self.deconv3_bn = nn.BatchNorm2d(c*2)        self.deconv4 = nn.ConvTranspose2d(c*2, c, 4, 2, 1)        self.deconv4_bn = nn.BatchNorm2d(c)        self.deconv5 = nn.ConvTranspose2d(c, 1, 4, 2, 1)    def forward(self, x):        x = F.relu(self.deconv1_bn(self.deconv1(x)))        x = F.relu(self.deconv2_bn(self.deconv2(x)))        x = F.relu(self.deconv3_bn(self.deconv3(x)))        x = F.relu(self.deconv4_bn(self.deconv4(x)))        x = F.tanh(self.deconv5(x))        return xclass Discriminator_CNN(nn.Module):    def __init__(self, c=128):        super(Discriminator_CNN, self).__init__()        self.conv1 = nn.Conv2d(1, c, 4, 2, 1)        self.conv1_bn = nn.BatchNorm2d(c)        self.conv2 = nn.Conv2d(c, c*2, 4, 2, 1)        self.conv2_bn = nn.BatchNorm2d(c*2)        self.conv3 = nn.Conv2d(c*2, c*4, 3, 2, 1)        self.conv3_bn = nn.BatchNorm2d(c*4)        self.conv4 = nn.Conv2d(c*4, c*8, 4, 2, 1)        self.conv4_bn = nn.BatchNorm2d(c*8)        self.conv5 = nn.Conv2d(c*8, 1, 2, 1, 0)    def forward(self, x):        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))        x = F.leaky_relu(self.conv4_bn(self.conv4(x)))        x = F.sigmoid(self.conv5(x))        return x