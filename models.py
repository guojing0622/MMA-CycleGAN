import torch
import torch.nn as nn

#from spectral import SpectralNorm
from attn import Multi_Attn

##############################
#        Generator
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=3):
        super(Generator, self).__init__()

        # Initial convolution block
        self.layerx1 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(in_channels, 64, 7),
                                     nn.InstanceNorm2d(64),
                                    nn.ReLU(inplace=True) )
        self.layery1 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(in_channels, 64, 7),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(inplace=True) )

        # Downsampling
        in_features = 64
        self.layerx2 = nn.Sequential(nn.Conv2d(in_features, in_features * 2,
                                     4, stride=2, padding=1),
                                     nn.InstanceNorm2d(in_features * 2),
                                     nn.ReLU(inplace=True))
        self.layery2 = nn.Sequential(nn.Conv2d(in_features, in_features * 2, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(in_features * 2),
                                     nn.ReLU(inplace=True))

        in_features = 2 * in_features #128
        self.layerx3 = nn.Sequential(nn.Conv2d(in_features, in_features*2, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(in_features*2),
                                     nn.ReLU(inplace=True))
        self.layery3 = nn.Sequential(nn.Conv2d(in_features, in_features*2, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(in_features*2),
                                     nn.ReLU(inplace=True))

        in_features = 2 * in_features #256
        # Residual blocks
        if res_blocks != 0:
            res_layers = [ResidualBlock(in_features)]
            for _ in range(1, res_blocks):
                res_layers += [ResidualBlock(in_features)]
        self.res_layersx = nn.Sequential(*res_layers)
        self.res_layersy = nn.Sequential(*res_layers)

        # Upsampling
        self.ma_layer1 = Multi_Attn('up', in_features, 2, 8, 1)
        self.ma_layer2 = Multi_Attn('up', in_features, 2, 8, 1)
        self.ma_layer3 = Multi_Attn('up', in_features, 2, 8, 1)

        out_features = in_features // 2 #128
        self.layerx4 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(out_features),
                                     nn.ReLU(inplace=True))
        self.layery4 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(out_features),
                                     nn.ReLU(inplace=True))

        in_features = out_features #128
        out_features = in_features // 2 #64
        self.layerx5 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(out_features),
                                     nn.ReLU(inplace=True))
        self.layery5 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(out_features),
                                     nn.ReLU(inplace=True))

        self.layerx6 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(out_features, out_channels, 7),
                                     nn.Tanh())
        self.layery6 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(out_features, out_channels, 7),
                                     nn.Tanh())


    def forward(self, x, y):
        x = self.layerx1(x);            y = self.layery1(y)
        x = self.layerx2(x);            y = self.layery2(y)
        x = self.layerx3(x);            y = self.layery3(y)
        x, y = self.ma_layer1(x, y)
        x = self.res_layersx(x);        y = self.res_layersy(y)
        x, _ = self.ma_layer2(x, x);    y, _ = self.ma_layer3(y, y)
        x = self.layerx4(x);            y = self.layery4(y)
        x = self.layerx5(x);            y = self.layery5(y)
        x = self.layerx6(x);            y = self.layery6(y)
        return x, y


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            if normalize:
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                          nn.InstanceNorm2d(out_filters)]
            else:
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(in_channels, 64, normalize=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)