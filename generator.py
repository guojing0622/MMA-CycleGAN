import torch.nn as nn
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
    def __init__(self, in_channels=5, out_channels=3, res_blocks=3):
        super(Generator, self).__init__()
        # Initial convolution block
        self.layer1 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(in_channels, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        # Downsampling
        in_features = 64
        self.layer2 = nn.Sequential(nn.Conv2d(in_features, in_features * 2,
                                     4, stride=2, padding=1),
                                     nn.InstanceNorm2d(in_features * 2),
                                     nn.ReLU(inplace=True))

        in_features = 2 * in_features #128
        self.layer3 = nn.Sequential(nn.Conv2d(in_features, in_features*2, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(in_features*2),
                                     nn.ReLU(inplace=True))

        in_features = 2 * in_features #256
        # Residual blocks
        if res_blocks != 0:
            res_layers = [ResidualBlock(in_features)]
            for _ in range(1, res_blocks):
                res_layers += [ResidualBlock(in_features)]
        self.res_layers = nn.Sequential(*res_layers)

        # Upsampling
        self.ma_layer1 = Multi_Attn('up', in_features, 2, 8, 1)
        self.ma_layer2 = Multi_Attn('up', in_features, 2, 8, 1)
        self.ma_layer3 = Multi_Attn('up', in_features, 2, 8, 1)

        out_features = in_features // 2 #128
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(out_features),
                                     nn.ReLU(inplace=True))

        in_features = out_features #128
        out_features = in_features // 2 #64
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(out_features),
                                     nn.ReLU(inplace=True))

        self.layer6 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(out_features, out_channels, 7),
                                     nn.Tanh())

    def forward(self, x, y):
        x = self.layer1(x);            y = self.layer1(y)
        x = self.layer2(x);            y = self.layer2(y)
        x = self.layer3(x);            y = self.layer3(y)
        x, y = self.ma_layer1(x, y)
        x = self.res_layers(x)
        x, _ = self.ma_layer2(x, x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
