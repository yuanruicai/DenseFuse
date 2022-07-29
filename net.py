import torch
import torch.nn as nn
import torch.nn.functional as F


reflection_padding = 1


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv3x3 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=1)

    def forward(self, *prev_features):
        # 数据格式 N,C,H,W
        input = torch.cat(prev_features, dim=1)
        input = self.reflection_pad(input)
        out = F.relu(self.conv3x3(input))
        return out


class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.DC1 = DenseLayer(16, 16)
        self.DC2 = DenseLayer(32, 16)
        self.DC3 = DenseLayer(48, 16)

    def forward(self, input):
        x1 = self.DC1(input)
        x2 = self.DC2(input, x1)
        x3 = self.DC3(input, x1, x2)
        output = torch.cat((input, x1, x2, x3), dim=1)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.C1 = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
        )
        self.DenseBlock = DenseBlock()

    def forward(self, input):
        x = self.C1(input)
        x = self.DenseBlock(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.C2 = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
        )
        self.C3 = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
        )
        self.C4 = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(32, 16, 3),
            nn.ReLU(),
        )
        self.C5 = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(16, 1, 3),
            nn.ReLU(),
        )

    def forward(self, input):
        x = self.C2(input)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x


class DenseFuse(nn.Module):
    def __init__(self):
        super(DenseFuse, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input):
        features = self.encoder(input)
        output = self.decoder(features)
        return output
