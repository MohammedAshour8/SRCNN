import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, in_channels):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.residual1 = ResidualBlock(64, 64)
        self.residual2 = ResidualBlock(64, 64)
        self.residual3 = ResidualBlock(64, 64)
        self.residual4 = ResidualBlock(64, 64)
        self.residual5 = ResidualBlock(64, 64)
        
        self.conv2 = nn.Conv2d(64, 75, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(75)
        self.relu2 = nn.ReLU(inplace=True)
        self.residual6 = ResidualBlock(75, 75)
        self.residual7 = ResidualBlock(75, 75)
        self.residual8 = ResidualBlock(75, 75)
        self.residual9 = ResidualBlock(75, 75)
        self.residual10 = ResidualBlock(75, 75)
        
        self.conv3 = nn.Conv2d(75, 75, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(75)
        self.relu3 = nn.ReLU(inplace=True)
        self.residual11 = ResidualBlock(75, 75)
        self.residual12 = ResidualBlock(75, 75)
        self.residual13 = ResidualBlock(75, 75)
        self.residual14 = ResidualBlock(75, 75)
        self.residual15 = ResidualBlock(75, 75)
        
        self.conv4 = nn.Conv2d(75, 75, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(75)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(75, in_channels, kernel_size=9, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu1(out)
        residual = out
        
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.residual3(out)
        out = self.residual4(out)
        out = self.residual5(out)
        out += residual
        
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.relu2(out)
        residual = out
        
        out = self.residual6(out)
        out = self.residual7(out)
        out = self.residual8(out)
        out = self.residual9(out)
        out = self.residual10(out)
        out += residual
        
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = self.relu3(out)
        residual = out
        
        out = self.residual11(out)
        out = self.residual12(out)
        out = self.residual13(out)
        out = self.residual14(out)
        out = self.residual15(out)
        out += residual
        
        out = self.conv4(out)
        out = self.conv4_bn(out)
        out = self.relu4(out)
        out = self.conv5(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out += x
        return out
