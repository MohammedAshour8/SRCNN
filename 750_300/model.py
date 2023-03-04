from torch import nn

class SRCNN(nn.Module):
    def __init__(self, in_channels):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=9, padding=4)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out