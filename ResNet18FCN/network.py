from torch import nn

# Take the basic block from the original ResNet18 network
from ResNet18.network import BasicBlock


class ResNet18FCN(nn.Module):
    def __init__(self, block=BasicBlock, channels=3, layers=None, num_classes=13):
        super(ResNet18FCN, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        self.expansion = 1
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=self.in_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        # Replace the avg_pool and fc layer
        self.conv5 = nn.Conv2d(512 * self.expansion, num_classes, kernel_size=1)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def make_layer(self, block, out_channels, blocks, stride=1):
        down_sample = None
        if stride != 1:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, self.expansion, down_sample)]
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        # * creates new list
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.up_sample(x)
        return x
