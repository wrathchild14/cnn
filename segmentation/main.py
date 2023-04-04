import torch

from resnet_18_fcn import ResNet18FCN
from segmentation.unet import UNet
from segmentation.utils import SegmentationDataset, train, test

if __name__ == '__main__':
    net = ResNet18FCN(in_channels=3, out_channels=13)
    # net = UNet(in_channels=3, out_channels=13)

    train_set = SegmentationDataset(train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    test_set = SegmentationDataset(train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    train(net=net, train_data=[train_set, train_loader])
    test(net=net, testset=test_set)
