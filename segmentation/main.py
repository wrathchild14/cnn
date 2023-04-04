import torch

from resnet_18_fcn import ResNet18FCN
from segmentation.unet import UNet
from segmentation.utils import SegmentationDataset, train, test

if __name__ == '__main__':
    # net = ResNet18FCN(in_channels=3, out_channels=13)
    net = UNet(in_channels=3, out_channels=13)
    trainset = SegmentationDataset(train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=4)

    testset = SegmentationDataset(train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)

    train(net=net, train_data=[trainset, trainloader])
    test(net=net, testset=testset)
