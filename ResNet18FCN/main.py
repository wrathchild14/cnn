import torch

from ResNet18FCN.segmentation_dataset import SegmentationDataset
from ResNet18FCN.utils import train, test
from network import ResNet18FCN

if __name__ == '__main__':
    net = ResNet18FCN()
    # net = UNet(in_channels=3, out_channels=13)
    trainset = SegmentationDataset(train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=4)

    testset = SegmentationDataset(train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)

    train(net=net, train_data=[trainset, trainloader])
    test(net=net, testset=testset)
