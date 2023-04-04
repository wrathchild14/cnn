import torch

from colorization.utils import ColorizationDataset, train, test
from segmentation.unet import UNet

if __name__ == '__main__':
    trainset = ColorizationDataset(train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=4)

    testset = ColorizationDataset(train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)

    net = UNet(in_channels=1, out_channels=3, colorization=True)
    train(net, train_data=[trainset, trainloader], epochs=5)
    test(net, testset)
