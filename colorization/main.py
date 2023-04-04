import torch

from colorization.utils import ColorizationDataset, train, test
from segmentation.unet import UNet

if __name__ == '__main__':
    net = UNet(in_channels=1, out_channels=3, skip_connection=True)

    train_set = ColorizationDataset(train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    test_set = ColorizationDataset(train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    train(net, train_data=[train_set, train_loader])
    test(net, test_set)
