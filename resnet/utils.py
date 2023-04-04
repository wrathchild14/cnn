import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm

BATCH_SIZE = 32


def test_bird_model(net, curr_data_loader, val_test="val"):
    criterion = nn.CrossEntropyLoss()

    num_images = len(curr_data_loader.dataset)
    gt_array = np.zeros(num_images)
    pred_array = np.zeros(num_images)

    correct = 0
    total = 0
    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(curr_data_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            gt_array[i * labels.size(0):(i + 1) * labels.size(0)] = labels.detach().cpu().numpy()
            pred_array[i * labels.size(0):(i + 1) * labels.size(0)] = predicted.detach().cpu().numpy()

    print('Accuracy of the network on %s images: %d %%' % (val_test,
                                                           100 * correct / total))
    print("Test loss: " + str(running_loss / (total / 4)))
    return gt_array, pred_array


def train_bird_model(net, loaded_data, batch_size, epochs, lr):
    BATCH_SIZE = batch_size  # temp
    train_set, train_loader, test_set, test_loader, val_set, val_loader = loaded_data

    net.cuda()
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr)

    # unneeded for now
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        with tqdm(total=len(train_set), desc='Epoch: ' + str(epoch) + "/" + str(epochs), unit='img') as prog_bar:
            for i, data in enumerate(train_loader, 0):
                # Get the inputs; Data is a tuple of (images, labels)
                inputs, labels = data
                # Transfer the images and labels to the GPU.
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Clear the saved gradients of the previous iteration
                optimizer.zero_grad()

                outputs = net(inputs)
                # Calculate the loss value
                loss = criterion(outputs, labels)
                # Calculate the gradients using backpropagation
                loss.backward()
                # Update the weights of the network using the chosen optimizer
                optimizer.step()

                prog_bar.set_postfix(**{'loss': loss.data.cpu().detach().numpy()})
                prog_bar.update(BATCH_SIZE)

        # scheduler.step()
        if epoch % 5 == 0:
            test_bird_model(net, val_loader)
    test_bird_model(net, test_loader)

    return net


def load_data():
    # Transforms can also be used for image augmentation - https://pytorch.org/vision/stable/transforms.html
    transform = transforms.Compose([transforms.ToTensor()])

    # transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.ImageFolder(root='./bird_data/train/', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4)

    test_set = torchvision.datasets.ImageFolder(root='./bird_data/test/', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)

    val_set = torchvision.datasets.ImageFolder(root='./bird_data/valid/', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)

    return train_set, train_loader, test_set, test_loader, val_set, val_loader


def display_data():
    fig = plt.figure(figsize=(24, 24))
    columns = 5
    rows = 4
    for i in range(columns * rows):
        input, label = train_set[np.random.randint(len(train_set))]
        img = input.detach().numpy().transpose((1, 2, 0))
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.set_title(train_set.classes[label], fontstyle='italic')
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    train_set, train_loader, test_set, test_loader, val_set, val_loader = load_data()
    display_data()
