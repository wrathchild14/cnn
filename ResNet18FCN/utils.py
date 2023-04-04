import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim, from_numpy
from tqdm import tqdm


def train(net, train_data, epochs=2, learning_rate=0.0001):
    trainset, trainloader = train_data
    net.cuda()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):  # loop over the dataset multiple times
        with tqdm(total=len(trainset), desc=str(epoch) + "/" + str(epochs), miniters=int(50), unit='img') as prog_bar:
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data["image"]
                labels = data["mask"]
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels[:, 0, :, :].long())
                loss.backward()
                optimizer.step()

                prog_bar.set_postfix(**{'loss': np.round(loss.data.cpu().detach().numpy(), 5)})
                prog_bar.update(4)


def test(net, testset):
    label_colors = np.array([
        [0, 0, 0],  # unlabeled
        [70, 70, 70],  # building
        [190, 153, 153],  # fence
        [250, 170, 160],  # other
        [220, 20, 60],  # pedestrian
        [153, 153, 153],  # pole
        [157, 234, 50],  # road line
        [128, 64, 128],  # road
        [244, 35, 232],  # sidewalk
        [107, 142, 35],  # vegetation
        [0, 0, 142],  # car
        [102, 102, 156],  # wall
        [220, 220, 0],  # traffic sign
    ])

    net.eval()

    fig = plt.figure(figsize=(24, 24))
    columns = 3
    rows = 5
    for i in range(rows):
        batch = testset[np.random.randint(len(testset))]
        img = batch['image']
        labels = batch['mask']
        img_tensor = from_numpy(img).cuda()
        img_tensor = img_tensor.unsqueeze(0)
        outputs = net(img_tensor)
        out_image = outputs[0, :, :, :].detach().cpu().numpy()
        seg_mask = np.argmax(out_image, axis=0)
        seg_mask_color = label_colors[seg_mask]
        gt_mask_color = label_colors[labels[0, :, :]]

        img = img.transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = labels[0, :, :]
        ax = fig.add_subplot(rows, columns, i * columns + 1)
        ax.set_title("Input", fontstyle='italic')
        plt.imshow(img)
        ax = fig.add_subplot(rows, columns, i * columns + 2)
        ax.set_title("Output", fontstyle='italic')
        plt.imshow(seg_mask_color)
        ax = fig.add_subplot(rows, columns, i * columns + 3)
        ax.set_title("GT", fontstyle='italic')
        plt.imshow(gt_mask_color)

    plt.show()
