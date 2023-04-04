from torch import nn, from_numpy, optim
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


class ColorizationDataset(Dataset):

    def __init__(self, train=True):
        self.all_images = sorted(glob.glob("./data/*"))
        self.images = self.all_images[:-500]
        self.test_images = self.all_images[-500:]
        if not train:
            self.images = self.test_images

        self.resize_shape = (256, 256)

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        channels = 3
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.array(image).reshape((image.shape[0], image.shape[1], channels)).astype(np.float32) / 255.0
        gray_image = np.array(gray_image).reshape((image.shape[0], image.shape[1], 1)).astype(np.float32) / 255.0

        image = np.transpose(image, (2, 0, 1))
        gray_image = np.transpose(gray_image, (2, 0, 1))
        return image, gray_image

    def __getitem__(self, idx):
        image, gray_image = self.transform_image(self.images[idx])
        sample = {'image': image, "gray_image": gray_image, 'idx': idx}

        return sample


def show_pictures():
    trainset = ColorizationDataset(train=True)
    fig = plt.figure(figsize=(24, 24))
    columns = 2
    rows = 5
    for i in range(rows):
        batch = trainset[np.random.randint(len(trainset))]
        img = batch['image'].transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = batch['gray_image'][0, :, :]
        fig.add_subplot(rows, columns, i * columns + 1)
        plt.imshow(img)
        fig.add_subplot(rows, columns, i * columns + 2)
        plt.imshow(gray, cmap='gray')
    plt.show()


def train(net, train_data, epochs=5, learning_rate=0.0001):
    trainset, trainloader = train_data
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        with tqdm(total=len(trainset), desc='Epoch: ' + str(epoch) + "/" + str(epochs), unit='img') as prog_bar:
            for i, data in enumerate(trainloader, 0):
                # get the inputs;
                gray_images = data["gray_image"]
                color_images = data["image"]

                # Get the inputs to the GPU
                gray_images = gray_images.cuda()
                color_images = color_images.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(gray_images)
                loss = nn.functional.l1_loss(outputs, color_images)
                loss.backward()
                optimizer.step()

                prog_bar.set_postfix(**{'loss': loss.data.cpu().detach().numpy()})
                prog_bar.update(4)


def test(net, testset):
    net.eval()
    fig = plt.figure(figsize=(24, 24))
    columns = 3
    rows = 5
    for i in range(rows):
        batch = testset[i + 80]
        img = batch['image']
        gray = batch['gray_image']
        img_tensor = from_numpy(gray).cuda()
        img_tensor = img_tensor.unsqueeze(0)
        outputs = net(img_tensor)
        out_image = outputs[0, :, :, :].detach().cpu().numpy().transpose((1, 2, 0))
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

        img = img.transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = gray[0, :, :]
        fig.add_subplot(rows, columns, i * columns + 1)
        plt.imshow(img)
        fig.add_subplot(rows, columns, i * columns + 2)
        plt.imshow(out_image)
        fig.add_subplot(rows, columns, i * columns + 3)
        plt.imshow(gray, cmap='gray')

    plt.show()
