import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim, from_numpy
from tqdm import tqdm
from torch.utils.data import Dataset
import glob


class SegmentationDataset(Dataset):

    def __init__(self, train=True):
        dataset_path = "./data/seg_data/" + ("train" if train else "test")
        self.images = sorted(glob.glob(dataset_path + "/*/*/CameraRGB/*.png"))
        self.masks = sorted(glob.glob(dataset_path + "/*/*/CameraSeg/*.png"))
        self.resize_shape = (320, 416)

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)[:, :, 2]

        channels = 3
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.array(image).reshape((image.shape[0], image.shape[1], channels)).astype(np.float32) / 255.0
        mask = np.array(mask).reshape((image.shape[0], image.shape[1], 1))

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.transform_image(self.images[idx], self.masks[idx])
        sample = {'image': image, "mask": mask, 'idx': idx}

        return sample


def train(net, train_data, epochs=2, learning_rate=0.0001):
    trainset, trainloader = train_data
    net.cuda()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)

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
