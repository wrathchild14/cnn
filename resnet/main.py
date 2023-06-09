from resnet_18 import ResNet18
from utils import train_bird_model, load_data

if __name__ == '__main__':
    net = ResNet18(num_classes=400)
    batch_size, epochs, learning_rate = 32, 10, 0.0001
    data = load_data()
    net = train_bird_model(net, loaded_data=data, batch_size=batch_size, epochs=epochs, lr=learning_rate)
