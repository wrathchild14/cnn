from networks import ExampleNet
from utils import train_bird_model, load_data

if __name__ == '__main__':
    net = ExampleNet()
    epochs, learning_rate = 3, 0.001
    data = load_data()
    net = train_bird_model(net, loaded_data=data, epochs=epochs, lr=0.001)
