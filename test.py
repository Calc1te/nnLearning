import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(25*25, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def convolve_processing(image):
    conv_kernel = torch.tensor([[ 0,  1,  1,  0],
                        [ 1,  2,  2,  1],
                        [ 1,  2,  2,  1],
                        [ 0,  1,  1,  0]],dtype=torch.float)
    conv_kernel = conv_kernel.view(1,1,4,4)
    image_tensor = torch.nn.functional.conv2d(image.unsqueeze(0), conv_kernel,padding=0)
    image_tensor = torch.clamp(image_tensor, min=0, max=255).squeeze(0)
    return image_tensor


def get_data_loader(is_train):
    def process_image(image):
        image = convolve_processing(image)
        
        return torch.tensor(image,dtype=torch.float)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(process_image)
    ])
    
    data_set = MNIST("./", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=64, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x,y = x.to(device), y.to(device)
            outputs = net.forward(x.view(-1, 25*25))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)
    
    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    for epoch in range(4):
        for (x, y) in train_data:
            x,y = x.to(device), y.to(device)
            net.zero_grad()
            output = net.forward(x.view(-1, 25*25))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for (n, (x, _)) in enumerate(test_data): #x:图像数据 _:忽略标签
        if n > 3:
            break
        x = x.to(device)
        predict = torch.argmax(net.forward(x[0].view(-1, 25*25)))
        plt.figure(n)
        plt.imshow(x[0].cpu().view(25, 25))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
