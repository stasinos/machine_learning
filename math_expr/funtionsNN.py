import torch
from torch import nn
import math
import matplotlib.pyplot as plt

from snake import Snake

torch.manual_seed(42)

# It is good to keep all numbers in 0..1,
# even mandatory for some loss functions.

# Range can be  modified, but corresponding changed must be made for each data class
RANGE = [0.0, 1.0]

# Learning Hyperparams
lr = 1e-3
num_epochs = 10
batch_size = 64

# Model structure
# Also tried ReLU, tanh seems to work better
# Note: For ReLU, add a Sigmoid at the very end
# to keep outputs in 0..1
class Distinguisher(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            Snake(64),
            # nn.Tanh(),
            nn.Linear(64, 64),
            Snake(64),
            # nn.Tanh(),
            nn.Linear(64, 64),
            Snake(64),
            # nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output


# Data for each Class function

# Dataset size
train_data_length = 64 * 1024

# Inputs and outputs are single values in 0..1,
# but must be in a 2D array anyway, to satisfy torch.
# So just fix the second dim to [,0]

class SineData:
    def __init__(self, dataset_size):
        self.name = "sin"
        self.size = dataset_size
        self.x = torch.zeros((dataset_size, 1))
        self.x[:, 0] = (RANGE[0] - RANGE[1]) * torch.rand(dataset_size) + RANGE[1]

        self.y = self.sine_in_01()

    def sine_in_01(self):
        # Calculate the sine, squeezed into 0..1
        # Inputs are 0..x squeezed to 0..1
        return 0.5 * (1 + torch.sin(self.x))

    def nnrange_to_realrange(self, z):
        # Unsqueezes back to -1..1
        return 2 * z - 1


class ExpData:
    def __init__(self, dataset_size):
        self.name = "exp"
        self.size = dataset_size
        self.x = torch.zeros((dataset_size, 1))
        self.x[:, 0] = (RANGE[0] - RANGE[1]) * torch.rand(dataset_size) + RANGE[1]

        self.y = self.exp_in_01()

    def exp_in_01(self):
        # Calculate the e^x, squeezed into 0..1
        return (torch.exp(self.x) - 1) / (math.e - 1)

    def nnrange_to_realrange(self, z):
        # Unsqueezes back to real range
        return z * (math.e - 1) + 1


class LnData:
    def __init__(self, dataset_size):
        self.name = "ln"
        self.size = dataset_size
        self.x = torch.zeros((dataset_size, 1))
        # ln(0) is not defined, so minimum value is ln(1e-4)
        self.x[:, 0] = (RANGE[0] + 1e-4 - RANGE[1]) * torch.rand(dataset_size) + RANGE[1]

        self.y = self.ln_in_01()

    def ln_in_01(self):
        # Calculate the ln(x), squeezed into 0..1
        self.max_v = torch.log(max(self.x[:, 0]))
        self.min_v = torch.log(min(self.x[:, 0]))
        return (torch.log(self.x) - self.min_v) / (self.max_v - self.min_v)

    def nnrange_to_realrange(self, z):
        # Unsqueezes back to real range
        self.max_v = torch.log(torch.tensor(RANGE[1], dtype=torch.float))
        self.min_v = torch.log(torch.tensor(RANGE[0] + 1e-4, dtype=torch.float))

        return z * (self.max_v - self.min_v) + self.min_v


class SincData:
    """ at this point this data is from the normalized sinc : sinc(pi*x)/(pi*x). It can be modified  for simple sinc"""
    def __init__(self, dataset_size):
        self.name = "sinc"
        self.size = dataset_size
        self.x = torch.zeros((dataset_size, 1))
        self.x[:, 0] = (RANGE[0] - RANGE[1]) * torch.rand(dataset_size) + RANGE[1]

        self.y = self.sinc_in_01()

    def sinc_in_01(self):
        # Calculate the sinc(pi*x), squeezed into 0..1
        self.max_v = torch.sinc(min(self.x[:, 0]))
        self.min_v = torch.sinc(max(self.x[:, 0]))

        return (torch.sinc(self.x) - self.min_v) / (self.max_v - self.min_v) * (1 - self.min_v) + self.min_v

    def nnrange_to_realrange(self, z):
        # Unsqueezes back to real range
        self.max_v = torch.sinc(torch.tensor(RANGE[0], dtype=torch.float) / math.pi)
        self.min_v = torch.sinc(torch.tensor(RANGE[1], dtype=torch.float) / math.pi)

        return (z - self.min_v) * (self.max_v - self.min_v) / (1 - self.min_v) + self.min_v


# Start training process ...
data = [ExpData(train_data_length), SineData(train_data_length), LnData(train_data_length), SincData(train_data_length)]
data_test = [ExpData(100), SineData(100), LnData(100), SincData(100)]

for i in range(len(data)):

    print(f'Starting training and testing for {data[i].name}\n')

    train_data = data[i].x
    train_labels = data[i].y

    train_loader = torch.utils.data.DataLoader(
        [(train_data[i], train_labels[i]) for i in range(train_data_length)],
        batch_size=batch_size, shuffle=True
    )

    model = Distinguisher()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for n, (x, y) in enumerate(train_loader):
            model.zero_grad()
            yy = model(x)
            loss = loss_fn(yy[0], y[0])
            loss.backward()
            optimizer.step()

            # Show loss
            if epoch % 1 == 0 and n == batch_size - 1:
                print(f"Epoch: {epoch} Loss: {loss} for {data[i].name}")

                test = data_test[i]
                x_samples = test.x
                y_samples = model(x_samples)

                y_plots = test.nnrange_to_realrange(y_samples[:, 0].detach())
                x_plots = x_samples[:, 0]

                plt.plot(x_plots, y_plots, ".")
                plt.savefig(f'{test.name}{epoch}.png')
                plt.close()

    print(f'\nEnd training and testing for {data[i].name}\n')
