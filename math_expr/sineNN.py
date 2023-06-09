import torch
from torch import nn

import math
import matplotlib.pyplot as plt


torch.manual_seed(42)

# It is good to keep all numbers in 0..1,
# even mandatory for some loss functions.
# So:
# Calculate the sine, squeezed into 0..1
# Inputs are 0..2pi squeezed to 0..1
def sine_in_01(x):
    return 1 + torch.sin(2*math.pi*x)/2
# Unsqueezes back to -1..1
def nnrange_to_realrange(x):
    return (x-1)*2

# Hyperparams

lr = 1e-3
num_epochs = 10
batch_size = 64

# Dataset

train_data_length = 64*1024
# Inputs and outputs are single values in 0..1,
# but must be in a 2D array anyway, to satisfy torch.
# So just fix the second dim to [,0]
train_data = torch.zeros( (train_data_length,1) )
train_data[:,0] = torch.rand( train_data_length )
# Labels, use sine_in_01() to squeeze to 0..1
train_labels = sine_in_01( train_data )
train_loader = torch.utils.data.DataLoader(
    [(train_data[i], train_labels[i]) for i in range(train_data_length)],
    batch_size=batch_size, shuffle=True
)

# Model structure
# Also tried ReLU, tanh seems to work better
# Note: For ReLU, add a Sigmoid at the very end
# to keep outputs in 0..1

class SineNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        output = self.model(x)
        return output

sineNN = SineNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(sineNN.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (x,y) in enumerate(train_loader):
        sineNN.zero_grad()
        yy = sineNN(x)
        loss = loss_fn( yy[0], y[0] )
        loss.backward()
        optimizer.step()

        # Show loss
        #if epoch % 10 == 0 and n == batch_size - 1:
        if epoch % 1 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss: {loss}")
            x_samples = torch.zeros( (100,1) )
            x_samples[:,0] = torch.rand( 100 )
            y_samples = sineNN(x_samples)
            y_plots = nnrange_to_realrange( y_samples[:,0].detach() )
            x_plots = 2 * math.pi * x_samples[:,0]
            plt.plot( x_plots, y_plots, "." )
            plt.savefig( "sine_{}.png".format(epoch) )
            plt.close()
