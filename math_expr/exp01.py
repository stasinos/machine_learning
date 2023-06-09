import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from executor import execute
from networks import FunctionApproximator, DoubleSin


# Hyperparams

lr = 1e-3
num_epochs = 15
batch_size = 64



##
## Train the network on data generated
## from sin(x) + sin(2x)
##

# plot, not used programmatically,
# but useful for visual inspection.

x = np.arange(-2*math.pi, 2*math.pi, 0.01)
execute( "sine_on_sine", x=x )


# make the training set

RANGE = [-10, 10]
torch.manual_seed(42)
def sine_in_01(x):
    xx = torch.sin(x) + torch.sin(2*x)
    return 0.5*( 1 + xx )
def nnrange_to_realrange(x):
    return 2*x-1

train_data_length = 64*1024*2
train_data = torch.zeros( (train_data_length,1) )
train_data[:, 0] = (RANGE[0] - RANGE[1]) * torch.rand(train_data_length) + RANGE[1]
train_labels = sine_in_01( train_data )
print(max(train_labels),flush=True)
print(min(train_labels),flush=True)
train_loader = torch.utils.data.DataLoader(
    [(train_data[i], train_labels[i]) for i in range(train_data_length)],
    batch_size=batch_size, shuffle=True
)


# Training loop


myNN = DoubleSin()
exp_name = "exp01"
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(myNN.parameters(), lr=lr)

now = datetime.now()
current_time = now.strftime( "%H:%M:%S" )
print( "STARTTIME " + current_time, flush=True )

for epoch in range(num_epochs):
    for n, (x,y) in enumerate(train_loader):
        myNN.zero_grad()
        yy = myNN(x)
        loss = loss_fn( yy[0], y[0] )
        loss.backward()
        optimizer.step()

        # Show loss
        #if epoch % 10 == 0 and n == batch_size - 1:
        if epoch % 1 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss: {loss}",flush=True)

            # x_samples[:,0] = torch.rand( 100 )
            x_samples = torch.zeros((200, 1))
            x_samples[:, 0] = (RANGE[0] - RANGE[1]) * torch.rand(200) + RANGE[1]

            y_samples = myNN(x_samples)

            y_plots = nnrange_to_realrange( y_samples[:,0].detach() )
            print(min(y_plots), max(y_plots),flush=True)
            x_plots = 2 * math.pi * x_samples[:,0]
            plt.plot( x_plots, y_plots, "." )
            plt.savefig( "{}_{}.png".format(exp_name,epoch) )
            plt.close()

#save model and final plot

x_samples = torch.zeros((200, 1))
x_samples[:, 0] = (RANGE[0] - RANGE[1]) * torch.rand(200) + RANGE[1]

y_samples = myNN(x_samples)
y_plots = nnrange_to_realrange( y_samples[:,0].detach() )
x_plots = 2 * math.pi * x_samples[:,0]
plt.plot( x_plots, y_plots, "." )
plt.savefig( "{}_final.png".format(exp_name) )
plt.close()

y1_samples = myNN.model1(x_samples)
y1_plots = nnrange_to_realrange( y1_samples[:,0].detach() )
plt.plot( x_plots, y1_plots, "." )
plt.savefig( "{}_model1.png".format(exp_name) )
plt.close()

y2_samples = myNN.model2(x_samples)
y2_plots = nnrange_to_realrange( y2_samples[:,0].detach() )
plt.plot( x_plots, y2_plots, "." )
plt.savefig( "{}_model2.png".format(exp_name) )
plt.close()

model_scripted = torch.jit.script( myNN.model1 )
model_scripted.save( "{}_model1.pt".format(exp_name) )
model_scripted = torch.jit.script( myNN.model2 )
model_scripted.save( "{}_model2.pt".format(exp_name) )
model_scripted = torch.jit.script( myNN )
model_scripted.save( "{}.pt".format(exp_name) )

now = datetime.now()
current_time = now.strftime( "%H:%M:%S" )
print( "ENDTIME " + current_time, flush=True )
