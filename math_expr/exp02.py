# Train the network on data generated
# specifically : sin(x) + sin(2x)
#              : sin(2x) + sin(3x)
#              : sin(x) + sin(5x)

import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from executor import execute
from networks import FunctionApproximator, DoubleSin, SingleSinTwice


# Hyperparams

lr = 1e-3
num_epochs = 10
batch_size = 32
RANGE = [-10, 10]


# Train the network on data generated
# from sin(x) + sin(2x)

# plot, not used programmatically,
# but useful for visual inspection.

x = np.arange( RANGE[0], RANGE[1], 0.01 )
execute("sine_on_sine", x=x)


# make the training set

torch.manual_seed(42)


def sine(x, a=1, b=1):
    return torch.sin(a*x) + torch.sin(b*x)


params = [[1, 2], [2, 3], [1, 5]]

for i in range(len(params)):

    alpha = params[i][0]
    beta = params[i][1]

    train_data_length = 512*1024*2
    train_data = torch.zeros((train_data_length, 1))
    train_data[:, 0] = (RANGE[1] - RANGE[0]) * torch.rand(train_data_length) + RANGE[0]
    train_labels = sine(x=train_data, a=alpha, b=beta)
    print(max(train_labels), flush=True)
    print(min(train_labels), flush=True)
    train_loader = torch.utils.data.DataLoader(
        [(train_data[i], train_labels[i]) for i in range(train_data_length)],
        batch_size=batch_size, shuffle=True
    )


    # Training loop
    myNN = SingleSinTwice(1,2)
    exp_name = "exp02"
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(myNN.parameters(), lr=lr)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("STARTTIME " + current_time, flush=True)

    for epoch in range(num_epochs):
        for n, (x, y) in enumerate(train_loader):
            myNN.zero_grad()
            yy = myNN(x)
            loss = loss_fn(yy[0], y[0])
            loss.backward()
            optimizer.step()

            # Show loss
            # if epoch % 10 == 0 and n == batch_size - 1:
            if epoch % 1 == 0 and n == batch_size - 1:
                print(f"Epoch: {epoch} Loss: {loss}", flush=True)
                '''
                # x_samples[:,0] = torch.rand( 100 )
                x_samples = torch.zeros((200, 1))
                x_samples[:, 0] = (RANGE[1] - RANGE[0]) * torch.rand(200) + RANGE[0]
    
                y_samples = myNN(x_samples)
    
                y_plots = y_samples[:,0].detach()
                print(min(y_plots), max(y_plots), flush=True)
                x_plots = 2 * math.pi * x_samples[:, 0]
                plt.plot(x_plots, y_plots, ".")
                plt.savefig("{}_{}.png".format(exp_name, epoch))
                plt.close()
                '''

    # save model and final plot
    torch_x = torch.arange(RANGE[0], RANGE[1], 0.01)
    x_samples = torch.zeros((len(torch_x), 1))
    x_samples[:, 0] = torch_x

    y_samples = myNN(x_samples)
    y_plots = y_samples[:, 0].detach()
    x_plots = 2 * math.pi * x_samples[:, 0]

    y_fun = torch.sin(alpha * x_samples[:, 0]) + torch.sin(beta * x_samples[:, 0])
    plt.plot(x_plots, y_plots, ".", label="predicted value")
    plt.plot(x_plots, y_fun, '.', color="green", alpha=0.4, label="true value")
    plt.title(f'Function sin({alpha}*x) + sin({beta}*x)')
    plt.legend()
    plt.savefig("{}_final.png".format(exp_name))
    plt.close()

    model_scripted = torch.jit.script(myNN)
    model_scripted.save("{}.pt".format(exp_name))

    for i, m in enumerate(myNN.get_internal()):
        y1_samples = m(x_samples)
        y1_plots = y1_samples[:, 0].detach()
        y1_fun = torch.sin(x_samples[:, 0])
        plt.plot(x_plots, y1_plots, ".", label="predicted value")
        plt.plot(x_plots, y1_fun, ".", color="green", alpha=0.4, label="true value")
        plt.title('Function sin(x)')
        plt.legend()
        plt.savefig("{}_model{}.png".format(exp_name, i))
        plt.close()
        model_scripted = torch.jit.script(m)
        model_scripted.save("{}_model{}.pt".format(exp_name, i))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("ENDTIME " + current_time, flush=True)
