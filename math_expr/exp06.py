# Train the network on data generated
# specifically : exp(sin(x))
#              : exp(ln(x))
#              : exp(sinc(x))

import math
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from networks import SinFunComposition as FunComposition

# Hyperparams

lr = 1e-3
num_epochs = 15
batch_size = 64

# make the training set

RANGE = [-10, 10]
torch.manual_seed(42)

train_data_length = 512 * 1024


class Exp:
    def __init__(self, data_len, function, fun_name, a=1):
        self.a = a
        self.data_len = data_len
        self.function = function

        self.x = torch.zeros((self.data_len, 1))
        if fun_name == "ln":  # logarithm is defined in (0, inf)
            self.x[:, 0] = (RANGE[1] - 0 + 1e-4) * torch.rand(self.data_len) + 0 + 1e-4
        else:
            self.x[:, 0] = (RANGE[1] - RANGE[0]) * torch.rand(self.data_len) + RANGE[0]

        self.y = self.expo()

    def expo(self):
        return torch.exp(self.function(self.a * self.x))


params = [1]
composition = [torch.sin, torch.log, torch.sinc]
composition_str = ["sin", "ln", "sinc"]

for i in range(len(composition)):

    alpha = params[0]

    print(f'\nStart training for exp({composition_str[i]}({alpha}*x))...\n')

    exp_train_data = Exp(a=alpha, function=composition[i], fun_name=composition_str[i], data_len=train_data_length)

    print(max(exp_train_data.y), flush=True)
    print(min(exp_train_data.y), flush=True)
    train_loader = torch.utils.data.DataLoader(
        [(exp_train_data.x[i], exp_train_data.y[i]) for i in range(train_data_length)],
        batch_size=batch_size, shuffle=True
    )
    # Training loop
    myNN = FunComposition(a=alpha)
    exp_name = "exp0" + str(i + 1) + "f"
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
            if epoch % 1 == 0 and n == batch_size - 1:
                print(f"Epoch: {epoch} Loss: {loss}", flush=True)

    # save model and final plot
    exp_test_data = Exp(a=alpha, function=composition[i], fun_name=composition_str[i], data_len=200)
    x_samples = exp_test_data.x
    y_samples = myNN(x_samples)
    y_plots = y_samples[:, 0].detach()
    x_plots = x_samples[:, 0]
    y_fun = torch.exp(composition[i](alpha * x_samples[:, 0]))
    plt.plot(x_plots, y_plots, ".", label="predicted value")
    plt.plot(x_plots, y_fun, '.', color="green", alpha=0.4, label="true value")
    plt.title(f'Function exp({composition_str[i]}({alpha}*x))')
    plt.legend()
    plt.savefig("{}_final.png".format(exp_name))
    plt.close()

    model_scripted = torch.jit.script(myNN)
    model_scripted.save("{}.pt".format(exp_name))

    first = True
    for j, m in enumerate(myNN.get_internal()):
        y1_samples = m(x_samples)
        y1_plots = y1_samples[:, 0].detach()
        if first:
            first = False
            y1_fun = torch.exp(x_samples[:, 0])
            plt.title('Function exp(x)')
        else:
            plt.title(f'Function {composition_str[i]}(x)')
            y1_fun = composition[i](x_samples[:, 0])

        plt.plot(x_plots, y1_plots, ".", label="predicted value")
        plt.plot(x_plots, y1_fun, ".", color="green", alpha=0.4, label="true value")
        plt.legend()
        plt.savefig("{}_model{}.png".format(exp_name, j))

        plt.close()
        model_scripted = torch.jit.script(m)
        model_scripted.save("{}_model{}.pt".format(exp_name, j))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("ENDTIME " + current_time, flush=True)

    print(f'\nEnd training for exp({composition_str[i]}({alpha}*x))...\n')
