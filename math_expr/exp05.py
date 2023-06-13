# Train the network on data generated
# specifically : exp(x) + exp(2x)
#              : exp(2x) + exp(3x)
#              : exp(x) + exp(5x)

import torch
import matplotlib.pyplot as plt
from datetime import datetime

from networks import SingleSinTwice as SingleFunTwice

# Hyperparams

lr = 1e-3
num_epochs = 15
batch_size = 64

# make the training set

RANGE = [-10, 10]
torch.manual_seed(42)

train_data_length = 512 * 1024


class Exp:
    def __init__(self, a, b, data_len):
        self.a = a
        self.b = b
        self.data_len = data_len

        self.x = torch.zeros((self.data_len, 1))
        self.x[:, 0] = (RANGE[0] - RANGE[1]) * torch.rand(self.data_len) + RANGE[0]

        self.y = self.expo()

    def expo(self):
        return torch.exp(self.a * self.x) + torch.exp(self.b * self.x)


# trying : exp(x) + exp(2x)
#        : exp(2x) + exp(3x)
#        : exp(x) + exp(5x)
params = [[1, 2], [2, 3], [1, 5]]

for i in range(len(params)):

    alpha = params[i][0]
    beta = params[i][1]

    print(f'\nStart training for exp({alpha}*x) + exp({beta}*x)...\n')

    exp_train_data = Exp(a=alpha, b=beta, data_len=train_data_length)

    print(max(exp_train_data.y), flush=True)
    print(min(exp_train_data.y), flush=True)
    train_loader = torch.utils.data.DataLoader(
        [(exp_train_data.x[i], exp_train_data.y[i]) for i in range(train_data_length)],
        batch_size=batch_size, shuffle=True
    )
    # Training loop
    myNN = SingleFunTwice(a=alpha, b=beta)
    exp_name = "exp0" + str(i + 1) + "e"
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

    # save model and final plot
    exp_test_data = Exp(a=alpha, b=beta, data_len=200)
    x_samples = exp_test_data.x
    y_samples = myNN(x_samples)
    y_plots = y_samples[:, 0].detach()
    x_plots = x_samples[:, 0]
    y_fun = torch.exp(alpha * x_samples[:, 0]) + torch.exp(beta * x_samples[:, 0])
    plt.plot(x_plots, y_plots, ".", label="predicted value")
    plt.plot(x_plots, y_fun, '.', color="green", alpha=0.4, label="true value")
    plt.title(f'Function exp({alpha}*x) + exp({beta}*x)')
    plt.legend()
    plt.savefig("{}_final.png".format(exp_name))
    plt.close()

    model_scripted = torch.jit.script(myNN)
    model_scripted.save("{}.pt".format(exp_name))

    for i, m in enumerate(myNN.get_internal()):
        y1_samples = m(x_samples)
        y1_plots = y1_samples[:, 0].detach()
        y1_fun = torch.exp(x_samples[:, 0])
        plt.plot(x_plots, y1_plots, ".", label="predicted value")
        plt.plot(x_plots, y1_fun, ".", color="green", alpha=0.4, label="true value")
        plt.title('Function exp(x)')
        plt.legend()
        plt.savefig("{}_model{}.png".format(exp_name, i))
        plt.close()
        model_scripted = torch.jit.script(m)
        model_scripted.save("{}_model{}.pt".format(exp_name, i))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("ENDTIME " + current_time, flush=True)

    print(f'\nEnd training for exp({alpha}*x) + exp({beta}*x)...')
