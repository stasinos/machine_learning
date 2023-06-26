# Train the network on data generated
# specifically : exp(x) + ln(x)
#              : exp(x) + sin(x)
#              : exp(x) + sinc(x)
#              : ln(x) + sin(x)
#              : ln(x) + sinc(x)
#              : sin(x) + sinc(x)

import torch
import matplotlib.pyplot as plt
from datetime import datetime

from networks import DoubleSin as Double

# Hyperparams

lr = 1e-3
num_epochs = 15
batch_size = 64

# make the training set

RANGE = [-10, 10]
torch.manual_seed(42)

train_data_length = 512 * 1024


class Fun:
    def __init__(self, fun1, fun2, fun1_name, fun2_name, data_len):
        self.data_len = data_len

        self.x = torch.zeros((self.data_len, 1))
        if fun1_name == "ln" or fun2_name == "ln":  # logarithm is defined in (0, inf)
            self.x[:, 0] = (RANGE[1] - 0 + 1e-4) * torch.rand(self.data_len) + 0 + 1e-4
        else:
            self.x[:, 0] = (RANGE[1] - RANGE[0]) * torch.rand(self.data_len) + RANGE[0]

        self.y = fun1(self.x) + fun2(self.x)


# composition = [torch.sin, torch.log, torch.sinc, torch.exp]
# composition_str = ["sin", "ln", "sinc", "exp"]

composition = [torch.sinc, torch.exp]
composition_str = ["sinc", "exp"]

count = 0

for i in range(len(composition)):
    for k in range(i+1, len(composition)):
        if i != k:

            print(f'\nStart training for {composition_str[i]}(x) + {composition_str[k]}(x)...\n')

            exp_train_data = Fun(fun1=composition[i], fun2=composition[k], fun1_name=composition_str[i],
                                 fun2_name=composition_str[k], data_len=train_data_length)

            print(max(exp_train_data.y), flush=True)
            print(min(exp_train_data.y), flush=True)
            train_loader = torch.utils.data.DataLoader(
                [(exp_train_data.x[i], exp_train_data.y[i]) for i in range(train_data_length)],
                batch_size=batch_size, shuffle=True
            )
            count += 1
            # Training loop
            myNN = Double()
            exp_name = "exp0" + str(count) + "g"
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
            exp_test_data = Fun(fun1=composition[i], fun2=composition[k], fun1_name=composition_str[i],
                                fun2_name=composition_str[k], data_len=200)
            x_samples = exp_test_data.x
            y_samples = myNN(x_samples)
            y_plots = y_samples[:, 0].detach()
            x_plots = x_samples[:, 0]
            y_fun = composition[i](x_samples[:, 0]) + composition[k](x_samples[:, 0])
            plt.plot(x_plots, y_plots, ".", label="predicted value")
            plt.plot(x_plots, y_fun, '.', color="green", alpha=0.4, label="true value")
            plt.title(f'Function {composition_str[i]}(x) + {composition_str[k]}(x)')
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
                    plt.title(f'Function {composition_str[i]}(x)')
                    y1_fun = composition[i](x_samples[:, 0])
                else:
                    plt.title(f'Function {composition_str[k]}(x)')
                    y1_fun = composition[k](x_samples[:, 0])

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

            print(f'\nEnd training for {composition_str[i]}(x) + {composition_str[k]}(x)...')
