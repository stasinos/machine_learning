from torch import nn
import math


# Model structure
# Also tried ReLU, tanh seems to work better
# Note: For ReLU, add a Sigmoid at the very end
# to keep outputs in 0..1

class FunctionApproximator( nn.Module ):

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


# A network that implements
# fn1(x) + fn2(2x)

class DoubleSin( nn.Module ):
    def __init__(self):
        super().__init__()
        self.model1 = FunctionApproximator()
        self.model2 = FunctionApproximator()
    def forward(self, x):
        output = self.model1(x) + self.model2(2*x)
        return output
