from torch import nn

from snake import Snake


# Model structure
# Also tried ReLU, tanh seems to work better
# Note: For ReLU, add a Sigmoid at the very end
# to keep outputs in 0..1

class FunctionApproximator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            Snake(64),
            nn.Linear(64, 64),
            Snake(64),
            nn.Linear(64, 64),
            Snake(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        output = self.model(x)
        return output


class SmallFunctionApproximator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            Snake(32),
            nn.Linear(32, 32),
            Snake(32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        output = self.model(x)
        return output


# A network that implements
# fn1(x) + fn2(2x)

class DoubleSin(nn.Module):
    def __init__(self, small=True):
        super().__init__()
        self.model1 = None
        self.model2 = None
        if small:
            self.model1 = SmallFunctionApproximator()
            self.model2 = SmallFunctionApproximator()
        else:
            self.model1 = FunctionApproximator()
            self.model2 = FunctionApproximator()

    def forward(self, x):
        output = self.model1(x) + self.model2(2*x)
        return output

    def get_internal(self):
        return [self.model1, self.model2]


class Double(nn.Module):
    def __init__(self, small=True):
        super().__init__()
        self.model1 = None
        self.model2 = None
        if small:
            self.model1 = SmallFunctionApproximator()
            self.model2 = SmallFunctionApproximator()
        else:
            self.model1 = FunctionApproximator()
            self.model2 = FunctionApproximator()

    def forward(self, x):
        output = self.model1(x) + self.model2(x)
        return output

    def get_internal(self):
        return [self.model1, self.model2]


class SingleSinTwice(nn.Module):
    def __init__(self, a, b, small=True):
        super().__init__()
        self.model = None
        if small:
            self.model = SmallFunctionApproximator()
        else:
            self.model = FunctionApproximator()
        self.a = a
        self.b = b

    def forward(self, x):
        output = self.model(self.a*x) + self.model(self.b*x)
        return output

    def get_internal(self):
        return [self.model]


class SinComposition(nn.Module):
    def __init__(self, a, small=True):
        super().__init__()
        self.model = None
        if small:
            self.model = SmallFunctionApproximator()
        else:
            self.model = FunctionApproximator()
        self.a = a

    def forward(self, x):
        output = self.model(self.model(self.a*x))
        return output

    def get_internal(self):
        return [self.model]


class SinFunComposition(nn.Module):
    def __init__(self, a, small=True):
        super().__init__()
        self.model1 = None
        self.model2 = None
        if small:
            self.model1 = SmallFunctionApproximator()
            self.model2 = SmallFunctionApproximator()
        else:
            self.model1 = FunctionApproximator()
            self.model2 = FunctionApproximator()

        self.a = a

    def forward(self, x):
        output = self.model1(self.model2(self.a*x))
        return output

    def get_internal(self):
        return [self.model1, self.model2]


class DoubleSin(nn.Module):
    def __init__(self, small=True):
        super().__init__()
        self.model1 = None
        self.model2 = None
        if small:
            self.model1 = SmallFunctionApproximator()
            self.model2 = SmallFunctionApproximator()
        else:
            self.model1 = FunctionApproximator()
            self.model2 = FunctionApproximator()

    def forward(self, x):
        output = self.model1(x) + self.model2(2*x)
        return output

    def get_internal(self):
        return [self.model1, self.model2]
