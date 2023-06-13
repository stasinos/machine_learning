from torch import nn
import math

from snake.activations import Snake


# Model structure
# Also tried ReLU, tanh seems to work better
# Note: For ReLU, add a Sigmoid at the very end
# to keep outputs in 0..1

class DeepApproximator( nn.Module ):

    def __init__(self, width=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, width),
            Snake(width),
            nn.Linear(width, width),
            Snake(width),
            nn.Linear(width, width),
            Snake(width),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        output = self.model(x)
        return output


class ShallowApproximator( nn.Module ):

    def __init__(self, width=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, width),
            Snake(width),
            nn.Linear(width, width),
            Snake(width),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        output = self.model(x)
        return output


# A network that implements
# a1*fn1(b1*x) + a2*fn2(b2*x)

class DoubleSin( nn.Module ):
    def __init__(self, a1=1, a2=1, b1=1, b2=1, depth=1, width=32):
        super().__init__()
        self.model1 = None
        self.model2 = None
        if depth == 2:
            self.model1 = DeepApproximator(width)
            self.model2 = DeepApproximator(width)
        else:
            self.model1 = ShallowApproximator(width)
            self.model2 = ShallowApproximator(width)
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2

    def forward(self, x):
        output = self.a1*self.model1(self.b1*x) + self.a2*self.model2(self.b2*x)
        return output

    def get_internal(self):
        return [self.model1,self.model2]


# A network that implements
# a1*fn(b1*x) + a2*fn(b2*x)

class SingleSinTwice( nn.Module ):
    def __init__(self, a1=1, a2=1, b1=1, b2=1, depth=1, width=32):
        super().__init__()
        self.model = None
        if depth == 2: self.model = DeepApproximator(width)
        else: self.model = ShallowApproximator(width)
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2

    def forward(self, x):
        output = self.a1*self.model(self.b1*x) + self.a2*self.model(self.b2*x)
        return output

    def get_internal(self):
        return [self.model]


# A network that implements
# a1*fn1( a2*fn2(b2*x) )

class Composer( nn.Module ):
    def __init__(self, a1=1, a2=1, b1=1, b2=1, depth=1, width=32):
        super().__init__()
        self.model1 = None
        self.model2 = None
        if depth == 2:
            self.model1 = DeepApproximator(width)
            self.model2 = DeepApproximator(width)
        else:
            self.model1 = ShallowApproximator(width)
            self.model2 = ShallowApproximator(width)
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2

    def forward(self, x):
        inner  = self.a2 * self.model2( self.b2 * x )
        output = self.a1 * self.model1(self.b1 * inner )
        return output

    def get_internal(self):
        return [self.model1,self.model2]
