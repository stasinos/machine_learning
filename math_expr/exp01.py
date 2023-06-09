import numpy as np
import math
from executor import execute

x = np.arange(-2*math.pi, 2*math.pi, 0.01)
execute( "sine_on_sine", x=x )
