import numpy as np
from executor import execute

execute( "poly" )
execute( "natlog" )
execute( "exp" )
execute( "sine" )
execute( "sinc" )
execute( "sinc_linear" )
execute( "sin_poly" )
execute( "sine_in_exp" )

x = np.arange(-1, 1, 0.01)
execute( "exp_in_sine", x=x )
execute( "exp_in_sine_on_poly", x=x )
execute( "exp_in_sine_in_poly", x=x )

execute( "sin_in_ln" )
