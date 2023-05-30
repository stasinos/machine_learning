import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def fn_exp( a, x0=0, x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    else:
        # Cannot simultaneously compose and translate
        # Translate in the inner function instead
        x0 = 0
    y = a*np.exp(x - x0)
    return x,y
    
def fn_ln( x0=0, x=None ):
    if x is None:
        x = np.arange(0.01, 10, 0.01)
    else:
        # Cannot simultaneously compose and translate
        # Translate in the inner function instead
        x0 = 0
    y = np.array( [np.log(a - x0) if a>0 else np.nan for a in x ] )
    return x,y

def fn_sin( a, b, x0=0, x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    else:
        # Do not simultaneously compose and translate
        # Translate in the inner function instead
        x0 = 0
    y = a*np.sin(b*x - x0)
    return x,y

def fn_sinc( x0=0, x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    else:
        # Do not simultaneously compose and translate
        # Translate in the inner function instead
        x0 = 0
    y = np.sinc(x - x0)
    return x,y

def fn_poly( A, x0=0, x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    else:
        # Do not simultaneously compose and translate
        # Translate in the inner function instead
        x0 = 0
    y = np.zeros(len(x))
    for i,xi in enumerate(x):
        for j,a in enumerate(A):
            y[i] += a*((xi-x0)**j)
    return x,y


def write_figure( title, x, y, xlim, ylim, pure=False ):
    plt.xlim(xlim)
    plt.ylim(ylim)
    fig = plt.figure()
    # setting the axes at the centre
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    if pure:
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    else:
        plt.title( title )
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    plt.plot( x, y )
    y_max = np.abs(ax.get_ylim()).max()
    ax.set_ylim(ymin=-y_max, ymax=y_max)
    plt.savefig( "{}.png".format(title) )
    plt.clf()
    # Make a binary BW image
    pic = Image.open("{}.png".format(title)).convert('1')
    pic.save( "{}-bw.png".format(title) )


xlim = [-10,10]
ylim = [-5,5]
pure = True
x0 = 0

x,y = fn_poly( [0,0.001,0.003,0.002] )
write_figure( "poly", x, y, xlim, ylim, pure )

x,y = fn_ln()
write_figure( "natlog", x, y, xlim, ylim, pure )

x,y = fn_exp( 1 )
write_figure( "exp", x, y, xlim, ylim, pure )

x,y = fn_sin( 2, 3 )
write_figure( "sine", x, y, xlim, ylim, pure )

x,y = fn_sinc()
write_figure( "sinc", x, y, xlim, ylim, pure )

x,y1 = fn_sinc(-2)
x,y2 = fn_poly([0,0.2])
write_figure( "sinc_linear", x, y1+y2, xlim, ylim, pure )

x,y1 = fn_poly([0,0.1,0.3,0.2])
x,y2 = fn_sin(5,10)
write_figure( "sin_poly", x, y1+y2, xlim, ylim, pure )

x,y = fn_sin(1,1)
y,z = fn_exp(1,x=y)
write_figure( "sine_in_exp", x, z, xlim, ylim, pure )

x = np.arange(-1, 1, 0.01)
x,y = fn_exp(1,x=x)
y,z = fn_sin(1,20,x=y)
write_figure( "exp_in_sine", x, z, xlim, ylim, pure )

x = np.arange(-1, 1, 0.01)
x,y = fn_exp(1,x=x)
y,z1 = fn_sin(1,20,x=y)
x,z2 = fn_poly([0,0.1,0.3,0.5],x=x)
write_figure( "exp_in_sine_on_poly", x, z1+z2, xlim, ylim, pure )

x = np.arange(-1, 1, 0.01)
x,y = fn_exp(1,x=x)
y,z = fn_sin(1,20,x=y)
z,w = fn_poly([0,0.1,0.3,0.5],x=z)
write_figure( "exp_in_sine_in_poly", x, w, xlim, ylim, pure )

x,y = fn_sin(1,1)
y,z = fn_ln(x=y)
write_figure( "sin_in_ln", x, z, xlim, ylim, pure )
