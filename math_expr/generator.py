import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def fn_exp( a, xtrans=0, ytrans=0 ):
    x = np.arange(-10, 10, 0.001)
    y = (ytrans) + a*np.exp(x-xtrans)
    return x,y
    
def fn_ln( xtrans=0, ytrans=0 ):
    x = np.arange(0.01, 10, 0.001)
    y = (ytrans) + np.log(x-xtrans)
    return x,y

def fn_sin( a, b, xtrans=0, ytrans=0 ):
    x = np.arange(-10, 10, 0.001)
    y = (ytrans) + a*np.sin(b*x-xtrans)
    return x,y

def fn_sinc( xtrans=0, ytrans=0 ):
    x = np.arange(-10, 10, 0.001)
    y = (ytrans) + np.sinc(x-xtrans)
    return x,y

def fn_poly( A, xtrans=0 ):
    x = np.arange(-10, 10, 0.001)
    y = np.zeros(len(x))
    for i,xi in enumerate(x):
        for j,a in enumerate(A):
            y[i] += a*(xi**j)
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
    plt.savefig( "{}.png".format(title) )
    plt.clf()
    # Make a binary BW image
    pic = Image.open("{}.png".format(title)).convert('1')
    pic.save( "{}-bw.png".format(title) )


xlim = [-10,10]
ylim = [-5,5]
pure = False#True
#xtrans = -2
#ytrans = -0.5
xtrans = 0
ytrans = 0

x = np.arange(-11, 11, 0.1)
a = 0.0002
b = 0.0003
c = 0.0001
y = (ytrans) + a*(x-xtrans)**3 + b*(x-xtrans)**2 + c*(x-xtrans)
write_figure( "poly", x, y, xlim, ylim, pure )

x,y = fn_ln()
write_figure( "natlog", x, y, xlim, ylim, pure )

x,y = fn_exp( 0.03 )
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

