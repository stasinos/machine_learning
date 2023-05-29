import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
pure = True
#xtrans = -2
#ytrans = -0.5
xtrans = 0
ytrans = 0

x = np.arange(-11, 11, 0.1)
a = 0.2
b = 0.3
c = 0.1
y = (ytrans) + a*(x-xtrans)**3 + b*(x-xtrans)**2 + c*(x-xtrans)
write_figure( "poly", x, y, xlim, ylim, pure )

x = np.arange(0.2, 11, 0.001)
y = (ytrans) + np.log(x-xtrans)
write_figure( "natlog", x, y, xlim, ylim, pure )

x = np.arange(-10, 10, 0.01)
a = 0.03
y = (ytrans) + a*np.exp(x-xtrans)
write_figure( "exp", x, y, xlim, ylim, pure )

x = np.arange(-11, 11, 0.001)
a = 2
b = 3
y = (ytrans) + a*np.sin(b*x-xtrans)
write_figure( "sine", x, y, xlim, ylim, pure )

x = np.arange(-11, 11, 0.01)
y = (ytrans) + np.sinc(x-xtrans)
write_figure( "sinc", x, y, xlim, ylim, pure )

x = np.arange(-11, 11, 0.001)
y = (ytrans) + np.cosh(x-xtrans)
write_figure( "cosh", x, y, xlim, ylim, pure )

x = np.arange(-11, 11, 0.01)
c = 0.2
y = (ytrans) + np.sinc(x-xtrans) + c*x
write_figure( "sinconlinear", x, y, xlim, ylim, pure )

x = np.arange(0.2, 11, 0.001)
a = 0.1
b = 10
y = (ytrans) + np.log(x-xtrans) + a*np.sin(b*x-xtrans)
write_figure( "sineonln", x, y, xlim, ylim, pure )

