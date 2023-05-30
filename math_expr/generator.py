import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def fn_exp( x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    y = np.exp(x)
    return x,y
    
def fn_ln( x=None ):
    if x is None:
        x = np.arange(0.01, 10, 0.01)
    y = np.array( [np.log(a) if a>0 else np.nan for a in x ] )
    return x,y

def fn_sin( x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    y = np.sin(x)
    return x,y

def fn_sinc( x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    y = np.sinc(x)
    return x,y

def fn_poly( A, x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    y = np.zeros(len(x))
    for i,xi in enumerate(x):
        for j,a in enumerate(A):
            y[i] += a*(xi**j)
    return x,y

def op_compose( fn_list, x=None ):
    if x is None:
        x = np.arange(-10, 10, 0.01)
    y = x
    for fn in reversed( fn_list ):
        if   fn == "fn_exp":  _,y = fn_exp(x=y)
        elif fn == "fn_ln":   _,y = fn_ln(x=y)
        elif fn == "fn_sin":  _,y = fn_sin(x=y)
        elif fn == "fn_sinc": _,y = fn_sinc(x=y)
        else:
            try:
                (f,A) = fn
                assert f == "fn_poly"
                _,y = fn_poly(A,x=y)
            except:
                assert 1==2
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

x,y = fn_poly( [0,0.001,0.003,0.002] )
write_figure( "poly", x, y, xlim, ylim, pure )

x,y = fn_ln()
write_figure( "natlog", x, y, xlim, ylim, pure )

x,y = fn_exp()
write_figure( "exp", x, y, xlim, ylim, pure )

x,y = op_compose( [("fn_poly",[0,2]),"fn_sin",("fn_poly",[0,3])] )
write_figure( "sine", x, y, xlim, ylim, pure )

x,y = fn_sinc()
write_figure( "sinc", x, y, xlim, ylim, pure )

x,y1 = op_compose( ["fn_sinc",("fn_poly",[2,1])] )
x,y2 = fn_poly([0,0.2])
write_figure( "sinc_linear", x, y1+y2, xlim, ylim, pure )

x,y1 = fn_poly([0,0.1,0.3,0.2])
x,y2 = op_compose( [("fn_poly",[0,5]),"fn_sin",("fn_poly",[0,10])] )
write_figure( "sin_poly", x, y1+y2, xlim, ylim, pure )

x,y = op_compose( ["fn_exp","fn_sin"] )
write_figure( "sine_in_exp", x, y, xlim, ylim, pure )

x = np.arange(-1, 1, 0.01)
x,y = op_compose( ["fn_sin",("fn_poly",[0,20]),"fn_exp"], x )
write_figure( "exp_in_sine", x, y, xlim, ylim, pure )

x = np.arange(-1, 1, 0.01)
x,y1 = op_compose( ["fn_sin",("fn_poly",[0,20]),"fn_exp"], x )
x,y2 = fn_poly([0,0.1,0.3,0.5],x=x)
write_figure( "exp_in_sine_on_poly", x, y1+y2, xlim, ylim, pure )

x = np.arange(-1, 1, 0.01)
x,y = op_compose( [("fn_poly",[0,0.1,0.3,0.5]),"fn_sin",("fn_poly",[0,20]),"fn_exp"], x )
write_figure( "exp_in_sine_in_poly", x, y, xlim, ylim, pure )

x,y = op_compose( ["fn_ln","fn_sin"] )
write_figure( "sin_in_ln", x, y, xlim, ylim, pure )
