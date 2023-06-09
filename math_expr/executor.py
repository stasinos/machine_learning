import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

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

def handle_fn_call( fn, param, x=None ):
    y = None
    if   fn == "exp":  _,y = fn_exp(x=x)
    elif fn == "ln":   _,y = fn_ln(x=x)
    elif fn == "sin":  _,y = fn_sin(x=x)
    elif fn == "sinc": _,y = fn_sinc(x=x)
    elif fn == "poly":
        A = json.loads(param)
        _,y = fn_poly(A,x=x)
    else:
        print(fn)
        assert 1 == 2
    return y

def execute_program_line( line, x=None ):
    functions = {}
    parameters= {}
    calls = []
    fn_names = ["exp","ln","sinc","sin","poly"]
    for fn_name in fn_names:
        index = line.rfind( fn_name )
        while (index > -1):
            if index not in functions.keys():
                # check for prior keys to avoid re-adding sin where sinc was found
                functions[index] = fn_name
            index = line.rfind( fn_name, 0, index )
    for k,v in sorted(functions.items()):
        calls.append( (k,v) )
    prev_index = None
    prev_fn_name = None
    calls1 = []
    for (index,fn_name) in calls:
        if prev_index is not None:
            prev_params1 = line.rfind( "(", 0, index )
            prev_params2 = line.rfind( ")", 0, index )
            prev_params = line[prev_params1+1:prev_params2]
            calls1.append( (prev_fn_name,prev_params) )
        prev_index = index
        prev_fn_name = fn_name
    last_params1 = line.rfind( "(" )
    last_params2 = line.rfind( ")" )
    last_params = line[last_params1+1:last_params2]
    calls1.append( (prev_fn_name,last_params) ) 
    if x is None: x = np.arange(-10, 10, 0.01)
    y = x
    for (fn_name,params) in reversed(calls1):
        y = handle_fn_call( fn_name, params, y )
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

def execute( program, x=None ):
    xlim = [-10,10]
    ylim = [-5,5]
    file = open( "graphs/{}.graph".format(program) )
    if x is None: x = np.arange(-10, 10, 0.01)
    y = np.zeros( x.shape[0] )
    for line in file.readlines():
        _,y1 = execute_program_line( line, x )
        y += y1
    write_figure( program, x, y, xlim, ylim, True )
    file.close()
