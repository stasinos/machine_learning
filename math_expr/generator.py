import matplotlib.pyplot as plt
import numpy as np


xtrans = -2
ytrans = -0.5
xlim = [-10,10]
ylim = [-5,5]

title = "poly"
x = np.arange(-11, 11, 0.1)
a = 0.2
b = 0.3
c = 0.1
y = (ytrans) + a*(x-xtrans)**3 + b*(x-xtrans)**2 + c*(x-xtrans)
plt.plot( x, y )
plt.title( title )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig( "{}.png".format(title) )
plt.clf()


title = "natlog"
x = np.arange(-1.9, 11, 0.001)
y = (ytrans) + np.log(x-xtrans)
plt.plot( x, y )
plt.title( title )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig( "{}.png".format(title) )
plt.clf()


title = "exp"
x = np.arange(-10, 10, 0.01)
a = 0.03
y = (ytrans) + a*np.exp(x-xtrans)
plt.plot( x, y )
plt.title( title )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig( "{}.png".format(title) )
plt.clf()


title = "sine"
x = np.arange(-11, 11, 0.001)
a = 2
b = 3
y = (ytrans) + a*np.sin(b*x-xtrans)
plt.plot( x, y )
plt.title( title )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig( "{}.png".format(title) )
plt.clf()


title = "sinc"
x = np.arange(-11, 11, 0.01)
y = (ytrans) + np.sinc(x-xtrans)
plt.plot( x, y )
plt.title( title )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig( "{}.png".format(title) )
plt.clf()


title = "cosh"
x = np.arange(-11, 11, 0.001)
y = (ytrans) + np.cosh(x-xtrans)
plt.plot( x, y )
plt.title( title )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig( "{}.png".format(title) )
plt.clf()


title = "sinconcarrier"
x = np.arange(-11, 11, 0.01)
c = 0.2
y = (ytrans) + np.sinc(x-xtrans) + c*x
plt.plot( x, y )
plt.title( title )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig( "{}.png".format(title) )
plt.clf()


title = "sineonln"
x = np.arange(-1.9, 11, 0.001)
a = 0.1
b = 10
y = (ytrans) + np.log(x-xtrans) + a*np.sin(b*x-xtrans)
plt.plot( x, y )
plt.title( title )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig( "{}.png".format(title) )
plt.clf()
