# Mathematical Expression Construction

Machine learning exercise on learning mathematical expressions by
observing graphs. The objective is to construct the mathematical
expression that when plotted, best matches the observed graph. The
most "elegant" solution must be found, so simply approximating with a
polyonym is not considered a good solution.

For the sake of simplicity, we shall restrict the scope of the
exercise to learning a "program" in a "programming language" that has
the following functions:

 - `exp(a)`: f(x) = a*exp(x), with real argument a

 - `sin(a,b)`: f(x) = a*sin(b*x), with real arguments a,b

 - `sinc(x0)`: f(x) = sinc(x-x0), with real argument x0

 - `poly(A)`: f(x) = the polynomial Sum[i in 0..len(A)-1]( A[i]*x^i ),
   with an array of integers as argument.
 
A "program" consists of multiple lines, where each line is exactly one
function call. The semantics of each line is the expression given
above. The semantics of the whole program is the graph for y=f(x),
for x in [-10,10], where f(x) is the summation of the semantics of
all lines.

For example, the following program:

```
sinc( -2 )
poly( [0,0.2] )
```

means the graph plotted by the equation
y = sinc(x) + 0.2*x
for x in [-10,10].
