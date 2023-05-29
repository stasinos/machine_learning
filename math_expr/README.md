# Mathematical Expression Construction

Machine learning exercise on learning mathematical expressions by
observing graphs. The objective is to construct the mathematical
expression that when plotted, best matches the observed graph. The
most "elegant" solution must be found, so simply approximating with a
polyonym is not considered a good solution.

For the sake of simplicity, we shall restrict the scope of the
exercise to learning a "program" in a "programming language" that has
the following functions:

 - `exp(a)`: returns $a \cdot e^x$, where $a$ is float argument
   and $x \in [-10,10]$

 - `sin(a,b)`: returns $a \cdot \sin(b\cdot x), where $a,b$ are
    float arguments and $x \in [-10,10]$

 - `sinc(x0)`: returns $sinc(x-x_0)$, where $x_0$ is float argument
    and $x \in [-10,10]$

 - `poly(A)`: returns $\sum_{i=0..len(A)-1} A_i \cdot x^i$,
    where argument $A$ is an array of floats and  $x \in [-10,10]$
 
A "program" consists of multiple lines, where each line is exactly one
function call. The semantics of each line is the expression given
above. The semantics of the whole program is the graph for y=f(x),
for x in [-10,10], where f(x) is the item-wise summation of the
arrays returned by all program lines.

For example, the following program:

```
sinc( -2 )
poly( [0,0.2] )
```

means the graph plotted by the equation
$y = sinc(x) + 0.2 \cdot x$ for $x \in [-10,10]$.


## Evaluation Function

We will assume that all test graphs are images of dimensions 640x480
and that the background is white and the actual graph is non-white.

We evaluate a program on a test graph _t_ as follows:

 - The program is executed and it produces a hypothesis graph _h_,
   with dimensions 640x480 and a white backfround.

 - Both _t_ and _h_ are converted to a 2D array, with 1-bit colour
   (zero for white and 1 for non-white).

 - The two arrays are item-wise logical-AND'ed, to get an array
   _t-h_ where all points of agreement are 1 and all points of
   disagreement are zero.

 - Since most of the figure is white, simply counting matches would
   always give too high a score. Instead, we first calculated q(t) as
   the fraction of points of t that are also part of h, and q(h) as
   the fraction of points of h that are also part of t. We then
   combine these two recall-like metrics into an f score-like
   evaluation in order to penalize painting everything black.

