# Mathematical Expression Construction

Machine learning exercise on learning mathematical expressions by
observing graphs. The objective is to construct the mathematical
expression that when plotted, best matches the observed graph. The
most "elegant" solution must be found, so simply approximating with a
polyonym is not considered a good solution.

For the sake of simplicity, we shall restrict the scope of the
exercise to learning a "program" in a "programming language" that
has the following functions:

 - `exp()`: returns $e^x$, for $x \in [-10,10]$

 - `ln()`: returns $\ln(x)$ for $x \in (0,10]$ and `NaN` for
   $x \in [-10,0]$

 - `sin()`: returns $\sin(x), for $x \in [-10,10]$

 - `sinc()`: returns $\mathrm{sinc}(x)$, for $x \in [-10,10]$

 - `poly(A)`: returns $\sum_{i} A_i \cdot x^i$,
    where argument $A$ is an array of floats and $x \in [-10,10]$

The functions return a `(x,y)` pair of numpy arrays that can be
directly used as inputs for `matplotlib`. All functions may also
be called with the optional keyword argument `x` which is an array
of floats to be used as domain instead of [-10,10].

The language also has the following operators that combine two or more
functions into a function which also returns a `(x,y)` pair of numpy
arrays:

 - `f,g,..,h`: returns the results of the composition
   $f \circ g \circ ... \circ h$

A "program" consists of multiple lines, where each line is exactly one
function call (or expression using the operators above). The semantics
of the whole program is the plot of all (x,y) pairs where
`x` is [-10,10] and `y` is the item-wise summation of the `y` arrays
returned by all program lines.

For example, the following program:

```
sinc()
poly( [0,0.2] )
```

means the graph plotted by the equation
$y = \mathrm{sinc}(x) + 0.2 \cdot x$ for $x \in [-10,10]$
whereas the program:

```
sin(), poly( [0,20] ), exp()
poly( [0,0.1,0.3,0.5] )
```

means the graph plotted by the equation
$y = \sin(20 \cdot e^x) + 0.5 \cdot x^{3} + 0.3 \cdot x^{2} + 0.1 \cdot x $
and the program:

```
poly( [0,1] )
poly( [0,0.1,0.3,0.5] ), exp(), sin(), poly( [0,20] )
```

means the graph plotted by the equation
$y = x + 0.1 \cdot e^{sin(20x)} + 0.3 \cdot e^{2 \cdot sin(20x)} + 0.5 \cdot e^{3 \cdot sin(20x)}$


## Evaluation Function

We will assume that all graphs are images of dimensions 640x480,
centered on (0,0), without axes or any other decorations,
and that the background is white and the graph is non-white.

We evaluate a program on a test graph _t_ as follows:

 - The program is executed and it produces a hypothesis graph _h_.

 - Both _t_ and _h_ are converted to a 2D array, with 1-bit colour
   (zero for white and 1 for non-white).

 - The two arrays are item-wise logical-AND'ed, to get an array
   _t-h_ where all points of agreement are 1 and all points of
   disagreement are zero.

 - Since most of the figure is white, simply counting matches would
   always give too high a score. Instead, we first calculated q(t) as
   the fraction of points of t that are also part of h, and q(h) as
   the fraction of points of h that are also part of t. We then
   combine these two recall-like metrics into an f-score-like
   evaluation in order to penalize extreme solutions like painting
   nothing or painting everything black.


