# Mathematical Expression Construction

Machine learning exercise on using PyTorch autodiff for program
synthesis. The objective is to complete a computer program where
one or more of the functions are left undefined and need to be
learned from examples. The complication is that the training set
has end-to-end inputs/outputs and not the inputs/outputs of the
undefined functions.

For the sake of simplicity, we shall restrict the scope of the
exercise to synthesising functions is a simple programming language
that plots mathematical expressions.


## Definition of the programming language

In its current form, this programming language has the following
functions:

 - `exp()`: returns $e^x$, for $x \in [-10,10]$

 - `ln()`: returns $\ln(x)$ for $x \in (0,10]$ and `NaN` for
   $x \in [-10,0]$

 - `sin()`: returns $\sin(x)$, for $x \in [-10,10]$

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


## Objective

The task of the _program synthesiser_ is to receive as input a program
in this language and the bitmap of the plot it generates. Although the
complete program is used to generate the plots, one or more of the
functions called by the program are undefined in what the synthesiser
sees and the synthesiser has to approximate them.

The solution that is more faithful to the original program must be
found, so simply approximating with a polyonym is not considered a
good solution.

For the purposes of this exercise the synthesier may assume that all
inputs are 640x480 bitmaps, centered on (0,0), without axes or any
other decorations, and that the background is white and the graph is
non-white.

An alternative way to think about this task is as an image
understanding exercise, where the the declarative program describes
the objects in the scene and the relationships between them. The AI
system receives an image and a partial description where some objects
are unknown. The AI system hypothesises about the nature of this
object based on its image and its relationship to other objects.

In future iterations of this exercise, it is possible that the AI
system is also allowed to amend the program itself, and not only fill
in funtion definitions.


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

It is expected that this evaluation function will not perform very
will, since it does _not_ behave as a heuristic. That is, the cost
does not gradually reduce as a hypothesis moves closer to the target.
Imagine a perfectly matching curve translated on x to be a little
higher than the target. This curve is evaluated with zero, and moving
it up or down a bit is again evaluated with zero: moving in the
correct direction does not increase its evaluation.

One possible exercise is to define, implement, and test a better
evaluation function.


## Hypothesis Formation and Evaluation

The hypothesis generated by the program synthesiser is evaluated by
first using the code in `executor.py` to produce a bitmap and then
using the code in `evaluator.py` to evaluate the similarity between
the input bitmap and the bitmap generated by executing the hypothesis.
In other words, there is no "golden truth" program, but rather a
target bitmap and any program that matches it is evaluated positively.

It should be noted that although some effort has been made to reduce
the different syntactic variations for expressing identical semantics,
this is not completely possible. At the very least, any poly can be
broken up into multiple lines. Furthermore, the graphs are evaluated
on a specific domain, there will be semantically different functions
that are indistiguishable within this domain. For example, exp() looks
like a straight line for small exponents.

# History

## Applications of AI, 2022-2023

[Tatiana Boura](https://github.com/tatiana-boura) and
[Konstantinos Chaldaiopoulos](https://github.com/KonstantinosChaldaiopoulos)
bravely undertook the first iteration of this exercise, while the
exercise was still being formed and greatly contributed to the exercise
definition itself while also carrying out experiments and writing an
excellent report based on these experiments. In this first iteration,
evaluation was performed on x,y pairs rather than bitmaps as there was
no satisfactory way to evaluate the similarity between the output of
a hypothesis and the target bitmap.
