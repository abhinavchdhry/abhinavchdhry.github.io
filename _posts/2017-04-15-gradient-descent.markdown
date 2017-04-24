---
layout: post
title:  "Understanding Gradient Descent: Part I"
tagline: "The math behind it all"
date:   2017-04-15 20:20:57 -0400
categories: jekyll update
---

Gradient descent is a powerful mathematical optimization technique on which many machine learning algorithms are built. This is the first part of an article aimed at understanding gradient descent from a mathematical and computational standpoint.

In this article, we look at the mathematical problem that gradient descent solves and how. It assumes that the reader is familiar with basic calculus and vector algebra. The next part deals with the computational aspects of gradient descent and its applications in machine learning.

### The Problem statement
Given a function $$ f : X \rightarrow R $$ where $$ X = (x_1, x_2, ... x_n) $$ is a n-dimensional vector, find a minima $$X^*$$ equation (a point for which $$ f(X^*) $$ is a minimum)

### 2 dimensional case
For the sake of simplicity and notation, we will first consider the case when the vector X is in 2-dimensional space i.e. $$ X = (x_1, x_2) $$. The graph for this function might look something like this (this one happens to be a plot of $$ z = sin(x) + sin(y) $$)

![Image](/images/3d-plot-sinx2-siny2.jpeg)

Imagine now that this graph represents the some part of surface of the earth and you happen to be standing at some point in it. Your goal is head to the deepest point possible (in other words, head to the point with the lowest elevation).

The first thing you’re probably going to think of (without even being aware of it) is this: which direction do I take the next step? Of course you won’t take a step uphill from that point. Simple logic tells us to head downwards along the direction of steepest decline (assuming of course that you don’t slip and fall). This is the exact thinking behind the gradient descent algorithm: **find the direction of maximum increasing (positive) gradient at the current point and take a small step in the opposite direction.**

Now, by the rules of calculus, a small change in the function $$ \Delta f $$ due to a small change in the vector $$ \Delta X = (\Delta x_1, \Delta x_2) $$ is given by:


$$
	\Delta f = \frac{\partial f}{\partial x_1}.\Delta x_1 + \frac{\partial f}{\partial x_2}.\Delta x_2
$$

or, in vector notation


$$
	\Delta f = \nabla f . \Delta X,  
	where \nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2})
$$


Our goal now is to choose the delta vector $$ \Delta X = (\Delta x_1, \Delta x_2) $$ so as to make sure that:
* The value $$ \Delta f $$ is negative (so that we know we are moving in the direction of decreasing gradient, or downwards)
* The magnitude of $$ \Delta f $$ is maximum along this direction

If we choose $$ \Delta f $$ as follows: $$ \Delta X = -\eta.\nabla f $$, then

$$ \Delta f = \mathbf{\nabla f}.(-\eta \mathbf{\nabla f}) = -\eta.(\mathbf{\nabla f.\nabla f}) = -\eta.(positive value) = negative $$

<br>
**The delta vector is now parallel to the gradient vector at that point** which according to the Cauchy-Schwartz inequality maximizes the magnitude of the dot product of the 2 vectors (hence maximizing $$ \left \| \Delta f \right \| $$ ). Note that the positive constant $$ \eta $$  does not change the direction of the vector, only scales its value. Since we are only looking to find the direction of maximization of descent, scaling the vector does not affect our goal.

$$ \eta $$ is also called the **learning rate**, a value that determines how large or small a step we take while descending. $$ \eta $$ should be small so as not to introduce error (if the step is too large, we might miss the minima). Also it should not be too small that the descent takes a long time.

And so we arrive at the update function for the variable vector X:


$$
	\mathbf{X_{new}} = \mathbf{X_{old}} - \eta.\mathbf{\nabla f}
$$

Although we have described the problem for the special case of 2 dimensions, it easily extends to the more general case of N dimensions. The formula above remains the same.
<br>
### Iterations and when to stop (TODO)

<br>
### Drawbacks
Note that, in general, a function can have multiple local minima and a global minima. Based on how the gradient descent algorithm is initialized (the initial value that we choose for the variable vector) and the structure of the function itself, we may end up at any one of these local minima. **So gradient descent does not always result in the global minima**.

<br>
### Proof (TODO)
According to the [Cauchy-Schwartz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality) for any 2 vectors $$ \mathbf{u} $$ and$$ \mathbf{v} $$ of the same size, $$ \| \mathbf{u}.\mathbf{v} \| \le \| \mathbf{u} \| . \| \mathbf{v} \| $$ and the equality happens when u and v are equal.

If we restrict our change vector $$ \Delta X $$ (or the incremental step) to be small, as restricted by $$ \| \Delta X \| \le \epsilon $$for some small $$ \epsilon > 0 $$ then according to the Cauchy-Schwartz inequality, the magnitude of change is

$$
| \Delta f | = | \mathbf{\nabla f }. \mathbf{\Delta X} |  \le | \mathbf{\nabla f} |.| \mathbf{\Delta X}| \le |\mathbf{\nabla f}|.\epsilon
$$


