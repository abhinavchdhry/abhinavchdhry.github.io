---
layout: post
title:  "Understanding Gradient Descent: Part II"
date:   2017-04-15 20:20:57 -0400
categories: jekyll update
---

In [Understanding Gradient Descent: Part I](/jekyll/update/2017/04/15/gradient-descent/), we talked about the problem of finding an extrema of a mathematical function using gradient descent. Here we look at how gradient descent is used in a machine learning context: linear regression. This article assumes that the reader is familiar with linear regression. If not, a tutorial on linear regression can be found [here]().

### The Problem
In its simplest form, linear regression boils down to expressing a continuous _target_ variable $$ y $$ as a linear combination of _m predictor_ variables (m >= 1) $$ \mathbf{X} = (x_1, x_2, ... x_m) $$ , thus ending up with a equation like:


$$ y_{pred} = w_1x_1 + w_2x_2 + ... + w_mx_m + b $$

or in vector notation,


$$ y_{pred} = \mathbf{w}.\mathbf{X} + b $$


In the case where there is only one predictor variable $$ x $$, linear regression can be seen as drawing a best-fit line through the scatterplot of 2-D (x, y) points.

[Image]

If our dataset consists of N datapoints $$ ( \mathbf{X}^{(1)}, y^{(1)}), (\mathbf{X}^{(2)}, y^{(2)}), ... (\mathbf{X}^{(N)}, Y^{(N)}) $$, then we define the cost function for our linear regression problem as:


$$ C = \frac{1}{2N} \sum_{i=1}^{N} (y^{(i)}_{pred} - y^{(i)})^2  $$


$$ or, C = \frac{1}{2N} \sum_{i=1}^{N} (\mathbf{w}.\mathbf{X}^{(i)} + b - y^{(i)})^2 $$

where the superscript _i_ denotes a datapoint, and the optimization problem we have to solve is:


$$ argmin_{W, b} (C) $$

Notice that in the cost function, the $$ \mathbf{X}^{(i)} $$ and the $$ y^{(i)} $$ are known values (these are just the datapoints that we have). The idea is to figure out the values of the unknowns $$ (w_1, w_2, ... w_m) $$ and $$ b $$ such that the value of the cost function $$ C $$ is _minimized_. This is where gradient descent comes to the rescue in finding the extremum (in this case, minimum) of the function $$ C(\mathbf{w}, b) $$


### Gradient descent to the rescue
Before delving into the details, let's first reformulate the above equations in a more convenient way (as is usually done in linear regression problems). Rather than consider the _intercept_ $$ b $$ as a separate unknown, we rename it to $$ w_0 $$ and make it a part of the unknown weights vector $$ \mathbf{w} $$. If all the $$ \mathbf{X_i} $$ vectors are also appended with a value $$ x_0 = 1 $$, such that any $$ \mathbf{X_i} $$ vector becomes:  $$ \mathbf{X_i} = (1, x_{i 1}, x_{i 2}, ..., x_{i m}) $$, then this transformation enables us to rewrite the cost function as:


$$ C = \frac{1}{2N} \sum_{i=1}^{N} (\mathbf{w}.\mathbf{X}^{(i)} - y^{(i)})^2 $$

where $$ \mathbf{w} = (w_0, w_1, w_2, ..., w_m) $$ is now a (m+1) dimensional unknown.

Rewriting the [the update step](/jekyll/update/2017/04/15/gradient-descent/#the-update-step) of gradient descent to suit our current problem,


$$ \mathbf{w_{new}} = \mathbf{w_{old}} - \eta \nabla C $$

This involves finding the gradient vector of $$ C $$ with respect to $$ \mathbf{w} $$:


$$ \nabla C = \frac{1}{N} \sum_{i=1}^{N} (\mathbf{w}.\mathbf{X}^{(i)} - y^{(i)}).\mathbf{X}^{(i)} $$

The summation indicates that to calculate the gradient vector $$ \nabla C $$, we need to loop through _all_ the datapoints. Since the $$ \mathbf{w} $$ changes at each update step, we need to recompute the gradient vector after each update step, and before proceeding with the next update. So if our algorithm does 50 updates before stopping, this results in a total of $$ 50N $$ iterations, which is prohibitively expensive in case of large datasets (for instance, a million datapoints). This is a major issue with gradient descent, and in practice, variants of gradient descent such as Stochastic Gradient descent and mini-batch gradient descent are used.

### The Code

