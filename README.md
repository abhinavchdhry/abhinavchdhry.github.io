# A practical guide to unsupervised learning
[Clustering](#lets-talk-clustering)
- [Part 1 - The Why](#the-why)
- [Part 2 - The What](#the-what)
- [Part 3 - The How](#the-how)

[Gradient Descent](#gradient-descent)

[About](#about)


## Let's talk clustering <a name="lets-talk-clustering"></a>

### Part 1 - The Why <a name="the-why"></a>
You just woke up from a 15 year coma and rediscover the internet. You open [Google News](news.google.com) to find out what has happened since and notice a column of "Top Stories" on the left. When you click on a top story topic that says "Donald Trump" and you are shown a collection of multiple news stories from different sources <i>all somehow related to this person you have no idea about</i>. As you explore the Top Stories and the various categories to your left, you realize that Google has somehow managed to go through many (if not all) news articles on the internet and grouped them based on topics. Interesting.

![alt text](https://abhinavchdhry.github.io/GoogleNews.JPG)

Case 2: 

All of these tasks that would seem to take days for any human to accomplish is relatively trivial for machines given a suite of algorithms called clustering algorithms. As the name suggests, the fundamental 

### Part 2 - The What <a name="the-what"></a>

### Part 3 - The How <a name="the-how"></a>

## K-Nearest Neighbor


<br>
<br>

# Understanding Gradient Descent <a name="gradient-descent"></a>

__The Problem statement__: Given a function ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20f%3A%20%5Cmathbf%7BX%7D%20%5Crightarrow%20R) where ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cmathbf%7BX%7D%20%3D%20%28x_1%2C%20x_2%2C%20...%2C%20x_n%29) is a n-dimensional vector, find a minima ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cmathbf%7BX%5E*%7D) (a point for which ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20f%28%5Cmathbf%7BX%5E*%7D%29) is a minimum)

__2 dimensional case__
For the sake of simplicity and notation, we will first consider the case when the vector <b>X</b> is in 2-dimensional space i.e. ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Ctextbf%7BX%7D%20%3D%20%28x_1%2C%20x_2%29). The graph for this function might look something like this (this one happens to be a plot of  ![equation](http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20z%20%3D%20%5Csin%28x%29+%5Csin%28y%29):

![alt-text](3d-plot-sinx2-siny2.JPG)

Imagine now that this graph represents the some part of surface of the earth and you happen to be standing at some point in it. Your goal is head to the deepest point possible (in other words, head to the point with the lowest elevation). 

The first thing you're probably going to think of (without even being aware of it) is this: which direction do I take the next step? Of course you won't take a step uphill from that point. Simple logic tells us to head downwards along the direction of steepest decline (assuming of course that you don't slip and fall). This is the exact thinking behind the gradient descent algorithm: __find the direction of maximum increasing (positive) gradient at the current point and take a small step in the opposite direction__.

Now, by the rules of calculus, a small change in the function ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5CDelta%20f) value due to a small change in the vector ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5CDelta%20%5Cmathbf%7BX%7D%20%3D%20%28%5CDelta%20x_1%2C%20%5CDelta%20x_2%29) is given by:

>>>![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5CLARGE%20%5CDelta%20f%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D.%5CDelta%20x_1%20+%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_2%7D.%5CDelta%20x_2)

or, in vector notation

>>> ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5CLARGE%20%5CDelta%20f%20%3D%20%5Cnabla%20%5Cmathbf%7Bf%7D%5Ccdot%20%5CDelta%20%5Cmathbf%7BX%7D),          where ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%5Cnabla%20%5Cmathbf%7Bf%7D%20%3D%20%28%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%2C%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_2%7D%29) and ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%5CDelta%20%5Cmathbf%7BX%7D%20%3D%20%28%5CDelta%20x_1%2C%20%5CDelta%20x_2%29)

Here ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B150%7D%20%5Cfrac%7B%5Cdelta%20f%7D%7B%5Cdelta%20x_1%7D%20%5C%20and%5C%20%5Cfrac%7B%5Cdelta%20f%7D%7B%5Cdelta%20x_2%7D) represent the partial derivatives of the function <b>f</b> w.r.t. the variables x1 and x2 respectively

Our goal now is to choose the delta vector ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5CDelta%20%5Cmathbf%7BX%7D%20%3D%20%28%5CDelta%20x_1%2C%20%5CDelta%20x_2%29) so as to make sure that:
- The value ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5CDelta%20f) is negative (so that we know we are moving in the direction of decreasing gradient, or downwards)
- The magnitude of ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5CDelta%20f) is maximum along this direction

If we choose ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%5CDelta%20f) as follows:
>>> ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%7B%5Ccolor%7BRed%7D%20%5CDelta%20%5Cmathbf%7BX%7D%20%3D%20-%5Ceta.%5Cnabla%20%5Cmathbf%7Bf%7D%7D).

then ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5CDelta%20f%20%3D%20%5Cnabla%20%5Cmathbf%7Bf%7D%20%5Ccdot%20%28-%5Ceta.%5Cnabla%20%5Cmathbf%7Bf%7D%29%20%3D%20-%5Ceta%20%28%5Cnabla%20%5Cmathbf%7Bf%7D%20%5Ccdot%5Cnabla%20%5Cmathbf%7Bf%7D%29%20%3D%20-%5Ceta.%28pos%20%5C%20value%29%20%3D%20neg%20%5C%20value)

__The delta vector is now parallel to the gradient vector at that point__ which according to the Cauchy-Schwartz inequality maximizes the magnitude of the dot product of the 2 vectors (hence maximizing ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%7C%5CDelta%20f%7C)). Note that the positive constant ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5Ceta) does not change the direction of the vector, only scales its value. Since we are only looking to find the direction of maximization of descent, scaling the vector does not affect our goal.

![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%5Ceta) is also called the __learning rate__, a value that determines how large or small a step we take while descending. ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%5Ceta) should be small so as not to introduce error (if the step is too large, we might miss the minima). Also it should not be too small that the descent takes a long time.

And so we arrive at the update function for the variable vector __X__
>>> ![equation](http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%7B%5Ccolor%7BRed%7D%20%5Cmathbf%7BX_%7Bnew%7D%7D%20%3D%20%5Cmathbf%7BX_%7Bold%7D%7D%20-%20%5Ceta.%5Cmathbf%7B%5Cnabla%20f%20%7D%7D)

Although we have described the problem for the special case of 2 dimensions, it easily extends to the more general case of N dimensions. The formula above remains the same.

#### Iterations and when to stop (TODO) 

#### Drawbacks
Note that, in general, a function can have multiple local minima and a global minima. Based on how the gradient descent algorithm is initialized (the initial value that we choose for the variable vector) and the structure of the function itself, we may end up at any one of these local minima. So __gradient descent does not always produce the global minima__.

### Proof (TODO)
According to the [Cauchy-Schwartz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality), for any 2 vectors <b>u</b> and <b>v</b> of the same size,
![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%5Cleft%20%7C%20%5Cmathbf%7Bu%7D%5Ccdot%5Cmathbf%7Bv%7D%20%5Cright%20%7C%20%5Cleq%20%5Cleft%20%7C%20%5Cmathbf%7Bu%7D%20%5Cright%20%7C.%20%5Cleft%20%7C%20%5Cmathbf%7Bv%7D%20%5Cright%20%7C) __and the equality happens when u and v are equal.__

__Why we need to maximize the value of delta f __

If we restrict our change vector delX (or the incremental step) to be small, as restricted by ![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%5Cleft%20%7C%20%5CDelta%20%5Cmathbf%7BX%7D%20%5Cright%20%7C%20%5Cleq%20%5Cepsilon%20%2C%5C%20for%20%5C%20some%20%5C%20small%20%5C%20%5Cepsilon%20%3E%200) then according to the Cauchy-Schwartz inequality, the magnitude of change is

![equation](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clarge%20%7C%5CDelta%20f%7C%20%3D%20%7C%5Cnabla%20%5Cmathbf%7Bf%7D%5Ccdot%20%5CDelta%20%5Cmathbf%7BX%7D%7C%20%5Cleq%20%7C%5Cnabla%20%5Cmathbf%7Bf%7D%7C%5Ccdot%20%7C%5CDelta%20%5Cmathbf%7BX%7D%7C%20%5Cleq%20%7C%5Cnabla%20%5Cmathbf%7Bf%7D%7C.%5Cepsilon)

So the maximum value of the magnitude of change 

## About <a name="about"></a>
