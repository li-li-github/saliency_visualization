# Back Propagation

This page will explain how back propagation is working in deep learning with a very 
[simple example](
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#warm-up-numpy
) provided in pytorch tutorial.

## The goal
Fit a 3rd order polynomial function to a sine function.

Fitting function
```math
y_apx = a + bx + cx^2 + dx^3
```

Target function
```math
y_target = sin(x)
```

We are going to use back propagation to find the best set of values for `a, b, c, d` 
that makes the fitting function well approximate the target function.


## Mathematical derivation
First we need to define a loss that represent how far our approximation is from the 
target values. Here we use squared difference or L2 loss in a sophisticated term.

$Loss = (y_apx - y_target)^2$


Our aim is to make the `Loss` as small as possible. (Given the same x,) `y_target` is 
fixed. What we can tune is `y_apx`. 
(by changing the values of `a, b, c, d` we can change the value of `y_apx`)

To find the minimum `Loss` respect to `y_apx`, the standard approach is to take a 
derivative. So here it's gonna be
```math
\frac{dLoss}{dy_apx} = 2(y_apx - y_target)
```
This is the gradient of `Loss` respect to `y_apx`.





