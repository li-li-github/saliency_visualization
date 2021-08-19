# Back Propagation

This page will explain how back propagation is working in deep learning with a very 
[simple example](
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#warm-up-numpy
) provided in pytorch tutorial.

## The challenge
Fit a 3rd order polynomial function to a sine function.

Fitting function

```math
y = a + bx + cx^2 + dx^3
```

Target function

```math
y = sin(x)
```


