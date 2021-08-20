import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)


def fn_3poly(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = fn_3poly(x, a, b, c, d)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')


x_plot = np.arange(-4, 4, 0.02)
fig, ax = plt.subplots(1)
ax.plot(x_plot, np.sin(x_plot))
ax.plot(x_plot, fn_3poly(x_plot, a, b, c, d))
ax.legend([r'$f(x)=\sin(x)$', r'$a + bx + cx^2 + dx^3$'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.show()


# Saliency map
def fn_3poly_dr(x):
    """direvative of fn_3poly"""
    return b + 2 * c * x + 3 * d * x **2


saliency = fn_3poly_dr(x_plot)
fig, ax = plt.subplots(1)
ax.plot(x_plot, np.sin(x_plot))
ax.plot(x_plot, fn_3poly(x_plot, a, b, c, d))
ax.scatter(
    x_plot, fn_3poly(x_plot, a, b, c, d),
    c=np.array(cm.tab10.colors[1]).reshape(1, -1), marker='.', 
    s=20 * (saliency - saliency.min())
)
ax.legend([r'$f(x)=\sin(x)$', '$a + bx + cx^2 + dx^3$\nthickness represents saliency'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.show()
fig.savefig('saliency.png')

