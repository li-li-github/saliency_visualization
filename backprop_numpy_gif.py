"""
This script creates a gif of the convergence during back propagation
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def fn_sin(x):
    return np.sin(x)


def fn_3d(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = fn_sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()
learning_rate = 1e-6

# Prepare a figure instance
fig, ax = plt.subplots(1)
fig.set_tight_layout(True)

print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

x_plot = np.arange(-4, 4, 0.02)
ax.plot(x_plot, fn_sin(x_plot))
line, = ax.plot(x_plot, fn_3d(x_plot, a, b, c, d))
ax.legend([r'$\sin(x)$', r'$a + bx + cx^2 + dx^3$'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(f'iteration {0}, loss ')
ax.set_ylim([-2, 2])


class BackPropagation:
    def __init__(self, fn, fn_arg, target, learn_rate=1e-6):
        self.fn = fn
        self.fn_arg = fn_arg
        self.learn_rate = learn_rate
        self.y_pred = self.fn(**self.fn_arg)
        self.y = target
        
    def update(self, i):        
        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (self.y_pred - self.y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * self.fn_arg['x']).sum()
        grad_c = (grad_y_pred * self.fn_arg['x'] ** 2).sum()
        grad_d = (grad_y_pred * self.fn_arg['x'] ** 3).sum()

        # Update weights
        self.fn_arg['a'] -= self.learn_rate * grad_a
        self.fn_arg['b'] -= self.learn_rate * grad_b
        self.fn_arg['c'] -= self.learn_rate * grad_c
        self.fn_arg['d'] -= self.learn_rate * grad_d
        self.y_pred = self.fn(**self.fn_arg)
        
        return self.fn_arg


fn_arg = {'x': x, 'a': a, 'b': b, 'c': c, 'd': d}
fn_arg_plot = {'x': x_plot, 'a': a, 'b': b, 'c': c, 'd': d}
backprop = BackPropagation(fn_3d, fn_arg, y)

total_iter = 2001
plot_every = 100
interval = 2
repeat_delay = 3000
waiting_frame = repeat_delay // 2


def update(i):
    y_pred = fn_3d(**fn_arg)      

    if i <= total_iter:
        _arg = backprop.update(i)
        if i % plot_every == 0:
            loss = np.square(y_pred - y).sum()
            label = f'iteration {i}, loss {loss:.2f}'
            print(label)
            ax.set_xlabel(label)
            for key, val in _arg.items():
                if key != 'x':
                    fn_arg_plot[key] = _arg[key]
            line.set_ydata(fn_3d(**fn_arg_plot))
            ax.legend(
                [r'$\sin(x)$',
                 f'${_arg["a"]:.3f} + {_arg["b"]:.3f}x + {_arg["c"]:.3f}x^2 + '
                 f'{_arg["d"]:.3f}x^3$'],
                loc='upper right'
            )
        
    return line, ax
    

anim = FuncAnimation(
    fig, update, frames=range(total_iter + waiting_frame), interval=interval, 
)

anim.save('images/backprop.gif', dpi=80, writer='imagemagick')




