import matplotlib.pyplot as plt
import numpy as np
import numdifftools as nd


def fn(x):
    return x**2


def fn_lin(x, x0):
    a = nd.Gradient(fn)([x0])
    b = -a * x0 + fn(x0)
    return a * x + b


x = np.arange(-3, 3, 0.1)

fig, ax = plt.subplots(1)
ax.plot(x, fn(x))
ax.plot(np.arange(0, 2, 0.1), fn_lin(np.arange(0, 2, 0.1), 1))
ax.plot(np.arange(-2, 0, 0.1), fn_lin(np.arange(-2, 0, 0.1), -1))
ax.legend(
    [r'$f(x)=x^2$', r'$\frac{df}{dx}$ positive', r'$\frac{df}{dx}$ negative']
)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.show()
fig.savefig('images/slope.png')
