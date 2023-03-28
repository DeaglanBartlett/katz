import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy.abc import x

seed = 123
nx = 300
frac_sigx = 0.1

np.random.seed(seed)

def make_data(name, f, xrange):

    print(name)

    expr = sympy.sympify(f)
    fun = sympy.lambdify(x, expr, modules='numpy')

    xdata = np.random.uniform(*xrange, nx)
    xdata = np.sort(xdata)
    
    # Truth
    ydata = fun(xdata)
    plt.plot(xdata, ydata)
    
    # Scatter
    sig = frac_sigx * np.std(ydata)
    print(sig)
    ydata = ydata + np.random.normal(size=nx) * sig
    plot_kwargs = {'fmt':'.', 'markersize':1, 'zorder':-1,
                 'capsize':1, 'elinewidth':0.5, 'color':'k', 'alpha':1}
    plt.errorbar(xdata,
                ydata,
                yerr=sig,
                **plot_kwargs)
                
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(name + ': ' + r'$%s$'%sympy.latex(expr))
    plt.tight_layout()
    
    plt.savefig(f'../figs/{name}.png')
    plt.clf()
    
    np.savetxt(f'../data/{name}.txt', np.vstack([xdata, ydata]).transpose())

    return

make_data('nguyen_6',
        'sin(x) + sin(x + x^2)',
        [-1, 1],
)
make_data('nguyen_8',
        'sqrt(x)',
        [0, 4],
)
make_data('korns_1',
        '1.57 + 2.43 * x',
        [-50, 50],
)
make_data('korns_4',
        '-2.3 + 0.13 * sin(x)',
        [-50, 50],
)
make_data('korns_6',
        '1.3 + 0.13 * sqrt(x)',
        [0, 50],
)
make_data('korns_7',
        '213.80940889 * (1 - exp(-0.54723748542 * x))',
        [0, 50],
)
make_data('korns_11',
        '6.87 + 11 * cos(2.73 * x^3)',
        [0, 50],
)
make_data('keijzer_1',
        '0.3 * x * sin( 2 * pi * x)',
        [-1, 1],
)
