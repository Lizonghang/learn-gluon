import numpy as np
import gluonbook as gb


def gd(lr, num_epochs, x):
    trace = [x]
    for i in range(num_epochs):
        # func: f(x)=x^2, f'(x)=2x
        x -= lr * 2 * x
        trace.append(x)
    return trace


def plot_trace(trace):
    n = max(abs(min(trace)), abs(max(trace)), 10)
    f_line = np.arange(-n, n, 0.1)
    gb.set_figsize()
    gb.plt.plot(f_line, [x * x for x in f_line])
    gb.plt.plot(trace, [x * x for x in trace], '-ro')
    gb.plt.xlabel("x")
    gb.plt.ylabel("f(x)=x^2")


if __name__ == "__main__":
    num_epochs = 10
    # lr = 1.1
    lr = 0.05
    init_value = 10.
    trace = gd(lr, num_epochs, init_value)
    plot_trace(trace)
    gb.plt.show()
