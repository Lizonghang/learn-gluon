import gluonbook as gb
from mxnet import nd

features, labels = gb.get_data_ch7()


def init_adadelta_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    delta_x_w = nd.zeros((features.shape[1], 1))
    delta_x_b = nd.zeros(1)
    return (s_w, delta_x_w), (s_b, delta_x_b)


def adadelta(params, states, hyperparams):
    eps = 1e-6
    for p, (s, delta_x) in zip(params, states):
        s[:] = hyperparams["rho"] * s + (1 - hyperparams["rho"]) * p.grad.square()
        g = ((delta_x + eps) / (s + eps)).sqrt() * p.grad
        p[:] = p - g
        delta_x[:] = hyperparams["rho"] * delta_x + (1 - hyperparams["rho"]) * g * g


if __name__ == "__main__":
    # gb.train_ch7(adadelta, init_adadelta_states(), {"rho": 0.9}, features, labels)
    gb.train_gluon_ch7("adadelta", {"rho": 0.9}, features, labels)
    gb.plt.show()
