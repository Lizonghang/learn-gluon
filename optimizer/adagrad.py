import gluonbook as gb
from mxnet import nd

features, labels = gb.get_data_ch7()


def init_adagrad_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return s_w, s_b


def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] = s + p.grad.square()
        p[:] = p - (hyperparams["learning_rate"] / (s + eps).sqrt()) * p.grad


if __name__ == "__main__":
    # version 1
    # gb.train_ch7(adagrad, init_adagrad_states(), {"learning_rate": 0.1}, features, labels)
    # version 2
    gb.train_gluon_ch7("adagrad", {"learning_rate": 0.1}, features, labels)
    gb.plt.show()
