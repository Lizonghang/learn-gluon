import gluonbook as gb
from mxnet import nd

features, labels = gb.get_data_ch7()


def init_rmsprop_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return s_w, s_b


def rmsprop(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] = hyperparams["gamma"] * s + (1 - hyperparams["gamma"]) * p.grad.square()
        p[:] = p - (hyperparams["learning_rate"] / (s + eps).sqrt()) * p.grad


if __name__ == "__main__":
    # gb.train_ch7(rmsprop, init_rmsprop_states(), {"learning_rate": 0.01, "gamma": 0.9}, features, labels)
    gb.train_gluon_ch7("rmsprop", {"learning_rate": 0.01, "gamma1": 0.9}, features, labels)
    gb.plt.show()
