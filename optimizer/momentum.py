import gluonbook as gb
from mxnet import nd

features, labels = gb.get_data_ch7()


def init_momentum_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    return v_w, v_b


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams["momentum"] * v + hyperparams["learning_rate"] * p.grad
        p[:] = p - v


if __name__ == "__main__":
    # version 1
    # gb.train_ch7(sgd_momentum, init_momentum_states(), {"learning_rate": 0.02, "momentum": 0.5}, features, labels)
    # version 2
    gb.train_gluon_ch7("sgd", {"learning_rate": 0.02, "momentum": 0.5}, features, labels)
    gb.plt.show()
