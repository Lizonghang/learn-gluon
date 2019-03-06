import gluonbook as gb
from mxnet import nd

features, labels = gb.get_data_ch7()


def init_adam_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (v_w, s_w), (v_b, s_b)


def adam(params, states, hyperparams):
    eps = 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = hyperparams["beta1"] * v + (1 - hyperparams["beta1"]) * p.grad
        s[:] = hyperparams["beta2"] * s + (1 - hyperparams["beta2"]) * p.grad.square()
        v_bias_corr = v / (1 - hyperparams["beta1"] ** hyperparams["t"])
        s_bias_corr = s / (1 - hyperparams["beta2"] ** hyperparams["t"])
        p[:] = p - hyperparams["learning_rate"] * v_bias_corr / (s_bias_corr + eps).sqrt()
    hyperparams["t"] += 1


if __name__ == "__main__":
    # gb.train_ch7(adam, init_adam_states(), {"learning_rate": 0.01, "beta1": 0.9, "beta2": 0.999, "t": 1},
    #              features, labels)
    gb.train_gluon_ch7("adam", {"learning_rate": 0.01, "beta1": 0.9, "beta2": 0.999}, features, labels)
    gb.plt.show()
