# coding=utf-8
import mxnet as mx
import gluonbook as gb
from mxnet import nd
from mxnet.gluon import rnn


corpus_indices, char_to_idx, idx_to_char, vocab_size = gb.load_data_jay_lyrics()
num_inputs = vocab_size
num_outputs = vocab_size
num_hiddens = 256
num_epochs = 160
num_steps = 35
batch_size = 32
lr = 1e2
clipping_theta = 1e-2
ctx = mx.cpu()


def get_param():
    def _get(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    # Update Gate Parameters
    W_xz = _get((num_inputs, num_hiddens))
    W_hz = _get((num_hiddens, num_hiddens))
    b_z = nd.zeros(num_hiddens, ctx=ctx)
    # Reset Gate Parameters
    W_xr = _get((num_inputs, num_hiddens))
    W_hr = _get((num_hiddens, num_hiddens))
    b_r = nd.zeros(num_hiddens, ctx=ctx)
    # Candidate Hidden State Parameters
    W_xh = _get((num_inputs, num_hiddens))
    W_hh = _get((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # Output Parameters
    W_hq = _get((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # Attach Gradient Space
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def init_gru_state(batch_size, num_hiddens, ctx):
    return nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)
        H_ = nd.tanh(nd.dot(X, W_xh) + R * nd.dot(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H


if __name__ == "__main__":
    pred_period = 1
    pred_len = 50
    prefixes = [u'分开', u'不分开']
    # version 1
    gb.train_and_predict_rnn(gru, get_param, init_gru_state, num_hiddens, vocab_size, ctx, corpus_indices,
                             idx_to_char, char_to_idx, False, num_epochs, num_steps, lr, clipping_theta,
                             batch_size, pred_period, pred_len, prefixes)
    # version 2
    # gru_layer = rnn.GRU(num_hiddens)
    # model = gb.RNNModel(gru_layer, vocab_size)
    # gb.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx,
    #                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
    #                                prefixes)
