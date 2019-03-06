# coding=utf-8
import gluonbook as gb
import mxnet as mx
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


def get_params():
    def _get(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    # Input Gate Parameters
    W_xi = _get((num_inputs, num_hiddens))
    W_hi = _get((num_hiddens, num_hiddens))
    b_i = nd.zeros(num_hiddens, ctx=ctx)
    # Forget Gate Parameters
    W_xf = _get((num_inputs, num_hiddens))
    W_hf = _get((num_hiddens, num_hiddens))
    b_f = nd.zeros(num_hiddens, ctx=ctx)
    # Output Gate Parameters
    W_xo = _get((num_inputs, num_hiddens))
    W_ho = _get((num_hiddens, num_hiddens))
    b_o = nd.zeros(num_hiddens, ctx=ctx)
    # Candidate Cell Parameters
    W_xc = _get((num_inputs, num_hiddens))
    W_hc = _get((num_hiddens, num_hiddens))
    b_c = nd.zeros(num_hiddens, ctx=ctx)
    # Output Parameters
    W_hq = _get((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # Create Gradient Space
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def init_lstm_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),
            nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx))


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    H, C = state
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_ = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_
        H = O * nd.tanh(C)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)


if __name__ == "__main__":
    pred_period = 1
    pred_len = 50
    prefixes = [u'分开', u'不分开']
    # version 1
    gb.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens, vocab_size, ctx, corpus_indices,
                             idx_to_char, char_to_idx, False, num_epochs, num_steps, lr, clipping_theta, batch_size,
                             pred_period, pred_len, prefixes)
    # version 2
    # lstm_layer = rnn.LSTM(num_hiddens)
    # model = gb.RNNModel(lstm_layer, vocab_size)
    # gb.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx,
    #                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
    #                                prefixes)
