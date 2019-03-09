# coding=utf-8
import random
import time
import math
import os
import zipfile
import mxnet as mx
import gluonbook as gb
from mxnet import nd, autograd
from mxnet.gluon import loss as gloss

num_hiddens = 256
num_epochs = 200
num_steps = 35
batch_size = 32
lr = 1e2
clipping_theta = 1e-2
ctx = mx.cpu()

zip_file = os.path.join(os.getcwd(), "data/lyric/jaychou_lyrics.txt.zip")
with zipfile.ZipFile(zip_file) as zin:
    with zin.open("jaychou_lyrics.txt") as f:
        corpus_chars = f.read().decode('utf-8')
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
corpus_indices = [char_to_idx[char] for char in corpus_chars]


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # seq = list(range(30))
    # for X, Y in data_iter_random(seq, batch_size=2, num_steps=6):
    #     print('X: ', X, '\nY: ', Y)
    #     break
    #
    # Output:
    #     X:
    #     [[12. 13. 14. 15. 16. 17.]
    #      [ 0.  1.  2.  3.  4.  5.]]
    #     <NDArray 2x6 @cpu(0)>
    #     Y:
    #     [[13. 14. 15. 16. 17. 18.]
    #      [ 1.  2.  3.  4.  5.  6.]]
    #     <NDArray 2x6 @cpu(0)>
    num_examples = (len(corpus_indices)-1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    # seq = list(range(30))
    # for X, Y in data_iter_consecutive(seq, batch_size=2, num_steps=6):
    #     print('X: ', X, '\nY: ', Y)
    #
    # Output:
    #     X:
    #     [[ 0.  1.  2.  3.  4.  5.]
    #      [15. 16. 17. 18. 19. 20.]]
    #     <NDArray 2x6 @cpu(0)>
    #     Y:
    #     [[ 1.  2.  3.  4.  5.  6.]
    #      [16. 17. 18. 19. 20. 21.]]
    #     <NDArray 2x6 @cpu(0)>
    #     X:
    #     [[ 6.  7.  8.  9. 10. 11.]
    #      [21. 22. 23. 24. 25. 26.]]
    #     <NDArray 2x6 @cpu(0)>
    #     Y:
    #     [[ 7.  8.  9. 10. 11. 12.]
    #      [22. 23. 24. 25. 26. 27.]]
    #     <NDArray 2x6 @cpu(0)>
    #     ...
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[:batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i:i+num_steps]
        Y = indices[:, i+1:i+num_steps+1]
        yield X, Y


def to_onehot(X, size):
    # X = nd.arange(10).reshape((2, 5))
    # inputs = to_onehot(X, vocab_size)
    # Output:
    #     In[6]: X.T
    #     Out[6]:
    #
    #     [[0. 5.]
    #      [1. 6.]
    #      [2. 7.]
    #      [3. 8.]
    #      [4. 9.]]
    #     <NDArray 5x2 @cpu(0)>
    #
    #     In[15]: inputs
    #     Out[14]:
    #     [
    #      [[1. 0. 0. ... 0. 0. 0.]
    #       [0. 0. 0. ... 0. 0. 0.]]
    #      <NDArray 2x1027 @cpu(0)>,
    #      [[0. 1. 0. ... 0. 0. 0.]
    #       [0. 0. 0. ... 0. 0. 0.]]
    #      <NDArray 2x1027 @cpu(0)>,
    #      [[0. 0. 1. ... 0. 0. 0.]
    #       [0. 0. 0. ... 0. 0. 0.]]
    #      <NDArray 2x1027 @cpu(0)>,
    #      [[0. 0. 0. ... 0. 0. 0.]
    #       [0. 0. 0. ... 0. 0. 0.]]
    #      <NDArray 2x1027 @cpu(0)>,
    #      [[0. 0. 0. ... 0. 0. 0.]
    #       [0. 0. 0. ... 0. 0. 0.]]
    #      <NDArray 2x1027 @cpu(0)>]
    return [nd.one_hot(x, size) for x in X.T]


def get_params():
    num_inputs = vocab_size
    num_outputs = vocab_size

    def _get(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    W_xh = _get((num_inputs, num_hiddens))
    W_hh = _get((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    W_hq = _get((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)


def rnn(inputs, state, params):
    # X = nd.arange(10).reshape((2, 5))
    # state = init_rnn_state(2, 256, mx.cpu())
    # inputs = to_onehot(X, vocab_size)
    # params = get_params()
    # outputs, state_new = rnn(inputs, state, params)
    # print(outputs, state_new)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        Y, state = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta, ctx):
    norm = nd.array([0.0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta, batch_size,
                          pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        loss_sum = 0.0
        start = time.time()
        for t, (X, Y) in enumerate(data_iter):
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                outputs, state = rnn(inputs, state, params)
                outputs = nd.concat(*outputs, dim=0)
                y = Y.T.reshape((-1,))
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)
            gb.sgd(params, lr, 1)
            loss_sum += l.asscalar()

        if (epoch+1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch+1, math.exp(loss_sum/(t+1)), time.time()-start))
            for prefix in prefixes:
                print('-', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens,
                                       vocab_size, ctx, idx_to_char, char_to_idx))


if __name__ == "__main__":
    pred_period = 1
    pred_len = 50
    prefixes = [u'分开', u'不分开']
    train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, True, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes)
