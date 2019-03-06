# coding=utf-8
import math
import time
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn

num_hiddens = 256
num_epochs = 200
num_steps = 35
batch_size = 32
lr = 1e2
clipping_theta = 1e-2
ctx = mx.cpu()
corpus_indices, char_to_idx, idx_to_char, vocab_size = gb.load_data_jay_lyrics()


class RNNModel(nn.Block):

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char, char_to_idx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        Y, state = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(init=init.Normal(0.01), ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(), "sgd", {"learning_rate": lr})
    for epoch in range(num_epochs):
        loss_sum = 0.0
        start = time.time()
        data_iter = gb.data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for t, (X, Y) in enumerate(data_iter):
            for s in state:
                s.detach()
            with autograd.record():
                output, state = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            params = [p.data() for p in model.collect_params().values()]
            gb.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)
            loss_sum += l.asscalar()

        if (epoch+1) % pred_period == 0:
            print 'epoch %d, perplexity %f, time %.2f sec' % (epoch+1, math.exp(loss_sum/(t+1)), time.time()-start)
            for prefix in prefixes:
                print '-', predict_rnn_gluon(prefix, pred_len, model, vocab_size, ctx, idx_to_char, char_to_idx)


if __name__ == "__main__":
    pred_period = 1
    pred_len = 50
    prefixes = [u'分开', u'不分开']
    rnn_layer = rnn.GRU(num_hiddens)
    model = RNNModel(rnn_layer, vocab_size)
    train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)