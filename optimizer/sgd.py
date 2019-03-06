import time
import gluonbook as gb
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import data as gdata

features, labels = gb.get_data_ch7()


def sgd(params, states, hyperparams):
    for param in params:
        param[:] -= hyperparams['learning_rate'] * param.grad


def train(trainer_fn, states, hyperparams, features, labels, batch_size, num_epochs):
    # version 1
    # w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    # b = nd.zeros(1)
    # w.attach_grad()
    # b.attach_grad()
    # version 2
    w_ = gluon.Parameter("w", shape=(features.shape[1], 1))
    w_.initialize()
    w_.set_data(nd.random.normal(scale=0.01, shape=(features.shape[1], 1)))
    b_ = gluon.Parameter("b", shape=(1,))
    b_.initialize()
    b_.set_data(nd.zeros(1))

    net = gb.linreg
    loss = gb.squared_loss
    trainer = gluon.Trainer([w_, b_], "sgd", hyperparams)

    def eval_loss():
        # version 1
        # return loss(net(features, w, b), labels).mean().asscalar()
        # version 2
        return loss(net(features, w_.data(), b_.data()), labels).mean().asscalar()

    ls = [eval_loss()]
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        start = time.time()
        for i, (X, y) in enumerate(data_iter):
            with autograd.record():
                # version 1
                # l = loss(net(X, w, b), y).mean()
                # version 2
                l = loss(net(X, w_.data(), b_.data()), y).mean()
            l.backward()
            # version 1
            # trainer_fn([w, b], states, hyperparams)
            # version 2
            trainer.step(1)
        ls.append(eval_loss())
        print 'epoch %d, loss: %f, %f sec per epoch' % (epoch+1, ls[-1], time.time()-start)

    gb.set_figsize()
    gb.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    gb.plt.xlabel('epoch')
    gb.plt.ylabel('loss')


def train_sgd(lr, batch_size, num_epochs):
    train(sgd, None, {'learning_rate': lr}, features, labels, batch_size, num_epochs)

if __name__ == "__main__":
    lr = 0.01
    batch_size = 256
    num_epochs = 100
    train_sgd(lr, batch_size, num_epochs)
    gb.plt.show()
