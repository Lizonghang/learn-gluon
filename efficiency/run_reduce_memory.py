from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
import os
import subprocess


def data_iter():
    num_batches = 100
    batch_size = 1024
    for i in range(num_batches):
        X = nd.random.normal(shape=(batch_size, 512))
        y = nd.ones((batch_size,))
        yield X, y


def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(64, activation="relu"),
            nn.Dense(32, activation="relu"),
            nn.Dense(1))
    return net


def get_memory():
    res = subprocess.check_output(["ps", "u", "-p", str(os.getpid())])
    return int(str(res).split()[15]) / 1e3


if __name__ == "__main__":
    net = get_net()
    net.initialize()
    for X, y in data_iter():
        break
    net(X).wait_to_read()
    loss = gloss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.01})

    l_sum = 0
    memory = get_memory()
    for X, y in data_iter():
        with autograd.record():
            l = loss(net(X), y)
        # note
        l_sum += l.mean().asscalar()
        l.backward()
        trainer.step(X.shape[0])
    nd.waitall()
    print 'increased memory: %f MB' % (get_memory() - memory)

    memory = get_memory()
    for X, y in data_iter():
        with autograd.record():
            l = loss(y, net(X))
        l.backward()
        trainer.step(X.shape[0])
    nd.waitall()
    print 'increased memory: %f MB' % (get_memory() - memory)
