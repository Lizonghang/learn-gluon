from mxnet import autograd, init, gluon
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
import mxnet as mx
import gluonbook as gb
import os


def get_lenet():
    net = nn.Sequential()
    net.add(
        nn.Conv2D(channels=16, kernel_size=3, activation="relu"),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=32, kernel_size=3, activation="relu"),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(32, activation="relu"),
        nn.Dense(16, activation="relu"),
        nn.Dense(10))
    return net


if __name__ == "__main__":
    batch_size = 256
    learning_rate = 0.5
    weight_decay = 0.005
    num_epochs = 5
    root = os.path.join(os.getcwd(), "data", "fashion-mnist")
    train_iter, val_iter = gb.load_data_fashion_mnist(batch_size, root=root)
    net = get_lenet()
    ctx = mx.cpu()
    net.initialize(init.Xavier(), ctx=ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer="sgd",
                            optimizer_params={
                                "learning_rate": learning_rate,
                                "wd": weight_decay
                            })
    for epoch in xrange(num_epochs):
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_ = net(X)
                l = loss(y_, y)
            l.backward()
            trainer.step(batch_size)
        acc = 0.0
        for X, y in val_iter:
            y_ = net(X)
            acc += (y_.argmax(axis=1) == y.astype("float32")).mean().asscalar()
        print 'epoch %d, acc %.3f' % (epoch+1, acc / len(val_iter))
