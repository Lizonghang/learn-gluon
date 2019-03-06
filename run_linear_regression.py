import random
from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn


def custom_data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


def linear_regression(X, w, b):
    return nd.dot(X, w) + b


def linear_regression_gluon():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    return net


def squared_loss(y_, y):
    loss = (y_ - y.reshape(y_.shape)) ** 2 / 2
    return loss.mean()


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


if __name__ == "__main__":
    num_inputs = 2
    num_examples = 1000
    batch_size = 10
    learning_rate = 0.01
    num_epochs = 10

    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    # version 1
    # w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    # b = nd.zeros(shape=(1,))
    # w.attach_grad()
    # b.attach_grad()
    # net = linear_regression
    # version 2
    net = linear_regression_gluon()
    net.initialize(init.Normal(sigma=0.01))

    # version 1
    # loss = squared_loss
    # version 2
    loss = gloss.L2Loss()

    # version 1
    # trainer = sgd
    # version 2
    trainer = gluon.Trainer(
        params=net.collect_params(),
        optimizer="sgd",
        optimizer_params={"learning_rate": learning_rate})

    for epoch in xrange(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                # version 1
                # l = loss(net(X, w, b), y)
                # version 2
                l = loss(net(X), y).mean()
            l.backward()
            # version 1
            # trainer([w, b], learning_rate, batch_size)
            # version 2
            # gradient will be normalized by `1/batch_size`, set batch_size to 1
            # if normalized loss manually with `loss = mean(loss)`.
            trainer.step(1)
        # version 1
        # train_loss = loss(net(features, w, b), labels).asscalar()
        # version 2
        train_loss = loss(net(features), labels).mean().asscalar()
        print 'epoch {}, loss {}'.format(epoch+1, train_loss)

    # version 2
    w = net[0].weight.data()
    b = net[0].bias.data()
    # version 1
    print
    print 'w {}, w_true {}'.format(w.reshape((num_inputs,)).asnumpy().tolist(), true_w)
    print 'b {}, b_true {}'.format(b.reshape((1,)).asnumpy().tolist(), true_b)
