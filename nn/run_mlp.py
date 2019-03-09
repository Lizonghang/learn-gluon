from mxnet import autograd, nd, init, gluon
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata
import gluonbook as gb
import matplotlib.pyplot as plt
import os


def softmax(X):
    X_exp = X.exp()
    return X_exp / X_exp.sum(axis=1, keepdims=True)


def simple_mlp(X, W, b):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


def simple_mlp_gluon():
    net = nn.Sequential()
    net.add(nn.Dense(10))
    return net


def cross_entropy_loss(y_, y):
    return -nd.pick(y_, y).log()


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# version 1
# def calc_acc(data_iter, net, W, b):
# version 2
def calc_acc(data_iter, net):
    acc = 0.
    for X, y in data_iter:
        # version 1
        # y_ = net(X, W, b)
        # version 2
        y_ = net(X)
        acc += (y_.argmax(axis=1) == y.astype("float32")).mean().asscalar()
    return acc / len(data_iter)


if __name__ == "__main__":
    batch_size = 32
    num_inputs = 784
    num_outputs = 10
    num_epochs = 5
    learning_rate = 0.01

    # version 1
    # num_workers = 1
    # root = os.path.join(os.getcwd(), "data", "fashion-mnist")
    # train_data = gdata.vision.FashionMNIST(root, train=True)
    # val_data = gdata.vision.FashionMNIST(root, train=False)
    # transformer = gdata.vision.transforms.ToTensor()
    # train_iter = gdata.DataLoader(train_data.transform_first(transformer),
    #                               batch_size=batch_size,
    #                               shuffle=True,
    #                               num_workers=num_workers)
    # val_iter = gdata.DataLoader(val_data.transform_first(transformer),
    #                             batch_size=batch_size,
    #                             shuffle=False,
    #                             num_workers=num_workers)
    # version 2
    # simplified data loader api
    root = os.path.join(os.getcwd(), "data", "fashion-mnist")
    train_iter, val_iter = gb.load_data_fashion_mnist(batch_size, root=root)

    # version 1
    # W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
    # b = nd.zeros(shape=(num_outputs,))
    # W.attach_grad()
    # b.attach_grad()
    # net = simple_mlp
    # version 2
    net = simple_mlp_gluon()
    net.initialize(init.Normal(sigma=0.01))

    # version 1
    # loss = cross_entropy_loss
    # version 2
    loss = gloss.SoftmaxCrossEntropyLoss()

    # version 1
    # trainer = sgd
    # version 2
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer="sgd",
                            optimizer_params={"learning_rate": learning_rate})

    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # version 1
                # l = loss(net(X, W, b), y)
                # version 2
                l = loss(net(X), y)
            l.backward()
            # version 1
            # trainer([W, b], learning_rate, batch_size)
            # version 2
            trainer.step(1)
        # version 1
        # train_acc = calc_acc(train_iter, net, W, b)
        # val_acc = calc_acc(val_iter, net, W, b)
        # version 2
        train_acc = calc_acc(train_iter, net)
        val_acc = calc_acc(val_iter, net)
        print('epoch %d, train acc %.3f, test acc %.3f' % (epoch + 1, train_acc, val_acc))

    X, y = None, None
    for X, y in val_iter:
        break
    true_labels = gb.get_fashion_mnist_labels(y.asnumpy())
    # version 1
    # pred_labels = gb.get_fashion_mnist_labels(net(X, W, b).argmax(axis=1).asnumpy())
    # version 2
    pred_labels = gb.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
    titles = [true + "\n" + pred for true, pred in zip(true_labels, pred_labels)]
    gb.show_fashion_mnist(X[:9], titles[:9])
    plt.show()
