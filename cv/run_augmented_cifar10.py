import sys
import time
import gluonbook as gb
import mxnet as mx
from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata, utils as gutils, loss as gloss


def load_cifar10(is_train, augs, batch_size):
    return gdata.DataLoader(
        dataset=gdata.vision.CIFAR10("../data/cifar10", train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx),
            features.shape[0])


def evaluate_accuracy(data_iter, net, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype("float32")
            acc += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()
    return acc.asscalar() / n


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print("training on:", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum = 0.0
        train_acc_sum = 0.0
        n = 0.0
        m = 0.0
        start = time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                ys_ = [net(X) for X in Xs]
                ls = [loss(y_, y) for y_, y in zip(ys_, ys)]
            for l in ls:
                l.backward()
            train_acc_sum += sum([(y_.argmax(axis=1) == y).sum().asscalar()
                                  for y_, y in zip(ys_, ys)])
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
              % (epoch+1, train_l_sum/n, train_acc_sum/m, test_acc, time.time()-start))


def train_with_data_aug(train_augs, test_augs, num_gpus, lr=0.001):
    batch_size = 256
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    net = gb.resnet18(10)
    net.initialize(init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=10)


if __name__ == "__main__":
    train_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.RandomFlipLeftRight(),
        gdata.vision.transforms.ToTensor()])
    test_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.ToTensor()])
    num_workers = 0 if sys.platform.startswith("win32") else 4
    train_with_data_aug(train_augs, test_augs, 2)
