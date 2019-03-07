import gluonbook as gb
import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import os
import time


def _get(shape, ctx=mx.cpu()):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)


def lenet(X, params):
    h1_conv = nd.Convolution(data=X, weight=params[0], bias=params[1], kernel=(3, 3), num_filter=20)
    h1_act = nd.relu(data=h1_conv)
    h1_pool = nd.Pooling(data=h1_act, pool_type="avg", kernel=(2, 2), stride=(2, 2))
    h2_conv = nd.Convolution(data=h1_pool, weight=params[2], bias=params[3], kernel=(5, 5), num_filter=50)
    h2_act = nd.relu(data=h2_conv)
    h2_pool = nd.Pooling(data=h2_act, pool_type="avg", kernel=(2, 2), stride=(2, 2))
    h2_flatten = nd.Flatten(data=h2_pool)
    h3_fc = nd.dot(h2_flatten, params[4]) + params[5]
    h3_act = nd.relu(data=h3_fc)
    h4_fc = nd.dot(h3_act, params[6]) + params[7]
    return h4_fc


def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params


def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])


def split_and_load(data, ctx):
    n = data.shape[0]
    k = len(ctx)
    m = n // k
    assert m * k == n, "examples is not divided by # devices."
    return [data[i*m: (i+1)*m].as_in_context(ctx[i]) for i in range(k)]


def train_batch(X, y, gpu_params, ctx, lr):
    gpu_Xs = split_and_load(X, ctx)
    gpu_ys = split_and_load(y, ctx)
    with autograd.record():
        ls = [loss(lenet(gpu_X, gpu_W), gpu_y)
              for gpu_X, gpu_y, gpu_W in zip(gpu_Xs, gpu_ys, gpu_params)]
    for l in ls:
        l.backward()
    for i in range(len(gpu_params[0])):
        allreduce([gpu_params[c][i].grad for c in range(len(ctx))])
    for params in gpu_params:
        gb.sgd(params, lr, X.shape[0])


def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size, root="../data/fashion-mnist")
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print("running on:", ctx)
    gpu_params = [get_params(params, c) for c in ctx]
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            train_batch(X, y, gpu_params, ctx, lr)
            nd.waitall()
        train_time = time.time() - start

        def net(x):
            return lenet(x, gpu_params[0])

        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print("epoch %d, time: %.1f sec, test acc: %.2f" % (epoch+1, train_time, test_acc))


if __name__ == "__main__":
    W1 = _get((20, 1, 3, 3))
    b1 = nd.zeros(20)
    W2 = _get((50, 20, 5, 5))
    b2 = nd.zeros(50)
    W3 = _get((800, 128))
    b3 = nd.zeros(128)
    W4 = _get((128, 10))
    b4 = nd.zeros(10)
    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    loss = gloss.SoftmaxCrossEntropyLoss()
    train(num_gpus=2, batch_size=1024, lr=0.2)
