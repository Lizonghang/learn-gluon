import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, utils as gutils, nn
import time


def resnet18(num_classes):
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(gb.Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(gb.Residual(num_channels))
        return blk

    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(),
            nn.Activation("relu"))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(),
            nn.Dense(num_classes))
    return net


def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size, root="../data/fashion-mnist")
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print("running on:", ctx)
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.split_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y)
                      for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        train_time = time.time() - start
        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print("epoch %d, time: %.1f sec, test acc: %.2f" % (epoch+1, train_time, test_acc))


if __name__ == "__main__":
    net = resnet18(10)
    train(num_gpus=2, batch_size=512, lr=0.1)
