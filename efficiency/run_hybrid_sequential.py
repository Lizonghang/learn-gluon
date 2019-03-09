from mxnet import nd, sym
from mxnet.gluon import nn
import time


def get_net():
    net = nn.HybridSequential()
    net.add(nn.Dense(256, activation="relu"),
            nn.Dense(128, activation="relu"),
            nn.Dense(2))
    return net


def benchmark(net, X):
    start = time.time()
    for i in range(1000):
        _ = net(X)
    nd.waitall()
    return time.time() - start


if __name__ == "__main__":
    X = nd.random.normal(scale=0.01, shape=(1, 512))
    net = get_net()
    net.initialize()
    print('before hybridizing: %.4f' % benchmark(net, X))
    net.hybridize()
    print('after hybridizing: %.4f' % benchmark(net, X))
    net.export('sym-hybrid-sequential')
