from mxnet import nd
from mxnet.gluon import nn


class Conv2D(nn.Block):

    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get("weight", shape=kernel_size)
        self.bias = self.params.get("bias", shape=(1,))

    def forward(self, x):
        return self._corr2d_multi_in_out(x, self.weight.data()) + self.bias.data()

    def _corr2d(self, X, K):
        h, w = K.shape
        Y = nd.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
        for i in xrange(Y.shape[0]):
            for j in xrange(Y.shape[1]):
                Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
        return Y

    def _corr2d_multi_in(self, X, K):
        return nd.add_n(*[self._corr2d(x, k) for x, k in zip(X, K)])

    def _corr2d_multi_in_out(self, X, K):
        return nd.stack(*[self._corr2d_multi_in(X, k) for k in K])


if __name__ == "__main__":
    X = nd.random.normal(scale=0.1, shape=(3, 28, 28))
    net = Conv2D((10, 3, 5, 5))
    net.initialize()
    y_ = net(X)
    assert y_.shape == (10, 24, 24)
