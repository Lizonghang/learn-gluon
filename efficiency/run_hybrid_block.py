from mxnet import nd
from mxnet.gluon import nn


class HybridNet(nn.HybridBlock):

    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # F: NDArray or Symbol
        # before calling net.hybridize(), type(F) is NDArray
        # after calling net.hybridize(), type(F) is Symbol
        print 'type(F) =', F
        x = F.relu(self.hidden(x))
        return self.output(x)


if __name__ == "__main__":
    X = nd.random.normal(shape=(1, 32))
    net = HybridNet()
    net.initialize()
    print 'before hybridize:',
    net(X)
    net.hybridize()
    print 'after hybridize:',
    net(X)
    print 'call again, no output'
    net(X)
