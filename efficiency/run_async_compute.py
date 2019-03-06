from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss
import os
import subprocess
import time


class Benchmark:

    def __init__(self, prefix=None):
        self.prefix = prefix + " " if prefix else ""

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print '%stime: %.4f sec' % (self.prefix, time.time() - self.start)


if __name__ == "__main__":
    with Benchmark("Workloads are queued."):
        x = nd.random.uniform(shape=(2000, 2000))
        y = nd.dot(x, x).sum()

    with Benchmark("Workloads are finished."):
        print "sum =", y.asscalar()

    print

    with Benchmark("Use wait_to_read()."):
        y = nd.dot(x, x).sum()
        y.wait_to_read()

    with Benchmark("Use nd.waitall()."):
        y = nd.dot(x, x).sum()
        nd.waitall()

    with Benchmark("Use asnumpy()."):
        y = nd.dot(x, x).sum()
        y.asnumpy()

    with Benchmark("Use asscalar()."):
        y = nd.dot(x, x).sum()
        y.asscalar()
