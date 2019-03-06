import gluonbook as gb
import mxnet as mx
from mxnet import nd


def run(X):
    return [nd.dot(X, X) for _ in range(10)]


def copy_to_cpu(l):
    return [X.copyto(mx.cpu()) for X in l]


if __name__ == "__main__":
    X = nd.random.normal(shape=(6000, 6000), ctx=mx.gpu(0))

    with gb.Benchmark("Run on GPU."):
        l = run(X)
        nd.waitall()

    with gb.Benchmark("Copy to CPU."):
        copy_to_cpu(l)
        nd.waitall()

    with gb.Benchmark("Run and copy in parallel."):
        l = run(X)
        copy_to_cpu(l)
        nd.waitall()
