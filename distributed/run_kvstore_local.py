import mxnet as mx
from mxnet import kv, nd, gpu

if __name__ == "__main__":
    # MXNet provides a key-value store to synchronize data among devices.
    # The following code initializes an ndarray associated with the key “weight” on a key-value store.
    print('running kvstore local')
    kv_local = kv.create("local")
    SHAPE = (2, 3)
    x = nd.random.uniform(shape=SHAPE)
    kv_local.init("params", x)
    print('=== init "params" ==={}'.format(x))
    # After initialization, we can pull the value to multiple devices.
    NUM_GPUS = 2
    ctx = [gpu(i) for i in range(NUM_GPUS)]
    y = [nd.zeros(shape=SHAPE, ctx=c) for c in ctx]
    kv_local.pull("params", out=y)
    print('=== pull "params" to {} ===\n{}'.format(ctx, y))
    # We can also push new data value into the store.
    # It will first sum the data on the same key and then overwrite the current value.
    z = [nd.ones(shape=SHAPE, ctx=c) for c in ctx]
    kv_local.push("params", z)
    print('=== push to "params" ===\n{}'.format(z))
    kv_local.pull("params", out=y)
    print('=== pull "params" ===\n{}'.format(y))
    # With push and pull we can define the allreduce function by
    # def allreduce(data, data_name, store):
    #     store.push(data_name, data)
    #     store.pull(data_name, out=data)

    # For each push command, KVStore applies the pushed value to the value stored by an updater.
    # The default updater is ASSIGN. You can replace the default to control how data is merged.
    def updater(key, input, stored):
        print("update on key: {}".format(key))
        stored += input
    kv_local._set_updater(updater)
    z = mx.nd.ones(shape=SHAPE)
    kv_local.push("params", z)
    print('=== push to "params" ===\n{}'.format(z))
    kv_local.pull("params", out=y)
    print('=== pull "params" ===\n{}'.format(y))
