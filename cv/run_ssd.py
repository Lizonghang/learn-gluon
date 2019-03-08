import os
import time
import gluonbook as gb
import mxnet as mx
from mxnet import gluon, image, autograd, contrib, init, nd
from mxnet.gluon import utils as gutils, loss as gloss, nn


def load_data_pikachu(batch_size, edge_size):
    data_dir = "../data/pikachu"
    train_iter = image.ImageDetIter(path_imgrec=os.path.join(data_dir, "train.rec"),
                                    path_imgidx=os.path.join(data_dir, "train.idx"),
                                    batch_size=batch_size,
                                    data_shape=(3, edge_size, edge_size),
                                    shuffle=True,
                                    rand_crop=1,
                                    min_object_covered=0.95,
                                    max_attempts=200)
    val_iter = image.ImageDetIter(path_imgrec=os.path.join(data_dir, "val.rec"),
                                  batch_size=batch_size,
                                  data_shape=(3, edge_size, edge_size),
                                  shuffle=False)
    return train_iter, val_iter


def class_predictor(num_anchors, num_classes):
    return nn.Conv2D(channels=num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_anchors):
    return nn.Conv2D(channels=num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    block.initialize()
    return block(x)


def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()


def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation("relu"))
    blk.add(nn.MaxPool2D(2))
    return blk


def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk


def blk_forward(X, blk, sizes, ratios, class_predictor, bbox_predictor):
    Y = blk(X)
    anchors = contrib.nd.MultiBoxPrior(Y, sizes=sizes, ratios=ratios)
    class_preds = class_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, class_preds, bbox_preds


class TinySSD(nn.Block):

    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            setattr(self, "blk_%d" % i, get_blk(i))
            setattr(self, "class_%d" % i, class_predictor(num_anchors, num_classes))
            setattr(self, "bbox_%d" % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors = [None] * 5
        class_preds = [None] * 5
        bbox_preds = [None] * 5
        for i in range(5):
            X, anchors[i], class_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, "blk_%d" % i), sizes[i], ratios[i],
                getattr(self, "class_%d" % i), getattr(self, "bbox_%d" % i))
        return (nd.concat(*anchors, dim=1),
                concat_preds(class_preds).reshape((0, -1, self.num_classes+1)),
                concat_preds(bbox_preds))


def calc_loss(class_preds, class_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = class_loss(class_preds, class_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox


def class_eval(class_preds, class_labels):
    return (class_preds.argmax(axis=-1) == class_labels).mean().asscalar()


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().mean().asscalar()


def predict(X):
    anchors, class_preds, bbox_preds = net(X.as_in_context(ctx))
    class_probs = class_preds.softmax().transpose((0, 2, 1))
    output = contrib.nd.MultiBoxDetection(class_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    return output[0, idx]


def display(img, output, threshold):
    fig = gb.plt.imshow(img.asnumpy())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        gb.show_bboxes(fig.axes, bbox, "%.2f" % score, "w")
    return fig


if __name__ == "__main__":
    lr = 0.2
    batch_size = 32
    wd = 5e-4
    edge_size = 256
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    ctx = mx.gpu(0)

    train_iter, val_iter = load_data_pikachu(batch_size, edge_size)
    train_iter.reshape(label_shape=(3, 5))

    net = TinySSD(num_classes=1)
    net.initialize(init=init.Xavier(), ctx=ctx)
    class_loss = gloss.SoftmaxCrossEntropyLoss()
    bbox_loss = gloss.L1Loss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr, "wd": wd})

    for epoch in range(20):
        acc, mae = 0, 0
        train_iter.reset()
        start = time.time()
        for i, batch in enumerate(train_iter):
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                anchors, class_preds, bbox_preds = net(X)
                bbox_labels, bbox_masks, class_labels = contrib.nd.MultiBoxTarget(
                    anchors, Y, class_preds.transpose((0, 2, 1)))
                l = calc_loss(class_preds, class_labels, bbox_preds, bbox_labels, bbox_masks)
            l.backward()
            trainer.step(batch_size)
            acc += class_eval(class_preds, class_labels)
            mae += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
        print("epoch %2d, class acc %.4f, bbox mae %.4f, time %.1f sec." % (
            epoch+1, acc/(i+1), mae/(i+1), time.time()-start))

    img = image.imread("../data/pikachu.jpg")
    feature = image.imresize(img, 256, 256).astype("float32")
    X = feature.transpose((2, 0, 1)).expand_dims(axis=0)
    output = predict(X)
    fig = display(img, output, threshold=0.3)
    # gb.plt.savefig("output.png")
    gb.plt.show()
