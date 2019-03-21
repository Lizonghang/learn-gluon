import sys
import time
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import gluon, image, nd, init, autograd, kv
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import utils as gutils
from mxnet.gluon import model_zoo, nn


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


class VOCSegDataset(gdata.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label):
        self.rgb_mean = nd.array([0.485, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label
        print("read " + str(len(self.features)) + " examples")

    def normalize_image(self, img):
        return (img.astype("float32") / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (img.shape[0] >= self.crop_size[0] and img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return feature.transpose((2, 0, 1)), voc_label_indices(label, self.colormap2label)

    def __len__(self):
        return len(self.features)


def read_voc_images(root, is_train):
    txt_fname = "%s/ImageSets/Segmentation/%s" % (root, "train.txt" if is_train else "val.txt")
    with open(txt_fname, "r") as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.imread("%s/JPEGImages/%s.jpg" % (root, fname))
        labels[i] = image.imread("%s/SegmentationClass/%s.png" % (root, fname))
    return features, labels


def voc_label_indices(colormap, colormap2label):
    colormap = colormap.astype("int32")
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, height, width):
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label


def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype="float32")
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
                 time.time() - start))
        net.save_parameters("/home/lizh/learn-gluon/models/fcn.params")


def predict(img, ctx=mx.cpu()):
    X = test_iter._dataset.normalize_image(img)
    X = X.transpose((2, 0, 1)).expand_dims(axis=0)
    pred = nd.argmax(net(X.as_in_context(ctx)), axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))


def evaluate_accuracy(data_iter, net, ctx=mx.cpu()):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def label2image(pred, ctx=mx.cpu()):
    colormap = nd.array(VOC_COLORMAP, ctx=ctx, dtype="uint8")
    X = pred.astype("int32")
    return colormap[X, :]


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


if __name__ == "__main__":
    """
    # On scheduler machine
    DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=3 DMLC_NUM_WORKER=6 \
        python ~/learn-gluon/distributed/run_kvstore_dist.py &

    # On other machines
    DMLC_ROLE=server DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=3 DMLC_NUM_WORKER=6 \
        python ~/learn-gluon/distributed/run_kvstore_dist.py &
    DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=3 DMLC_NUM_WORKER=6 \
        python ~/learn-gluon/distributed/run_kvstore_dist.py -g 0 &
    DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=3 DMLC_NUM_WORKER=6 \
        python ~/learn-gluon/distributed/run_kvstore_dist.py -g 1 &
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, default=1)
    parser.add_argument("-p", "--load-parameters", type=int, default=0)
    parser.add_argument("-n", "--num-epochs", type=int, default=100)
    parser.add_argument("-l", "--learning-rate", type=float, default=0.01)
    parser.add_argument("-w", "--weight-decay", type=float, default=0.001)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    args, unknown = parser.parse_known_args()

    crop_size = (320, 480)
    num_classes = 21
    ctx = [mx.gpu(args.gpu)]

    colormap2label = nd.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256+colormap[1]) * 256 + colormap[2]] = i

    kvstore = kv.create("dist_async")
    num_workers = 0 if sys.platform.startswith("win32") else 4
    voc_train = VOCSegDataset(True, crop_size, "/home/lizh/learn-gluon/data/VOC2012", colormap2label)
    voc_test = VOCSegDataset(False, crop_size, "/home/lizh/learn-gluon/data/VOC2012", colormap2label)
    train_iter = gdata.DataLoader(voc_train, args.batch_size, shuffle=True, last_batch="discard", num_workers=num_workers)
    test_iter = gdata.DataLoader(voc_test, args.batch_size, last_batch="discard", num_workers=num_workers)

    pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True, root="/home/lizh/learn-gluon/models")
    net = nn.HybridSequential()
    for layer in pretrained_net.features[:-2]:
        net.add(layer)
    net.add(nn.Conv2D(num_classes, kernel_size=1),
            nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))
    net[-2].initialize(init.Xavier())
    net[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 64)))
    net.collect_params().reset_ctx(ctx)

    if args.train:
        if args.load_parameters:
            net.load_parameters("/home/lizh/learn-gluon/models/fcn.params")
        num_epochs = args.num_epochs
        lr = args.learning_rate
        wd = args.weight_decay
        loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
        trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr, "wd": wd}, kvstore=kvstore)
        train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
    else:
        ctx = mx.gpu(0)
        net.collect_params().reset_ctx(ctx)
        net.load_parameters("/home/lizh/learn-gluon/models/fcn.params")
        test_images, test_labels = read_voc_images("/home/lizh/learn-gluon/data/VOC2012", is_train=False)
        n = 4
        imgs = []
        for i in range(n):
            crop_rect = (0, 0, 480, 320)
            X = image.fixed_crop(test_images[i], *crop_rect)
            pred = label2image(predict(X, ctx), ctx)
            imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
        show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)
        plt.savefig("result.png")
