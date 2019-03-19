import sys
import gluonbook as gb
import numpy as np
import mxnet as mx
from mxnet import gluon, image, nd, init
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn


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


def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = X.transpose((2, 0, 1)).expand_dims(axis=0)
    pred = nd.argmax(net(X.as_in_context(mx.gpu(0))), axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))


def label2image(pred):
    colormap = nd.array(gb.VOC_COLORMAP, ctx=mx.gpu(0), dtype="uint8")
    X = pred.astype("int32")
    return colormap[X, :]


if __name__ == "__main__":
    crop_size = (320, 480)
    lr = 1e-1
    wd = 1e-3
    batch_size = 128
    num_epochs = 100
    num_classes = 21
    num_workers = 0 if sys.platform.startswith("win32") else 4
    ctx = [mx.gpu(0), mx.gpu(1)]

    colormap2label = nd.zeros(256 ** 3)
    for i, colormap in enumerate(gb.VOC_COLORMAP):
        colormap2label[(colormap[0]*256+colormap[1]) * 256 + colormap[2]] = i

    voc_train = VOCSegDataset(True, crop_size, "../data/VOC2012", colormap2label)
    voc_test = VOCSegDataset(False, crop_size, "../data/VOC2012", colormap2label)
    train_iter = gdata.DataLoader(voc_train, batch_size, shuffle=True, last_batch="discard", num_workers=num_workers)
    test_iter = gdata.DataLoader(voc_test, batch_size, last_batch="discard", num_workers=num_workers)

    pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True, root="../models")
    net = nn.HybridSequential()
    for layer in pretrained_net.features[:-2]:
        net.add(layer)
    net.add(nn.Conv2D(num_classes, kernel_size=1),
            nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))
    net[-2].initialize(init.Xavier())
    net[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 64)))

    loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr, "wd": wd})
    gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

    net.save_parameters("../models/fcn.params")

    # test_images, test_labels = read_voc_images("../data/VOC2012", is_train=False)
    # n = 4
    # imgs = []
    # for i in range(n):
    #     crop_rect = (0, 0, 480, 320)
    #     X = image.fixed_crop(test_images[i], *crop_rect)
    #     pred = label2image(predict(X))
    #     imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
    # gb.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)
    # gb.plt.show()
