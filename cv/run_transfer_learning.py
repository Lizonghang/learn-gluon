import gluonbook as gb
import mxnet as mx
from mxnet import init, gluon
from mxnet.gluon import data as gdata, model_zoo, loss as gloss


def train_fine_tunning(net, lr, batch_size, num_epochs):
    train_iter = gdata.DataLoader(train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gdata.DataLoader(test_imgs.transform_first(test_augs), batch_size)
    ctx = [mx.gpu(0), mx.gpu(1)]
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr, "wd": 0.001})
    gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)


if __name__ == "__main__":
    train_imgs = gdata.vision.ImageFolderDataset("../data/hotdog/train")
    test_imgs = gdata.vision.ImageFolderDataset("../data/hotdog/test")
    train_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.RandomResizedCrop(224),
        gdata.vision.transforms.RandomFlipLeftRight(),
        gdata.vision.transforms.ToTensor(),
        gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.Resize(256),
        gdata.vision.transforms.CenterCrop(224),
        gdata.vision.transforms.ToTensor(),
        gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True, root="../models")
    finetune_net = model_zoo.vision.resnet18_v2(classes=2)
    finetune_net.features = pretrained_net.features
    finetune_net.features.collect_params().setattr("grad_req", "null")
    finetune_net.output.initialize(init=init.Xavier())
    finetune_net.output.collect_params().setattr("lr_mult", 10)
    train_fine_tunning(finetune_net, lr=0.01, batch_size=128, num_epochs=5)
