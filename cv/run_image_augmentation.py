import gluonbook as gb
from mxnet import image
from mxnet.gluon import data as gdada


def show_images(imgs, num_rows, num_cols, scale=2.):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = gb.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


if __name__ == "__main__":
    img = image.imread("../data/cat.jpg")
    apply(img, gdada.vision.transforms.RandomFlipLeftRight())
    apply(img, gdada.vision.transforms.RandomFlipTopBottom())
    apply(img, gdada.vision.transforms.RandomResizedCrop(size=(200, 200), scale=(0.1, 1.0), ratio=(0.5, 2.0)))
    apply(img, gdada.vision.transforms.RandomBrightness(brightness=0.5))
    apply(img, gdada.vision.transforms.RandomHue(hue=0.5))
    apply(img, gdada.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    apply(img, gdada.vision.transforms.Compose([
        gdada.vision.transforms.RandomFlipLeftRight(),
        gdada.vision.transforms.RandomFlipTopBottom(),
        gdada.vision.transforms.RandomResizedCrop(size=(200, 200), scale=(0.1, 1.0), ratio=(0.5, 2.0)),
        gdada.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    ]))
    gb.plt.show()
