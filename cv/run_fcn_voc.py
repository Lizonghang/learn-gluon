import os
import random
import gluoncv
import gluonbook as gb
import mxnet as mx
from mxnet import nd, image
from colormath.color_objects import AdobeRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


def get_colormap2label():
    colormap2label = nd.zeros(256 ** 3)
    for i, colormap in enumerate(gb.VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


def load_and_crop_image(root, files, crop_size):
    imgs = []
    for file in files:
        fp = os.path.join(root, file)
        img = image.imread(fp)
        h, w = img.shape[:2]
        ratio = h / w
        if crop_size[0] / crop_size[1] < ratio:
            w = crop_size[1]
            h = w * ratio
        else:
            h = crop_size[0]
            w = h / ratio
        image.imresize(img, int(w), int(h))
        img, rect = image.random_crop(img, crop_size)
        imgs.append(img)
    return imgs


def normalize_and_reshape_images(imgs):
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    batch_shape = (len(imgs), imgs[0].shape[2], imgs[0].shape[0], imgs[0].shape[1])
    batch_imgs = nd.zeros(batch_shape)
    for idx, img in enumerate(imgs):
        normalized = (img.astype("float32") / 255 - rgb_mean) / rgb_std
        reshaped = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
        batch_imgs[idx] = reshaped
    return batch_imgs


def label2image(pred_classes):
    colormap = nd.array(gb.VOC_COLORMAP, dtype="uint8")
    X = pred_classes.astype("int32")
    return colormap[X, :]


def gen_colormap(pred):
    h, w = pred.shape[1], pred.shape[2]
    pred = nd.argmax(pred, axis=0)
    pred = pred.reshape((h, w))
    colormap = label2image(pred)
    return colormap


def get_click_class(classmap, pos):
    y, x = map(int, pos)
    class_idx = classmap[x, y].astype("int32").asscalar()
    return gb.VOC_CLASSES[class_idx]


def get_click_pixel(img, pos):
    y, x = map(int, pos)
    return img[x, y, :].asnumpy().astype("int32").tolist()


def delta_e_cie(pixels):
    rgbs = [AdobeRGBColor(*pix, is_upscaled=True) for pix in pixels]
    labs = [convert_color(rgb, LabColor) for rgb in rgbs]
    print(rgbs)
    print(labs)
    for i in range(len(files)-1):
        for j in range(i+1, len(files)):
            delta_e = delta_e_cie2000(labs[i], labs[j])
            print("color difference between %9s and %9s is: %.1f" %
                  (classes[random_idxes[i]], classes[random_idxes[j]], delta_e))


if __name__ == "__main__":
    ctx = mx.cpu()
    files = ["catdog.jpg", "cars.jpg", "horseman.jpg", "motorbike.jpg"]
    classes = ["cat", "bus", "person", "motorbike"]
    num_rows, num_cols = 3, len(files)
    figsize = (num_cols * 2.5, num_rows * 2)
    crop_size = (480, 480)

    net = gluoncv.model_zoo.get_fcn_resnet101_voc(pretrained=True, ctx=ctx)

    imgs = load_and_crop_image("../data", files, crop_size)
    Xs = normalize_and_reshape_images(imgs)
    preds = net.evaluate(Xs.as_in_context(ctx))
    preds = preds.as_in_context(mx.cpu())
    preds.wait_to_read()

    colormap_list = []
    classmap_list = []
    for pred in preds:
        colormap = gen_colormap(pred)
        classmap = gb.voc_label_indices(colormap, get_colormap2label())
        colormap_list.append(colormap)
        classmap_list.append(classmap)

    verified = False
    first_time = True
    pixels = []

    blank_img = nd.ones((crop_size[0], crop_size[1], 3), dtype="int32") * 255
    _, axes = gb.plt.subplots(num_rows, num_cols, figsize=figsize)
    while not verified:
        for i in range(num_rows):
            for j in range(num_cols):
                axes[i][j].imshow(blank_img.asnumpy())
                axes[i][j].axes.get_xaxis().set_visible(False)
                axes[i][j].axes.get_yaxis().set_visible(False)
                axes[i][j].axes.set_title("")

        if first_time:
            text = "Start authentication, click the objects in order."
            gb.plt.suptitle(text)
            first_time = False
        else:
            text = "[Reject] Authentication Failed !"
            gb.plt.suptitle(text)

        pixels.clear()
        random_idxes = list(range(len(files)))
        random.shuffle(random_idxes)
        for i, idx in enumerate(random_idxes):
            classmap = classmap_list[idx]
            colormap = colormap_list[idx]
            img = imgs[idx]
            axes[0][i].axes.set_title("click the %s." % classes[idx])
            axes[0][i].axes.imshow(img.asnumpy())
            pos = gb.plt.ginput(n=1, timeout=-1)[0]
            class_name = get_click_class(classmap, pos)
            pix = get_click_pixel(img, pos)
            pixels.append(pix)
            if class_name is not classes[idx]:
                break
            gb.plt.suptitle("Correct Click %d times." % (i + 1))
            if num_rows >= 2:
                axes[1][i].axes.imshow(colormap.asnumpy())
                axes[1][i].axes.set_title("colormap")
            if num_rows >= 3:
                colormask = nd.ones((crop_size[0], crop_size[1], 3), dtype="int32")
                for c in range(3):
                    colormask[:, :, c] *= pix[c]
                axes[2][i].axes.imshow(colormask.asnumpy())
                axes[2][i].axes.set_title("RGB:" + str(pix))
            if i == len(files) - 1:
                gb.plt.suptitle("[Success] Accessing to Service ...")
                verified = True

    delta_e_cie(pixels)
    gb.plt.show()
