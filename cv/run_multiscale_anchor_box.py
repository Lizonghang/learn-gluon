import gluonbook as gb
from mxnet import image, nd, contrib


def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_w, fmap_h))
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    fig = gb.plt.imshow(img)
    gb.show_bboxes(fig.axes, anchors[0]*bbox_scale)


if __name__ == "__main__":
    img = image.imread("../data/catdog.jpg").asnumpy()
    h, w = img.shape[:2]
    gb.set_figsize()
    display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
    gb.plt.show()
    display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
    gb.plt.show()
    display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
    gb.plt.show()
