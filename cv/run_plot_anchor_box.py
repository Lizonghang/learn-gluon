import gluonbook as gb
import numpy as np
np.set_printoptions(2)
from mxnet import contrib, gluon, image, nd


def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_value=None):
        if obj is None:
            obj = default_value
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = gb.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va="center", ha="center", fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


if __name__ == "__main__":
    img = image.imread("../data/catdog.jpg").asnumpy()
    h, w = img.shape[:2]
    X = nd.random.uniform(shape=(1, 3, h, w))
    Y = contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    boxes = Y.reshape((h, w, 5, 4))
    gb.set_figsize()
    bbox_scale = nd.array((w, h, w, h))
    fig = gb.plt.imshow(img)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ["s=0.75, r=1", "s=0.5, r=1", "s=0.25, r=1", "s=0.75, r=2", "s=0.75, r=0.5"])
    gb.plt.show()
