import gluonbook as gb
from mxnet import nd, image, contrib


if __name__ == "__main__":
    ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = nd.array([[0, 0.1, 0.2, 0.3],
                        [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98],
                        [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])
    img = image.imread("../data/catdog.jpg").asnumpy()
    h, w = img.shape[:2]
    bbox_scale = nd.array((w, h, w, h))
    fig = gb.plt.imshow(img)
    gb.show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ["dog", "cat"], "chocolate")
    gb.show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    gb.plt.show()

    labels = contrib.nd.MultiBoxTarget(anchors.expand_dims(axis=0),
                                       ground_truth.expand_dims(axis=0),
                                       nd.zeros((1, 3, 5)))
