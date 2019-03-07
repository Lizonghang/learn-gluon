import gluonbook as gb
from mxnet import image, nd, contrib


if __name__ == "__main__":
    anchors = nd.array([[0.1, 0.08, 0.52, 0.92],
                        [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91],
                        [0.55, 0.2, 0.9, 0.88]])
    offset_preds = nd.array([0]*anchors.size)
    cls_probs = nd.array([[0, 0, 0, 0],
                          [0.9, 0.8, 0.7, 0.1],
                          [0.1, 0.2, 0.3, 0.9]])
    img = image.imread("../data/catdog.jpg").asnumpy()
    h, w = img.shape[:2]
    bbox_scale = nd.array((w, h, w, h))
    fig = gb.plt.imshow(img)
    gb.show_bboxes(fig.axes, anchors*bbox_scale, ["dog=0.9", "dog=0.8", "dog=0.7", "cat=0.9"])
    gb.plt.show()

    output = contrib.nd.MultiBoxDetection(cls_probs.expand_dims(axis=0),
                                          offset_preds.expand_dims(axis=0),
                                          anchors.expand_dims(axis=0),
                                          nms_threshold=0.5)

    fig = gb.plt.imshow(img)
    for i in output[0].asnumpy():
        if i[0] == -1:
            continue
        label = ("dog=", "cat=")[int(i[0])] + str(i[1])
        gb.show_bboxes(fig.axes, [nd.array(i[2:])*bbox_scale], label)
    gb.plt.show()
