import gluonbook as gb
from mxnet import image


def bbox_to_rect(bbox, color):
    return gb.plt.Rectangle(xy=(bbox[0], bbox[1]),
                            width=bbox[2]-bbox[0],
                            height=bbox[3]-bbox[1],
                            fill=False,
                            edgecolor=color,
                            linewidth=2)


if __name__ == "__main__":
    gb.set_figsize()
    img = image.imread("../data/catdog.jpg").asnumpy()
    fig = gb.plt.imshow(img)
    # bbox = [x1,y1,x2,y2]
    dog_bbox = [60, 45, 378, 516]
    cat_bbox = [400, 112, 655, 493]
    fig.axes.add_patch(bbox_to_rect(dog_bbox, "blue"))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, "red"))
    gb.plt.show()
