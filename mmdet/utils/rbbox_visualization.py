import cv2
import numpy as np
import math

from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val

color = [(0, 204, 255), (255, 204, 102), (0, 204, 0), (255, 0, 102)]
color_idx = {'crane':0, 'small_ship':1, 'middle_ship':2, 'large_ship': 3}

def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)


def imshow_det_rbboxes(img,
                       bboxes,
                       labels,
                       class_names=None,
                       score_thr=0,
                       bbox_color='green',
                       text_color='green',
                       thickness=1,
                       font_scale=0.5,
                       show=True,
                       win_name='',
                       wait_time=0,
                       out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or
            (n, 6).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 6
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 6
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        draw_color = color[label]

        xc, yc, w, h, ag, p = bbox.tolist()
        wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
        hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        ps = np.int0(np.array([p1, p2, p3, p4]))
        cv2.drawContours(img, [ps], -1, draw_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 5:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (int(p1[0]), int(p1[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.25, draw_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
