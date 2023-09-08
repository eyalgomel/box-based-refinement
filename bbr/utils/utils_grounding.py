import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes

# This code is borrowed from https://github.com/talshaharabany/what-is-where-by-looking


def intensity_to_rgb(intensity, cmap="cubehelix", normalize=False):
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype("float32") * 255.0


def generate_bbox(cam, threshold=0.5, nms_threshold=0.05, max_drop_th=0.5):
    heatmap = intensity_to_rgb(cam, normalize=True).astype("uint8")
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

    thr_val = threshold * np.max(gray_heatmap)

    _, thr_gray_heatmap = cv2.threshold(gray_heatmap, int(thr_val), 255, cv2.THRESH_TOZERO)
    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        contours, _ = cv2.findContours(thr_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        proposals = [cv2.boundingRect(c) for c in contours]
        proposals = [(x, y, w, h) for (x, y, w, h) in proposals if h * w > 0.05 * cam.shape[0] * cam.shape[0]]
        if len(proposals) > 0:
            proposals_with_conf = [thr_gray_heatmap[y : y + h, x : x + w].mean() / 255 for (x, y, w, h) in proposals]
            inx = torchvision.ops.nms(
                torch.tensor(proposals).float(),
                torch.tensor(proposals_with_conf).float(),
                nms_threshold,
            )
            estimated_bbox = torch.cat(
                (
                    torch.tensor(proposals).float()[inx],
                    torch.tensor(proposals_with_conf)[inx].unsqueeze(dim=1),
                ),
                dim=1,
            ).tolist()
            estimated_bbox = [
                (x, y, x + w, y + h, conf)
                for (x, y, w, h, conf) in estimated_bbox
                if conf > max_drop_th * np.max(proposals_with_conf)
            ]
        else:
            estimated_bbox = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
    else:
        estimated_bbox = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
    return estimated_bbox


def IoU(boxA, boxB):
    # order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def isCorrect(boxes_gt, boxes_pred, iou_thr=0.5):
    for box_pred in boxes_pred:
        for box_gt in boxes_gt:
            if IoU(box_pred, box_gt) >= iou_thr:
                return 1
    return 0


def isCorrectHit(bbox_annot, heatmap, orig_img_shape):
    H, W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    max_loc = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)

    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1
    return 0


def union(bbox):
    if len(bbox) == 0:
        return []
    if isinstance(bbox[0], float) or isinstance(bbox[0], int):
        bbox = [bbox]

    if isinstance(bbox, np.ndarray):
        maxes = np.max(bbox, axis=0)
        mins = np.min(bbox, axis=0)
    elif isinstance(bbox, torch.Tensor):
        maxes = torch.max(bbox, axis=0)[0]
        mins = torch.min(bbox, axis=0)[0]

    return [[mins[0], mins[1], maxes[2], maxes[3]]]


def normal_box_to_image(box, h, w, orig_h=None, orig_w=None):
    box_i = box.clone()
    if orig_h or orig_w:
        box_i[:, 0] = box_i[:, 0] / orig_w
        box_i[:, 1] = box_i[:, 1] / orig_h
        box_i[:, 2] = box_i[:, 2] / orig_w
        box_i[:, 3] = box_i[:, 3] / orig_h

    box_i[:, 0] = box_i[:, 0] * w
    box_i[:, 1] = box_i[:, 1] * h
    box_i[:, 2] = box_i[:, 2] * w
    box_i[:, 3] = box_i[:, 3] * h
    return box_i.to(torch.int16)


def draw_boxes_on_image(img, boxes_and_colors, width=3):
    for color, boxes in boxes_and_colors.items():
        img = draw_bounding_boxes(img, boxes, colors=color, width=width)
    return img
