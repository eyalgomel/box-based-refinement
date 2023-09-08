from copy import deepcopy
import CLIP.clip as clip
import cv2
import hydra
import numpy as np
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms.functional as F_tv
from bbr.datasets.coco_20k import get_coco_val_dataset
from bbr.datasets.flickr import get_flickr1K_dataset
from bbr.datasets.referit_loader import get_referit_test_dataset
from bbr.datasets.visual_genome import get_VGtest_dataset
from bbr.datasets.voc import get_voc07_train_dataset, get_voc12_train_dataset
from bbr.models.model import MultiModel
from bbr.models.vgg16 import BboxRegressor, BboxRegressorResnet50DINO
from bbr.utils.utils import no_tuple
from bbr.utils.utils_grounding import (
    generate_bbox,
    isCorrect,
    isCorrectHit,
    normal_box_to_image,
    union,
)
from einops import rearrange
from LOST.networks import dino_forward as dino_lost_forward
from LOST.networks import get_model as get_lost_dino_model
from LOST.object_discovery import lost
from omegaconf import DictConfig
from TokenCut.networks import dino_forward as dino_tokencut_forward
from TokenCut.networks import get_model as get_tokencut_dino_model
from TokenCut.object_discovery import ncut
from torchvision import transforms
from torchvision.ops import box_convert, box_iou
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from moveseg.segmenter import build_segmenter


def norm_z(z):
    return z / z.norm(dim=1).unsqueeze(dim=1)


def norm_img(img):
    img = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = img * 0.5 + 0.5
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def norm_mask(mask):
    mask = mask.squeeze().detach().cpu().numpy()
    return mask


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap * 0.8 + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def load_blip_img(raw_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 384

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def convert_move_mask_to_box(mask, thresh=0.5, get_all_boxes=False):
    K = 1 if get_all_boxes else 1
    assert mask.ndim == 2, f"{mask.ndim=} {mask.shape=}"
    mask = mask.detach().cpu()
    thresholded_image = (mask > thresh) / 1
    thresholded_image = (thresholded_image.numpy() * 255).astype(np.uint8)
    ret, connected_components = cv2.connectedComponents(thresholded_image)

    # box = []
    wh = []
    boxes = []
    largest_size = 0
    zero_mask = np.zeros_like(connected_components, dtype=np.uint8)
    min_size = (0.05 * mask.shape[0]) ** 2

    for label in range(1, ret):
        mask = zero_mask.copy()
        mask[connected_components == label] = 255
        x, y, w, h = cv2.boundingRect(mask)
        if w * h > min_size:
            if K == 1:
                if w * h >= largest_size:
                    largest_size = w * h
                    boxes = [[x, y, x + w - 1, y + h - 1]]
            else:
                wh.append(w * h)
                boxes.append([x, y, x + w - 1, y + h - 1])

    if len(boxes) == 0:
        return np.array([[0, 0, mask.shape[0] - 1, mask.shape[0] - 1]])

    if K == 1:
        return np.array(boxes)

    else:
        wh = np.array(wh)
        order = np.argsort(wh)[::-1]
        boxes = np.array(boxes)[order][:K]

    return boxes


def infer_box_detector(h2b_model, orig_h, orig_w, heatmap, threshold=0.5):
    pred_boxes_norm, pred_labels = h2b_model(heatmap.expand(-1, 3, -1, -1))
    pred_prob = F.softmax(pred_labels, -1)
    masks = pred_prob[..., :-1].squeeze(-1) > threshold
    pred_boxes_norm = pred_boxes_norm[masks].detach().cpu()
    pred_boxes_norm = box_convert(pred_boxes_norm, "cxcywh", "xyxy")
    pred_boxes = normal_box_to_image(pred_boxes_norm, h=orig_h, w=orig_w)
    return pred_boxes_norm, pred_boxes


def inference_bbox_distillation(ds, h2b_model, heatmap_model, clip_model, cfg, **kwargs):
    pbar = tqdm(ds)
    logger = kwargs.get("logger")
    epoch = kwargs.get("epoch", {}).get("global")

    results = {
        "h2b_acc": [],
        "cv2_acc": [],
        "cv2_point_acc": [],
    }

    for batch_idx, inputs in enumerate(pbar):
        real_imgs, meta, size = inputs
        real_imgs = real_imgs.cuda()
        if len(list(meta.keys())) == 0:
            continue
        orig_image_size = [int(size[0]), int(size[1])]
        orig_h, orig_w = orig_image_size
        for sen_idx, sen in enumerate(meta.keys()):
            item = meta[sen]

            gt_box = item["bbox"]
            if ds.dataset.name == "flickr":
                title = no_tuple(item["sentences"])
            else:
                title = item["sentences"][0]
            gt_box = torch.stack(*gt_box, 1).to(torch.float).clone()

            text = clip.tokenize(title).to("cuda")
            with torch.enable_grad():
                z_text = norm_z(clip_model.encode_text(text))
                clip_model.zero_grad()
            curr_image = real_imgs.repeat(z_text.shape[0], 1, 1, 1)
            heatmap = heatmap_model(curr_image, z_text).mean(dim=0, keepdims=True)

            np_heatmap = heatmap.squeeze().detach().cpu().numpy()
            hm_size = heatmap.shape[-1]

            cv2_bbox_confidences = generate_bbox(np_heatmap)
            cv2_boxes_norm = torch.stack([torch.tensor(i[:4]) / hm_size for i in cv2_bbox_confidences])

            gt_box_norm = normalize_bbox(orig_image_size, gt_box)

            if h2b_model is not None:
                pred_boxes_norm, _ = infer_box_detector(h2b_model, orig_h, orig_w, heatmap)
                bbox_correct = isCorrect(union(gt_box_norm), union(pred_boxes_norm), iou_thr=0.5)
                results["h2b_acc"].append(bbox_correct)

            bbox_correct = isCorrect(union(gt_box_norm), union(cv2_boxes_norm), iou_thr=0.5)
            results["cv2_acc"].append(bbox_correct)

            point_correct = isCorrectHit(gt_box, np_heatmap, orig_image_size)
            results["cv2_point_acc"].append(point_correct)

        acc_h2b = 100.0 * np.mean(results["h2b_acc"])
        acc_cv2 = 100.0 * np.mean(results["cv2_acc"])
        acc_point = 100.0 * np.mean(results["cv2_point_acc"])

        acc_desc = f"box_acc h2b:{acc_h2b:.2f} | box_acc cv2:{acc_cv2:.2f} | point_acc:{acc_point:.2f}"
        pbar.set_description(acc_desc)

    if logger is not None:
        logger.add_scalar(
            "Validation/Accuracy_box/h2b",
            100.0 * np.mean(results["h2b_acc"]),
            epoch * len(ds),
        )
        logger.add_scalar(
            "Validation/Accuracy_box/cv2",
            100.0 * np.mean(results["cv2_acc"]),
            epoch * len(ds),
        )
        logger.add_scalar(
            "Validation/Accuracy_point/cv2",
            100.0 * np.mean(results["cv2_point_acc"]),
            epoch * len(ds),
        )

    return {k: 100 * np.mean(v) for k, v in results.items()}


@torch.no_grad()
def inference_bbox_lost(dataloader, box_model, dino_model, cfg, **kwargs):
    logger = kwargs.get("logger", None)
    epoch = kwargs.get("epoch", 0)
    device = kwargs.get("device", "cuda:0")
    pbar = tqdm(dataloader)

    pred_and_gt_corloc_scores = []
    lost_and_gt_corloc_scores = []

    for batch_idx, inputs in enumerate(pbar):
        real_imgs = [i.to(device) for i in inputs[0]]
        gt_bbox = inputs[1]
        gt_sizes = inputs[2]

        resized_imgs = torch.stack(
            [F.interpolate(im[None], (304, 304), mode="bilinear", align_corners=True).squeeze() for im in real_imgs]
        )
        use_qkv_feats = cfg.use_qkv_feats
        resized_feats, qkv_feats_resized = dino_lost_forward(dino_model, resized_imgs, use_qkv_feats=use_qkv_feats)

        feats = []
        for img in real_imgs:
            size_im = (
                img.shape[0],
                int(np.ceil(img.shape[1] / 16) * 16),
                int(np.ceil(img.shape[2] / 16) * 16),
            )
            padded = torch.zeros(size_im).to(device)
            padded[:, : img.shape[1], : img.shape[2]] = img
            feat, _ = dino_lost_forward(dino_model, padded[None])
            feats.append(feat)

        lost_pred_xyxy = []
        for feat, img in zip(feats, real_imgs):
            h_featmap = int(np.ceil(img.shape[1] / 16))
            w_featmap = int(np.ceil(img.shape[2] / 16))
            pred, A, scores, seed = lost(
                feat.detach(),
                [h_featmap, w_featmap],
                [16, 16],
                img.shape,
                k_patches=100,
            )
            lost_pred_xyxy.append(torch.as_tensor(pred)[None])

        lost_pred_xyxy = torch.stack(lost_pred_xyxy).to("cpu")

        if use_qkv_feats:
            expanded_feats = qkv_feats_resized
        else:
            expanded_feats = rearrange(resized_feats, "b h w -> b 1 h w").expand(-1, 3, -1, -1)

        if box_model is not None:
            pred_boxes, pred_labels = box_model(expanded_feats)

            pred = {}
            pred["pred_logits"] = pred_labels
            pred["pred_boxes"] = pred_boxes

            pred_boxes_xyxy_norm = box_convert(pred["pred_boxes"], "cxcywh", "xyxy")
            pred_boxes_xyxy = torch.stack(
                [normal_box_to_image(box, h=h, w=w) for box, (h, w) in zip(pred_boxes_xyxy_norm, gt_sizes)]
            ).to(torch.int16)
            pred_boxes_prob = F.softmax(pred_labels, -1)[..., :-1].squeeze(-1)
            argmax_box = pred_boxes_prob.argmax(1, keepdim=True)
            pred_highest_score_box_xyxy = pred_boxes_xyxy[
                torch.zeros_like(pred_boxes_prob).scatter(1, argmax_box, value=1).to(torch.bool)
            ][:, None, :].to("cpu")

            # Compute metrics
            pred_and_gt_ious = [
                box_iou(pboxes_xyxy, gt_box) for pboxes_xyxy, gt_box in zip(pred_highest_score_box_xyxy, gt_bbox)
            ]
            pred_and_gt_corloc = torch.stack([torch.any(iou >= 0.5) * 1 for iou in pred_and_gt_ious])
            pred_and_gt_corloc_scores.extend(pred_and_gt_corloc.float().numpy())

        lost_and_gt_ious = [box_iou(pboxes_xyxy, gt_box) for pboxes_xyxy, gt_box in zip(lost_pred_xyxy, gt_bbox)]
        lost_and_gt_corloc = torch.stack([torch.any(iou >= 0.5) * 1 for iou in lost_and_gt_ious])
        lost_and_gt_corloc_scores.extend(lost_and_gt_corloc.float().numpy())

        pbar.set_description(
            f"""Epoch: {epoch:3d} | CorLoc h: {np.mean(pred_and_gt_corloc_scores).item() * 100:05.1f} | CorLoc LOST: {np.mean(lost_and_gt_corloc_scores) * 100:05.1f}"""
        )

    if logger is not None:
        logger.add_scalar(
            "Validation/CorLoc_LOST_gt",
            np.mean(lost_and_gt_corloc_scores),
            epoch * len(dataloader),
        )
        logger.add_scalar(
            "Validation/CorLoc_h_gt",
            np.mean(pred_and_gt_corloc_scores),
            epoch * len(dataloader),
        )

    return np.mean(lost_and_gt_corloc_scores)


@torch.no_grad()
def inference_bbox_tokencut(dataloader, box_model, dino_model, cfg, **kwargs):
    logger = kwargs.get("logger", None)
    epoch = kwargs.get("epoch", 0)
    device = kwargs.get("device", "cuda:0")
    pbar = tqdm(dataloader)

    pred_and_gt_corloc_scores = []
    tokencut_and_gt_corloc_scores = []

    for batch_idx, inputs in enumerate(pbar):
        real_imgs = [i.to(device) for i in inputs[0]]
        gt_bbox = inputs[1]
        gt_sizes = inputs[2]

        resized_imgs = torch.stack(
            [F.interpolate(im[None], (304, 304), mode="bilinear", align_corners=True).squeeze() for im in real_imgs]
        )
        use_qkv_feats = cfg.use_qkv_feats
        resized_feats, qkv_feats_resized = dino_tokencut_forward(dino_model, resized_imgs, use_qkv_feats=use_qkv_feats)

        feats = []
        for img in real_imgs:
            size_im = (
                img.shape[0],
                int(np.ceil(img.shape[1] / 16) * 16),
                int(np.ceil(img.shape[2] / 16) * 16),
            )
            padded = torch.zeros(size_im).to(device)
            padded[:, : img.shape[1], : img.shape[2]] = img

            feat, qkv_feats = dino_tokencut_forward(dino_model, padded[None], use_qkv_feats=False)
            feats.append(feat)

        tokencut_pred_xyxy = []
        for feat, img in zip(feats, real_imgs):
            h_featmap = int(np.ceil(img.shape[1] / 16))
            w_featmap = int(np.ceil(img.shape[2] / 16))
            pred, _, _, _, _, _ = ncut(
                feat.detach(),
                [h_featmap, w_featmap],
                [16, 16],
                img.shape,
                0.2,  # args.tau,
                1e-5,  # args.eps,
            )
            tokencut_pred_xyxy.append(torch.as_tensor(pred)[None])

        tokencut_pred_xyxy = torch.stack(tokencut_pred_xyxy).to("cpu")

        if use_qkv_feats:
            expanded_feats = qkv_feats_resized
        else:
            expanded_feats = rearrange(resized_feats, "b h w -> b 1 h w").expand(-1, 3, -1, -1)

        if box_model is not None:
            pred_boxes, pred_labels = box_model(expanded_feats)

            pred = {}
            pred["pred_logits"] = pred_labels
            pred["pred_boxes"] = pred_boxes

            pred_boxes_xyxy_norm = box_convert(pred["pred_boxes"], "cxcywh", "xyxy")
            pred_boxes_xyxy = torch.stack(
                [normal_box_to_image(box, h=h, w=w) for box, (h, w) in zip(pred_boxes_xyxy_norm, gt_sizes)]
            ).to(torch.int16)
            pred_boxes_prob = F.softmax(pred_labels, -1)[..., :-1].squeeze(-1)
            argmax_box = pred_boxes_prob.argmax(1, keepdim=True)
            pred_highest_score_box_xyxy = pred_boxes_xyxy[
                torch.zeros_like(pred_boxes_prob).scatter(1, argmax_box, value=1).to(torch.bool)
            ][:, None, :].to("cpu")

            # Compute metrics
            pred_and_gt_ious = [
                box_iou(pboxes_xyxy, gt_box) for pboxes_xyxy, gt_box in zip(pred_highest_score_box_xyxy, gt_bbox)
            ]
            pred_and_gt_corloc = torch.stack([torch.any(iou >= 0.5) * 1 for iou in pred_and_gt_ious])
            pred_and_gt_corloc_scores.extend(pred_and_gt_corloc.float().numpy())

        tokencut_and_gt_ious = [
            box_iou(pboxes_xyxy, gt_box) for pboxes_xyxy, gt_box in zip(tokencut_pred_xyxy, gt_bbox)
        ]
        tokencut_and_gt_corloc = torch.stack([torch.any(iou >= 0.5) * 1 for iou in tokencut_and_gt_ious])
        tokencut_and_gt_corloc_scores.extend(tokencut_and_gt_corloc.float().numpy())

        pbar.set_description(
            f"""Epoch: {epoch:3d} | CorLoc h: {np.mean(pred_and_gt_corloc_scores) * 100:05.1f} | CorLoc TokenCut: {np.mean(tokencut_and_gt_corloc_scores) * 100:05.1f}"""
        )

    if logger is not None:
        logger.add_scalar(
            "Validation/CorLoc_TokenCut_gt",
            np.mean(tokencut_and_gt_corloc_scores),
            epoch * len(dataloader),
        )
        logger.add_scalar(
            "Validation/CorLoc_h_gt",
            np.mean(pred_and_gt_corloc_scores),
            epoch * len(dataloader),
        )

    return np.mean(tokencut_and_gt_corloc_scores)


@torch.inference_mode()
def inference_bbox_move(dataloader, box_model, move_model, **kwargs):
    logger = kwargs.get("logger", None)
    epoch = kwargs.get("epoch", 0)
    device = kwargs.get("device", None)
    pbar = tqdm(dataloader)

    pred_and_gt_corloc_scores = []
    move_and_gt_corloc_scores = []

    for batch_idx, inputs in enumerate(pbar):
        real_imgs = [i.to(device) for i in inputs[0]]

        gt_bbox = inputs[1]
        gt_sizes = inputs[2]
        resized_imgs = []

        patch_size = move_model.module.feature_extractor.patch_size
        move_masks = []
        for img in real_imgs:
            h, w = torch.as_tensor(img.shape[-2:]).numpy()
            img = F_tv.pad(
                img,
                padding_mode="reflect",
                padding=(
                    0,
                    0,
                    (patch_size - w % patch_size) % patch_size,
                    (patch_size - h % patch_size) % patch_size,
                ),
            )
            resized_imgs.append(F.interpolate(img[None], (304, 304), mode="bilinear", align_corners=True).squeeze())
            mask, _, _ = move_model(img[None])
            mask = mask[:, :, :h, :w]
            move_masks.append(mask[0])

        resized_imgs = torch.stack(resized_imgs)
        resized_masks, _, _ = move_model(resized_imgs)

        move_pred_xyxy = []
        for move_mask in move_masks:
            with torch.no_grad():
                pred = convert_move_mask_to_box(move_mask[0])
                move_pred_xyxy.append(torch.as_tensor(pred))

        move_pred_xyxy = torch.stack(move_pred_xyxy).to("cpu")

        expanded_masks = resized_masks.expand(-1, 3, -1, -1)
        if box_model is not None:
            pred_boxes, pred_labels = box_model(expanded_masks)

            pred = {}
            pred["pred_logits"] = pred_labels
            pred["pred_boxes"] = pred_boxes

            pred_boxes_xyxy_norm = box_convert(pred["pred_boxes"], "cxcywh", "xyxy")
            pred_boxes_xyxy = torch.stack(
                [normal_box_to_image(box, h=h, w=w) for box, (h, w) in zip(pred_boxes_xyxy_norm, gt_sizes)]
            ).to(torch.int16)
            pred_boxes_prob = F.softmax(pred_labels, -1)[..., :-1].squeeze(-1)
            argmax_box = pred_boxes_prob.argmax(1, keepdim=True)
            pred_highest_score_box_xyxy = pred_boxes_xyxy[
                torch.zeros_like(pred_boxes_prob).scatter(1, argmax_box, value=1).to(torch.bool)
            ][:, None, :].to("cpu")

            # Compute metrics
            pred_and_gt_ious = [
                box_iou(pboxes_xyxy, gt_box) for pboxes_xyxy, gt_box in zip(pred_highest_score_box_xyxy, gt_bbox)
            ]
            pred_and_gt_corloc = torch.stack([torch.any(iou >= 0.5) * 1 for iou in pred_and_gt_ious])
            pred_and_gt_corloc_scores.extend(pred_and_gt_corloc.float().numpy())

        move_and_gt_ious = [box_iou(pboxes_xyxy, gt_box) for pboxes_xyxy, gt_box in zip(move_pred_xyxy, gt_bbox)]
        move_and_gt_corloc = torch.stack([torch.any(iou >= 0.5) * 1 for iou in move_and_gt_ious])
        move_and_gt_corloc_scores.extend(move_and_gt_corloc.float().numpy())

        pbar.set_description(
            f"""Epoch: {epoch:3d} | CorLoc h: {np.mean(pred_and_gt_corloc_scores) * 100:05.1f} | CorLoc MOVE: {np.mean(move_and_gt_corloc_scores) * 100:05.1f}"""
        )
        if logger is None:
            continue

    if logger is not None:
        logger.add_scalar(
            "Validation/CorLoc_MOVE_gt",
            np.mean(move_and_gt_corloc_scores),
            epoch * len(dataloader),
        )
        logger.add_scalar(
            "Validation/CorLoc_h_gt",
            np.mean(pred_and_gt_corloc_scores),
            epoch * len(dataloader),
        )

    return np.mean(move_and_gt_corloc_scores)


def normalize_bbox(orig_image_size, gt_box):
    gt_box_norm = gt_box.clone()
    gt_box_norm[:, 0] = gt_box_norm[:, 0] / orig_image_size[1]
    gt_box_norm[:, 1] = gt_box_norm[:, 1] / orig_image_size[0]
    gt_box_norm[:, 2] = gt_box_norm[:, 2] / orig_image_size[1]
    gt_box_norm[:, 3] = gt_box_norm[:, 3] / orig_image_size[0]
    return gt_box_norm


def run_inference_task(cfg, model, box_model, dataset, clip_model):
    if cfg.task == "grounding":
        inference_bbox_distillation(dataset, box_model, model, clip_model, cfg)
    elif cfg.task.startswith("od"):
        if cfg.task == "od_lost":
            cfg = DictConfig({"use_qkv_feats": True})
            inference_bbox_lost(dataset, box_model, model, cfg)
        elif cfg.task == "od_tokencut":
            cfg = DictConfig({"use_qkv_feats": True})
            inference_bbox_tokencut(dataset, box_model, model, cfg)
        elif cfg.task == "od_move":
            inference_bbox_move(dataset, box_model, model)
    else:
        raise NotImplementedError(f"Take {cfg.task} isn't supported!")


def load_dataset(cfg):
    loader_args = {}
    if cfg.data.dataset == "flickr":
        testset = get_flickr1K_dataset(cfg.data)
    elif cfg.data.dataset == "referit":
        testset = get_referit_test_dataset(cfg.data)
    elif cfg.data.dataset == "vg":
        testset = get_VGtest_dataset(cfg.data)
    elif cfg.data.dataset == "VOC07":
        testset = get_voc07_train_dataset(cfg=cfg.data, test=True)
        loader_args["collate_fn"] = testset.collate_fn_test
    elif cfg.data.dataset == "VOC12":
        testset = get_voc12_train_dataset(cfg=cfg.data, test=True)
        loader_args["collate_fn"] = testset.collate_fn_test
    elif cfg.data.dataset == "coco20k":
        testset = get_coco_val_dataset(cfg=cfg.data, test=True)
        loader_args["collate_fn"] = testset.collate_fn_test
    else:
        raise NotImplementedError(f"Dataset {cfg.data.dataset} isn't supported!")

    dataset = torch.utils.data.DataLoader(
        testset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        **loader_args,
    )

    return dataset


def load_box_model(cfg, device):
    if cfg.box_model_path is None:
        return None
    if cfg.task == "grounding":
        box_model = BboxRegressor(num_boxes=cfg.num_boxes)
    elif cfg.task.startswith("od"):
        box_model = BboxRegressorResnet50DINO(num_boxes=cfg.num_boxes)
    else:
        raise NotImplementedError(f"Take {cfg.task} isn't supported!")

    box_model = torch.nn.DataParallel(box_model, list(range(cfg.gpu_num)))
    box_model.load_state_dict(torch.load(cfg.box_model_path))
    box_model = box_model.module.eval().to(device)
    return box_model


def load_model(cfg, device):
    if cfg.task == "grounding":
        model = (
            torch.nn.DataParallel(MultiModel(image_size=cfg.data.image_size), list(range(cfg.gpu_num)))
            .to(device)
            .eval()
        )
        model_state_dict = torch.load(cfg.model_path)
        model.load_state_dict(model_state_dict)
    elif cfg.task == "od_lost":
        model = get_lost_dino_model("vit_small", 16, 2, device, trainable=False).eval()
        model.load_state_dict(torch.load(cfg.model_path))
    elif cfg.task == "od_tokencut":
        model = get_tokencut_dino_model("vit_small", 16, device, trainable=False).eval()
        model.load_state_dict(torch.load(cfg.model_path))
    elif cfg.task == "od_move":
        model = build_segmenter(cfg.move_segmenter).to(device)
        random_segmenter = deepcopy(model.segmenter_head)
        model.random_segmenter = random_segmenter
        model = torch.nn.DataParallel(model, list(range(cfg.gpu_num))).to(device).eval()
        model.load_state_dict(torch.load(cfg.model_path))
    else:
        raise NotImplementedError(f"Take {cfg.task} isn't supported!")

    return model


@hydra.main(version_base=None, config_path="../../config/inference", config_name="inference")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Original model - f
    model = load_model(cfg, device)

    # Box model - h
    box_model = load_box_model(cfg, device)

    # Data loading
    dataset = load_dataset(cfg)

    clip_model = load_clip(cfg, device)

    # Inference
    run_inference_task(cfg, model, box_model, dataset, clip_model)


def load_clip(cfg, device):
    clip_model = None
    if cfg.task == "grounding":
        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    return clip_model


if __name__ == "__main__":
    main()
