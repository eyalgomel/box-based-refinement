import gc
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import CLIP.clip as clip
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from bbr.datasets.coco import get_coco_dataset
from bbr.datasets.flickr import get_flickr_dataset
from bbr.datasets.referit_loader import get_referit_dataset
from bbr.datasets.visual_genome import get_VG_dataset, get_VGtest_dataset
from bbr.inference.run_inference import inference_bbox_distillation, norm_z
from bbr.models.model import MultiModel
from bbr.models.vgg16 import BboxRegressor
from bbr.utils.utils import interpret_new
from bbr.utils.utils_grounding import generate_bbox
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion as DETRLoss
from einops import repeat
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert, remove_small_boxes
from torchvision.transforms.functional import resize
from torchvision.utils import draw_bounding_boxes, make_grid
from tqdm import tqdm, trange


def save_models(box_model: nn.Module, heatmap_model: nn.Module, val_results: dict, kwargs: dict):
    model_acc = max(val_results["h2b_acc"], val_results["cv2_acc"])
    save_model = False

    if val_results["h2b_acc"] > kwargs["best_accuracy_h"]:
        kwargs["best_accuracy_h"] = val_results["h2b_acc"]
        save_model = True

    if val_results["cv2_point_acc"] > kwargs["cv2_point_acc"]:
        kwargs["cv2_point_acc"] = val_results["cv2_point_acc"]
        save_model = True

    if model_acc > kwargs["best_accuracy"]:
        kwargs["best_accuracy"] = model_acc
        save_model = True

    if save_model:
        model_name = f"epoch_{kwargs['epoch']['global']}_acc_{model_acc:.2f}".replace(".", "_")
        box_model_path = Path(kwargs["best_model_path"]).with_name(f"box_{model_name}").with_suffix(".pt").as_posix()
        torch.save(box_model.eval(), box_model_path)
        heatmap_model_path = (
            Path(kwargs["best_model_path"]).with_name(f"heatmap_{model_name}").with_suffix(".pt").as_posix()
        )
        torch.save(heatmap_model.eval(), heatmap_model_path)


def get_adam_opt(cfg: DictConfig, model: nn.Module):
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    return optimizer


def get_box_model(cfg: DictConfig):
    num_boxes = cfg.training.num_boxes
    model = BboxRegressor(num_boxes=num_boxes)
    model = torch.nn.DataParallel(model, list(range(cfg.training.gpu_num))).cuda()
    if cfg.training.pretrained_box is not None:
        resumed_model = torch.load(cfg.training.pretrained_box)
        model.load_state_dict(resumed_model)
        print(f"## Loaded model: {cfg.training.pretrained_box}")

    return model


def get_dataloaders(cfg: DictConfig):
    if cfg.data.train.dataset == "coco":
        train_dataset = get_coco_dataset(cfg=cfg.data.train)
    elif cfg.data.train.dataset == "vg":
        train_dataset = get_VG_dataset(cfg=cfg.data.train)

    test_dataloader = None
    if cfg.data.val.dataset == "flickr":
        _, testset = get_flickr_dataset(cfg=cfg.data.val, only_test=True)
    elif cfg.data.val.dataset == "referit":
        _, testset = get_referit_dataset(cfg.data.val, only_test=True)
    elif cfg.data.val.dataset == "vg":
        _, testset = get_VGtest_dataset(cfg.data.val)
    else:
        raise NotImplementedError(f"{cfg.data.val.dataset} isn't supported!")

    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return train_dataset, test_dataloader


def training_step(
    box_model: nn.Module,
    dataloader: DataLoader,
    opts: Dict[str, torch.optim.Optimizer],
    criterion: nn.Module,
    cfg: DictConfig,
    heatmap_model: nn.Module,
    clip_model: nn.Module,
    training_box_model: bool = True,
    **kwargs,
):
    logger = kwargs["logger"]
    epoch = kwargs["epoch"]["global"]
    device = kwargs["device"]

    training_heatmap_model = not training_box_model
    print("Training BOX ::" if training_box_model else "Training HEATMAP ::")

    if training_box_model:
        loss_keys = {k: kwargs["epoch"]["global"] for k in criterion.weight_dict.keys()}
    elif training_heatmap_model:
        loss_keys = {k: kwargs["epoch"]["global"] for k in criterion.weight_dict.keys()}
        loss_keys.update({k: kwargs["epoch"]["heatmap"] for k in cfg.wsg_loss_weights.keys()})

    running_losses = {k: {"mean": 0, "batches": []} for k in loss_keys.keys()}
    pbar = tqdm(dataloader)
    opt = opts["box_opt"] if training_box_model else opts["heatmap_opt"]

    for batch_idx, inputs in enumerate(pbar):
        opts["box_opt"].zero_grad(set_to_none=True)
        opts["heatmap_opt"].zero_grad(set_to_none=True)

        real_imgs, text = inputs[0].to(device), inputs[1]
        img_size = real_imgs.shape[-1]
        text = clip.tokenize(text).to(device)
        z_text = norm_z(clip_model.encode_text(text))
        curr_image = real_imgs.expand(z_text.shape[0], *real_imgs.shape[1:])

        losses = 0
        if training_box_model:
            with torch.no_grad():
                heatmap_model.eval()
                box_model.train()
                heatmaps = heatmap_model(curr_image, z_text).detach()

        elif training_heatmap_model:
            heatmap_model.train()
            box_model.eval()

            heatmaps = heatmap_model(curr_image, z_text)

            real_imgs_224 = F.interpolate(real_imgs.detach(), size=(224, 224), mode="bilinear", align_corners=True)
            cam = interpret_new(real_imgs_224, text.detach(), clip_model, device).detach().float()
            cam = F.interpolate(cam, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=True)
            clip_cam_loss = F.mse_loss(heatmaps, cam)
            M = F.interpolate(heatmaps, size=(224, 224), mode="bilinear", align_corners=True)
            z_fr = norm_z(clip_model.encode_image(real_imgs_224 * M))
            z_bg = norm_z(clip_model.encode_image(real_imgs_224 * (1 - M)))
            regularization = M.mean()
            fr_loss = (1 - (z_fr @ z_text.T)).mean()
            bg_loss = torch.abs(z_bg @ z_text.T).mean()

            wsg_loss_dict = {
                "wsg_reg": cfg.wsg_loss_factor * cfg.wsg_loss_weights.wsg_reg * regularization,
                "wsg_clip": cfg.wsg_loss_factor * cfg.wsg_loss_weights.wsg_clip * clip_cam_loss,
                "wsg_bg": cfg.wsg_loss_factor * cfg.wsg_loss_weights.wsg_bg * bg_loss.mean(),
                "wsg_fr": cfg.wsg_loss_factor * cfg.wsg_loss_weights.wsg_fr * fr_loss,
            }
            losses = sum(wsg_loss_dict.values())

            for k, v in wsg_loss_dict.items():
                running_losses[k]["batches"].append(v.item())
                running_losses[k]["mean"] = np.mean(running_losses[k]["batches"])

        target = []
        union_boxes = (cfg.pred_union and training_heatmap_model) and np.random.rand() > (1 - cfg.pred_union)
        with torch.no_grad():
            for hm in heatmaps.detach():
                target.append(
                    get_suzuki_boxes(
                        cfg,
                        device,
                        img_size,
                        hm,
                        threshold=cfg.hm_mask_threshold if training_heatmap_model else cfg.box_mask_threshold,
                        union_boxes=union_boxes,
                    )
                )

        pred_boxes, pred_labels = box_model(repeat(heatmaps, "b 1 h w -> b 3 h w"))

        pred = {}
        pred["pred_logits"] = pred_labels
        pred["pred_boxes"] = pred_boxes

        # bounding box matching loss - taken from DETR
        loss_dict = criterion(pred, target)
        weight_dict = criterion.weight_dict

        for k in loss_dict.keys() - weight_dict.keys():
            del loss_dict[k]

        scaled_loss_dict = {k: v * weight_dict[k] for k, v in loss_dict.items()}

        losses += sum(scaled_loss_dict.values())

        for k, v in scaled_loss_dict.items():
            running_losses[k]["batches"].append(v.item())
            running_losses[k]["mean"] = np.mean(running_losses[k]["batches"])

        pbar.set_description(
            f"""Epoch: {epoch:3d} | {'| '.join([f'{k}: {v["mean"]:.2f}' for k,v in running_losses.items()])}"""
        )

        for k, v in running_losses.items():
            logger.add_scalar(f"Training/{k}", v["mean"], loss_keys[k] * len(dataloader) + batch_idx)

        with torch.no_grad():
            if batch_idx == 0:
                plot_images(
                    heatmaps.squeeze().detach().cpu(),
                    real_imgs.cpu(),
                    pred,
                    target,
                    logger,
                    epoch,
                    batch_idx,
                )

        losses.backward()
        opt.step()


def get_suzuki_boxes(cfg, device, img_size, heatmap, threshold=0.5, union_boxes=False):
    bboxes_confidences = generate_bbox(heatmap.squeeze().detach().cpu().numpy(), threshold=threshold)
    bboxes_confidences = torch.as_tensor(bboxes_confidences)[: cfg.num_boxes]
    bboxes = bboxes_confidences[:, :4] / img_size

    keep = remove_small_boxes(bboxes, min_size=0.05)
    bboxes = bboxes[keep]

    if union_boxes and len(bboxes) > 0:
        xy_min = torch.amin(bboxes, dim=(0))[:2]
        xy_max = torch.amax(bboxes, dim=(0))[2:]
        bboxes = torch.cat((xy_min, xy_max), dim=0)[None]

    labels = torch.zeros((len(bboxes)), dtype=torch.long)
    # convert from xyxy to cxcywh
    cxcywh_bboxes = box_convert(bboxes, "xyxy", "cxcywh")
    return {"boxes": cxcywh_bboxes.to(device), "labels": labels.to(device)}


@torch.no_grad()
def plot_images(heatmaps, real_imgs, pred, gt, logger, epoch, idx, threshold=0.5):
    im_size = heatmaps.shape[-1]
    gt_boxes = [(box_convert(i["boxes"], "cxcywh", "xyxy") * im_size).type(torch.int16) for i in gt]
    pred_boxes = (box_convert(pred["pred_boxes"], "cxcywh", "xyxy") * im_size).type(torch.int16)

    prob = F.softmax(pred["pred_logits"], -1)
    masks = prob[..., :-1].squeeze(-1) > threshold

    imgs = []
    for heatmap, real_img, pred_box, gt_box, mask in zip(heatmaps, real_imgs, pred_boxes, gt_boxes, masks):
        hm = (heatmap * 255).unsqueeze(0).type(torch.uint8)
        real_img = (((real_img * 0.5) + 0.5) * 255).type(torch.uint8).cpu()
        img = draw_bounding_boxes(hm, gt_box, colors="green", width=2)
        img = draw_bounding_boxes(img, pred_box[mask], colors="red", width=2)
        img = torch.cat([img, real_img], 2)
        img = resize(img, (100, 200))
        imgs.append(img)

    imgs_grid = make_grid(imgs[:32], nrow=4)
    logger.add_image(
        f"Training/Images/{idx}",
        imgs_grid,
        global_step=epoch,
    )


@torch.no_grad()
def evaluation_step(
    box_model: nn.Module,
    heatmap_model: nn.Module,
    dataloader: DataLoader,
    clip_model: nn.Module,
    cfg: DictConfig,
    **kwargs,
):
    results = inference_bbox_distillation(
        dataloader,
        box_model.eval(),
        heatmap_model.eval(),
        clip_model.eval(),
        cfg,
        **kwargs,
    )

    return results


def train(
    box_model: nn.Module,
    train_dataset: Dataset,
    test_dataloader: DataLoader,
    opts: Dict[str, torch.optim.Optimizer],
    clip_model: nn.Module,
    heatmap_model: nn.Module,
    cfg: DictConfig,
    **kwargs,
):
    weight_dict = cfg.training.box_loss_weights
    matcher = HungarianMatcher(
        cost_class=weight_dict["loss_ce"],
        cost_bbox=weight_dict["loss_bbox"],
        cost_giou=weight_dict["loss_giou"],
    )
    eos_coef = 0.1
    losses = ["labels", "boxes"]

    detr_critertion = DETRLoss(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses,
    ).to(kwargs["device"])

    for epoch in trange(cfg.training.epochs):
        idx = np.random.choice(range(len(train_dataset)), cfg.training.samples_per_epoch, replace=False)
        sampler = SubsetRandomSampler(idx)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
        )

        kwargs["epoch"]["global"] = epoch

        training_box_model = (epoch < cfg.training.box_warmup_epochs) or (
            (epoch - cfg.training.box_warmup_epochs) % (cfg.training.box.steps + cfg.training.heatmap.steps)
            >= cfg.training.heatmap.steps
        )

        training_step(
            box_model,
            train_dataloader,
            opts,
            detr_critertion,
            cfg.training,
            heatmap_model.eval(),
            clip_model.eval(),
            training_box_model,
            **kwargs,
        )

        if not training_box_model:
            kwargs["epoch"]["heatmap"] += 1

        if (epoch % cfg.training.evaluation_freq == 0) or epoch == cfg.training.epochs - 1:
            val_results = evaluation_step(
                box_model.eval(),
                heatmap_model.eval(),
                test_dataloader,
                clip_model.eval(),
                cfg.inference,
                **kwargs,
            )
            save_models(box_model.eval(), heatmap_model.eval(), val_results, kwargs)

        torch.cuda.empty_cache()
        gc.collect()


@hydra.main(version_base=None, config_path="../../config/train", config_name="wsg")
def main(cfg: DictConfig) -> None:
    os.chdir(get_original_cwd())
    timestamp = datetime.today().strftime("%Y%m%d%H%M")

    log_dir = Path(cfg.logs.results_dir) / f"{timestamp}_{cfg.exp_name}"
    logger = SummaryWriter(log_dir)
    OmegaConf.save(cfg, (log_dir / "config.yaml"))
    best_model_path = log_dir / "model/best.pt"
    best_model_path.parent.mkdir(exist_ok=True)

    # Models
    box_model = get_box_model(cfg)
    box_opt = get_adam_opt(cfg.training.box, box_model)
    train_dataset, test_dl = get_dataloaders(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)

    heatmap_model = (
        torch.nn.DataParallel(MultiModel(image_size=cfg.training.image_size), list(range(cfg.training.gpu_num)))
        .cuda()
        .eval()
    )
    heatmap_model.load_state_dict(torch.load(cfg.training.heatmap_model))

    heatmap_opt = get_adam_opt(cfg.training.heatmap, heatmap_model)

    global_args = {
        "device": device,
        "logger": logger,
        "epoch": defaultdict(lambda: 0),
        "best_accuracy": 0,
        "best_accuracy_h": 0,
        "cv2_point_acc": 0,
        "best_model_path": best_model_path.as_posix(),
    }

    train(
        box_model=box_model,
        heatmap_model=heatmap_model.eval(),
        train_dataset=train_dataset,
        test_dataloader=test_dl,
        opts={"box_opt": box_opt, "heatmap_opt": heatmap_opt},
        clip_model=clip_model.eval(),
        cfg=cfg,
        **global_args,
    )


if __name__ == "__main__":
    main()
