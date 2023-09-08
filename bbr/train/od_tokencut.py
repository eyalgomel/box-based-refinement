import gc
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from einops import rearrange
from hydra.utils import get_original_cwd
from bbr.models.vgg16 import BboxRegressorResnet50DINO
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm, trange

from bbr.datasets.coco_20k import get_coco_dataset, get_coco_val_dataset
from bbr.datasets.voc import get_voc07_train_dataset, get_voc12_train_dataset
from bbr.inference.run_inference import inference_bbox_tokencut
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion as DETRLoss
from TokenCut.networks import dino_forward
from TokenCut.networks import get_model as get_dino_model
from TokenCut.object_discovery import ncut


def dino_feature_loss_fn(dino_model, dino_fixed_model, real_imgs):
    feat1 = dino_model(real_imgs)
    feat2 = dino_fixed_model(real_imgs).detach()
    loss = F.mse_loss(feat1, feat2)
    return loss


def save_models(
    box_model: nn.Module, dino_model: nn.Module, val_results: float, kwargs: dict, always_save: bool = False
):
    save_model = False
    if val_results >= kwargs["best_corloc_tokencut"]:
        kwargs["best_corloc_tokencut"] = val_results
        save_model = True

    if save_model or always_save:
        model_name = f"epoch_{kwargs['epoch']}_corloc_{val_results}".replace(".", "_")
        box_model_path = Path(kwargs["best_model_path"]).with_name(f"box_{model_name}").with_suffix(".pt").as_posix()
        torch.save(box_model.eval(), box_model_path)
        dino_model_path = Path(kwargs["best_model_path"]).with_name(f"dino_{model_name}").with_suffix(".pt").as_posix()
        torch.save(dino_model.eval(), dino_model_path)


def get_adam_opt(cfg: DictConfig, model: nn.Module):
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    return optimizer


def get_box_model(cfg: DictConfig):
    num_boxes = cfg.training.num_boxes
    model = BboxRegressorResnet50DINO(num_boxes=num_boxes, num_classes=cfg.training.num_classes)

    model = torch.nn.DataParallel(model, list(range(cfg.training.gpu_num))).cuda()
    if cfg.training.pretrained_box is not None:
        resumed_model = torch.load(cfg.training.pretrained_box)
        model.load_state_dict(resumed_model)
        print(f"## Loaded model: {cfg.training.pretrained_box}")

    return model


def get_dataloaders(cfg: DictConfig):
    if cfg.data.train.dataset == "coco":
        train_dataset = get_coco_dataset(cfg=cfg.data.train)
    elif cfg.data.train.dataset == "VOC07":
        train_dataset = get_voc07_train_dataset(cfg=cfg.data.train)
    elif cfg.data.train.dataset == "VOC12":
        train_dataset = get_voc12_train_dataset(cfg=cfg.data.train)
    else:
        raise NotImplementedError(f"{cfg.data.train.dataset} isn't supported!")

    test_dataloader = None
    if cfg.data.val.dataset == "coco":
        testset = get_coco_val_dataset(cfg=cfg.data.val)
    elif cfg.data.val.dataset == "VOC07":
        testset = get_voc07_train_dataset(cfg=cfg.data.val, test=True)
    elif cfg.data.val.dataset == "VOC12":
        testset = get_voc12_train_dataset(cfg=cfg.data.val, test=True)
    else:
        raise NotImplementedError(f"{cfg.data.val.dataset} isn't supported!")

    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=testset.collate_fn_test,
    )

    return train_dataset, test_dataloader


def training_step(
    box_model: nn.Module,
    dataloader: DataLoader,
    opts: Dict[str, torch.optim.Optimizer],
    criterion: nn.Module,
    cfg: DictConfig,
    dino_model: nn.Module,
    dino_fixed_model: nn.Module,
    training_box_model: bool = True,
    **kwargs,
):
    logger = kwargs["logger"]
    epoch = kwargs["epoch"]
    dino_epoch = kwargs["dino_epoch"]
    device = kwargs["device"]

    training_dino_model = not training_box_model
    print("Training BOX ::" if training_box_model else "Training DINO ::")

    loss_keys = {k: kwargs["epoch"] for k in criterion.weight_dict.keys()}
    if training_dino_model:
        loss_keys["dino_feature_loss"] = dino_epoch

    running_losses = {k: {"mean": 0, "batches": []} for k in loss_keys.keys()}
    pbar = tqdm(dataloader)
    opt = opts["box_opt"] if training_box_model else opts["dino_opt"]

    if training_box_model:
        dino_model.eval()
        box_model.train()
        for param in box_model.module.model.parameters():
            param.requires_grad = True

    elif training_dino_model:
        dino_model.train()
        box_model.eval()
        for param in box_model.module.model.parameters():
            param.requires_grad = False

    for batch_idx, inputs in enumerate(pbar):
        opts["box_opt"].zero_grad(set_to_none=True)
        opts["dino_opt"].zero_grad(set_to_none=True)

        real_imgs = inputs[0].to(device)
        gt_bbox = inputs[1]
        use_qkv_feats = cfg.use_qkv_feats

        losses = 0

        if training_box_model:
            with torch.no_grad():
                feats, qkv_feats = dino_forward(dino_model, real_imgs, use_qkv_feats=use_qkv_feats)
                feats = feats.detach()
                if use_qkv_feats:
                    qkv_feats = qkv_feats.detach()

        elif training_dino_model:
            feats, qkv_feats = dino_forward(dino_model, real_imgs, use_qkv_feats=use_qkv_feats)
            dino_feature_loss = cfg.dino_loss_coef * dino_feature_loss_fn(dino_model, dino_fixed_model, real_imgs)
            losses += dino_feature_loss

        tokencut_pred_xyxy = []
        for feat in feats:
            with torch.no_grad():
                pred, _, _, _, _, _ = ncut(
                    feat[None].detach(),
                    [real_imgs.shape[-1] // 16, real_imgs.shape[-1] // 16],
                    [16, 16],
                    real_imgs.shape[1:],
                    0.2,  # args.tau,
                    1e-5,  # args.eps,
                )
                tokencut_pred_xyxy.append(torch.as_tensor(pred)[None])

        tokencut_pred_xyxy = torch.stack(tokencut_pred_xyxy).to("cpu")

        tokencut_target = [
            {
                "boxes": box_convert(tokencut_box / real_imgs.shape[-1], "xyxy", "cxcywh").to(device),
                "labels": torch.as_tensor([0]).to(device),  # using only object/no object label
            }
            for tokencut_box in tokencut_pred_xyxy
        ]

        if use_qkv_feats:
            expanded_feats = qkv_feats
        else:
            expanded_feats = rearrange(feats, "b h w -> b 1 h w").expand(-1, 3, -1, -1)
        pred_boxes, pred_labels = box_model(expanded_feats)

        pred = {}
        pred["pred_logits"] = pred_labels
        pred["pred_boxes"] = pred_boxes
        pred_boxes_xyxy = (box_convert(pred["pred_boxes"], "cxcywh", "xyxy") * real_imgs.shape[-1]).type(torch.int16)
        pred_boxes_prob = F.softmax(pred_labels, -1)[..., :-1].squeeze(-1)
        argmax_box = pred_boxes_prob.argmax(1, keepdim=True)
        pred_highest_score_box_xyxy = pred_boxes_xyxy[
            torch.zeros_like(pred_boxes_prob).scatter(1, argmax_box, value=1).to(torch.bool)
        ][:, None, :].to("cpu")

        # bounding box matching loss - taken from DETR
        loss_dict = criterion(pred, tokencut_target)
        weight_dict = criterion.weight_dict

        for k in loss_dict.keys() - weight_dict.keys():
            del loss_dict[k]

        scaled_loss_dict = {k: v * weight_dict[k] for k, v in loss_dict.items()}

        losses += sum(scaled_loss_dict.values())

        if training_dino_model:
            scaled_loss_dict.update({"dino_feature_loss": dino_feature_loss})

        for k, v in scaled_loss_dict.items():
            running_losses[k]["batches"].append(v.item())
            running_losses[k]["mean"] = np.mean(running_losses[k]["batches"])

        pbar.set_description(
            f"""Epoch: {epoch:3d} | {'| '.join([f'{k}: {v["mean"]:.2f}' for k,v in running_losses.items()])}"""
        )

        for k, v in running_losses.items():
            logger.add_scalar(f"Training/{k}", v["mean"], loss_keys[k] * len(dataloader) + batch_idx)

        # Compute metrics
        with torch.no_grad():
            pred_and_gt_ious = [
                box_iou(pboxes_xyxy, gt_box) for pboxes_xyxy, gt_box in zip(pred_highest_score_box_xyxy, gt_bbox)
            ]
            pred_and_gt_corloc = torch.stack([torch.any(iou >= 0.5) * 1 for iou in pred_and_gt_ious])
            logger.add_scalar(
                "Training/CorLoc_h_gt",
                pred_and_gt_corloc.float().mean().item() * 100,
                epoch * len(dataloader) + batch_idx,
            )

            tokencut_and_gt_ious = [
                box_iou(pboxes_xyxy, gt_box) for pboxes_xyxy, gt_box in zip(tokencut_pred_xyxy, gt_bbox)
            ]
            tokencut_and_gt_corloc = torch.stack([torch.any(iou >= 0.5) * 1 for iou in tokencut_and_gt_ious])
            logger.add_scalar(
                "Training/CorLoc_TokenCut_gt",
                tokencut_and_gt_corloc.float().mean().item() * 100,
                epoch * len(dataloader) + batch_idx,
            )

        losses.backward()
        opt.step()

    torch.cuda.empty_cache()
    gc.collect()


@torch.inference_mode()
def evaluation_step(
    box_model: nn.Module,
    dino_model: nn.Module,
    dataloader: DataLoader,
    cfg: DictConfig,
    **kwargs,
):
    result = inference_bbox_tokencut(
        dataloader,
        box_model.eval(),
        dino_model.eval(),
        cfg,
        **kwargs,
    )

    return result


def train(
    box_model: nn.Module,
    train_dataset: Dataset,
    test_dataloader: DataLoader,
    opts: Dict[str, torch.optim.Optimizer],
    dino_model: nn.Module,
    dino_fixed_model: nn.Module,
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
        num_classes=cfg.training.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses,
    ).to(kwargs["device"])

    for epoch in trange(cfg.training.epochs):
        idx = np.random.choice(
            range(len(train_dataset)), min(cfg.training.samples_per_epoch, len(train_dataset)), replace=False
        )
        sampler = SubsetRandomSampler(idx)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
            collate_fn=train_dataset.collate_fn_train,
        )

        kwargs["epoch"] = epoch

        training_box_model = (epoch < cfg.training.box_warmup_epochs) or (
            (epoch - cfg.training.box_warmup_epochs) % (cfg.training.box.steps + cfg.training.dino.steps)
            >= cfg.training.dino.steps
        )

        training_step(
            box_model,
            train_dataloader,
            opts,
            detr_critertion,
            cfg.training,
            dino_model.eval(),
            dino_fixed_model.eval(),
            training_box_model,
            **kwargs,
        )

        if (epoch % cfg.training.evaluation_freq == 0 and epoch > 0) or epoch == cfg.training.epochs - 1:
            val_results = evaluation_step(
                box_model.eval(),
                dino_model.eval(),
                test_dataloader,
                cfg.inference,
                **kwargs,
            )
            if epoch % 2 == 0 and epoch > 0:
                save_models(box_model.eval(), dino_model.eval(), val_results, kwargs)

        if not training_box_model:
            kwargs["dino_epoch"] += 1


@hydra.main(version_base=None, config_path="../../config/train", config_name="od_tokencut")
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

    dino_model = get_dino_model("vit_small", 16, device, trainable=True)
    dino_fixed_model = get_dino_model("vit_small", 16, device, trainable=False)

    dino_opt = get_adam_opt(cfg.training.dino, dino_model)

    global_args = {
        "device": device,
        "logger": logger,
        "epoch": defaultdict(lambda: 0),
        "dino_epoch": 0,
        "best_corloc_tokencut": 0,
        "best_model_path": best_model_path.as_posix(),
    }

    train(
        box_model=box_model,
        dino_model=dino_model.eval(),
        dino_fixed_model=dino_fixed_model.eval(),
        train_dataset=train_dataset,
        test_dataloader=test_dl,
        opts={"box_opt": box_opt, "dino_opt": dino_opt},
        cfg=cfg,
        **global_args,
    )


if __name__ == "__main__":
    main()
