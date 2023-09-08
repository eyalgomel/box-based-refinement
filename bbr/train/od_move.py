import gc
import os
from collections import defaultdict
from copy import deepcopy
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
from hydra.utils import get_original_cwd
from bbr.models.poly_lr import PolynomialLR
from bbr.models.vgg16 import BboxRegressorResnet50DINO
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert
from tqdm import tqdm, trange

from bbr.datasets.coco_20k import get_coco_dataset, get_coco_val_dataset
from bbr.datasets.voc import get_voc07_train_dataset, get_voc12_train_dataset
from bbr.inference.run_inference import convert_move_mask_to_box, inference_bbox_move
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion as DETRLoss
from moveseg.segmenter import build_segmenter


def move_feature_loss_fn(x, y, fn):
    if fn == "mse":
        loss = F.mse_loss(x, y)
    elif fn == "bce":
        loss = F.binary_cross_entropy(x, y)
    elif fn == "mse_sigmoid":
        loss = F.mse_loss(x, y, reduce=False) * y.detach()
        loss = loss.mean()
    else:
        raise NotImplementedError
    return loss


def save_models(
    box_model: nn.Module, move_model: nn.Module, val_results: float, kwargs: dict, always_save: bool = False
):
    save_model = False
    if val_results >= kwargs["best_corloc_move"]:
        kwargs["best_corloc_move"] = val_results
        save_model = True

    if save_model or always_save:
        model_name = f"epoch_{kwargs['epoch']}_corloc_{val_results}".replace(".", "_")
        box_model_path = Path(kwargs["best_model_path"]).with_name(f"box_{model_name}").with_suffix(".pt").as_posix()
        torch.save(box_model.eval(), box_model_path)
        move_model_path = Path(kwargs["best_model_path"]).with_name(f"move_{model_name}").with_suffix(".pt").as_posix()
        torch.save(move_model.eval(), move_model_path)


def get_adam_opt(cfg: DictConfig, model: nn.Module):
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    return optimizer


def get_adamw_opt(cfg: DictConfig, model: nn.Module):
    optimizer = optim.AdamW(
        list(model.parameters()),
        lr=cfg.lr,
        betas=cfg.betas,
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


def get_dataloaders(cfg: DictConfig, collate_test=True):
    train_dataset = None
    if "train" in cfg.data:
        if cfg.data.train.dataset == "coco":
            train_dataset = get_coco_dataset(cfg=cfg.data.train, test=False)
        elif cfg.data.train.dataset == "VOC07":
            train_dataset = get_voc07_train_dataset(cfg=cfg.data.train, test=False)
        elif cfg.data.train.dataset == "VOC12":
            train_dataset = get_voc12_train_dataset(cfg=cfg.data.train, test=False)
        else:
            raise NotImplementedError(f"{cfg.data.train.dataset} isn't supported!")

    test_dataloader = None
    if cfg.data.val.dataset == "coco":
        testset = get_coco_val_dataset(cfg=cfg.data.val, test=True)
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
        collate_fn=testset.collate_fn_test if collate_test else testset.collate_fn_train,
    )

    return train_dataset, test_dataloader


def training_step(
    box_model: nn.Module,
    dataloader: DataLoader,
    opts: Dict[str, torch.optim.Optimizer],
    criterion: nn.Module,
    cfg: DictConfig,
    move_model: nn.Module,
    training_box_model: bool = True,
    **kwargs,
):
    logger = kwargs["logger"]
    epoch = kwargs["epoch"]
    move_epoch = kwargs["move_epoch"]
    device = kwargs["device"]
    scheduler = kwargs["move_scheduler"]

    training_move_model = not training_box_model
    print("Training BOX ::" if training_box_model else "Training MOVE ::")

    loss_keys = {k: kwargs["epoch"] for k in criterion.weight_dict.keys()}

    if training_move_model:
        loss_keys["move_feature_loss"] = move_epoch
        loss_keys["move_reg_loss"] = move_epoch

    running_losses = {k: {"mean": 0, "batches": []} for k in loss_keys.keys()}
    pbar = tqdm(dataloader)
    opt = opts["box_opt"] if training_box_model else opts["move_opt"]

    if training_box_model:
        move_model.eval()
        box_model.train()
        for param in box_model.module.model.parameters():
            param.requires_grad = True

    elif training_move_model:
        move_model.train()
        box_model.eval()
        for param in box_model.module.model.parameters():
            param.requires_grad = False

    for batch_idx, inputs in enumerate(pbar):
        opts["box_opt"].zero_grad(set_to_none=True)
        opts["move_opt"].zero_grad(set_to_none=True)

        real_imgs = inputs[0].to(device)
        losses = 0

        if training_box_model:
            with torch.no_grad():
                move_masks, move_masks_fixed, feat1 = move_model(real_imgs)
                move_masks, move_masks_fixed = move_masks.detach(), move_masks_fixed.detach()

        elif training_move_model:
            move_masks, move_masks_fixed, feat1 = move_model(real_imgs)
            feat1, feat2 = move_masks, move_masks_fixed

            move_feature_loss = cfg.move_loss_coef * move_feature_loss_fn(feat1, feat2, cfg.move_loss_fn)
            move_reg_loss = cfg.move_reg_coef * feat1.mean()

            losses += move_feature_loss
            losses += move_reg_loss

        if cfg.move.train_multi_box:
            move_pred_xyxy = []
            for move_mask in move_masks_fixed.detach().cpu():
                with torch.no_grad():
                    preds = convert_move_mask_to_box(move_mask[0], get_all_boxes=True)
                    move_pred_xyxy.append(torch.as_tensor(preds, dtype=torch.float32))
        else:
            move_pred_xyxy = []
            for move_mask in move_masks_fixed.detach().cpu():
                with torch.no_grad():
                    pred = convert_move_mask_to_box(move_mask[0])
                    move_pred_xyxy.append(torch.as_tensor(pred, dtype=torch.float32))

            move_pred_xyxy = torch.stack(move_pred_xyxy).to("cpu")

        move_target = [
            {
                "boxes": box_convert(move_boxes / real_imgs.shape[-1], "xyxy", "cxcywh").to(device),
                "labels": torch.zeros((len(move_boxes)), dtype=torch.long).to(
                    device
                ),  # using only object/no object label
            }
            for move_boxes in move_pred_xyxy
        ]

        if epoch < cfg.box_warmup_epochs:
            expanded_move_masks = move_masks_fixed.expand(-1, 3, -1, -1)
        else:
            expanded_move_masks = move_masks.expand(-1, 3, -1, -1)

        pred_boxes, pred_labels = box_model(expanded_move_masks)

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
        loss_dict = criterion(pred, move_target)
        weight_dict = criterion.weight_dict

        for k in loss_dict.keys() - weight_dict.keys():
            del loss_dict[k]

        scaled_loss_dict = {k: v * weight_dict[k] for k, v in loss_dict.items()}

        losses += sum(scaled_loss_dict.values())

        if training_move_model:
            scaled_loss_dict.update({"move_feature_loss": move_feature_loss})
            scaled_loss_dict.update({"move_reg_loss": move_reg_loss})

        for k, v in scaled_loss_dict.items():
            running_losses[k]["batches"].append(v.item())
            running_losses[k]["mean"] = np.mean(running_losses[k]["batches"])

        pbar.set_description(
            f"""Epoch: {epoch:3d} | {'| '.join([f'{k}: {v["mean"]:.2f}' for k,v in running_losses.items()])}"""
        )

        for k, v in running_losses.items():
            logger.add_scalar(f"Training/{k}", v["mean"], loss_keys[k] * len(dataloader) + batch_idx)

        losses.backward()
        opt.step()

    if scheduler is not None and training_move_model:
        scheduler.step()

    torch.cuda.empty_cache()
    gc.collect()


@torch.inference_mode()
def evaluation_step(
    box_model: nn.Module,
    move_model: nn.Module,
    dataloader: DataLoader,
    **kwargs,
):
    result = inference_bbox_move(
        dataloader,
        box_model.eval(),
        move_model.eval(),
        **kwargs,
    )

    return result


def train(
    box_model: nn.Module,
    train_dataset: Dataset,
    test_dataloader: DataLoader,
    opts: Dict[str, torch.optim.Optimizer],
    move_model: nn.Module,
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
            (epoch - cfg.training.box_warmup_epochs) % (cfg.training.box.steps + cfg.training.move.steps)
            >= cfg.training.move.steps
        )

        training_step(
            box_model,
            train_dataloader,
            opts,
            detr_critertion,
            cfg.training,
            move_model.eval(),
            training_box_model,
            **kwargs,
        )

        val_results = 0
        if (epoch > cfg.training.box_warmup_epochs - 1) and (
            (epoch % cfg.training.evaluation_freq == 0 and epoch > 0) or epoch == cfg.training.epochs - 1
        ):
            val_results = evaluation_step(
                box_model.eval(),
                move_model.eval(),
                test_dataloader,
                **kwargs,
            )
            if epoch > 0:
                save_models(box_model.eval(), move_model.eval(), val_results, kwargs)

        if (epoch == (cfg.training.box_warmup_epochs - 1)) and epoch != 0:
            save_models(box_model.eval(), move_model.eval(), val_results, kwargs)

        if not training_box_model:
            kwargs["move_epoch"] += 1


@hydra.main(version_base=None, config_path="../../config/train", config_name="od_move")
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

    move_model_state_dict = torch.load(cfg.training.move.model_path)
    move_model = build_segmenter(cfg.training.move.segmenter).to(device)
    random_segmenter = deepcopy(move_model.segmenter_head)
    move_model.load_state_dict(move_model_state_dict)

    move_model.random_segmenter = random_segmenter

    for param in move_model.random_segmenter.parameters():
        param.requires_grad = True
    for param in move_model.feature_extractor.parameters():
        param.requires_grad = False
    for param in move_model.segmenter_head.parameters():
        param.requires_grad = False

    move_opt = get_adamw_opt(cfg.training.move, move_model.random_segmenter)

    move_scheduler = None
    if cfg.training.move.use_scheduler:
        print("Using PolyLR scheduler !")
        move_scheduler = PolynomialLR(
            move_opt, total_iters=cfg.training.epochs, power=cfg.training.move.scheduler.power
        )

    move_model = torch.nn.DataParallel(move_model, list(range(cfg.training.gpu_num))).cuda().eval()

    global_args = {
        "device": device,
        "logger": logger,
        "epoch": defaultdict(lambda: 0),
        "move_epoch": 0,
        "best_corloc_move": 0,
        "best_model_path": best_model_path.as_posix(),
        "move_scheduler": move_scheduler,
    }

    train(
        box_model=box_model,
        move_model=move_model.eval(),
        train_dataset=train_dataset,
        test_dataloader=test_dl,
        opts={"box_opt": box_opt, "move_opt": move_opt},
        cfg=cfg,
        **global_args,
    )


if __name__ == "__main__":
    main()
