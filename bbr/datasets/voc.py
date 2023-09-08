import collections
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import cv2
import torch
from omegaconf import DictConfig

from bbr.datasets.tfs import get_coco20k_transform
from bbr.utils.utils_grounding import normal_box_to_image

from xml.etree.ElementTree import parse as ET_parse
from xml.etree.ElementTree import Element as ET_Element


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="trainval"):
        self.root_path = f"{root}"
        splits_dir = os.path.join(self.root_path, "ImageSets", "Main")
        split_f = os.path.join(splits_dir, split.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]
        image_dir = os.path.join(self.root_path, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(self.root_path, "Annotations")
        self.targets = [os.path.join(target_dir, x + ".xml") for x in file_names]
        self.annotations = [
            self.parse_voc_xml(ET_parse(target).getroot())["annotation"]["object"] for target in self.targets
        ]

        assert len(self.images) == len(self.targets)

        self.transform = transform
        print("num of data:{}".format(len(self.images)))

    def parse_voc_xml(self, node: ET_Element) -> Dict[str, Any]:
        # Taken fron torchvision.datasets.voc
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def __getitem__(self, index):
        img_path = self.images[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = (img.shape[0], img.shape[1])
        gt_size = (img.shape[0], img.shape[1])

        img = self.transform(img)
        annotation = self.annotations[index]
        gt_bbxs = []

        for o in range(len(annotation)):
            annotation[o]["name"]
            # gt_clss.append(gt_cls)
            obj = annotation[o]["bndbox"]
            x1y1x2y2 = [
                int(obj["xmin"]),
                int(obj["ymin"]),
                int(obj["xmax"]),
                int(obj["ymax"]),
            ]
            x1y1x2y2[0] -= 1
            x1y1x2y2[1] -= 1
            gt_bbxs.append(x1y1x2y2)

        gt_bbox = np.stack(gt_bbxs).astype(np.float32)
        gt_bbox_norm = normal_box_to_image(
            torch.as_tensor(gt_bbox), h=img.shape[-2], w=img.shape[-1], orig_h=h, orig_w=w
        )

        return (img, gt_bbox_norm, gt_size, Path(img_path).name)

    def __len__(self):
        return len(self.images)

    def collate_fn_train(self, batch):
        images = list()
        boxes = list()
        sizes = list()
        names = list()

        for b in batch:
            images.append(b[0])
            boxes.append(torch.as_tensor(b[1], dtype=torch.float32))
            sizes.append(b[2])
            names.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, sizes, names

    def collate_fn_test(self, batch):
        images = list()
        boxes = list()
        sizes = list()
        names = list()

        for b in batch:
            images.append(b[0])
            boxes.append(torch.as_tensor(b[1], dtype=torch.float32))
            sizes.append(b[2])
            names.append(b[3])

        return images, boxes, sizes, names


def get_voc07_train_dataset(cfg: DictConfig, test=False):
    datadir = cfg.data_path
    transform_train, transform_test = get_coco20k_transform(image_size=cfg.image_size, test=test)

    transform = transform_test if test else transform_train
    ds_train = ImageLoader(datadir, split="trainval", transform=transform)
    return ds_train


def get_voc12_train_dataset(cfg: DictConfig, test=False):
    datadir = cfg.data_path
    transform_train, transform_test = get_coco20k_transform(image_size=cfg.image_size, test=test)

    transform = transform_test if test else transform_train
    ds_train = ImageLoader(datadir, split="trainval", transform=transform)
    return ds_train
