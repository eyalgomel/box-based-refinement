import json
import os
import pickle
import cv2

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from bbr.datasets.tfs import get_flickr_transform


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", img_path=False):
        self.imgs_data_folder = os.path.join(root, "VG_Annotations", "imgs_data.pickle")
        self.splits_folder = os.path.join(root, "VG_Annotations", "data_splits.pickle")
        self.annotations_folder = os.path.join(root, "VG_Annotations", "region_descriptions.json")
        with open(self.annotations_folder, "rb") as f:
            self.annotations = json.load(f)
        with open(self.splits_folder, "rb") as f:
            self.splits = pickle.load(f, encoding="latin1")
        with open(self.imgs_data_folder, "rb") as f:
            self.imgs_data = pickle.load(f, encoding="latin1")
        self.img_folder = os.path.join(root, "VG_Images")

        self.img_path = img_path
        self.transform = transform
        self.files = list(self.splits[split])
        self.split = split
        self.annotations = sync_data(self.files, self.annotations, self.imgs_data, split)
        self.files = list(self.annotations.keys())
        print("num of data:{}".format(len(self.files)))

    def __getitem__(self, index):
        item = str(self.files[index])
        img_path = os.path.join(self.img_folder, item + ".jpg")
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image_sizes = (img.shape[0], img.shape[1])
        img = self.transform(img)
        ann = self.annotations[int(item)]
        if self.split == "train":
            region_id = np.random.randint(0, len(ann))
            return img, "image of " + ann[region_id]["phrase"].lower()
        out = {}
        for i in range(0, len(ann)):
            tmp = {}
            tmp["sentences"] = "image of " + ann[i]["phrase"].lower()
            bbox = [
                [
                    int(ann[i]["x"]),
                    int(ann[i]["y"]),
                    int(ann[i]["x"]) + int(ann[i]["width"]),
                    int(ann[i]["y"]) + int(ann[i]["height"]),
                ]
            ]
            tmp["bbox"] = bbox
            if (bbox[0][3] - bbox[0][1]) * (bbox[0][2] - bbox[0][0]) > 0.05 * image_sizes[0] * image_sizes[1]:
                out[str(i)] = tmp

        if self.img_path:
            return img, out, image_sizes, img_path
        return img, out, image_sizes

    def __len__(self):
        return len(self.files)

    @property
    def name(self):
        return "vg"


def get_VG_dataset(cfg: DictConfig):
    datadir = cfg.data_path
    transform_train, transform_test = get_flickr_transform(image_size=cfg.image_size)
    ds_train = ImageLoader(datadir, split="train", transform=transform_train)
    return ds_train


def get_VGtest_dataset(cfg: DictConfig):
    datadir = cfg.val_path
    transform_train, transform_test = get_flickr_transform(image_size=cfg.image_size)
    ds_test = ImageLoader(datadir, split="test", transform=transform_test)
    return ds_test


def sync_data(files, annotations, imgs_data, split="train"):
    out = {}
    for ann in tqdm(annotations):
        if ann["id"] in files:
            tmp = []
            for item in ann["regions"]:
                if len(item["phrase"].split(" ")) < 80:
                    tmp.append(item)
            out[ann["id"]] = tmp
    return out
