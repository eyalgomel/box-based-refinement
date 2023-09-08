import os
import pickle
import cv2

import numpy as np
import torch
from omegaconf import DictConfig

from bbr.datasets.tfs import get_flickr_transform


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", img_path=False):
        annt_path = os.path.join(root, "annotations", split + ".pickle")
        with open(annt_path, "rb") as f:
            self.annotations = pickle.load(f, encoding="latin1")
        self.files = list(self.annotations.keys())
        print("num of data:{}".format(len(self.files)))
        self.transform = transform
        self.split = split
        self.img_folder = os.path.join(root, "ReferIt_Images")
        self.img_path = img_path

    def __getitem__(self, index):
        item = str(self.files[index])
        folder = (2 - len(str(int(item) // 1000))) * "0" + str(int(item) // 1000)
        img_path = os.path.join(self.img_folder, folder, "images", item + ".jpg")
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image_sizes = (img.shape[0], img.shape[1])
        img = self.transform(img)
        ann = self.annotations[item]["annotations"]
        if self.split == "train":
            region_id = np.random.randint(0, len(ann))
            return img, "image of " + ann[region_id]["query"]
        out = {}
        for i in range(0, len(ann)):
            tmp = {}
            bbox = ann[i]["bbox"]
            if (bbox[0][3] - bbox[0][1]) * (bbox[0][2] - bbox[0][0]) > 0.05 * image_sizes[0] * image_sizes[1]:
                tmp["sentences"] = "image of " + ann[i]["query"]
                tmp["bbox"] = bbox
                out[str(i)] = tmp
        if self.img_path:
            return img, out, image_sizes, img_path
        return img, out, image_sizes

    def __len__(self):
        return len(self.files)

    @property
    def name(self):
        return "referit"


def get_referit_test_dataset(cfg: DictConfig):
    datadir = cfg.val_path
    transform_train, transform_test = get_flickr_transform(image_size=cfg.image_size)
    ds_test = ImageLoader(datadir, split="test", transform=transform_test)
    return ds_test


def get_referit_dataset(cfg: DictConfig, only_test=False):
    datadir = cfg.data_path
    transform_train, transform_test = get_flickr_transform(image_size=cfg.image_size)
    ds_test = ImageLoader(datadir, split="val", transform=transform_test)
    ds_train = None
    if not only_test:
        ds_train = ImageLoader(datadir, split="train", transform=transform_train)
    return ds_train, ds_test
