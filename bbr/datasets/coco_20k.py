import os
import cv2

import numpy as np
import torch
import tempfile
import zipfile
from omegaconf import DictConfig
from pycocotools.coco import COCO

from bbr.datasets.tfs import get_coco20k_transform
from bbr.utils.utils_grounding import normal_box_to_image
from bbr.utils.utils import get_root_path


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    if len(anno) == 0:
        return False
    if _has_only_empty_bbox(anno):
        return False

    return True


def get_dict_coco(coco_obj, image_objects, names, supercats):
    dictionary = {}
    for n, img_obj in enumerate(image_objects):
        img_id, height, width, _ = (
            img_obj["id"],
            img_obj["height"],
            img_obj["width"],
            img_obj["file_name"],
        )
        annotations = []
        bbox_annIds = coco_obj.getAnnIds(imgIds=img_id)
        bbox_anns = coco_obj.loadAnns(bbox_annIds)
        for bbox_ann in bbox_anns:
            bbox = bbox_ann["bbox"]
            x_min, x_max, y_min, y_max = [
                bbox[0],
                bbox[0] + bbox[2],
                bbox[1],
                bbox[1] + bbox[3],
            ]
            bbox = [x_min, y_min, x_max, y_max]
            bbox_norm = [x_min / width, y_min / height, x_max / width, y_max / height]
            category_id = bbox_ann["category_id"]
            supercategory = supercats[category_id]
            category_name = names[category_id]
            object_id = bbox_ann["id"]
            segmentation = bbox_ann["segmentation"]
            bbox_entity = {
                "image_id": img_id,
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "supercategory": supercategory,
                "category": category_name,
                "category_id": category_id,
                "obj_id": object_id,
                "segmentation": segmentation,
            }
            annotations.append(bbox_entity)
        dictionary[str(img_id)] = {
            "size": (height, width, 3),
            "annotations": annotations,
        }
    return dictionary


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.img_folder = os.path.join(root, "images")
        self.ann_20k = os.path.join(get_root_path(), "datasets/coco_instances_train2014_sel20k.zip")
        self.instances_file = self.extract_json_to_temp(self.ann_20k)

        coco_obj = COCO(self.instances_file)
        categories = coco_obj.loadCats(coco_obj.getCatIds())
        names = {}
        supercats = {}
        for cat in categories:
            names[cat["id"]] = cat["name"]
            supercats[cat["id"]] = cat["supercategory"]
        imageIds = coco_obj.getImgIds()
        image_objects = coco_obj.loadImgs(imageIds)
        self.split = {str(o["id"]): "train" if "train" in o["file_name"] else "val" for o in image_objects}
        self.categories = {cat["id"]: cat["name"] for cat in coco_obj.cats.values()}
        self.annotations = get_dict_coco(coco_obj, image_objects, names, supercats)
        self.annotations = {k: v for k, v in self.annotations.items() if has_valid_annotation(v["annotations"])}

        self.files = list(self.annotations.keys())
        self.transform = transform
        print("num of data:{}".format(len(self.files)))

    def __getitem__(self, index):
        item = str(self.files[index])
        split = self.split[item]
        img_string = f"COCO_{split}2014_" + "0" * (12 - len(item)) + item + ".jpg"
        img_path = os.path.join(self.img_folder, split + "2014", img_string)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = (img.shape[0], img.shape[1])
        gt_size = (img.shape[0], img.shape[1])

        img = self.transform(img)

        gt_annot = self.annotations[item]["annotations"]

        coco_gt_bbox = np.stack([a["bbox"] for a in gt_annot])
        coco_gt_bbox_norm = normal_box_to_image(
            torch.as_tensor(coco_gt_bbox), h=img.shape[-2], w=img.shape[-1], orig_h=h, orig_w=w
        )

        return (img, coco_gt_bbox_norm, gt_size, item)

    def __len__(self):
        return len(self.files)

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

    def extract_json_to_temp(self, zip_path):
        temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        json_filename = [filename for filename in os.listdir(temp_dir) if filename.endswith(".json")][0]
        json_path = os.path.join(temp_dir, json_filename)

        return json_path


def get_coco_dataset(cfg: DictConfig, test=False):
    datadir = cfg.data_path
    transform_train, _ = get_coco20k_transform(image_size=cfg.image_size, test=test)

    ds_train = ImageLoader(datadir, transform=transform_train)
    return ds_train


def get_coco_val_dataset(cfg: DictConfig, test=False):
    datadir = cfg.val_path
    _, transform_test = get_coco20k_transform(image_size=cfg.image_size, test=test)

    ds_test = ImageLoader(datadir, transform=transform_test)
    return ds_test
