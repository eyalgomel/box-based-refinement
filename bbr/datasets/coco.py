import os
import cv2

import numpy as np
import torch
from omegaconf import DictConfig
from pycocotools.coco import COCO

from bbr.datasets.tfs import get_flickr_transform


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    if len(anno) == 0:
        return False
    if _has_only_empty_bbox(anno):
        return False

    return True


def get_dict_coco(coco_obj, image_objects, caption_objects, names, supercats):
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
        capsAnnIds = caption_objects.getAnnIds(imgIds=img_id)
        caps = caption_objects.loadAnns(capsAnnIds)
        sentences = [cap["caption"] for cap in caps]
        dictionary[str(img_id)] = {
            "size": (height, width, 3),
            "queries": sentences,
            "captions": sentences,
            "annotations": annotations,
        }
    return dictionary


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", mode="sentence"):
        self.captions_file = os.path.join(root, "annotations", "captions_" + split + "2014.json")
        self.instances_file = os.path.join(root, "annotations", "instances_" + split + "2014.json")
        self.img_folder = os.path.join(root, "images", split + "2014")

        coco_obj = COCO(self.instances_file)
        caption_objects = COCO(self.captions_file)
        categories = coco_obj.loadCats(coco_obj.getCatIds())
        names = {}
        supercats = {}
        for cat in categories:
            names[cat["id"]] = cat["name"]
            supercats[cat["id"]] = cat["supercategory"]
        imageIds = coco_obj.getImgIds()
        image_objects = coco_obj.loadImgs(imageIds)

        self.annotations = get_dict_coco(coco_obj, image_objects, caption_objects, names, supercats)
        self.annotations = {k: v for k, v in self.annotations.items() if has_valid_annotation(v["annotations"])}
        self.files = list(self.annotations.keys())

        self.transform = transform
        self.split = split
        self.mode = mode

    def __getitem__(self, index):
        item = str(self.files[index])
        img_path = os.path.join(self.img_folder, f"COCO_{self.split}2014_" + "0" * (12 - len(item)) + item + ".jpg")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        size = (img.shape[0], img.shape[1])

        img = self.transform(img)
        queries = self.annotations[item]["queries"]
        annotations = self.annotations[item]["annotations"]

        if self.mode == "categories":
            text = self._get_random_category(annotations, prefix="image of a ")
        elif self.mode == "sentence":
            text = self._get_random_masked_caption(queries)
        else:
            raise NotImplementedError
        if self.split != "train":
            filtered_annot = [
                {"bbox": ann["bbox"], "category": ann["category"]} for ann in annotations if len(ann["bbox"]) > 0
            ]
            return img, filtered_annot, size

        return img, text

    def _get_random_masked_caption(self, queries, prefix="image of "):
        region_id = np.random.randint(0, len(queries))
        a = len(queries[region_id].split(" "))
        mask = np.random.randint(0, 5, a) > 0
        text_list = np.array(queries[region_id].split(" "))[mask].tolist()
        text = " ".join(text_list)
        text = prefix + text
        return text

    def _get_random_category(self, annotations, prefix="image of "):
        category = np.random.choice([ann["category"] for ann in annotations])
        category = prefix + category
        return category

    def collate_fn_test(self, batch):
        images = list()
        annot = list()
        sizes = list()

        for b in batch:
            images.append(b[0])
            annot.append(b[1])
            sizes.append(b[2])

        images = torch.stack(images, dim=0)

        return images, annot, sizes

    def __len__(self):
        if self.split == "val":
            return 10000
        return len(self.files) * 1


def get_coco_dataset(cfg: DictConfig):
    datadir = cfg.data_path
    mode = cfg.get("mode", "sentence")
    transform_train, transform_test = get_flickr_transform(image_size=cfg.image_size)
    ds_train = ImageLoader(datadir, split="train", transform=transform_train, mode=mode)
    return ds_train
