import os
import random
import xml.etree.ElementTree as ET
from random import shuffle
import cv2

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from bbr.datasets.tfs import get_flickr_transform
from bbr.utils.utils_grounding import union


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", img_path=False):
        self.annotations_folder = os.path.join(root, "flickr30k_entities", "Annotations")
        self.img_folder = os.path.join(root, "flickr30k_images", "flickr30k_images")
        self.sentences_folder = os.path.join(root, "flickr30k_entities", "Sentences")

        split_file = os.path.join(root, split + ".txt")
        self.f = open(split_file, "r")
        self.files = self.f.readlines()

        self.transform = transform
        self.split = split
        self.img_path = img_path
        if split == "train":
            self.data = get_train_data(self.annotations_folder, self.sentences_folder, self.files)
            self.files = list(self.data.keys())
        else:
            self.data = get_test_data(self.annotations_folder, self.sentences_folder, self.files)
            self.files = list(self.data.keys())
        print("num of data:{}".format(len(self.files)))

    def __getitem__(self, index):
        item = self.files[index].strip("\n")
        meta = self.data[item]
        img_path = os.path.join(self.img_folder, item + ".jpg")
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image_sizes = (img.shape[0], img.shape[1])
        img = self.transform(img)
        if self.split == "train":
            sen = random.choice(list(meta.keys()))
            curr_meta = meta[sen]
            shuffle(curr_meta)
            opt = curr_meta[:2]
            return img, opt[0][0]
        if self.img_path:
            return img, meta, image_sizes, img_path
        return img, meta, image_sizes

    def __len__(self):
        return len(self.files) * 1

    @property
    def name(self):
        return "flickr"


def get_train_data(annotations_folder, sentences_folder, files):
    out_data = {}
    for file in tqdm(files):
        ann_path = os.path.join(annotations_folder, file.strip("\n") + ".xml")
        sen_path = os.path.join(sentences_folder, file.strip("\n") + ".txt")
        ann = get_annotations(ann_path)
        sen_list = get_sentence_data(sen_path)
        out_sen = {}
        for ix_sen, sen in enumerate(sen_list):
            items = ann["boxes"].keys()
            tmp = []
            for ix, phrase in enumerate(sen["phrases"]):
                if phrase["phrase_id"] in items:
                    curr_bboxes = ann["boxes"][phrase["phrase_id"]]
                    tmp.append(("image of " + phrase["phrase"].lower(), curr_bboxes))
            if len(tmp) > 1:
                out_sen[str(ix_sen)] = tmp
        if len(out_sen.keys()) > 1:
            out_data[file.strip("\n")] = out_sen
    return out_data


def get_test_data(annotations_folder, sentences_folder, files):
    out_data = {}
    for file in tqdm(files):
        ann_path = os.path.join(annotations_folder, file.strip("\n") + ".xml")
        sen_path = os.path.join(sentences_folder, file.strip("\n") + ".txt")
        ann = get_annotations(ann_path)
        sen_list = get_sentence_data(sen_path)
        out_items = {}
        items = ann["boxes"].keys()
        for ix, item in enumerate(items):
            item_data = {}
            tmp = []
            for ix_sen, sen in enumerate(sen_list):
                for phrase in sen["phrases"]:
                    if (
                        phrase["phrase_id"] == item
                        and area_bbox(ann["boxes"][phrase["phrase_id"]][0]) > 0.05 * ann["width"] * ann["height"]
                    ):
                        tmp.append("image of " + phrase["phrase"].lower())
                        item_data["bbox"] = union(torch.tensor(ann["boxes"][phrase["phrase_id"]]).numpy())
            if len(tmp) > 0:
                item_data["sentences"] = tmp
                out_items[item] = item_data
        out_data[file.strip("\n")] = out_items
    return out_data


def area_bbox(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_flickr_dataset(cfg: DictConfig, only_test=False):
    datadir = cfg.data_path
    transform_train, transform_test = get_flickr_transform(image_size=cfg.image_size)
    ds_test = ImageLoader(datadir, split="val", transform=transform_test)
    ds_train = None
    if not only_test:
        ds_train = ImageLoader(datadir, split="train", transform=transform_train)
    return ds_train, ds_test


def get_flickr1K_dataset(cfg: DictConfig):
    datadir = cfg.val_path
    _, transform_test = get_flickr_transform(cfg.image_size)
    ds_test = ImageLoader(datadir, split="test", transform=transform_test)
    return ds_test


def norm_bbox(curr_bbox, image_sizes, resize, shift, max_value):
    x = curr_bbox[0]
    y = curr_bbox[1]
    bbox_width = curr_bbox[2] - curr_bbox[0]
    bbox_height = curr_bbox[3] - curr_bbox[1]
    image_width, image_height = image_sizes
    left_bottom_x = np.maximum(x / image_width * resize - shift, 0)
    left_bottom_y = np.maximum(y / image_height * resize - shift, 0)

    right_top_x = np.minimum((x + bbox_width) / image_width * resize - shift, max_value)
    right_top_y = np.minimum((y + bbox_height) / image_height * resize - shift, max_value)
    return left_bottom_x, left_bottom_y, right_top_x, right_top_y


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset
    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to
    """
    with open(fn, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data["phrases"].append(
                {
                    "first_word_index": index,
                    "phrase": phrase,
                    "phrase_id": p_id,
                    "phrase_type": p_type,
                }
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset
    input:
      fn - full file path to the annotations file to parse
    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info = {"boxes": {}, "scene": [], "nobox": []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in anno_info["boxes"]:
                    anno_info["boxes"][box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text) - 1
                ymin = int(box_container[0].findall("ymin")[0].text) - 1
                xmax = int(box_container[0].findall("xmax")[0].text) - 1
                ymax = int(box_container[0].findall("ymax")[0].text) - 1
                anno_info["boxes"][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    anno_info["nobox"].append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    anno_info["scene"].append(box_id)

    return anno_info
