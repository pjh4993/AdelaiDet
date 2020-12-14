#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
from torch.utils.data.dataset import Dataset
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    supp_set_id = predictions[0]['supp_set_id']
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret, supp_set_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--json-file", default="", type=str)
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    origin_dicts = list(DatasetCatalog.get(args.dataset))
    origin_metadata = MetadataCatalog.get(args.dataset)

    if len(args.json_file) != 0:
        new_meta = origin_metadata.as_dict()
        new_meta.pop('evaluator_type')
        if 'json_file' in new_meta:
            new_meta.pop('json_file')
        if 'image_root' in new_meta:
            image_root = new_meta['image_root']
            new_meta.pop('image_root')
        if 'dirname' in new_meta:
            image_root = os.path.join(new_meta['dirname'], 'JPEGImages')
            new_meta.pop('dirname')
        new_meta['name'] = 'visualize'
        register_coco_instances("visualize", new_meta ,args.json_file, image_root)
        dicts = list(DatasetCatalog.get("visualize"))
        metadata = MetadataCatalog.get(args.dataset)
        if hasattr(metadata , 'thing_dataset_id_to_contiguous_id') is False:
            metadata.thing_dataset_id_to_contiguous_id = {
                k: k for k in range(len(metadata.thing_classes))
            }

    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        img = cv2.resize(img, (dic['width'], dic['height']))
        basename = os.path.basename(dic["file_name"])

        predictions, supp_set_id = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        vis_imgs = [vis_pred, vis_gt]
        for img_id in supp_set_id:
            if isinstance(img_id, int):
                sup_img = cv2.imread(os.path.join(os.path.dirname(dic["file_name"]), '{:012d}.jpg'.format(img_id)), cv2.IMREAD_COLOR)[:,:,::-1]
            else:
                sup_img = cv2.imread(os.path.join(os.path.dirname(dic["file_name"]), '{}.jpg'.format(img_id)), cv2.IMREAD_COLOR)[:,:,::-1]
            sup_dic = [x for x in origin_dicts if x['image_id'] == img_id]
            vis = Visualizer(sup_img, origin_metadata)
            vis_sup = vis.draw_dataset_dict(sup_dic[0]).get_image()
            ratio = dic['height'] / vis_sup.shape[0]
            vis_sup = cv2.resize(vis_sup, (int(vis_sup.shape[1] * ratio), dic['height']))
            vis_imgs.append(vis_sup)

        concat = np.concatenate(vis_imgs, axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
