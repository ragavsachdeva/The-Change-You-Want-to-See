import io
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from easydict import EasyDict
from einops import rearrange
from PIL import Image
from pycocotools import mask as coco_mask_utils

def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """
    Reads a yaml and returns it as an easy dict.
    """
    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)


def single_coco_annotation_to_mask_image(annotation, image_shape_as_hw):
    """
    Converts a single object annotation which can be polygons, uncompressed RLE,
    or RLE to binary mask.
    """
    h, w = image_shape_as_hw
    segm = annotation["segmentation"]
    if type(segm) == list:
        rles = coco_mask_utils.frPyObjects(segm, h, w)
        rle = coco_mask_utils.merge(rles)
    elif type(segm["counts"]) == list:
        rle = coco_mask_utils.frPyObjects(segm, h, w)
    else:
        rle = annotation["segmentation"]
    m = coco_mask_utils.decode(rle)
    return m


def coco_annotations_to_mask_np_array(list_of_annotations, image_shape_as_hw):
    """
    Given a list of object annotations, returns a single binary mask.
    """
    mask = np.zeros(image_shape_as_hw, dtype=bool)
    for annotation in list_of_annotations:
        object_mask = single_coco_annotation_to_mask_image(annotation, image_shape_as_hw)
        mask = np.maximum(object_mask, mask)
    return mask


def cache_data_triton(path_to_dataset, path_to_file, machine):
    """
    When running jobs on triton/slurm, automatically cache
    data in the temp folder.
    """
    if machine not in ["triton", "slurm"]:
        return os.path.join(path_to_dataset, path_to_file)
    caching_location = os.path.join("/tmp/", path_to_file)
    if not os.path.exists(caching_location):
        Path(caching_location).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(path_to_dataset, path_to_file), caching_location)
        os.sync()
    return caching_location
