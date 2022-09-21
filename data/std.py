from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import kornia as K
import numpy as np
import shapely.geometry
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

import utils.geometry


class StdDataset(Dataset):
    def __init__(self, path_to_dataset, split, method, image_transformation="identity"):
        assert split == "test"
        self.split = split
        self.path_to_dataset = path_to_dataset
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.image_transformation = image_transformation
        self.annotations = self.get_annotations()
        self.image_ids = list(self.annotations.keys())

    def get_annotations(self):
        return np.load(
            os.path.join(self.path_to_dataset, "annotations.npy"), allow_pickle=True
        ).item()

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def get_target_annotations_in_coco_format(self, bboxes):
        coco_annotations = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            four_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            coco_annotation = {
                "bbox": [*four_corners[0], *four_corners[2]],
                "segmentation": [four_corners.reshape(-1)],
            }
            coco_annotations.append(coco_annotation)
        return coco_annotations

    def random_perspective(self, image_as_tensor, annotations, type_of_image, image_index):
        aug = K.augmentation.RandomPerspective(p=1.0, return_transform=True)
        precomputed_augmentation_path = os.path.join(
            self.path_to_dataset, f"projective_augmentations/{type_of_image}/{image_index}.params"
        )
        image_as_tensor = rearrange(image_as_tensor, "... -> 1 ...")
        if os.path.exists(precomputed_augmentation_path):
            augmentation_params = torch.load(precomputed_augmentation_path)
        else:
            aug_params = aug.generate_parameters(image_as_tensor.shape)
            augmentation_params = {"projective": aug_params}
            Path(precomputed_augmentation_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(augmentation_params, precomputed_augmentation_path)
        image_as_tensor, transformation = aug(
            image_as_tensor, params=augmentation_params["projective"]
        )
        for annotation in annotations:
            bbox = rearrange(torch.Tensor(annotation["bbox"]), "four -> 1 four")
            bbox = K.geometry.bbox.transform_bbox(transformation, bbox)[0]
            annotation["bbox"] = bbox
            annotation[
                "segmentation"
            ] = utils.geometry.convert_shapely_polygon_into_coco_segmentation(
                shapely.geometry.box(*bbox)
            )
        return image_as_tensor.squeeze(), annotations

    def __len__(self):
        """
        Returns the number of testing images.
        """
        return len(self.image_ids)

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        """
        Returns 3 things:
        a reference image (3xRxC),
        a query image (3xRxC), and
        a list of coco-format change annotations corresponding to the query image.
        """
        image_id = self.image_ids[item_index]
        annotation = self.get_target_annotations_in_coco_format(self.annotations[image_id])

        image1_as_tensor = self.read_image_as_tensor(
            os.path.join(self.path_to_dataset, f"{image_id}.png")
        )
        image2_as_tensor = self.read_image_as_tensor(
            os.path.join(self.path_to_dataset, f"{image_id}_2.png")
        )
        if self.image_transformation == "projective":
            image1_as_tensor, target_annotations_1 = self.random_perspective(
                image1_as_tensor, deepcopy(annotation), "image1", item_index
            )
            image2_as_tensor, target_annotations_2 = self.random_perspective(
                image2_as_tensor, deepcopy(annotation), "image2", item_index
            )
        else:
            target_annotations_1 = annotation
            target_annotations_2 = annotation
        return {
            "image1": image1_as_tensor.squeeze(),
            "image2": image2_as_tensor.squeeze(),
            "image1_target_annotations": target_annotations_1,
            "image2_target_annotations": target_annotations_2,
            "image1_target_region_as_coco_annotation": {
                "segmentation": utils.geometry.convert_shapely_polygon_into_coco_segmentation(
                    shapely.geometry.box(0, 0, *image1_as_tensor.shape[::-1])
                )
            },
            "image2_target_region_as_coco_annotation": {
                "segmentation": utils.geometry.convert_shapely_polygon_into_coco_segmentation(
                    shapely.geometry.box(0, 0, *image2_as_tensor.shape[::-1])
                )
            },
        }
