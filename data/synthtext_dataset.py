import os
from copy import deepcopy
from pathlib import Path

import h5py
import kornia as K
import numpy as np
import shapely.geometry
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

import utils.geometry


class SynthTextDataset(Dataset):
    def __init__(self, path_to_dataset, split, method, image_transformation="identity"):
        assert split == "test"
        self.split = split
        self.path_to_dataset = path_to_dataset
        self.synthetic_images = h5py.File(os.path.join(path_to_dataset, "custom.h5"), "r")
        self.image_transformation = image_transformation
        self.original_image_names, self.synthetic_image_names = self.get_paths_of_test_images()
        self.marshal_getitem_data = self.import_method_specific_functions(method)

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def get_paths_of_test_images(self):
        h5_keys = sorted(self.synthetic_images["data"].keys())
        synthetic_image_names = []
        original_image_names = []
        bg_img_directory_path = os.path.join(self.path_to_dataset, "bg_img")
        files_in_bg_img_directory = os.listdir(bg_img_directory_path)
        for key in h5_keys:
            synthetic_image_names.append(key)
            synth_image_name_and_extension = key.split(".")
            original_image_name_with_extension = [
                filename
                for filename in files_in_bg_img_directory
                if filename.split(".")[0] == synth_image_name_and_extension[0]
            ][0]
            original_image_names.append(original_image_name_with_extension)
        return original_image_names, synthetic_image_names

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def get_target_annotations_in_coco_format(self, bboxes):
        """
        bboxes.shape: 2x4xN
            2 -> x,y
            4 -> four corners (clockwise, starting from top left)
            N -> number of boxes
        """
        bboxes = np.array(bboxes)
        if len(bboxes.shape) == 2:
            bboxes = bboxes[..., np.newaxis]
        bboxes = rearrange(bboxes, "two four n -> n four two")
        annotations = []
        for bbox in bboxes:
            x, y = bbox[:, 0], bbox[:, 1]
            annotation = {
                "bbox": [min(x), min(y), max(x), max(y)],
                "segmentation": [bbox.reshape(-1)],
            }
            annotations.append(annotation)
        return annotations

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
        return len(self.synthetic_image_names)

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
        original_image_name = self.original_image_names[item_index]
        original_image_path = os.path.join(self.path_to_dataset, f"bg_img/{original_image_name}")
        original_image_as_tensor = self.read_image_as_tensor(original_image_path)
        synth_image_name = self.synthetic_image_names[item_index]
        synth_image_as_tensor = (
            K.image_to_tensor(self.synthetic_images["data"][synth_image_name][...])
            .squeeze()
            .float()
            / 255.0
        )
        original_image_as_tensor = K.geometry.transform.resize(
            original_image_as_tensor, synth_image_as_tensor.shape[-2:]
        )
        change_bboxes = self.synthetic_images["data"][synth_image_name].attrs["wordBB"]
        target_annotations = self.get_target_annotations_in_coco_format(change_bboxes)
        if self.image_transformation == "projective":
            original_image_as_tensor, target_annotations_1 = self.random_perspective(
                original_image_as_tensor, deepcopy(target_annotations), "original", item_index
            )
            synth_image_as_tensor, target_annotations_2 = self.random_perspective(
                synth_image_as_tensor, deepcopy(target_annotations), "synth", item_index
            )
        else:
            target_annotations_1 = target_annotations
            target_annotations_2 = target_annotations
        return {
            "image1": original_image_as_tensor.squeeze(),
            "image2": synth_image_as_tensor.squeeze(),
            "image1_target_annotations": target_annotations_1,
            "image2_target_annotations": target_annotations_2,
            "image1_target_region_as_coco_annotation": {
                "segmentation": utils.geometry.convert_shapely_polygon_into_coco_segmentation(
                    shapely.geometry.box(0, 0, *original_image_as_tensor.shape[::-1])
                )
            },
            "image2_target_region_as_coco_annotation": {
                "segmentation": utils.geometry.convert_shapely_polygon_into_coco_segmentation(
                    shapely.geometry.box(0, 0, *original_image_as_tensor.shape[::-1])
                )
            },
        }
