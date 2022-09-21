import os
import pickle
import random
from copy import deepcopy

import kornia as K
import numpy as np
from einops import rearrange
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

import utils.general
import utils.geometry
from data.augmentation import AugmentationPipeline
from utils.general import cache_data_triton

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


class InpatinedCocoDataset(Dataset):
    def __init__(self, path_to_dataset, split, method, image_transformation, machine="local"):
        self.path_to_dataset = path_to_dataset
        train_val_test_split = self.get_train_val_test_split(split)
        self.indicies = train_val_test_split[split]
        self.split = split
        self.machine = machine
        self.inpainted_image_names = self.get_inpainted_image_names()
        self.image_augmentations = AugmentationPipeline(
            mode=split, path_to_dataset=path_to_dataset, image_transformation=image_transformation
        )
        self.marshal_getitem_data = self.import_method_specific_functions(method)

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def get_train_val_test_split(self, split):
        train_val_test_split_file_path = os.path.join(self.path_to_dataset, "data_split.pkl")
        if os.path.exists(train_val_test_split_file_path):
            with open(train_val_test_split_file_path, "rb") as file:
                return pickle.load(file)
        indices_of_coco_images = np.load(os.path.join(self.path_to_dataset, "list_of_indices.npy"))
        np.random.shuffle(indices_of_coco_images)
        if split == "test":
            train_val_test_split = {
                "test": indices_of_coco_images,
            }
        else:
            number_of_images = len(indices_of_coco_images)
            number_of_train_images = int(0.95 * number_of_images)
            train_val_test_split = {
                "train": indices_of_coco_images[:number_of_train_images],
                "val": indices_of_coco_images[number_of_train_images:],
            }
        with open(train_val_test_split_file_path, "wb") as file:
            pickle.dump(train_val_test_split, file)
        return train_val_test_split

    def get_inpainted_image_names(self):
        filenames_as_list = list(os.listdir(os.path.join(self.path_to_dataset, "inpainted")))
        inpainted_image_names = dict()
        for filename in filenames_as_list:
            index = int(filename.split("_")[0])
            if index in inpainted_image_names.keys():
                inpainted_image_names[index].append(filename)
            else:
                inpainted_image_names[index] = [filename]
        return inpainted_image_names

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def get_inpainted_objects_bitmap_from_image_path(self, image_path, bit_length):
        if "inpainted" not in image_path:
            return 0
        bitmap_string = image_path.split("mask")[1].split(".")[0]
        if bitmap_string == "":
            return (2**bit_length) - 1
        return int(bitmap_string)

    def add_random_objects(self, image_as_tensor, item_index):
        all_indices_except_current = list(range(item_index)) + list(
            range(item_index + 1, len(self.indicies))
        )
        random_image_index = random.choice(all_indices_except_current)
        index = self.indicies[random_image_index]
        original_image = self.read_image_as_tensor(
            cache_data_triton(self.path_to_dataset, f"images_and_masks/{index}.png", self.machine)
        )
        annotation_path = cache_data_triton(
            self.path_to_dataset, f"metadata/{index}.npy", self.machine
        )
        annotations = np.load(annotation_path, allow_pickle=True)
        (
            original_image_resized_to_current,
            annotations_resized,
        ) = utils.geometry.resize_image_and_annotations(
            original_image, image_as_tensor.shape[-2:], annotations
        )
        annotation_mask = utils.general.coco_annotations_to_mask_np_array(
            annotations_resized, image_as_tensor.shape[-2:]
        )
        image_as_tensor = rearrange(image_as_tensor, "c h w -> h w c")
        original_image_resized_to_current = rearrange(
            original_image_resized_to_current, "c h w -> h w c"
        )
        image_as_tensor[annotation_mask] = original_image_resized_to_current[annotation_mask]
        return rearrange(image_as_tensor, "h w c -> c h w"), annotations_resized

    def __len__(self):
        """
        Returns the number of training/validation/testing images.
        """
        return len(self.indicies)

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        index = self.indicies[item_index]
        image_filenames = [f"images_and_masks/{index}.png"]
        for inpainted_image_name in self.inpainted_image_names[index]:
            image_filenames.append("inpainted/" + inpainted_image_name)
        if self.split == "test":
            # this if condition is important to enforce fixed test set
            image1_image_path, image2_image_path = image_filenames
        else:
            image1_image_path, image2_image_path = random.sample(image_filenames, 2)
        if random.random() < 0.5 or self.split == "test":
            annotation_path = cache_data_triton(
                self.path_to_dataset, f"metadata/{index}.npy", self.machine
            )
            image1_image_as_tensor = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, image1_image_path, self.machine)
            )
            image2_image_as_tensor = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, image2_image_path, self.machine)
            )
            image2_image_as_tensor = K.geometry.transform.resize(
                image2_image_as_tensor, image1_image_as_tensor.shape[-2:]
            )
            annotations = np.load(annotation_path, allow_pickle=True)
            image1_image_inpainted_objects = self.get_inpainted_objects_bitmap_from_image_path(
                image1_image_path, len(annotations)
            )
            image2_image_inpainted_objects = self.get_inpainted_objects_bitmap_from_image_path(
                image2_image_path, len(annotations)
            )
            changed_objects = image1_image_inpainted_objects ^ image2_image_inpainted_objects
            change_objects_indices = np.array(
                [x == "1" for x in bin(changed_objects)[2:].zfill(len(annotations))]
            )
            annotations = annotations[change_objects_indices]
        else:
            image1_image_as_tensor = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, image1_image_path, self.machine)
            )
            image2_image_as_tensor = deepcopy(image1_image_as_tensor)
            image2_image_as_tensor, annotations = self.add_random_objects(
                image2_image_as_tensor, item_index
            )
            if random.random() < 0.5:
                image1_image_as_tensor, image2_image_as_tensor = (
                    image2_image_as_tensor,
                    image1_image_as_tensor,
                )

        (
            image1_image_as_tensor,
            image2_image_as_tensor,
            transformed_image1_target_annotations,
            transformed_image2_target_annotations,
            target_image1_region_as_coco_annotation,
            target_image2_region_as_coco_annotation,
        ) = self.image_augmentations(
            image1_image_as_tensor,
            image2_image_as_tensor,
            annotations,
            index,
        )

        return {
            "image1": image1_image_as_tensor.squeeze(),
            "image2": image2_image_as_tensor.squeeze(),
            "image1_target_annotations": transformed_image1_target_annotations,
            "image2_target_annotations": transformed_image2_target_annotations,
            "image1_target_region_as_coco_annotation": target_image1_region_as_coco_annotation,
            "image2_target_region_as_coco_annotation": target_image2_region_as_coco_annotation,
        }
