from argparse import ArgumentParser

import kornia as K
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from data.datamodule import DataModule
from models.centernet_with_coam import CenterNetWithCoAttention
from utils.general import get_easy_dict_from_yaml_file


def import_dataloader_collate_fn(method):
    if method == "centernet":
        from models.centernet_with_coam import dataloader_collate_fn
    else:
        raise NotImplementedError(f"Unknown method {method}")
    return dataloader_collate_fn


def suppress_non_maximum(bboxes):
    maximum = np.ones(len(bboxes))
    results = []
    for i in range(len(bboxes)):
        if not maximum[i]:
            continue
        results.append(bboxes[i])
        for j in range(i + 1, len(bboxes)):
            left_object = shapely.geometry.box(*bboxes[i])
            right_object = shapely.geometry.box(*bboxes[j])
            iou = left_object.intersection(right_object).area / left_object.union(right_object).area
            if iou > 0.5:
                maximum[j] = 0
            if right_object.intersection(left_object).area / right_object.area > 0.5:
                maximum[j] = 0
    return results


class SinglePair(Dataset):
    def __init__(self, method):
        self.path_to_image1 = "demo_images/img1.png"
        self.path_to_image2 = "demo_images/img2.jpg"
        self.split = "test"
        self.marshal_getitem_data = self.import_method_specific_functions(method)

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def read_image_as_tensor(self, path_to_image):
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def get_dummy_target_annotations_in_coco_format(self):
        # replace this if GT annotations are known
        x, y, w, h = 0, 0, 5, 5
        four_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        coco_annotation = {
            "bbox": [*four_corners[0], *four_corners[2]],
            "segmentation": [four_corners.reshape(-1)],
        }
        return [coco_annotation]

    def __len__(self):
        return 1

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        image1 = self.read_image_as_tensor(self.path_to_image1)
        image2 = self.read_image_as_tensor(self.path_to_image2)
        return {
            "image1": image1,
            "image2": image2,
            "image1_target_annotations": self.get_dummy_target_annotations_in_coco_format(),
            "image2_target_annotations": self.get_dummy_target_annotations_in_coco_format(),
        }


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = CenterNetWithCoAttention.add_model_specific_args(parser)
    parser = DataModule.add_data_specific_args(parser)
    parser.add_argument("--load_weights_from", type=str, default=None)
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args()
    args = EasyDict(vars(args))
    configs = get_easy_dict_from_yaml_file(args.config_file)

    # copy cmd line configs into config file configs, overwriting any duplicates
    for key in args.keys():
        if args[key] is not None:
            configs[key] = args[key]
        elif key not in configs.keys():
            configs[key] = args[key]

    dataset = SinglePair(method="centernet")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=import_dataloader_collate_fn("centernet"),
    )

    model = CenterNetWithCoAttention(configs)
    model.eval()

    for batch_with_single_item in dataloader:
        left_predicted_bboxes, right_predicted_bboxes, _, _ = model.test_step(
            batch_with_single_item, 0
        )
        left_predicted_bboxes = left_predicted_bboxes[0][0].detach().numpy()
        right_predicted_bboxes = right_predicted_bboxes[0][0].detach().numpy()
        top_n_bboxes = 10
        left_predicted_bboxes = suppress_non_maximum(
            left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:top_n_bboxes, :4]
        )
        right_predicted_bboxes = suppress_non_maximum(
            right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:top_n_bboxes, :4]
        )
        figure, subplots = plt.subplots(1, 2)
        subplots[0].imshow(K.tensor_to_image(batch_with_single_item["left_image"][0]))
        subplots[1].imshow(K.tensor_to_image(batch_with_single_item["right_image"][0]))
        for index, bboxes in zip([0, 1], [left_predicted_bboxes, right_predicted_bboxes]):
            for bbox in bboxes:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=1, edgecolor="g", facecolor="none"
                )
                subplots[index].add_patch(rect)
        for subplot in subplots:
            subplot.axis("off")
        plt.show()
