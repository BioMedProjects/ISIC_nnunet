import shutil

import numpy as np
import batchgenerators.utilities.file_and_folder_operations as ff_operations

from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti


class TaskCreatorNNUnet:

    def __init__(
            self,
            training_source_img_dir_path,
            training_source_label_dir_path,
            validation_source_img_dir_path,
            validation_source_label_dir_path,
            target_dir_path
    ):
        self.training_source_img_dir_path = training_source_img_dir_path
        self.training_source_label_dir_path = training_source_label_dir_path

        self.validation_source_img_dir_path = validation_source_img_dir_path
        self.validation_source_label_dir_path = validation_source_label_dir_path

        self.target_dir_path = target_dir_path

    def create_task(self, task_name, training_cases):
        target_base = join(self.target_dir_path, task_name)

        target_imagesTr = join(target_base, "imagesTr")
        target_imagesTs = join(target_base, "imagesTs")
        target_labelsTs = join(target_base, "labelsTs")
        target_labelsTr = join(target_base, "labelsTr")

        ff_operations.maybe_mkdir_p(target_imagesTr)
        ff_operations.maybe_mkdir_p(target_labelsTs)
        ff_operations.maybe_mkdir_p(target_imagesTs)
        ff_operations.maybe_mkdir_p(target_labelsTr)

        for tr in training_cases:
            unique_name = tr[:-17]
            input_segmentation_file = join(self.training_source_label_dir_path, tr)
            input_image_file = join(self.training_source_img_dir_path, (unique_name + '.jpg'))

            output_image_file = join(target_imagesTr, unique_name)
            output_seg_file = join(target_labelsTr, unique_name)

            convert_2d_image_to_nifti(
                input_image_file,
                output_image_file,
                is_seg=False
            )

            convert_2d_image_to_nifti(
                input_segmentation_file,
                output_seg_file,
                is_seg=True,
                transform=lambda x: (x == 255).astype(int)
            )

        testing_cases = subfiles(self.validation_source_label_dir_path, suffix='.png', join=False)

        for ts in testing_cases:
            unique_name = ts[:-17]
            input_segmentation_file = join(self.validation_source_label_dir_path, ts)
            input_image_file = join(self.validation_source_img_dir_path, (unique_name + '.jpg'))

            output_image_file = join(target_imagesTs, unique_name)
            output_seg_file = join(target_labelsTs, unique_name)

            convert_2d_image_to_nifti(
                input_image_file,
                output_image_file,
                is_seg=False
            )
            convert_2d_image_to_nifti(
                input_segmentation_file,
                output_seg_file,
                is_seg=True,
                transform=lambda x: (x == 255).astype(int)
            )
        generate_dataset_json(
            join(target_base, 'dataset.json'),
            target_imagesTr,
            target_imagesTs,
            ('Red', 'Green', 'Blue'),
            labels={0: 'background', 1: 'melanoma'},
            dataset_name=task_name,
            license='hands off!'
        )
