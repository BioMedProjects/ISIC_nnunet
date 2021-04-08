import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

if __name__ == '__main__':
    
    base = 'data'

    task_name = 'Task1_ISIC'
    target_base = join('', task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)


    labels_dir_tr = join(base, 'ISIC2018_Task1_Training_GroundTruth')
    images_dir_tr = join(base, 'ISIC2018_Task1-2_Training_Input')
    training_cases = subfiles(labels_dir_tr, suffix='.png', join=False)


    for t in training_cases:
        unique_name = t[:-17]  
        input_segmentation_file = join(labels_dir_tr, t)
        input_image_file = join(images_dir_tr, (unique_name + '.jpg'))

        output_image_file = join(target_imagesTr, unique_name)  
        output_seg_file = join(target_labelsTr, unique_name)

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)


        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 255).astype(int))

    labels_dir_ts = join(base, 'ISIC2018_Task1_Validation_GroundTruth')
    images_dir_ts = join(base, 'ISIC2018_Task1-2_Validation_Input')
    testing_cases = subfiles(labels_dir_ts, suffix='.png', join=False)
    for ts in testing_cases:
        unique_name = ts[:-17]
        input_segmentation_file = join(labels_dir_ts, ts)
        input_image_file = join(images_dir_ts, (unique_name + '.jpg'))

        output_image_file = join(target_imagesTs, unique_name)
        output_seg_file = join(target_labelsTs, unique_name)

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 255).astype(int))

    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Red', 'Green', 'Blue'),
                          labels={1: 'street'}, dataset_name=task_name, license='hands off!')

    