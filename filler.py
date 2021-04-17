import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
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

    labels_dir_tr = join(base, 'ISIC2018_Task1_Training_GroundTruth')
    images_dir_tr = join(base, 'ISIC2018_Task1-2_Training_Input')

    training_cases = subfiles(labels_dir_tr, suffix='.png', join=False)
    case_to_fix = subfiles(target_labelsTr, suffix='.nii.gz', join=False)

    label_suffix = '_segmentation.png'

    for case in case_to_fix:
        # ISIC_0000000.nii.gz
        unique_case_name = case[:-7]
        path_to_file = join(target_labelsTr, case)

        fileSize = os.path.getsize(path_to_file)

        if fileSize == 0:
            print("Converting case: ", unique_case_name)
            # ISIC_0000000_segmentation.png
            label_name = unique_case_name + label_suffix
            input_label_file = join(labels_dir_tr, label_name)
            input_image_file = join(images_dir_tr, (unique_case_name + '.jpg'))

            output_image_file = join(target_imagesTr, unique_case_name)  
            output_seg_file = join(target_labelsTr, unique_case_name)

            convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)


            convert_2d_image_to_nifti(input_label_file, output_seg_file, is_seg=True,
                                    transform=lambda x: (x == 255).astype(int))
 
    