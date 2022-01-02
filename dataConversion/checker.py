from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

# WSTAW SCIEZKE
base = 'data'
task_name = 'Task001_ISIC'
folder = join('', task_name)

FILES_TO_MISS = 4

target_imagesTr = join(folder, "imagesTr")
target_labelsTr = join(folder, "labelsTr")

labels_dir_tr = join(base, 'ISIC2018_Task1_Training_GroundTruth')
images_dir_tr = join(base, 'ISIC2018_Task1-2_Training_Input')

assert isfile(join(folder, "dataset.json")), "There needs to be a dataset.json file in folder, folder=%s" % folder
assert isdir(target_imagesTr), "There needs to be a imagesTr subfolder in folder, folder=%s" % folder
assert isdir(target_labelsTr), "There needs to be a labelsTr subfolder in folder, folder=%s" % folder

dataset = load_json(join(folder, "dataset.json"))
training_cases = dataset['training']
num_modalities = len(dataset['modality'].keys())
test_cases = dataset['test']
expected_train_identifiers = [i['image'].split("/")[-1][:-7] for i in training_cases]
expected_test_identifiers = [i.split("/")[-1][:-7] for i in test_cases]

## check training set
nii_files_in_imagesTr = subfiles(target_imagesTr, suffix=".nii.gz", join=False)
nii_files_in_labelsTr = subfiles(target_labelsTr, suffix=".nii.gz", join=False)

training_cases = subfiles(labels_dir_tr, suffix='.png', join=False)
case_to_fix = subfiles(target_labelsTr, suffix='.nii.gz', join=False)

label_suffix = '_segmentation.png'

# check all cases
if len(expected_train_identifiers) != len(np.unique(expected_train_identifiers)): raise RuntimeError("found duplicate training cases in dataset.json")

counter = 0

print("Verifying training set")
for c in expected_train_identifiers:
    counter += 1
    if counter < FILES_TO_MISS:
        continue

    print("checking case", c)
    # check if all files are present
    expected_label_file = join(folder, "labelsTr", c + ".nii.gz")
    expected_image_files = [join(folder, "imagesTr", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
    if isfile(expected_label_file):
        pass
    else:
        print("could not find label file for case %s. Expected file: \n%s" % (c, expected_label_file))

        label_name = c + label_suffix
        input_label_file = join(labels_dir_tr, label_name)
        input_image_file = join(images_dir_tr, (c + '.jpg'))

        output_image_file = join(target_imagesTr, c)  
        output_seg_file = join(target_labelsTr, c)

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        convert_2d_image_to_nifti(input_label_file, output_seg_file, is_seg=True,
                                transform=lambda x: (x == 255).astype(int))
        print("Convering case ", c)


    if all([isfile(i) for i in expected_image_files]):
        pass
    else:
        print("some image files are missing for case %s. Expected files:\n %s" % (c, expected_image_files))
        label_name = c + label_suffix
        input_label_file = join(labels_dir_tr, label_name)
        input_image_file = join(images_dir_tr, (c + '.jpg'))

        output_image_file = join(target_imagesTr, c)  
        output_seg_file = join(target_labelsTr, c)

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        convert_2d_image_to_nifti(input_label_file, output_seg_file, is_seg=True,
                                transform=lambda x: (x == 255).astype(int))
        print("Convering case ", c)



