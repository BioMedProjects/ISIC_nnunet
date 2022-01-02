import os
import glob

from preprocessing import get_features, preprocess, get_attributes_df
from divider import Divider
from creator import TaskCreatorNNUnet


file_dir = 'result.json'
attributes_dir = 'ISIC2018_Task2_Training_GroundTruth_v3'

TARGET_NUMBER = 1

divider = Divider(TARGET_NUMBER)
task_creator = TaskCreatorNNUnet(
    training_source_img_dir_path="ISIC2018_Task1-2_Training_Input",
    training_source_label_dir_path="ISIC2018_Task1_Training_GroundTruth",
    validation_source_img_dir_path="ISIC2018_Task1-2_Validation_Input",
    validation_source_label_dir_path="ISIC2018_Task1_Validation_GroundTruth",
    target_dir_path="../../raid/ai_biomed"
)

data, names = get_features(file_dir)
reduced_data = preprocess(data)

cropped_names = [name[:-7] for name in names]

attributes_files = [os.path.basename(x) for x in glob.glob(attributes_dir + '**/*.png')]
attributes_names = ['milia_like_cyst', 'negative_network', 'streaks', 'globules', 'pigment_network']

attributes_df = get_attributes_df(cropped_names, attributes_files, attributes_names)

cluster_labels, representatives = divider.get_representatives(reduced_data)
print(representatives)

representatives_names = [cropped_names[i] for i in representatives]

attributes_df['cluster'] = cluster_labels
divider.get_ISIC_atributes_summary(attributes_df)
