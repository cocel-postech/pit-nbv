# Author : Doyu Lim (2023/7)
# Split shapenetcore.v1 into train/validation/test according to csv file.

import os
import pandas as pd
import shutil

def split_dataset(dataset_path, csv_file_path):

    # Create data save folder
    train_folder = os.path.join(dataset_path, 'train')
    test_folder = os.path.join(dataset_path, 'test')
    valid_folder = os.path.join(dataset_path, 'valid')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_file_path)
    train=0
    test=0
    valid=0

    # save into train, test, valid
    for index, row in df.iterrows():
        id = row['id']                              # 6 digit
        synsetId = str(row['synsetId']).zfill(8)    # 8 digit
        #subSynsetId = row['subSynsetId']           # 8 digit
        modelId = str(row['modelId'])               # 30 string
        data_split = row['split']                   # test/train/valid

        # Set original dir and target dir path
        source_model_folder = os.path.join(dataset_path, 'raw', synsetId, modelId)
        
        if data_split == 'train':
            target_model_folder = os.path.join(train_folder, synsetId, modelId)
            train += 1
        elif data_split == 'test':
            target_model_folder = os.path.join(test_folder, synsetId, modelId)
            test += 1
        elif data_split == 'val':
            target_model_folder = os.path.join(valid_folder, synsetId, modelId)
            valid += 1
        else:
            raise ValueError(f"Invalid data split label '{data_split}' in row {index}")

        # copy original dir to target dir
        shutil.copytree(source_model_folder, target_model_folder)

    print("Dataset split and saved successfully!")
    print(f"# of train={train}, test={test}, valid={valid}")


if __name__ == '__main__':

    dataset_path = '/media/doyu/SLAM/data/ShapeNetCore.v1'
    csv_file_path = 'all.csv'

    split_dataset(dataset_path, csv_file_path)