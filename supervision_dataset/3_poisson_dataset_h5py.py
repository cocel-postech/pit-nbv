# Author : Doyu Lim (2024/5)
# PCD, CSV -> HDF5 : Generate HDF5 file

import os
import h5py
import open3d as o3d
import numpy as np
import pandas as pd
import fcntl

class dataset_generator_shapenet():

    def __init__(self, pointLowerBound, modelDir, gtDir, poissonDir, outputPath, datasetScale, maxIter, scoreBound=None):
        self.pointLowerBound = pointLowerBound
        self.modelDir = modelDir
        self.gtDir = gtDir
        self.poissonDir = poissonDir
        self.outputPath = outputPath
        self.datasetScale = datasetScale
        self.maxIter = maxIter
        self.scoreBound = scoreBound

        ### Dataset HDF Preparation by Dataset Group Type ###########################
        main_file = h5py.File(self.outputPath, 'w')

        self.train_group = main_file.create_group('training_group')
        self.train_group.attrs['type'] = 'training'

        self.valid_group = main_file.create_group('validation_group')
        self.valid_group.attrs['type'] = 'validation'

        self.test_group = main_file.create_group('test_group')
        self.test_group.attrs['type'] = 'test'
        #############################################################################

        print('Start making dataset', outputPath)

        self.get_data('train', self.train_group)
        self.get_data('test', self.test_group)
        self.get_data('valid', self.valid_group)
        main_file.close()


    def get_data(self, mode, sub_group):

        print(f'mode : {mode}')

        idx = 0
        low_idx = 0

        class_list = os.listdir(os.path.join(self.modelDir, mode))
        for class_id in class_list:
            model_list = os.listdir(os.path.join(modelDir, mode, class_id))
            for i, model_id in enumerate(model_list):
                
                if i % self.datasetScale != 0:
                    continue
                print(f'processing {model_id} ...')

                objPath = os.path.join(self.modelDir, mode, class_id, model_id, 'model.obj')
                obj = str(objPath).encode('utf-8')

                gtPath = os.path.join(self.gtDir, mode, 'gt', model_id, 'gt.pcd')
                gt = str(gtPath).encode('utf-8')
                if not os.path.exists(gtPath):
                    print(f'skip {model_id} since gt cloud is not exist')
                    continue
                
                dataDir = os.path.join(self.poissonDir, mode, model_id)
                length = len(os.listdir(dataDir))
                if length!=23 and length!=32 :
                    print(f'skip {model_id} since data sequence is not enough : {length}')
                    continue

                for i in range(self.maxIter): # gt, pcd, nbv, pose
                    iter = i+1
                    saveFlag = True
                    partialPath = os.path.join(dataDir, str(iter)+'.pcd')
                    if not os.path.exists(partialPath) :
                        print(f'skip {model_id}/{iter} since partial cloud does not exist')
                        saveFlag = False
                    else :
                        partialCloud = o3d.io.read_point_cloud(partialPath)
                        pointNum = len(np.asarray(partialCloud.points))
                        # print(f'{model_id}/{iter} points : {pointNum}')
                        if pointNum < self.pointLowerBound :
                            saveFlag = False
                            low_idx += 1
                            print(f'skip {model_id}/{iter} since partial pcd points number is {pointNum}.')
                    partial = str(partialPath).encode('utf-8')

                    nbvPath = os.path.join(dataDir, str(iter)+'_nbv.csv')
                    nbv = str(nbvPath).encode('utf-8')

                    posePath = os.path.join(dataDir, 'pose.csv')
                    if not os.path.exists(posePath) :
                        print(f'skip {model_id} since pose file does not exist')
                        saveFlag = False
                    else:
                        pose = str(posePath).encode('utf-8')
                        with open(posePath, 'r') as file:
                            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
                            df = pd.read_csv(file)
                            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                        score = np.array(df['coverageScore']).reshape(-1, 1)
                        scoreDif = score[iter]-score[iter-1]
                        if self.scoreBound and scoreDif < self.scoreBound :
                            print(f'skip {model_id}/{iter} since scoreDif < 1')
                            saveFlag=False

                    if saveFlag:
                        data = [str(model_id).encode('utf-8'), iter, partial, nbv, gt, pose, obj]
                        sub_group.create_dataset(name=str(idx).zfill(10), data = data)
                        idx += 1

        print(f'[{mode}] data num : {idx}')
        print(f'[{mode}] rejected data num : {low_idx}')

if __name__ == '__main__':

    maxIter = 10
    pointLowerBound = 100
    modelDir = '/media/owner/SLAM/data/ShapeNetCore.v1'
    gtDir = "/media/owner/SLAM/data/ShapeNetCore.v1/Render"
    poissonDir = '/media/owner/CoCEL/ShapeNetCore.v1/poisson_dataset_overlap'
    outputPath = "/media/owner/CoCEL/nbv_dataset/owner_poisson_overlap_ShapeNet_nbv_dataset_scale_10_sDif_1.hdf5"
    datasetScale = 10   # sclae n (dataset size 1/n)
    scoreBound = 1

    df = dataset_generator_shapenet(pointLowerBound, modelDir, gtDir, poissonDir, outputPath, datasetScale, maxIter, scoreBound)

    print("Done successfully")