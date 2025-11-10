# Author : Doyu Lim (2024)
# Dataloader

import torch
import torch.utils.data
import numpy as np
import h5py
import pandas as pd
import open3d as o3d
import ast
import fcntl

class Poisson_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, mode='training', inputSample=3000, gtSample=5000, normal=False):

        self.dataset_path = dataset_path
        self.mode = mode
        self.inputSample = inputSample
        self.gtSample = gtSample
        self.normal = normal

    def __getitem__(self, idx):
        
        info_file = h5py.File(self.dataset_path, mode='r')
        idx=str(idx).zfill(10)

        info_location = '/' + self.mode + '_group/' + idx
        info = list(info_file[info_location])

        # model_id iter, partial, nbv, gt, pose

        model = str(info[0], 'utf-8')
        iter = int(info[1])
        partialPath = str(info[2], 'utf-8')
        nbvPath = str(info[3], 'utf-8')
        gtPath = str(info[4], 'utf-8')
        posePath = str(info[5], 'utf-8')
        objPath = str(info[6], 'utf-8')

        # 1. gtPoints
        gtCloud = o3d.io.read_point_cloud(gtPath)
        # self.plot_pcd(gtCloud, normal=False)
        gtPoints = np.asarray(gtCloud.points).astype(np.float64)   # (5000, 3)
        # gtPoints = self.randomSample(gtPoints, self.gtSample)         # (sample_gt, 3)

        # 2. partialPoints with Normals
        partialCloud = o3d.io.read_point_cloud(partialPath)
        partialPoints = np.asarray(partialCloud.points).astype(np.float64)     # (original, 3)
        if self.normal:
            partialCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # self.plot_pcd(partialCloud, normal=True)
            center = partialCloud.get_center()
            # print(f'center {center}')
            partialCloud.orient_normals_towards_camera_location(camera_location=center)
            partialCloud.normals = o3d.utility.Vector3dVector(np.asarray(partialCloud.normals) * -1)
            # self.plot_pcd(partialCloud, normal=True)
            partialNormals = np.asarray(partialCloud.normals).astype(np.float64)   # (original, 3)
            partialPoints = np.hstack((partialPoints, partialNormals))               # (original, 6)
        partialPoints = self.randomSample(partialPoints, self.inputSample)          # (inputSample, 3 or 6)

        # 3. nbv each for partialPoints : if Rejected, fill with 0
        nbvList = []
        df = pd.read_csv(nbvPath)
        accepted_rows = df[df['viewStatus'] == 'Accepted']
        for idx, row in accepted_rows.iterrows():
            x = row[' positionX']
            y = row[' positionY']
            z = row[' positionZ']
            normalx = row[' normalX']
            normaly = row[' normalY']
            normalz = row[' normalZ']
            nbvList.append((x, y, z, normalx, normaly, normalz))

        if len(nbvList) > 0:
            nbvCandidate = np.array(nbvList)
        else:
            nbvCandidate = np.empty((0, 6))  # initialize with empty array
        
        nbvNum = nbvCandidate.shape[0]
        if nbvNum < 10:
            dif = 10 - nbvCandidate.shape[0]
            zero_rows = np.zeros((dif, 6))
            if nbvCandidate.shape[0] == 0:
                nbvCandidate = zero_rows
            else:
                nbvCandidate = np.vstack((nbvCandidate, zero_rows))
        # print(f'[Dataloader] {model}/{iter} nbvC shape ', nbvCandidate.shape)

        # 4. pose for model (selected nbv during dataset generation)
        with open(posePath, 'r') as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
            df = pd.read_csv(file)
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        position = np.array([list(map(float, pos.strip('[],').replace(',', '').split())) for pos in df['Position']])
        direction = np.array([list(map(float, dir.strip('[],').replace(',', '').split())) for dir in df['Direction']])
        pose = np.hstack((position, direction))
        score = np.array(df['coverageScore']).reshape(-1, 1)
        curnbv = pose[iter]
        curPose = pose[iter-1]
        curScore = score[iter-1]
        scoreDif = score[iter]-score[iter-1]
        dist = self.nearDist(partialCloud, curnbv[:3])
        dist = np.expand_dims(dist, axis=0)

        # iteration one-hot encoding
        oneHotIter = np.zeros(10)
        oneHotIter[iter-1] = 1

        return model, gtPoints, partialPoints, curnbv, curScore, oneHotIter, pose, score, nbvCandidate, scoreDif, dist, objPath, nbvNum

    def __len__(self):
        file = h5py.File(self.dataset_path, mode='r')
        self.len = len(file.get('/' + self.mode + '_group/'))
        file.close
        return self.len
    
    def plot_pcd(self, pcd, normal=True):
        o3d.visualization.draw_geometries([pcd], point_show_normal=normal)
    
    def get_normals(self, pcd):

        normal_estimator = pcd.make_NormalEstimation()

        kdtree = pcd.make_kdtree()
        normal_estimator.set_SearchMethod(kdtree)
        normal_estimator.set_KSearch(50)
        normals = normal_estimator.compute()

        return normals

    def randomSample(self, pcd, n):

        idx = np.random.permutation(pcd.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
        return pcd[idx[:n]]

    def farthestSample(self, inCloud, npoint): #fps
        """
        Input:
            xyz: pointcloud data, [N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
            newPoitns: [npoint, 3]
        """
        if not isinstance(inCloud, torch.Tensor):
            xyz = torch.tensor(inCloud[:,:3], dtype=torch.float32)

        N, C = xyz.shape
        centroids = torch.zeros(npoint, dtype=torch.long)
        distance = torch.ones(N) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long).item()
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :].view(1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1].item()
        newPoints = inCloud[centroids, :]
        return newPoints

    def nearDist(self, inCloud, position):
        points = np.asarray(inCloud.points)
        distances = np.linalg.norm(points - position, axis=1)
        minDist = np.min(distances)
        return minDist

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

if __name__ == '__main__':

    dataPath = "/media/owner/CoCEL/nbv_dataset/owner_poisson_overlap_ShapeNet_nbv_dataset_scale_10_sDif_1.hdf5"
    idx = 1
    mode = 'validation' # validation, training, test

    ShapeNet_dataset = Poisson_dataset(dataset_path=dataPath, mode=mode)

    # getitem
    model, gtPoints, partialPoints, curnbv, curScore, oneHotIter, pose, score, nbvC, sDif, dist, objPath, nbvNum = ShapeNet_dataset[idx]
    print('model\t', model)
    print('partialPoints\t', partialPoints.shape, partialPoints.dtype)
    print('gtPoints\t', gtPoints.shape, gtPoints.dtype)
    print('nbvCandidate\t', nbvC.shape, '\n', nbvC)
    print('pose\t', pose.shape, '\n', pose)
    print('score\t', score.shape, '\n', score)
    print('oneHotIter\t', oneHotIter)
    print('curScore\t', curScore)
    print('curnbv\t', curnbv)
    print('dist\t', dist)
    print('scoreDif\t', sDif)

    # len
    length = len(ShapeNet_dataset)
    print('len\t', length)