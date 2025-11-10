# Author : Doyu Lim (2024)
# Simulate reconstruction process through PIT-NBV

import argparse
import os
import csv
import time
import random
import subprocess
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader_poisson import Poisson_dataset
import util.poisson_util as poisson_util
from network.pct_v import PointTransformer_View

def print_data(i_batch, model_id, gt_pcd, partial_pcd, nbv, score, iter, poseList, scoreList, nbvC, sDif):
    print('\n--------------------------------- data view ---------------------------------\n')
    print('batch idx \t', i_batch)
    print('model_id \t', model_id.shape, model_id)      # Batch X 1
    print('gt_pcd \t', gt_pcd.shape, gt_pcd[0][0])      # Batch X 5000 X 3
    print('partial_pcd    \t', partial_pcd.shape)       # Batch X (# of gt points) X 6
    print('nbv_pose \t', nbv.shape, nbv)                # Batch X 6
    print('nbv scroe \t', score.shape, score)           # Batch X 1
    print('iteration \t', iter.shape, iter)             # Batch X 10
    print('all pose\t', poseList.shape, poseList)       # Batch X 11 X 6
    print('all score\t', scoreList.shape, scoreList)    # Batch X 11 X 1
    print('nbv candidate\t', nbvC.shape)                # Batch X 10 X 6
    print('score Diff\t', sDif.shape, sDif)             # Batch X 1
    print('\n-----------------------------------------------------------------------------\n\n')


def testShapeNet(args, model, device, test_dataloader, logDir, nbvPositions, scaleFactor):

    idx = 0
    model.eval()
    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_dataloader):
            
            if (i_batch % args.test_term) != 0:
                continue

            idx += 1

            i_batch = int(i_batch)+1
            batch_model_id = np.array(batch_data[0])
            batch_gt_pcd = np.array(batch_data[1])
            batch_partial_pcd = np.array(batch_data[2])
            batch_nbv = np.array(batch_data[3])
            batch_score = np.array(batch_data[4])
            batch_iter = np.array(batch_data[5])
            batch_poseList = np.array(batch_data[6])
            batch_scoreList = np.array(batch_data[7])
            batch_nbvC = np.array(batch_data[8])
            batch_sDif = np.array(batch_data[9])
            batch_dist = np.array(batch_data[10])
            batch_objPath = np.array(batch_data[11])

            print_data(i_batch, batch_model_id, batch_gt_pcd, batch_partial_pcd, batch_nbv, \
                batch_score, batch_iter, batch_poseList, batch_scoreList, batch_nbvC, batch_sDif)

            target = batch_model_id[0]
            objPath = batch_objPath[0]
            gtPoints = batch_gt_pcd[0]

            simFlag = False
            while not simFlag:
                simFlag = simulation(args, target, model, device, logDir, nbvPositions, objPath, gtPoints, scaleFactor)
            

def testMesh(args, model, device, logDir, nbvPositions, meshPath, scaleFactor):

    target=args.data
    gtPoints = np.array([])

    model.eval()
    with torch.no_grad():
        for i in range(args.Iter):
            simFlag = False
            while not simFlag:
                simFlag = simulation(args, target, model, device, logDir, nbvPositions, meshPath, gtPoints, scaleFactor)

def simulation(args, target, model, device, logDir, nbvPositions, meshPath, gtPoints, scaleFactor):

    if args.save :
        savePath = os.path.join(logDir, args.test_file)
        isExist = os.path.exists(savePath)
        csv_file = open(savePath, 'a+')
        csv_writer = csv.writer(csv_file)
        if not isExist:
            header = ['model_id', 'simulation time', 'inference time'] + [str(i) for i in range(1, args.Round + 1)]
            csv_writer.writerow(header)
        print(f'save file {savePath}')

    if args.saveNpIt :
        npitPath = os.path.join(logDir, 'npit_'+args.test_file)
        ititPath = os.path.join(logDir, 'itit_'+args.test_file)
        npit_isExist = os.path.exists(npitPath)
        itit_isExist = os.path.exists(ititPath)
        npit_file = open(npitPath, 'a+')
        itit_file = open(ititPath, 'a+')
        npit_writer = csv.writer(npit_file)
        itit_writer = csv.writer(itit_file)
        if not npit_isExist:
            header = ['model_id', 'points num', 'inference time']
            npit_writer.writerow(header)
        if not itit_isExist:
            header = ['model_id', 'simulation time', 'inference time'] + [str(i) for i in range(1, args.Round + 1)]
            itit_writer.writerow(header)

    mesh = o3d.io.read_triangle_mesh(meshPath)
    center, mesh = poisson_util.meshCenter(mesh)

    if gtPoints.size == 0: # except shapenet
        gtCloud = mesh.sample_points_uniformly(number_of_points=500000)
    else : # shapenet
        gtCloud = o3d.geometry.PointCloud()
        gtCloud.points = o3d.utility.Vector3dVector(gtPoints)
        gtCloud.translate(-center)

    aabb = gtCloud.get_axis_aligned_bounding_box()
    size = aabb.get_max_bound() - aabb.get_min_bound()
    print(f'Before preprocessing object size (x,y,z) {size}')

    if args.data == 'shapenet': # rotate
        R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        center = mesh.get_center()
        mesh.rotate(R, center=center)
        gtCloud.rotate(R, center=center)

    # scailing
    if scaleFactor != False :
        maxL = np.max(size)
        scaleFactor = 2*args.sphereRadius*scaleFactor / maxL
        #scaleFactor = 2*scaleFactor / maxL
        center = mesh.get_center()
        mesh.scale(scaleFactor, center=center)
        gtCloud.scale(scaleFactor, center=center)

    aabb = gtCloud.get_axis_aligned_bounding_box()
    size = aabb.get_max_bound() - aabb.get_min_bound()
    print(f'After preprocessing object size (x,y,z) {size}')
    print(f'After preprocessing AABB {aabb}')
    print(f'After preprocessing center {mesh.get_center()}')

    if args.viz :
        mesh.compute_vertex_normals()
        #o3d.visualization.draw_geometries([mesh])
        #o3d.visualization.draw_geometries([gtCloud])
        #o3d.visualization.draw_geometries([gtCloud, mesh])

    # initial viewpoint
    isAvailable = False
    trial = 0
    while trial < 50:
        trial += 1
        camera, direction = poisson_util.genInitialPose(args.sphereRadius)
        initCloud = poisson_util.rayCasting(mesh, camera, direction, \
                        yaw=0.0, sfov=args.sfov, minDist=args.RCminDist, maxDist=args.RCmaxDist)
        if initCloud is not None:
            isAvailable = True
            break

    if not isAvailable:
        print("No available point!")
        if args.save : csv_writer.writerow([target, 0.0, 0.0] + [0.0]*20)
        return True

    # simulation process
    prevCloud = o3d.geometry.PointCloud()
    cover = []
    infTimes = []
    prevCamera = []
    prevDirection = []
    prevCube = None
    simFlag = True
    startTime = time.time()
    yaw = 0.0
    for iter in range(args.Round):
        print('\niter', int(iter)+1)
        print('camera parameter (nbv) :', camera)
        
        if simFlag == False :
            print('Invalid view! Skip this iteration')
            curCloud = prevCloud
            simFlag = True
        else: curCloud = poisson_util.rayCasting(mesh, camera, direction, \
                        yaw=yaw, sfov=args.sfov, minDist=args.RCminDist, maxDist=args.RCmaxDist)
        
        if args.viz :
            if curCloud == None: poisson_util.vizCloud(prevCloud, prevCloud)
            else : poisson_util.vizCloud(curCloud, prevCloud)

        # accCloud
        accCloud = o3d.geometry.PointCloud()
        if curCloud == None:
            curCloud = prevCloud
            accCloud = prevCloud
        else : accCloud = prevCloud + curCloud
        print('accCloud', accCloud)

        # downsampling
        voxel_size = 0.01
        curCloud = curCloud.voxel_down_sample(voxel_size)
        accCloud = accCloud.voxel_down_sample(voxel_size)
        prevCloud = accCloud
        print('downsampled', accCloud)

        # compute coverage score
        score = poisson_util.getCoverageScore(gtCloud, accCloud, threshold=0.01)
        cover.append(score)
        print(f'[Coverage Score] {score}%')

        # get NBV through network
        infStartTime = time.time()

        accPoints = np.asarray(accCloud.points).astype(np.float64)      # (original, 3)
        accPoints = poisson_util.resample(accPoints, args.sample_input)
        accPoints = np.expand_dims(accPoints, axis=0)
        accPoints = torch.from_numpy(accPoints).to(device).float()

        Snet = model(accPoints, device=device)

        camera = np.array([Snet[0][0], Snet[0][1], Snet[0][2]])
        direction = np.array([Snet[0][3], Snet[0][4], Snet[0][5]])
        print(f'Before VCB : {camera} {direction}')

        # VCB
        camera, direction = poisson_util.getNBV_VCB_5DOF_all(camera, direction, accCloud, args.VCBoptDist)
        camera, direction, yaw = poisson_util.getNBV_VCB_6DOF_nearDensity(camera, direction, accCloud, h_fov=args.h_fov, w_fov=args.w_fov, optDist=args.VCBoptDist, yaw_range_deg=180, num_yaw_steps=90)
        print(f'After VCB : {camera} {direction} / yaw {yaw}')

        infTime = time.time() - infStartTime
        infTimes.append(infTime)
        print(f'[Inference Time] {infTime} sec')


        if camera is None:
            print('[Err] camera is None!')
            return False
        # view constraint
        if poisson_util.isCameraOnObj(gtCloud, camera, threshold=0.01):
            print('camera is on the surface')
            simFlag = False
        # if poisson_util.isCameraInObj(mesh, camera, threshold=0.01):
        #     print('camera is in the object')
        if poisson_util.isCameraInAABB(aabb, camera):
            print('camera is in the bbox')
        if poisson_util.isVisitedBefore(camera, prevCamera, threshold=0.00):
            print('visited before')
            simFlag = False

        prevCamera.append(camera)
        prevDirection.append(direction)
        if args.saveNpIt : npit_writer.writerow([target, len(accCloud.points), infTime])

        if args.viz:
            print('viz acc cloud and nbv')
            poisson_util.viz_cloud_wCams(accCloud, prevCamera, prevDirection, sphere_radius=0.02, arrow_length=0.3)

    # save data
    simTime = time.time() - startTime
    print(f'\n[Simulation Time] {simTime} sec\n')
    avgInfTime = sum(infTimes)/len(infTimes)
    if args.save : csv_writer.writerow([target, simTime, avgInfTime] + cover)
    if args.saveNpIt : itit_writer.writerow([target, simTime, avgInfTime] + infTimes)

    if args.viz:
        print('Viz cloud with NBV points')
        poisson_util.viz_cloud_wCams(accCloud, prevCamera, prevDirection, sphere_radius=0.02, arrow_length=0.3)

    return True

def main(args):

    # Dataset
    shapenet_dataset_test = Poisson_dataset(dataset_path=args.data_path, mode='test', \
                                            inputSample=args.sample_input*5, gtSample=args.sample_gt)
    test_dataloader = DataLoader(dataset=shapenet_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = args.gpu
    print(f'[device] {device}')
    print(f'[small FOV] {args.sfov}')
    print(f'[limited depth] {args.ldepth}')
    print(f'[dataset] {args.data}')
    print(f'[test file] {args.test_file}')

    # Set parameters
    if args.ldepth:
        args.RCminDist = 0.2
        args.RCmaxDist = 0.3
        args.VCBoptDist = 0.25
    else:
        args.RCminDist = 0
        args.RCmaxDist = 100
        args.VCBoptDist = 1.0

    if args.sfov:
        args.h_fov=30
        args.w_fov=22.73
    else :
        args.h_fov=67.82
        args.w_fov=53.51

    # Set model parameters
    nbvPositions = None
    model = PointTransformer_View(in_dim=3, out=6).to(device)
    logDir = args.log_dir
    ptPath = logDir+'/model/epoch_'+str(args.loadEpoch)+'.pt'
    input_shape = (args.sample_input, 3)

    # trained model load
    weightPath = os.path.join(ptPath)
    print(f'Load trained model {weightPath}')
    model.load_state_dict(torch.load(weightPath, map_location=torch.device(device)))

    if args.data == 'shapenet':
        scaleFactor = False
        args.test_file =f'shapenet_{args.loadEpoch}_{args.test_file}'
        testShapeNet(args, model, device, test_dataloader, logDir, nbvPositions, scaleFactor)

    elif args.data == 'stanford':
        scaleFactor = 0.7
        testTerm = f'_{args.loadEpoch}_{args.test_file}'
        args.test_file = 'bunny'+testTerm
        meshPath = 'data/bunny.ply'
        testMesh(args, model, device, logDir, nbvPositions, meshPath, scaleFactor)

    elif args.data == 'csail':
        scaleFactor = 0.7
        testTerm = f'_{args.loadEpoch}_{args.test_file}'
        args.test_file = 'bird'+testTerm
        meshPath = 'data/bird.obj'
        testMesh(args, model, device, logDir, nbvPositions, meshPath, scaleFactor)

    print('Done Successfully')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/media/owner/CoCEL/nbv_dataset/owner_poisson_ShapeNet_nbv_dataset_scale_10.hdf5')
    parser.add_argument('--log_dir', default='log/240902')
    parser.add_argument('--loadEpoch', type=int, default=500)
    parser.add_argument('--data', type=str, default='stanford') # test obejct (shapenet/stanford/csail)
    parser.add_argument('--test_term', type=int, default=10) # One simulation for every 10 (i.e., only one sim per object)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_input', type=int, default=1024)
    parser.add_argument('--sample_gt', type=int, default=5000)
    parser.add_argument('--gpu', default='cpu')
    parser.add_argument('--viz', type=bool, default=True)

    # save
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--test_file', type=str, default='test.csv')
    parser.add_argument('--saveNpIt', type=bool, default=False) # save relationship btw # of points and inference time 
    parser.add_argument('--Round', type=int, default=20)
    parser.add_argument('--Iter', type=int, default=20)

    # sensor specification
    parser.add_argument('--sfov', type=bool, default=True)      # FOV
    parser.add_argument('--ldepth', type=bool, default=False)   # working range
    parser.add_argument('--sphereRadius', type=float, default=1.0)  # Sphere radius for initial viewpoint

    args = parser.parse_args()

    main(args)