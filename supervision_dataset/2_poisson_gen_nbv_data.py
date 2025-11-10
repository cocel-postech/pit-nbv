# Author : Doyu Lim (2024/5)
# OBJ -> PCD, CSV : Generate NBV data using Poisson autoscanning algorithm

import os
import sys
import numpy as np
import argparse
import open3d as o3d
import random
import time
import csv
import subprocess
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import util.poisson_util as poisson_util

def colorize(cloud, color):
    colors = np.array([color for _ in range(len(np.asarray(cloud.points)))])  # red
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud

def get_coverage_score(gt_cloud, acc_cloud, threshold=0.01):
    tree = o3d.geometry.KDTreeFlann(acc_cloud)
    overlap_count = 0
    for point in gt_cloud.points:
        [k, idx, dist] = tree.search_radius_vector_3d(point, threshold)
        if k > 0:
            overlap_count += 1
    score = overlap_count/len(np.asarray(gt_cloud.points))*100
    return round(score, 2)

def genInitialPose(radius):

    theta = np.random.uniform(0, 2 * np.pi)  # 0-2pi
    phi = np.random.uniform(0, np.pi)  # 0-2pi

    # point on sphere
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    position = np.array([x, y, z])

    # point to origin(0,0,0)
    direction = -position / np.linalg.norm(position)  # normalize

    return position, direction

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def getNBV(df, prevCamera, threshold):

    accepted_df = df[df['viewStatus'] == 'Accepted']

    for idx, row in accepted_df.iterrows():
        x = row[' positionX']
        y = row[' positionY']
        z = row[' positionZ']
        normalx = row[' normalX']
        normaly = row[' normalY']
        normalz = row[' normalZ']

        camera = [x, y, z]
        direction = [normalx, normaly, normalz]

        if all(distance(camera, prev_camera) > threshold for prev_camera in prevCamera):
            return idx+1, camera, direction
    
    # highest score
    if accepted_df.empty:
        print("No accepted views found.")
        return None, None, None  # handling execption
    first_accepted_row = accepted_df.iloc[0]

    x = first_accepted_row[' positionX']
    y = first_accepted_row[' positionY']
    z = first_accepted_row[' positionZ']
    normalx = first_accepted_row[' normalX']
    normaly = first_accepted_row[' normalY']
    normalz = first_accepted_row[' normalZ']

    idx = 1
    camera = [x, y, z]
    direction = [normalx, normaly, normalz]

    return idx, camera, direction
    
def simulate(args, model, objPath, gtPath):

    savePath = os.path.join(args.saveDir, model)
    os.makedirs(savePath, exist_ok=True)
    
    if args.save:
        csv_file = open(os.path.join(savePath, 'pose.csv'), 'w+')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['iter', 'Position', 'Direction', 'coverageScore'])

    # Simulation Start
    start_time = time.time()

    # load gt cloud (# of points = 5000)
    gtCloud = o3d.io.read_point_cloud(gtPath)
    if args.viz :
        print(f'viz gtCloud {gtPath}')
        o3d.visualization.draw_geometries([gtCloud])

    # load mesh
    mesh = o3d.io.read_triangle_mesh(objPath)
    mesh.compute_vertex_normals()
    # obj_cloud = mesh.sample_points_poisson_disk(number_of_points=10000)

    prevCloud = o3d.geometry.PointCloud()
    prevCamera = []
    camera, direction = genInitialPose(1) # initial camera pose
    for i in range(args.maxIter+1):

        iter = i+1
        print('\niter', iter)
        print('[NBV] Position', camera)
        print('[NBV] Direction', direction)

        # 1. Ray Casting
        curCloud = poisson_util.rayCasting(mesh, camera, direction)
        if args.viz :
            print('viz curCloud')
            o3d.visualization.draw_geometries([curCloud])

        # 2. Accumulated Cloud
        accCloud = o3d.geometry.PointCloud()
        print('curCloud', curCloud)
        print('prevCloud', prevCloud)
        if not prevCamera : # first iter
            if args.viz :
                print('viz curCloud(red)')
                curCloud = colorize(curCloud, [1, 0, 0]) # red
                o3d.visualization.draw_geometries([curCloud, prevCloud], point_show_normal=False)
            accCloud = curCloud
            print('accCloud (curCloud)', accCloud)
        else:
            if args.viz :
                print('viz curCloud(red) and prevCloud(blue)')
                curCloud = colorize(curCloud, [1, 0, 0]) # red
                if len(np.asarray(prevCloud.points)):
                    prevCloud = colorize(prevCloud, [0, 0, 1]) # blue
                o3d.visualization.draw_geometries([curCloud, prevCloud], point_show_normal=False)

            accPoints = np.vstack((prevCloud.points, curCloud.points))
            accCloud.points = o3d.utility.Vector3dVector(accPoints)
            print('accCloud (curCloud + prevCloud)', accCloud)

        # 3. Downsampling
        voxel_size = 0.001
        accCloud = accCloud.voxel_down_sample(voxel_size)
        prevCloud = accCloud
        print('Downsampled accCloud', accCloud)

        # 4. Compute Coverage Score
        score = get_coverage_score(gtCloud, accCloud, threshold=0.01)
        print(f'[Coverage Score] {score}%')

        if args.viz :
            o3d.visualization.draw_geometries([accCloud])
        
        # 5. Save Cloud and NBV information
        if args.save :
            print('save cloud and nbv information for iter', iter)
            o3d.io.write_point_cloud(os.path.join(savePath, str(iter)+'.pcd'), accCloud)
            csv_writer.writerow([iter, camera, direction, score])

        if iter == args.maxIter+1:
            continue
        
        # 6. Poisson Autoscanning to get NBV
        comPath = args.poissonPath
        comNormalTag = '_normal'
        comGroundFlag = '0'
        comVizFlag = str(args.viz) # '0'
        comVirtualFlag = '1'
        command = ' '.join([comPath, savePath+'/', str(iter), comNormalTag, comGroundFlag, comVizFlag, comVirtualFlag])
        print(f'poisson autoscanning command : {command}')
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Command output:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("Error occurred while running command:")
            print(e.stderr)

        # 7. NBV update
        nbvPath = os.path.join(savePath, str(iter)+'_nbv.csv')
        df = pd.read_csv(nbvPath)

        nbvIdx, camera, direction = getNBV(df, prevCamera, threshold=0.1)
        if nbvIdx == None: # poisson algorithm fail
            return False # return to first iter
        prevCamera.append(camera)
        print(f'[NBV] {nbvIdx}-th Highest NBV score was selected')

    sim_time = time.time() - start_time
    print(f'\n[Simulation Time] {sim_time} sec\n')

    if args.save:
        csv_file.close()

    # Delete useless file
    delPath = os.path.join(savePath, 'poisson_field.raw')
    try:
        if os.path.exists(delPath):
            os.remove(delPath)
            print(f"File {delPath} has been deleted successfully.")
        else:
            print(f"File {delPath} does not exist.")
    except Exception as e:
        print(f"Error occurred while trying to delete the file {delPath}: {e}")

    for files in os.listdir(savePath):
         if files.endswith('_normal.ply'):
            delNormal = os.path.join(savePath, files)
            try:
                os.remove(delNormal)
                print(f"Deleted file: {delNormal}")
            except Exception as e:
                print(f"Failed to delete {delNormal}: {e}")
    return True



def main(args):
    
    if not os.path.exists(args.saveDir):
        os.makedirs(args.saveDir)

    class_list = os.listdir(os.path.join(args.ShapeNetv1Dir, dataType))
    print(f'Class list ({len(class_list)}) : {class_list}')

    for class_id in class_list:

        model_list = os.listdir(os.path.join(args.ShapeNetv1Dir, dataType, class_id))
        print(f'Model list ({len(model_list)}) : {model_list}')

        for i, model in enumerate(model_list):

            if i % args.datasetScale != 0:
                continue
            
            obj_path = os.path.join(args.ShapeNetv1Dir, dataType, class_id, model, 'model.obj')

            savePath = os.path.join(args.saveDir, model)
            if os.path.exists(savePath):
                length = len(os.listdir(savePath))
                if length==23 or length==32 :
                    print(f'Skip {model} since already processed')
                    continue
            
            gt_path = os.path.join(args.gtDir, model, 'gt.pcd')
            if not os.path.exists(gt_path):
                print(f'Skip {model} since gt.pcd is not exist')
                continue
            
            result = simulate(args, model, obj_path, gt_path)
            if not result :
                simulate(args, model, obj_path, gt_path)


    print('Done Successfully')

if __name__ == '__main__':

    dataType='test' # train, valid, test

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataType', default=dataType)
    parser.add_argument('--ShapeNetv1Dir', default='/media/owner/SLAM/data/ShapeNetCore.v1/')
    parser.add_argument('--gtDir', default='/media/owner/SLAM/data/ShapeNetCore.v1/Render/' + dataType + '/gt')
    parser.add_argument('--saveDir', default='/media/owner/CoCEL/ShapeNetCore.v1/poisson_dataset_overlap/' + dataType)
    parser.add_argument('--poissonPath', default='/home/owner/autoscanning/build/autoscanning')
    parser.add_argument('--maxIter', type=int, default=10)
    parser.add_argument('--datasetScale', type=int, default=10) # sclae n (dataset size 1/n)

    parser.add_argument('--viz', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=True)

    args = parser.parse_args()

    main(args)