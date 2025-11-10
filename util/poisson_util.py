# Author : Doyu Lim (2024)

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import scipy.stats

# centering
def meshCenter(mesh):
    center = mesh.get_center()
    mesh.translate(-center)
    return center, mesh

def cloudCenter(cloud):
    center = cloud.get_center()
    cloud.translate(-center)
    return cloud

# coverage score
def getCoverageScore(gt_cloud, acc_cloud, threshold=0.01, viz=False):
    tree = o3d.geometry.KDTreeFlann(acc_cloud)
    overlap_count = 0
    overlap_points = []
    for point in gt_cloud.points:
        [k, idx, dist] = tree.search_radius_vector_3d(point, threshold)
        if k > 0:
            overlap_count += 1
            overlap_points.append(point)
    score = overlap_count/len(np.asarray(gt_cloud.points))*100

    if(viz):
        overlap_cloud = o3d.geometry.PointCloud()
        overlap_cloud.points = o3d.utility.Vector3dVector(np.array(overlap_points))
        acc_cloud.paint_uniform_color([0, 0, 1])          # blue
        overlap_cloud.paint_uniform_color([1, 0, 0])      # red
        print('accCloud(blue) and overlapCloud(red)')
        o3d.visualization.draw_geometries([acc_cloud, overlap_cloud])

    return round(score, 2)

# scaling
def scaleCloud(inCloud, scale_factor):
    points = np.asarray(inCloud.points)
    points *= scale_factor
    inCloud.points = o3d.utility.Vector3dVector(points)
    return inCloud

def resample(pcd, n):
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

def safe_normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# ray casting
def rayCasting(mesh, viewpoint, direction, yaw=0.0, sfov=True, minDist=0, maxDist=100):

    ### 1. Set viewpoint
    viewpoint = np.array(viewpoint)  # Camera position
    direction = np.array(direction)  # Ray direction (looking at)
    norm = np.linalg.norm(direction)
    if norm == 0:
        print("direction vector norm ZERO! skip this viewpoint raycasting")
        return None
    else: direction = direction / norm  # Normalize direction

    ### 2. Build camera axes (z = forward, x = right, y = up)
    if yaw == 0.0: # shortcut
        up_vector = [0, 1, 0]
    else:
        z_axis = direction
        if np.abs(z_axis[2]) < 0.9:
            temp = np.array([0, 0, 1])
        else:
            temp = np.array([1, 0, 0])

        x_axis = np.cross(temp, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # Apply yaw rotation around z-axis
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        # Rotate x and y
        x2 = R_yaw @ x_axis[:2]
        y2 = R_yaw @ y_axis[:2]

        rotated_x = np.append(x2, x_axis[2])
        rotated_y = np.append(y2, y_axis[2])

        # Now rotated_y is the new up vector
        up_vector = rotated_y.tolist()

    ### 3. Convert mesh to tensor mesh and add to ray casting scene
    scene = o3d.t.geometry.RaycastingScene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(tmesh)

    if sfov: # small fov
        _fov_deg=30
        _width_px=1280
        _height_px=960
    else: # large fov
        _fov_deg=67.82 #90
        _width_px=640 # 1280
        _height_px=480 # 960

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=_fov_deg,
            center=direction,
            eye=viewpoint,
            up=up_vector,
            width_px=_width_px,
            height_px=_height_px,
        )

    ### 4. Perform ray casting
    ans = scene.cast_rays(rays)

    ### 5. Check for hits and calculate hit points and distance
    hit = ans['t_hit'].isfinite()
    t_hit_np = ans['t_hit'].numpy()
    
    # Distance range filter
    valid_hit = hit & (t_hit_np >= minDist) & (t_hit_np <= maxDist)
    valid_hit_np = valid_hit.numpy()

    print(f'Hit rays: {hit.numpy().sum()}')  # Number of hit rays
    print(f'In the sensing range: {valid_hit_np.sum()}')  # Number of valid hits within the range

    if valid_hit_np.any():
        rays_np = rays.numpy()
        # Properly index the rays_np array with valid_hit_np before using it for calculations
        points = rays_np[valid_hit_np][:, :3] + rays_np[valid_hit_np][:, 3:] * t_hit_np[valid_hit_np].reshape((-1, 1))
        visibleCloud = o3d.geometry.PointCloud()
        visibleCloud.points = o3d.utility.Vector3dVector(points)
        return visibleCloud
    else:
        print("No hits detected, please verify the ray origin and direction.")
        return None

# collision
def isCameraInAABB(aabb, point):
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    return (min_bound[0] <= point[0] <= max_bound[0] and
            min_bound[1] <= point[1] <= max_bound[1] and
            min_bound[2] <= point[2] <= max_bound[2])

def isCameraOnObj(gtCloud, camera, threshold=0.05):

    points = np.asarray(gtCloud.points)
    if points is None or camera is None :
        return False

    distances = np.linalg.norm(points - camera, axis=1)
    min_distance = np.min(distances)

    return min_distance <= threshold

def isCameraInObj(mesh, camera, threshold=0.05):

    if mesh.is_empty() or camera is None:
        return False

    # Convert legacy mesh to tensor mesh
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create RaycastingScene and add mesh
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(tmesh)

    # occupancy check
    camera_tensor = o3d.core.Tensor(camera.reshape((1, 3)), dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(camera_tensor)

    # inside: occupancy = 1, outside: 0
    return occupancy.numpy()[0] > threshold

def isVisitedBefore(camera, prevCamera, threshold=0.01):
    for prevPos in prevCamera:
        if distance(camera, prevPos) <= threshold:
            return True
    return False

def genInitialPose(radius):

    theta = np.random.uniform(0, 2 * np.pi)  # 0~2π
    phi = np.random.uniform(0, np.pi)  # 0~π

    # point on sphere
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    position = np.array([x, y, z])

    # point to origin(0,0,0)
    direction = -position / np.linalg.norm(position)  # normalize

    return position, direction

# ------------------- VCB ------------------------------------------------------
def getNBV_VCB_5DOF_all(position, direction, accCloud, optDist=1.0):
    
    points = np.asarray(accCloud.points)
    distances = np.linalg.norm(points - position, axis=1)
    nearPoint = points[np.argmin(distances)]

    direction = direction / np.linalg.norm(direction)

    adjusted_position = nearPoint - direction * optDist

    return adjusted_position, direction

# FOV modeling : 2D horizontal/vertical rectangle + search best yaw
def getNBV_VCB_6DOF_nearDensity(position, direction, accCloud, h_fov, w_fov, optDist, yaw_range_deg=180, num_yaw_steps=30):
    """
    Args:
        position (np.array): (3,) Viewpoint position
        direction (np.array): (3,) Current viewing direction (unit vector)
        accCloud (Open3D point cloud): Accumulated point cloud
        h_fov (float): Vertical FOV in degrees
        w_fov (float): Horizontal FOV in degrees
        optDist (float): Target distance to closest point
        yaw_range_deg (float): Total yaw rotation range in degrees (default (-90)-(+90) degrees)
        num_yaw_steps (int): Number of yaw steps within yaw_range (default 30)

    Returns:
        adjusted_position (np.array): New viewpoint position
        best_direction (np.array): New viewing direction (after yaw)
        best_yaw_angle (float): Yaw rotation angle (in radians)
    """
    
    # shift
    #position, direction = getNBV_VCB_5DOF_all(position, direction, accCloud, optDist)

    points = np.asarray(accCloud.points)
    direction = direction / np.linalg.norm(direction)
    vectors = points - position

    h_fov_rad = np.radians(h_fov / 2)
    w_fov_rad = np.radians(w_fov / 2)

    # Create orthonormal basis (x, y, z) for the camera
    z_axis = direction
    # Choose an arbitrary vector not colinear with z
    temp = np.array([0, 0, 1]) if np.abs(z_axis[2]) < 0.9 else np.array([1, 0, 0])
    x_axis = np.cross(temp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # yaw sweep settings
    yaw_half_range = np.radians(yaw_range_deg) / 2
    yaw_angles = np.linspace(-yaw_half_range, yaw_half_range, num_yaw_steps) # rad

    min_local_density = np.inf
    best_nearest_point = None
    best_yaw = 0

    for yaw_angle in yaw_angles:
        # 2D yaw rotation matrix
        R_yaw_2d = np.array([
            [np.cos(yaw_angle), -np.sin(yaw_angle)],
            [np.sin(yaw_angle),  np.cos(yaw_angle)]
        ])
        rotated_x = np.hstack((R_yaw_2d @ x_axis[:2], x_axis[2]))
        rotated_y = np.hstack((R_yaw_2d @ y_axis[:2], y_axis[2]))

        # Project vectors onto rotated camera frame
        proj_x = np.dot(vectors, rotated_x)
        proj_y = np.dot(vectors, rotated_y)
        proj_z = np.dot(vectors, z_axis)

        # Find points within FOV
        valid_indices = (proj_z > 0) & (np.abs(proj_x) <= proj_z * np.tan(w_fov_rad)) & (np.abs(proj_y) <= proj_z * np.tan(h_fov_rad))

        # Calculate density (including both points in & out of FOV) around nearest point in FOV
        r = 0.05
        if np.any(valid_indices):
            distances = np.linalg.norm(vectors[valid_indices], axis=1)
            min_distance = np.min(distances)
            nearest_point = points[valid_indices][np.argmin(distances)] # in FOV

            # local density
            neighbor_distances = np.linalg.norm(points - nearest_point, axis=1)
            local_density = np.sum(neighbor_distances < r) - 1  # exclude nearest point itself
            #local_density /= points.shape[0] # normalize
            
            if local_density < min_local_density:
                #print(f'Update best yaw: {np.degrees(yaw_angle):.2f} deg / Local density: {local_density}')
                min_local_density = local_density
                best_nearest_point = nearest_point
                best_yaw = yaw_angle

        # else : # no valid points
        #     #print("[VCB] No valid points after yaw optimization! Using nearest (out of FOV) point.")
        #     distances = np.linalg.norm(vectors, axis=1)
        #     min_distance = -0.01
        #     nearest_point = points[np.argmin(distances)] # ouf of FOV

    if best_nearest_point is None:
        # fallback (in case all min_distance <= 0)
        best_nearest_point = points[np.argmin(np.linalg.norm(vectors, axis=1))]
        print("[Warning] No valid point inside FOV. Falling back to nearest point.")

    adjusted_position = best_nearest_point - direction * optDist
    return adjusted_position, direction, best_yaw


def getNewCloud(curCloud, prevCloud, threshold=0.0001):

    if not len(np.asarray(prevCloud.points)):
        return curCloud

    kdtree = o3d.geometry.KDTreeFlann(prevCloud)

    cur_points = np.asarray(curCloud.points)
    new_points = []
    for point in cur_points:
        [_, idx, distances] = kdtree.search_knn_vector_3d(point, 1)
        if distances[0] >= threshold:
            new_points.append(point)

    if new_points:
        new_points = np.array(new_points)
    else:
        new_points = np.empty((0, 3))

    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(new_points)
    return new_cloud

# ------------------- Viz ------------------------------------------------------
def vizCloud(curCloud, prevCloud): # cur <-> prev if needed
    new_cloud = o3d.geometry.PointCloud()
    if len(np.asarray(curCloud.points)):
        new_cloud = getNewCloud(curCloud, prevCloud, threshold=0.0001)
        if len(np.asarray(new_cloud.points)):
            new_cloud = colorize(new_cloud, [0, 0, 1]) # cur cloud color
    if len(np.asarray(prevCloud.points)):
        prevCloud = colorize(prevCloud, [0.5, 0.5, 0.5]) # prev cloud color
    o3d.visualization.draw_geometries([new_cloud, prevCloud], point_show_normal=False)

def colorize(cloud, color):
    colors = np.array([color for _ in range(len(np.asarray(cloud.points)))])
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud

def compute_rotation_matrix(from_vec, to_vec):
    v = np.cross(from_vec, to_vec)
    c = np.dot(from_vec, to_vec)
    s = np.linalg.norm(v)
    if s == 0:
        return np.identity(3)  # I if parallel
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

def viz_cloud_wCams(cloud, cameras, directions, sphere_radius=0.01, arrow_length=0.05):
    """
    cloud : open3d.geometry.PointCloud
    cameras : (n, 3) numpy array of camera positions
    directions : (n, 3) numpy array of normalized direction vectors
    """

    cam_color=(0, 0, 0)
    line_color=(0, 0, 0)
    cloud_color=(0.6, 0.6, 0.6)

    cloud.paint_uniform_color(cloud_color)
    geometries = [cloud]

    for i, (cam, dir_vec) in enumerate(zip(cameras, directions)):
        # 1. camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.paint_uniform_color(cam_color)
        sphere.translate(cam)
        geometries.append(sphere)

        # 2. direction normalization
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            print(f'{i}-th direction vector norm ZERO!')
            continue
        else:
            dir_vec = dir_vec / norm
        #end = cam + dir_vec * arrow_length

        # 3. direction arrow
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.003,
            cone_radius=0.006,
            cylinder_height=arrow_length * 0.8,
            cone_height=arrow_length * 0.2
        )
        arrow.paint_uniform_color(cam_color)

        # direction
        rot_matrix = compute_rotation_matrix(np.array([0, 0, 1]), dir_vec)
        arrow.rotate(rot_matrix, center=(0, 0, 0))
        arrow.translate(cam)

        geometries.append(arrow)
    
    '''# connection between cams
    if len(cameras) >= 2:
        # indices 0-1, 1-2, 2-3, ...
        lines = [[i, i + 1] for i in range(len(cameras) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(cameras),
            lines=o3d.utility.Vector2iVector(lines))
        colors = np.tile(np.array(line_color), (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)'''

    o3d.visualization.draw_geometries(geometries, window_name="Cameras with Directions and PointCloud")

if __name__ == '__main__':
    viz = True
    obj_path = 'path/to/mesh'

    camera = [-1, 0, 0]
    direction = [0, 0, 0]
    mesh = o3d.io.read_triangle_mesh(obj_path)
    curCloud = rayCasting(mesh, camera, direction)
    if viz :
        print('viz curCloud')
        o3d.visualization.draw_geometries([curCloud])