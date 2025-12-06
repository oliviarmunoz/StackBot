import numpy as np
import trimesh

from pydrake.all import (
    Concatenate,
    RigidTransform,
    RotationMatrix,
    PointCloud, 
    Rgba
)
import matplotlib.pyplot as plt

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation_module.poses import *

def remove_table_points(pc: PointCloud) -> PointCloud:
            xyzs = pc.xyzs()
            mask = xyzs[2, :] > 0.01
            filtered = PointCloud(mask.sum(), fields=pc.fields())
            filtered.mutable_xyzs()[:] = xyzs[:, mask]
            if pc.has_rgbs():
                filtered.mutable_rgbs()[:] = pc.rgbs()[:, mask]
            if pc.has_normals():
                filtered.mutable_normals()[:] = pc.normals()[:, mask]
            return filtered

def make_point_clouds(station, context, diagram, meshcat, cup_bodies, cup_names):
    N_SAMPLE_POINTS = 150

    mesh = trimesh.load(f"assets/cup.obj", force="mesh")
    model_points_O = mesh.sample(count=N_SAMPLE_POINTS).T  # 3 x N, object frame

    # Drake PointCloud for visualization (optional)
    model_point_cloud = PointCloud(N_SAMPLE_POINTS)
    model_point_cloud.mutable_xyzs()[:] = model_points_O

    # sample N_SAMPLE_POINTS from the mesh and then turn those points into a numpy array
    points = mesh.sample(count=N_SAMPLE_POINTS)

    # create a `PointCloud` object from the numpy array
    point_cloud = PointCloud(N_SAMPLE_POINTS)
    point_cloud.mutable_xyzs()[:] = points.T

    cameras = ["camera0", "camera1", "camera2"]
    station_context = diagram.GetSubsystemContext(station, context)

    # Visualize rgb output
    fig, axes = plt.subplots(
        1, len(cameras), figsize=(5 * len(cameras), 4), constrained_layout=True
    )
    for ax, cam in zip(axes, cameras):
        img = station.GetOutputPort(f"{cam}.rgb_image").Eval(station_context)
        arr = np.array(img.data, copy=False).reshape(img.height(), img.width(), -1)
        im = ax.imshow(arr)
        ax.set_title(f"{cam} rgb image")
        ax.axis("off")

    plt.savefig("cups_point_cloud_rgb.png")

    # Visualize depth output
    fig, axes = plt.subplots(
        1, len(cameras), figsize=(5 * len(cameras), 4), constrained_layout=True
    )
    for ax, cam in zip(axes, cameras):
        img = station.GetOutputPort(f"{cam}.depth_image").Eval(station_context)
        depth_img = np.array(img.data, copy=False).reshape(img.height(), img.width(), -1)
        depth_img = np.ma.masked_invalid(depth_img)
        img = ax.imshow(depth_img, cmap="magma")
        ax.set_title(f"{cam} depth image")
        ax.axis("off")

    plt.savefig("cups_point_cloud_depth.png")

    plant = station.plant()
    plant_context = diagram.GetSubsystemContext(plant, context)
    world_frame = plant.world_frame()

    # List of cup bodies
    cropped_point_clouds = []
    scene_pcls = []

    for body, name in zip(cup_bodies, cup_names):
        # Get current pose of the cup in world frame
        X_WC = plant.CalcRelativeTransform(plant_context, world_frame, body.body_frame())

        # Draw the true axis
        AddMeshcatTriad(
            meshcat,
            f"truth/{name}",
            X_PT=X_WC,
            length=0.07,   
            radius=0.003,
        )
        # Create a crop region around the cup (adjust offset to cup size)
        lower = X_WC.translation() - np.array([0.07, 0.07, 0.15]) 
        upper = X_WC.translation() + np.array([0.07, 0.07, 0.01])

        # Get camera point clouds
        pc0 = diagram.GetOutputPort("camera0_point_cloud").Eval(context)
        pc1 = diagram.GetOutputPort("camera1_point_cloud").Eval(context)
        pc2 = diagram.GetOutputPort("camera2_point_cloud").Eval(context)

        # Crop each camera’s point cloud
        pc0_crop = pc0.Crop(lower_xyz=lower, upper_xyz=upper)
        pc1_crop = pc1.Crop(lower_xyz=lower, upper_xyz=upper)
        pc2_crop = pc2.Crop(lower_xyz=lower, upper_xyz=upper)

        # Concatenate crops from all cameras
        cup_pc = Concatenate([pc0_crop, pc1_crop, pc2_crop])

        # Downsample for efficiency
        cup_pc = cup_pc.VoxelizedDownSample(0.005)

        # Remove points below table
        cup_pc = remove_table_points(cup_pc)
        cropped_point_clouds.append(cup_pc)
        scene_pcls.append(np.asarray(cup_pc.xyzs()))

        meshcat.SetLineSegments(
            f"bounding_{name}",
            lower.reshape(3,1),
            upper.reshape(3,1),
            1.0,
            Rgba(0, 1, 0),
        )

    colors = [Rgba(1, 0, 0), Rgba(0, 1, 0), Rgba(0, 0, 1)] 

    for cup_pc, color, name in zip(cropped_point_clouds, colors, cup_names):
        meshcat.SetObject(f"cup_{name}_point_cloud", cup_pc, point_size=0.02, rgba=color)
        
    return model_points_O, world_frame, scene_pcls
