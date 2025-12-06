
from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)

from manipulation.icp import IterativeClosestPoint

MAX_ITERATIONS = 25

def crop_point_clouds_run_icp(cup_names, cup_bodies, plant_context, world_frame, plant, diagram, context, model_points_O, meshcat, scene_pcls):
    cup_icp_results = {}
    
    for name, body, scene_pcl_np in zip(cup_names, cup_bodies, scene_pcls):
        # Get true pose
        X_WC_true = plant.CalcRelativeTransform(plant_context, world_frame, body.body_frame())
        # cup_pos = X_WC_true.translation()
        
        # Crop point clouds
        # offset = np.array([0.07, 0.07, 0.15])
        # lower = cup_pos - offset
        # upper = cup_pos + offset
        
        # pc0 = diagram.GetOutputPort("camera0_point_cloud").Eval(context)
        # pc1 = diagram.GetOutputPort("camera1_point_cloud").Eval(context)
        # pc2 = diagram.GetOutputPort("camera2_point_cloud").Eval(context)
        
        # pc0_crop = pc0.Crop(lower_xyz=lower, upper_xyz=upper)
        # pc1_crop = pc1.Crop(lower_xyz=lower, upper_xyz=upper)
        # pc2_crop = pc2.Crop(lower_xyz=lower, upper_xyz=upper)
        
        # cup_pc = Concatenate([pc0_crop, pc1_crop, pc2_crop])
        # cup_pc = cup_pc.VoxelizedDownSample(0.005)
        # scene_point_cloud = remove_table_points(cup_pc)
        # scene_pcl_np = np.asarray(scene_point_cloud.xyzs())
        
        # Run ICP
        centroid_W = scene_pcl_np.mean(axis=1)
        X_WO_initial = RigidTransform(RotationMatrix(), centroid_W)
        
        X_WO_hat, _ = IterativeClosestPoint(
            p_Om=model_points_O,
            p_Ws=scene_pcl_np,
            X_Ohat=X_WO_initial,
            meshcat=meshcat,
            meshcat_scene_path=f"icp/{name}_cup",
            max_iterations=MAX_ITERATIONS,
        )
        
        # Convert to base_link frame
        X_CO_offset = X_WC_true.inverse().multiply(X_WO_hat)
        X_OC_offset = X_CO_offset.inverse()
        X_WC_hat = X_WO_hat.multiply(X_OC_offset)
        
        cup_icp_results[name] = X_WC_hat
        print(f"Cup '{name}' ICP pose: {X_WC_hat.translation()}")
    return cup_icp_results