
from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.icp import IterativeClosestPoint

def crop_point_clouds_run_icp(cup_names, cup_bodies, plant_context, world_frame, plant, model_points_O, meshcat, scene_pcls, MAX_ITERATIONS):
    cup_icp_results = {}
    cup_true_poses = {}
    
    for name, body, scene_pcl_np in zip(cup_names, cup_bodies, scene_pcls):
        # Get true pose
        X_WC_true = plant.CalcRelativeTransform(plant_context, world_frame, body.body_frame())
        cup_true_poses[name] = X_WC_true 
        
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
        
        AddMeshcatTriad(
            meshcat,
            f"icp/{name}_init",
            X_PT=X_WO_initial,
            length=0.05,
            radius=0.002,
        )
        
        # Convert to base_link frame
        X_CO_est = X_WC_true.inverse().multiply(X_WO_hat)
        X_OC_est = X_CO_est.inverse()
        X_WC_hat = X_WO_hat.multiply(X_OC_est)
        
        cup_icp_results[name] = X_WC_hat
        print(f"Cup '{name}' ICP pose: {X_WC_hat.translation()}")
        
        AddMeshcatTriad(
            meshcat,
            f"icp/{name}_final",
            X_PT=X_WC_hat,
            length=0.07,
            radius=0.003,
        )
    
    return cup_icp_results, cup_true_poses 