import numpy as np

from pydrake.all import (
    RigidTransform,
    RollPitchYaw,
)

from manipulation_module.poses import *

def make_keyframes(X_WGinitial, cup_icp_results):
    opened = 0.1
    closed = 0.05

    cup_order = ["left", "right", "top"]
    dt = 0.5

    gripper_poses = []
    finger_cmds = []
    sample_times = []

    t = 0.0
    current_X_WG = X_WGinitial

    gripper_poses.append(current_X_WG)
    finger_cmds.append(opened)
    sample_times.append(t)
    
    xy_list = []
    for name in cup_order:
        X_WC_hat = cup_icp_results[name]
        xy_list.append(X_WC_hat.translation()[:2])
    xy_list = np.array(xy_list)
    center_xy = xy_list.mean(axis=0)

    # Orientation for all cups in the pyramid (lying on their side)
    rpy_cup_goal = RollPitchYaw(np.deg2rad([270, 0, -90]))
    
    pyramid_positions = get_pyramid_positions(center_xy)

    cup_goal_poses = {
        "left":  RigidTransform(rpy_cup_goal, pyramid_positions[0]),
        "right": RigidTransform(rpy_cup_goal, pyramid_positions[1]),
        "top":   RigidTransform(rpy_cup_goal, pyramid_positions[2]),
    }

    for name in cup_order:
        print(f"Planning segment for cup: {name}")

        X_WC_initial = cup_icp_results[name]   # ICP estimate
        X_WC_goal    = cup_goal_poses[name]      # pyramid pose

        # Grasp at initial pose
        X_CG, X_WGpick = design_grasp_pose(X_WC_initial)
        X_WGprepick = design_pregrasp_pose(X_WGpick, dz=0.15)

        # Gripper at goal pose
        X_WGgoal = X_WC_goal @ X_CG
        X_WGpregoal = design_pregoal_pose(X_WGgoal, dz=0.10)
        X_WGpostgoal = design_postgoal_pose(X_WGgoal, dz=0.15)

        X_WGsafe = make_safe_pose_above_workspace(current_X_WG, z_safe=0.5)
        X_WGabove_cup = make_above_pose(X_WGprepick, dz=0.15)
        X_WGabove_goal = make_above_pose(X_WGpregoal, dz=0.15)

        local_sequence = [
            (X_WGsafe,        opened),
            (X_WGabove_cup,   opened),
            (X_WGprepick,     opened),
            (X_WGpick,        opened),
            (X_WGpick,        closed),
            (X_WGprepick,     closed),
            (X_WGabove_cup,   closed),
            (X_WGsafe,        closed),
            (X_WGabove_goal,  closed),
            (X_WGpregoal,     closed),
            (X_WGgoal,        closed),
            (X_WGgoal,        opened),
            (X_WGpostgoal,    opened),
            (X_WGabove_goal,  opened),
            (X_WGsafe,        opened),
        ]

        for X_WG, wsg in local_sequence:
            t += dt
            gripper_poses.append(X_WG)
            finger_cmds.append(wsg)
            sample_times.append(t)

        current_X_WG = local_sequence[-1][0]

    # Return to initial pose at the end
    X_WGsafe_final = make_safe_pose_above_workspace(current_X_WG, z_safe=0.5)
    t += dt
    gripper_poses.append(X_WGsafe_final)
    finger_cmds.append(opened)
    sample_times.append(t)

    t += dt
    gripper_poses.append(X_WGinitial)
    finger_cmds.append(opened)
    sample_times.append(t)

    finger_values = np.array([finger_cmds])
    return gripper_poses, finger_values, sample_times