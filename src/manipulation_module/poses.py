import numpy as np

from pydrake.all import (
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RollPitchYaw,
)

def design_grasp_pose(X_WC: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
    """
    Design grasp pose for a cup given X_WC (world-from-cup-base_link).

    We define a nominal grasp in the cup frame C, then apply a manual
    correction along WORLD x, mapped back into C.

    Returns:
      X_CG : cup-base_link-from-gripper
      X_WG : world-from-gripper (grasp pose)
    """
    # Gripper orientation relative to cup frame
    R_CG = RollPitchYaw(0, np.pi/2, 0).ToRotationMatrix()

    # Nominal position of gripper palm in C frame
    p_CG = np.array([0.0, 0.0, 0.05])

    # Manual correction in WORLD x
    x_correction_world = -0.04 

    d_W = np.array([x_correction_world, 0.0, 0.0])

    # Express that correction in C frame so it rotates with the cup
    R_WC = X_WC.rotation().matrix()
    R_CW = R_WC.T          
    d_C = R_CW @ d_W     

    # Apply the correction in C frame
    p_CG = p_CG + d_C
    X_CG = RigidTransform(R_CG, p_CG)
    X_WG = X_WC @ X_CG

    return X_CG, X_WG


def design_pregrasp_pose(X_WG: RigidTransform, dz: float = 0.15):
    """
    Move above the grasp position, straight in world z by +dz.
    """
    p = X_WG.translation().copy()
    p[2] += dz
    return RigidTransform(X_WG.rotation(), p)

def design_goal_poses(X_WC: RigidTransform, X_CG: RigidTransform):
    """
    Place cup at a different location on the table.
    X_WC: world-from-cup-base_link
    X_CG: cup-base_link-from-gripper
    """
    # Get current position
    current_pos = X_WC.translation()
    
    # Move to a different spot
    p_WCgoal = current_pos + np.array([0.3, 0.0, 0.0])
    R_WCgoal = X_WC.rotation()
    
    X_WCgoal = RigidTransform(R_WCgoal, p_WCgoal)
    X_WGgoal = X_WCgoal @ X_CG
    
    return X_WGgoal

def design_pregoal_pose(X_WG: RigidTransform, dz: float = 0.10):
    """
    Hover above goal before placing, in world z.
    """
    p = X_WG.translation().copy()
    p[2] += dz
    return RigidTransform(X_WG.rotation(), p)

def design_postgoal_pose(X_WG: RigidTransform, dz: float = 0.15):
    """
    Retract after placing, straight up in world z.
    """
    p = X_WG.translation().copy()
    p[2] += dz
    return RigidTransform(X_WG.rotation(), p)

def make_trajectory(X_Gs: list[RigidTransform], finger_values: np.ndarray, sample_times: list[float]):
    """
    Create smooth trajectory through keyframes.
    """
    robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
    robot_velocity_trajectory = robot_position_trajectory.MakeDerivative()
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return robot_velocity_trajectory, traj_wsg_command

def get_pyramid_positions():
    """
    Define positions for a 3-cup pyramid (2 base cups + 1 top cup).
    Returns positions for [left_base, right_base, top].
    """
    # Center of pyramid on table
    pyramid_center = np.array([0.0, 0.0, 0.0])
    
    base_spacing = 0.12  
    z_base = 0.15
    
    left_base = pyramid_center + np.array([-base_spacing/2, 0, z_base])
    right_base = pyramid_center + np.array([base_spacing/2, 0, z_base])
    
    z_top = 0.34  
    top_pos = pyramid_center + np.array([0, 0, z_top])
    
    return [left_base, right_base, top_pos]

scenario_path = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"

def create_pick_place_trajectory(X_WC_initial, X_WGinitial, goal_position, cup_orientation):
    """
    Create trajectory keyframes for picking and placing one cup.
    
    Args:
        X_WC_initial: Initial cup pose (from ICP)
        X_WGinitial: Initial gripper pose
        goal_position: Target position [x, y, z]
        cup_orientation: Desired cup orientation at goal
    """
    # Design grasp poses
    X_CG, X_WGpick = design_grasp_pose(X_WC_initial)
    X_WGprepick = design_pregrasp_pose(X_WGpick, dz=0.15)
    
    # Design goal poses with specified position and orientation
    p_WCgoal = goal_position
    R_WCgoal = cup_orientation
    X_WCgoal = RigidTransform(R_WCgoal, p_WCgoal)
    X_WGgoal = X_WCgoal @ X_CG
    X_WGpregoal = design_pregoal_pose(X_WGgoal, dz=0.10)
    X_WGpostgoal = design_postgoal_pose(X_WGgoal, dz=0.15)
    
    # Safe waypoints
    safe_height_pos = np.array([0.0, 0.0, 0.5])
    X_WGsafe = RigidTransform(X_WGinitial.rotation(), safe_height_pos)
    
    above_cup_pos = X_WGprepick.translation() + np.array([0.0, 0.0, 0.15])
    X_WGabove = RigidTransform(X_WGprepick.rotation(), above_cup_pos)
    
    above_goal_pos = X_WGpregoal.translation() + np.array([0.0, 0.0, 0.15])
    X_WGabove_goal = RigidTransform(X_WGpregoal.rotation(), above_goal_pos)
    
    opened = 0.1
    closed = 0.05
    
    keyframes = [
        ("safe_high", X_WGsafe, opened),
        ("above_cup", X_WGabove, opened),
        ("prepick", X_WGprepick, opened),
        ("pick", X_WGpick, opened),
        ("pick_close", X_WGpick, closed),
        ("pick_lift", X_WGprepick, closed),
        ("lift_high", X_WGabove, closed),
        ("safe_transport", X_WGsafe, closed),
        ("above_goal", X_WGabove_goal, closed),
        ("pregoal", X_WGpregoal, closed),
        ("goal", X_WGgoal, closed),
        ("goal_open", X_WGgoal, opened),
        ("postgoal", X_WGpostgoal, opened),
        ("above_goal2", X_WGabove_goal, opened),
        ("safe_return", X_WGsafe, opened),
    ]
    
    return keyframes
