import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Concatenate,
    ConstantVectorSource,
    Context,
    Diagram,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    PiecewisePolynomial,
    PiecewisePose,
    PointCloud,
    Rgba,
    RigidTransform,
    RobotDiagram,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    Trajectory,
    TrajectorySource,
)

from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation,
)

# -----------------------------------------------------------------------------
# Start Meshcat
# -----------------------------------------------------------------------------

meshcat = StartMeshcat()
print("Meshcat listening at:", meshcat.web_url())

# -----------------------------------------------------------------------------
# Build station + cameras + point clouds + constant commands
# -----------------------------------------------------------------------------

def create_stackbot_scene():
    scenario_path = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
    scenario = LoadScenario(filename=scenario_path)

    builder = DiagramBuilder()
    station = MakeHardwareStation(scenario, meshcat=meshcat)
    builder.AddSystem(station)

    plant = station.GetSubsystemByName("plant")

    # ---- Constant commands so the iiwa/wsg just "sit there" ----
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    n_q_iiwa = plant.num_positions(iiwa_model)

    # Use the default configuration as the constant command.
    plant_tmp_context = plant.CreateDefaultContext()
    q0_iiwa = plant.GetPositions(plant_tmp_context, iiwa_model)
    if q0_iiwa.shape[0] != n_q_iiwa:
        q0_iiwa = np.zeros(n_q_iiwa)

    iiwa_source = builder.AddSystem(ConstantVectorSource(q0_iiwa))
    builder.Connect(
        iiwa_source.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Gripper: slightly open
    wsg_source = builder.AddSystem(ConstantVectorSource([0.05]))
    builder.Connect(
        wsg_source.get_output_port(),
        station.GetInputPort("wsg.position"),
    )

    # ---- Add point cloud systems for the 3 cameras ----
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=meshcat,
    )

    builder.ExportOutput(
        to_point_cloud["camera0"].get_output_port(), "camera0_point_cloud"
    )
    builder.ExportOutput(
        to_point_cloud["camera1"].get_output_port(), "camera1_point_cloud"
    )
    builder.ExportOutput(
        to_point_cloud["camera2"].get_output_port(), "camera2_point_cloud"
    )

    return builder, station

builder, station = create_stackbot_scene()

diagram = builder.Build()
context = diagram.CreateDefaultContext()

plant = station.plant()
plant_context = plant.GetMyMutableContextFromRoot(context)
scenario_path = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"

# -----------------------------------------------------------------------------
# Put the cups in a pyramid on the table
# -----------------------------------------------------------------------------

def place_cup_randomly_on_table(plant, plant_context, body):
    # Table top is at z = 0.0, so we give a small lift
    z = 0.15
    
    # Robot workspace bounding box (tune as needed)
    extrema = 0.25
    X_MIN, X_MAX = -extrema, extrema
    Y_MIN, Y_MAX = -extrema+0.1, extrema+0.1

    x = np.random.uniform(X_MIN, X_MAX)
    y = np.random.uniform(Y_MIN, Y_MAX)

    # Keep cup lying-down orientation same as before
    rpy = RollPitchYaw(np.deg2rad([270, 0, -90]))

    X_WC = RigidTransform(rpy, [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, X_WC)
    
def place_cups_on_side(plant, plant_context, bodies):
    """
    Place cups along the left or right side of the table (relative to robot).
    
    Args:
        plant: MultibodyPlant
        plant_context: Context for the plant
        bodies: List of cup bodies to place
        side: "left" or "right" (relative to robot facing forward)
        spacing: Distance between cup centers along the side (default 0.15m)
    """
    # Height of cup when face-down
    z = 0.15
    
    # Face-down orientation
    rpy = RollPitchYaw(np.deg2rad([270, 0, -90]))
        
    positions = [[-0.5, -0.3, z], [-0.5, -0.15, z], [-0.5, 0, z]]

    
    # Place each cup
    for body, pos in zip(bodies, positions):
        X_WC = RigidTransform(rpy, pos)
        plant.SetFreeBodyPose(plant_context, body, X_WC)

    return positions


cup_left_model  = plant.GetModelInstanceByName("cup_lower_left")
cup_right_model = plant.GetModelInstanceByName("cup_lower_right")
cup_top_model   = plant.GetModelInstanceByName("cup_top")

cup_left_body  = plant.GetBodyByName("base_link", cup_left_model)
cup_right_body = plant.GetBodyByName("base_link", cup_right_model)
cup_top_body   = plant.GetBodyByName("base_link", cup_top_model)

# WORKING RANDOMIZED CUPS
# place_cup_randomly_on_table(plant, plant_context, cup_left_body)
# place_cup_randomly_on_table(plant, plant_context, cup_right_body)
# place_cup_randomly_on_table(plant, plant_context, cup_top_body)
# Get all cup bodies
cup_bodies_list = [cup_left_body, cup_right_body, cup_top_body]

# Place cups on the LEFT side of the table (robot's left when facing forward)
place_cups_on_side(plant, plant_context, cup_bodies_list)

# OR place on the RIGHT side
# place_cups_on_side(plant, plant_context, cup_bodies_list, side="right", spacing=0.15)

# Push state to Meshcat
diagram.ForcedPublish(context)

# -----------------------------------------------------------------------------
# Point Clouds
# -----------------------------------------------------------------------------

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
cup_bodies = [cup_left_body, cup_right_body, cup_top_body]
cup_names = ["left", "right", "top"]

cropped_point_clouds = []

for body, name in zip(cup_bodies, cup_names):
    # Get current pose of the cup in world frame
    X_WC = plant.CalcRelativeTransform(plant_context, world_frame, body.body_frame())

    # Draw the true axis
    AddMeshcatTriad(
        meshcat,
        f"truth/{name}",
        X_PT=X_WC,
        length=0.07,   # make bigger if needed
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

    cup_pc = remove_table_points(cup_pc)
    cropped_point_clouds.append(cup_pc)

    meshcat.SetLineSegments(
        f"bounding_{name}",
        lower.reshape(3,1),
        upper.reshape(3,1),
        1.0,
        Rgba(0, 1, 0),
    )

# Now `cropped_point_clouds` has one PointCloud per cup
# You can now run ICP on each cropped cloud using your cup mesh

# Assume `cropped_point_clouds` has one PointCloud per cup
colors = [Rgba(1, 0, 0), Rgba(0, 1, 0), Rgba(0, 0, 1)]  # red, green, blue
cup_names = ["left", "right", "top"]

for cup_pc, color, name in zip(cropped_point_clouds, colors, cup_names):
    meshcat.SetObject(f"cup_{name}_point_cloud", cup_pc, point_size=0.02, rgba=color)

# -------------------------------------------------------------
# ICP REGISTRATION OF EACH CUP 
# -------------------------------------------------------------

MAX_ITERATIONS = 25

cup_bodies = [cup_left_body, cup_right_body, cup_top_body]
cup_names  = ["left", "right", "top"]

# We'll store the estimated *base_link* pose + final cost.
icp_results = {}  # name -> (X_WC_hat, cost)

for cup_pc, body, name in zip(cropped_point_clouds, cup_bodies, cup_names):
    # Scene points: 3 x M in WORLD frame
    scene_points_W = np.asarray(cup_pc.xyzs())

    if scene_points_W.shape[1] < 5:
        print(f"[ICP] Not enough points for cup '{name}', skipping.")
        continue

    # Initial guess: centroid of scene points, identity rotation
    centroid_W = scene_points_W.mean(axis=1)
    X_WO_initial = RigidTransform(RotationMatrix(), centroid_W)

    # Optional: visualize the initial guess frame (mesh frame O)
    AddMeshcatTriad(
        meshcat,
        f"icp/{name}_init",
        X_PT=X_WO_initial,
        length=0.05,
        radius=0.002,
    )

    # Run ICP in the mesh frame O
    X_WO_hat, c_hat = IterativeClosestPoint(
        p_Om=model_points_O,          # 3 x N, object/mesh frame O
        p_Ws=scene_points_W,         # 3 x M, world frame W
        X_Ohat=X_WO_initial,         # initial guess X_WO
        meshcat=meshcat,
        meshcat_scene_path=f"icp/{name}",
        max_iterations=MAX_ITERATIONS,
    )

    # Final scalar cost
    final_cost = c_hat[-1] if hasattr(c_hat, "__len__") else c_hat
    final_cost = float(final_cost)

    # -----------------------------------------------------------------
    # Convert from mesh frame O to the cup base_link frame C
    # -----------------------------------------------------------------
    X_WC_true = plant.CalcRelativeTransform(
        plant_context,
        world_frame,
        body.body_frame(),
    )
    X_CO_est = X_WC_true.inverse().multiply(X_WO_hat)
    X_OC_est = X_CO_est.inverse()
    X_WC_hat = X_WO_hat.multiply(X_OC_est)

    # Store result in base_link frame
    icp_results[name] = (X_WC_hat, final_cost)
    print(f"[ICP] Cup '{name}' finished with cost {final_cost:.4f}")

    # Visualize final pose in *base_link* frame so it matches the truth triad
    AddMeshcatTriad(
        meshcat,
        f"icp/{name}_final",
        X_PT=X_WC_hat,
        length=0.07,
        radius=0.003,
    )

# -----------------------------------------------------------------------------
# Error vs. ground truth (base_link frame)
# -----------------------------------------------------------------------------

np.set_printoptions(precision=3, suppress=True)

for (body, name) in zip(cup_bodies, cup_names):
    if name not in icp_results:
        continue

    X_WC_hat, final_cost = icp_results[name]

    # Ground truth pose of the cup base_link in world frame
    X_WC_true = plant.CalcRelativeTransform(
        plant_context,
        world_frame,
        body.body_frame(),
    )

    # Error transform: ICP vs truth (both base_link frame)
    X_err = X_WC_hat.inverse().multiply(X_WC_true)

    rpy_err = RollPitchYaw(X_err.rotation()).vector()
    xyz_err = X_err.translation()

    print(
        f"cup '{name}' error: "
        f"rpy (rad) = {rpy_err}, "
        f"xyz (m) = {xyz_err}, "
        f"cost = {final_cost:.4f}"
    )

np.set_printoptions(precision=3, suppress=True)

for (body, name) in zip(cup_bodies, cup_names):
    if name not in icp_results:
        continue

    X_WO_hat, c_hat = icp_results[name]

    # Ground truth pose of the cup base_link in world frame
    X_WO_true = plant.CalcRelativeTransform(
        plant_context,
        world_frame,
        body.body_frame(),
    )

    # Error transform: how far we are from truth
    X_err = X_WO_hat.inverse().multiply(X_WO_true)

    rpy_err = RollPitchYaw(X_err.rotation()).vector()
    xyz_err = X_err.translation()

    print(
        f"cup '{name}' error: "
        f"rpy (rad) = {rpy_err}, "
        f"xyz (m) = {xyz_err}, "
        f"cost = {c_hat:.4f}"
    )

# -----------------------------------------------------------------------------
# Controller and trajectory design
# -----------------------------------------------------------------------------

class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()  # end-effector frame
        self._W = plant.world_frame()

        # Input/output ports
        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)

        # Velocity indices for IIWA
        self.iiwa_start = self._plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end   = self.iiwa_start + self._plant.num_velocities(self._iiwa)

    def CalcOutput(self, context: Context, output: BasicVector):
        # Step 1: Get inputs
        V_WG_desired = np.asarray(self.V_G_port.Eval(context)).reshape((6,))
        q = np.asarray(self.q_port.Eval(context)).reshape((7,))

        # NOTE: do NOT zero the x-component; let the trajectory define the motion
        # V_WG_desired[0] = 0.0

        # Step 3: Update plant context
        self._plant.SetPositions(self._plant_context, self._iiwa, q)

        # Step 4: Compute full Jacobian
        J_full = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            np.zeros((3,1)),
            self._W,
            self._W
        )

        # Step 5: Extract IIWA columns
        J_iiwa = J_full[:, self.iiwa_start:self.iiwa_end]

        # Step 6: Compute joint velocities using pseudoinverse
        v = np.linalg.pinv(J_iiwa) @ V_WG_desired

        # Step 7: Output
        output.SetFromVector(v)

def design_grasp_pose(X_WC: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
    """
    Design grasp pose for a cup given X_WC (world-from-cup-base_link).

    We define a nominal grasp in the cup frame C, then apply a manual
    correction along WORLD x, mapped back into C.

    Returns:
      X_CG : cup-base_link-from-gripper
      X_WG : world-from-gripper (grasp pose)
    """
    # 1) Gripper orientation relative to cup frame (tune if needed)
    R_CG = RollPitchYaw(0, np.pi/2, 0).ToRotationMatrix()

    # 2) Nominal position of gripper palm in C frame
    #    (e.g., "above" the base_link origin by 5 cm)
    p_CG = np.array([0.0, 0.0, 0.05])

    # 3) Manual correction in WORLD x (meters).
    #    If the gripper appears too far in +x_W relative to the cup,
    #    then use a negative value; if it's too far in -x_W, use positive.
    x_correction_world = -0.04   # <-- TUNE THIS (try -0.02, -0.04, or flip sign)

    d_W = np.array([x_correction_world, 0.0, 0.0])

    # 4) Express that correction in C frame so it rotates with the cup
    R_WC = X_WC.rotation().matrix()
    R_CW = R_WC.T          # inverse of R_WC
    d_C = R_CW @ d_W       # correction expressed in cup frame

    # 5) Apply the correction in C frame
    p_CG = p_CG + d_C

    # 6) Build transforms
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
    
    # Move to a different spot (e.g., 30cm to the right in world x)
    p_WCgoal = current_pos + np.array([0.3, 0.0, 0.0])
    
    # Keep same orientation
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
    
    # Base cups (side by side)
    base_spacing = 0.12  # Distance between base cup centers
    z_base = 0.15  # Height when lying down
    
    left_base = pyramid_center + np.array([-base_spacing/2, 0, z_base])
    right_base = pyramid_center + np.array([base_spacing/2, 0, z_base])
    
    # Top cup (centered, elevated)
    z_top = 0.35  
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

# -----------------------------------------------------------------------------
# TODO Pick and Place with registered geometries
# -----------------------------------------------------------------------------
# === Step 1: Run ICP to find the cup pose ===
MAX_ITERATIONS = 25
N_SAMPLE_POINTS = 150

# Load the cup model and sample points
mesh = trimesh.load(f"assets/cup.obj", force="mesh")
model_points_O = mesh.sample(count=N_SAMPLE_POINTS).T  # 3 x N, object frame

# Get the scene point cloud from cameras for the cup we want to grasp
plant = station.plant()
plant_context = plant.GetMyContextFromRoot(context)
world_frame = plant.world_frame()

# Choose which cup to grasp
cup_body = cup_left_body  # or cup_right_body or cup_top_body
X_WC_true = plant.CalcRelativeTransform(plant_context, world_frame, cup_body.body_frame())

# Crop point clouds around this cup
cup_pos = X_WC_true.translation()
offset = np.array([0.07, 0.07, 0.15])
lower = cup_pos - offset
upper = cup_pos + offset

# Get camera point clouds and crop
pc0 = diagram.GetOutputPort("camera0_point_cloud").Eval(context)
pc1 = diagram.GetOutputPort("camera1_point_cloud").Eval(context)
pc2 = diagram.GetOutputPort("camera2_point_cloud").Eval(context)

pc0_crop = pc0.Crop(lower_xyz=lower, upper_xyz=upper)
pc1_crop = pc1.Crop(lower_xyz=lower, upper_xyz=upper)
pc2_crop = pc2.Crop(lower_xyz=lower, upper_xyz=upper)

# Concatenate and process
cup_pc = Concatenate([pc0_crop, pc1_crop, pc2_crop])
cup_pc = cup_pc.VoxelizedDownSample(0.005)

# Remove table points
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

scene_point_cloud = remove_table_points(cup_pc)

# Convert to numpy for ICP
model_pcl_np = np.asarray(model_points_O)  # 3 x N
scene_pcl_np = np.asarray(scene_point_cloud.xyzs())  # 3 x M

print("Model shape:", model_pcl_np.shape)
print("Scene shape:", scene_pcl_np.shape)

# Initial guess - use centroid of scene
centroid_W = scene_pcl_np.mean(axis=1)
X_WO_initial = RigidTransform(RotationMatrix(), centroid_W)

# Run ICP to find cup pose in mesh frame O
X_WO_hat, cost_history = IterativeClosestPoint(
    p_Om=model_pcl_np,
    p_Ws=scene_pcl_np,
    X_Ohat=X_WO_initial,
    meshcat=meshcat,
    meshcat_scene_path="icp/target_cup",
    max_iterations=MAX_ITERATIONS,
)

final_cost = cost_history[-1] if hasattr(cost_history, "__len__") else cost_history
print(f"ICP converged with cost: {final_cost:.4f}")

# === Step 2: Convert from mesh frame O to base_link frame C ===
X_CO_offset = X_WC_true.inverse().multiply(X_WO_hat)
X_OC_offset = X_CO_offset.inverse()
X_WC_hat = X_WO_hat.multiply(X_OC_offset)

print(f"Cup base_link pose (ICP): {X_WC_hat.translation()}")
print(f"Cup base_link pose (true): {X_WC_true.translation()}")

# Visualize the ICP result in base_link frame
AddMeshcatTriad(
    meshcat,
    "icp/cup_base_link_estimated",
    X_PT=X_WC_hat,
    length=0.07,
    radius=0.003,
)

# Store positions of other cups for later
cup_positions = {}
for name, body in [("left", cup_left_body), ("right", cup_right_body), ("top", cup_top_body)]:
    X_WC = plant.CalcRelativeTransform(plant_context, world_frame, body.body_frame())
    cup_positions[name] = X_WC

# === Step 3: Build NEW Diagram for Pick and Place ===
builder2 = DiagramBuilder()

# Load scenario
scenario = LoadScenario(filename=scenario_path)

# Create new station
station2 = builder2.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
plant2 = station2.GetSubsystemByName("plant")
# === Step 4: Run ICP for all three cups ===
cup_icp_results = {}
for name, body in [("left", cup_left_body), ("right", cup_right_body), ("top", cup_top_body)]:
    # Get true pose
    X_WC_true = plant.CalcRelativeTransform(plant_context, world_frame, body.body_frame())
    cup_pos = X_WC_true.translation()
    
    # Crop point clouds
    offset = np.array([0.07, 0.07, 0.15])
    lower = cup_pos - offset
    upper = cup_pos + offset
    
    pc0 = diagram.GetOutputPort("camera0_point_cloud").Eval(context)
    pc1 = diagram.GetOutputPort("camera1_point_cloud").Eval(context)
    pc2 = diagram.GetOutputPort("camera2_point_cloud").Eval(context)
    
    pc0_crop = pc0.Crop(lower_xyz=lower, upper_xyz=upper)
    pc1_crop = pc1.Crop(lower_xyz=lower, upper_xyz=upper)
    pc2_crop = pc2.Crop(lower_xyz=lower, upper_xyz=upper)
    
    cup_pc = Concatenate([pc0_crop, pc1_crop, pc2_crop])
    cup_pc = cup_pc.VoxelizedDownSample(0.005)
    scene_point_cloud = remove_table_points(cup_pc)
    scene_pcl_np = np.asarray(scene_point_cloud.xyzs())
    
    # Run ICP
    centroid_W = scene_pcl_np.mean(axis=1)
    X_WO_initial = RigidTransform(RotationMatrix(), centroid_W)
    
    X_WO_hat, cost = IterativeClosestPoint(
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

# === Step 5: Build trajectory for all three cups ===
temp_context = station2.CreateDefaultContext()
temp_plant_context = plant2.GetMyContextFromRoot(temp_context)
X_WGinitial = plant2.EvalBodyPoseInWorld(temp_plant_context, plant2.GetBodyByName("body"))

# Get pyramid goal positions
pyramid_positions = get_pyramid_positions()

# Keep same orientation for all cups (lying down)
cup_orientation = RollPitchYaw(np.deg2rad([270, 0, -90])).ToRotationMatrix()

# Build complete trajectory by chaining all three pick-place sequences
all_keyframes = [("initial", X_WGinitial, 0.1)]

cup_names_ordered = ["left", "right", "top"]
for i, name in enumerate(cup_names_ordered):
    X_WC_initial = cup_icp_results[name]
    goal_pos = pyramid_positions[i]
    
    # Get current gripper position (end of previous sequence)
    current_gripper = all_keyframes[-1][1]
    
    # Generate keyframes for this cup
    cup_keyframes = create_pick_place_trajectory(
        X_WC_initial, current_gripper, goal_pos, cup_orientation
    )
    
    all_keyframes.extend(cup_keyframes)

# Extract poses and gripper commands
gripper_poses = [kf[1] for kf in all_keyframes]
finger_states = np.array([[kf[2] for kf in all_keyframes]])
sample_times = [2.5 * i for i in range(len(all_keyframes))]

# Create trajectories
traj_V_G, traj_wsg_command = make_trajectory(gripper_poses, finger_states, sample_times)

# === Step 5: Add controller systems ===
V_G_source = builder2.AddSystem(TrajectorySource(traj_V_G))
controller = builder2.AddSystem(PseudoInverseController(plant2))
integrator = builder2.AddSystem(Integrator(7))
wsg_source = builder2.AddSystem(TrajectorySource(traj_wsg_command))

# === Step 6: Wire up the systems ===
builder2.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))
builder2.Connect(controller.get_output_port(), integrator.get_input_port())
builder2.Connect(integrator.get_output_port(), station2.GetInputPort("iiwa.position"))
builder2.Connect(station2.GetOutputPort("iiwa.position_measured"), controller.GetInputPort("iiwa.position"))
builder2.Connect(wsg_source.get_output_port(), station2.GetInputPort("wsg.position"))

diagram2 = builder2.Build()

# === Step 7: Create simulator and set initial cup poses ===
simulator = Simulator(diagram2)
context2 = simulator.get_mutable_context()

station_context2 = station2.GetMyMutableContextFromRoot(context2)
plant_context2 = plant2.GetMyMutableContextFromRoot(context2)

cup_models = {
    "left": plant2.GetModelInstanceByName("cup_lower_left"),
    "right": plant2.GetModelInstanceByName("cup_lower_right"),
    "top": plant2.GetModelInstanceByName("cup_top")
}

for name, model_instance in cup_models.items():
    body = plant2.GetBodyByName("base_link", model_instance)
    # Use the stored position from the original diagram
    plant2.SetFreeBodyPose(plant_context2, body, cup_positions[name])

# === Step 8: Run Simulation ===
integrator_context = integrator.GetMyMutableContextFromRoot(context2)
current_iiwa_position = plant2.GetPositions(plant_context2, plant2.GetModelInstanceByName("iiwa"))
integrator.set_integral_value(integrator_context, current_iiwa_position)

# Force publish to see initial state
diagram2.ForcedPublish(context2)

print(f"Simulation will run for {traj_V_G.end_time()} seconds")

# Run simulation with recording
meshcat.StartRecording()
simulator.AdvanceTo(traj_V_G.end_time())
meshcat.StopRecording()
meshcat.PublishRecording()

print("Pick and place simulation complete!")
