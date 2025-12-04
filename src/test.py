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
# 1. Patch old Deepnote paths to the current working directory
# -----------------------------------------------------------------------------

def patch_file_paths():
    """
    The project files still reference file:///datasets/_deepnote_work.
    Replace that prefix with the current working directory so Drake
    can find the table / cups / directives / scenario.
    """
    cwd = Path.cwd()
    old_prefix = "file:///datasets/_deepnote_work"
    new_prefix = f"file://{cwd}"

    files_to_patch = [
        Path("directives/bimanual_IIWA14_with_table.dmd.yaml"),
        Path("directives/camera_directives.dmd.yaml"),
        Path("scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"),
        Path("assets/model_cup.sdf"),
    ]

    for path in files_to_patch:
        if not path.exists():
            print(f"[patch_file_paths] Warning: {path} not found, skipping")
            continue
        text = path.read_text()
        if old_prefix in text:
            path.write_text(text.replace(old_prefix, new_prefix))
            print(f"[patch_file_paths] Patched paths in {path}")
        else:
            # Nothing to do, but good to know.
            print(f"[patch_file_paths] No old prefix found in {path}")

# patch_file_paths()

# -----------------------------------------------------------------------------
# 2. Start Meshcat
# -----------------------------------------------------------------------------

meshcat = StartMeshcat()
print("Meshcat listening at:", meshcat.web_url())

# -----------------------------------------------------------------------------
# 3. Build station + cameras + point clouds + constant commands
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

# -----------------------------------------------------------------------------
# 4. Put the cups in a pyramid on the table
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

cup_left_model  = plant.GetModelInstanceByName("cup_lower_left")
cup_right_model = plant.GetModelInstanceByName("cup_lower_right")
cup_top_model   = plant.GetModelInstanceByName("cup_top")

cup_left_body  = plant.GetBodyByName("base_link", cup_left_model)
cup_right_body = plant.GetBodyByName("base_link", cup_right_model)
cup_top_body   = plant.GetBodyByName("base_link", cup_top_model)

# # Orientation: same as in your notebook (lying on their side)
# rpy_cup = RollPitchYaw(np.deg2rad([270, 0, -90]))

# # Base row on table (y = 0), spacing in x.
# # z ~ 0.2 puts them roughly on the table; adjust if they float / intersect.
# X_W_left  = RigidTransform(rpy_cup, [-0.06, 0.0, 0.20])
# X_W_right = RigidTransform(rpy_cup, [ 0.06, 0.0, 0.20])

# # Top cup centered above them, a bit higher
# X_W_top   = RigidTransform(rpy_cup, [ 0.00, 0.0, 0.35])

# plant.SetFreeBodyPose(plant_context, cup_left_body,  X_W_left)
# plant.SetFreeBodyPose(plant_context, cup_right_body, X_W_right)
# plant.SetFreeBodyPose(plant_context, cup_top_body,   X_W_top)


# WORKING RANDOMIZED CUPS
place_cup_randomly_on_table(plant, plant_context, cup_left_body)
place_cup_randomly_on_table(plant, plant_context, cup_right_body)
place_cup_randomly_on_table(plant, plant_context, cup_top_body)

# Push state to Meshcat
diagram.ForcedPublish(context)

# -----------------------------------------------------------------------------
# 5. Point Clouds (Notebook 4 - Geometric Pose Estimation)
# -----------------------------------------------------------------------------

N_SAMPLE_POINTS = 150

# load your intitials with trimesh.load(...) as a mesh.
# To do this, you should make sure to use the kwargs force="mesh".
# See the docs for more info at https://trimesh.org/
mesh = trimesh.load(f"assets/cup.obj", force="mesh")



model_points_O = mesh.sample(count=N_SAMPLE_POINTS).T  # 3 x N, object frame

# Drake PointCloud for visualization (optional)
model_point_cloud = PointCloud(N_SAMPLE_POINTS)
model_point_cloud.mutable_xyzs()[:] = model_points_O



# sample N_SAMPLE_POINTS from the mesh and then turn those points into a numpy array (you might need to transpose)
# You should make use of the `sample` method of the mesh object.
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

    # # Optional: visualize bounding box in Meshcat
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
# TODO REGISTER POINT CLOUD GEOMETRY WITH ICP
# -------------------------------------------------------------

# -------------------------------------------------------------
# 5.5 ICP REGISTRATION OF EACH CUP
# -------------------------------------------------------------

# -------------------------------------------------------------
# 5.5 ICP REGISTRATION OF EACH CUP (FIXED)
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
    # Convert from mesh frame O to the cup base_link frame C:
    #   X_WC_true = world-from-base_link (plant)
    #   X_WO_hat  = world-from-mesh (ICP)
    #   X_CO_est  = (X_WC_true)^{-1} * X_WO_hat   (mesh-in-base_link)
    #   X_OC_est  = (X_CO_est)^{-1}
    #   X_WC_hat  = X_WO_hat * X_OC_est           (predicted base_link pose)
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
    # TODO (OLIVIA) create debug feature to visualize TRIADS (FIND AND ERADICATE)
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


# -----------------------------------------------------------------------------
# TODO Pick and Place with registered geometries
# -----------------------------------------------------------------------------

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
# 6. Run a short simulation to test collisions & point clouds
# -----------------------------------------------------------------------------

# simulator = Simulator(diagram, context)
# simulator.Initialize()

# # A couple seconds is enough to see gravity / contact effects
# simulator.AdvanceTo(15.0)
simulator = Simulator(diagram, context)
# context = simulator.get_mutable_context()
# diagram.ForcedPublish(context)

meshcat.StartRecording()
simulator.AdvanceTo(5.0)
meshcat.StopRecording()
meshcat.PublishRecording()

print("Scene setup complete: iiwa + wsg + table + pyramid cups + 3 depth cameras with point clouds.")
