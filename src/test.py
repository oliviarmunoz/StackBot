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
    X_MIN, X_MAX = -0.35, 0.35
    Y_MIN, Y_MAX = -0.35, 0.35
    
    x = np.random.uniform(X_MIN, X_MAX)
    y = np.random.uniform(Y_MIN, Y_MAX)

    # Keep cup lying-down orientation same as before
    rpy = RollPitchYaw(np.deg2rad([270, 0, -90]))

    X_WC = RigidTransform(rpy, [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, X_WC)

def place_cup_noncolliding(plant, plant_context, body, max_tries=100):
    """
    Sample random poses for the cup until Drake reports that
    the placement does NOT collide with the robot, table, or other cups.
    """
    for attempt in range(max_tries):
        # Sample random position
        z = 0.15
        X_MIN, X_MAX = -0.35, 0.35
        Y_MIN, Y_MAX = -0.35, 0.35
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        rpy = RollPitchYaw(np.deg2rad([270, 0, -90]))
        X_WC = RigidTransform(rpy, [x, y, z])

        # Try pose
        plant.SetFreeBodyPose(plant_context, body, X_WC)

        # Query collisions
        query_object = plant.get_geometry_query_input_port().Eval(plant_context)
        penetrations = query_object.ComputePointPairPenetration()

        if len(penetrations) == 0:
            # SUCCESS!
            return X_WC

    raise RuntimeError("Could not find a collision-free placement after many tries.")


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

N_SAMPLE_POINTS = 1500

# load your intitials with trimesh.load(...) as a mesh.
# To do this, you should make sure to use the kwargs force="mesh".
# See the docs for more info at https://trimesh.org/
mesh = trimesh.load(f"assets/cup.obj", force="mesh")

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
