import os
from pathlib import Path

import numpy as np

from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    ConstantVectorSource,
)
from pydrake.math import RigidTransform, RollPitchYaw

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

patch_file_paths()

# -----------------------------------------------------------------------------
# 2. Start Meshcat
# -----------------------------------------------------------------------------

meshcat = StartMeshcat()

# -----------------------------------------------------------------------------
# 3. Build station + cameras + point clouds + constant commands
# -----------------------------------------------------------------------------

def create_stackbot_scene():
    scenario_path = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
    scenario = LoadScenario(filename=scenario_path)
    station = MakeHardwareStation(scenario, meshcat)

    builder = DiagramBuilder()
    station = builder.AddSystem(station)

    plant = station.plant()

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

cup_left_model  = plant.GetModelInstanceByName("cup_lower_left")
cup_right_model = plant.GetModelInstanceByName("cup_lower_right")
cup_top_model   = plant.GetModelInstanceByName("cup_top")

cup_left_body  = plant.GetBodyByName("base_link", cup_left_model)
cup_right_body = plant.GetBodyByName("base_link", cup_right_model)
cup_top_body   = plant.GetBodyByName("base_link", cup_top_model)

# Orientation: same as in your notebook (lying on their side)
rpy_cup = RollPitchYaw(np.deg2rad([270, 0, -90]))

# Base row on table (y = 0), spacing in x.
# z ~ 0.2 puts them roughly on the table; adjust if they float / intersect.
X_W_left  = RigidTransform(rpy_cup, [-0.10, 0.0, 0.20])
X_W_right = RigidTransform(rpy_cup, [ 0.10, 0.0, 0.20])

# Top cup centered above them, a bit higher
X_W_top   = RigidTransform(rpy_cup, [ 0.00, 0.0, 0.35])

plant.SetFreeBodyPose(plant_context, cup_left_body,  X_W_left)
plant.SetFreeBodyPose(plant_context, cup_right_body, X_W_right)
plant.SetFreeBodyPose(plant_context, cup_top_body,   X_W_top)

# Push state to Meshcat
diagram.ForcedPublish(context)

# -----------------------------------------------------------------------------
# 5. Run a short simulation to test collisions & point clouds
# -----------------------------------------------------------------------------

simulator = Simulator(diagram, context)
simulator.Initialize()

# A couple seconds is enough to see gravity / contact effects
simulator.AdvanceTo(15.0)

print("Scene setup complete: iiwa + wsg + table + pyramid cups + 3 depth cameras with point clouds.")
