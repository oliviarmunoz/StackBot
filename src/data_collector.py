import os
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from pydrake.all import (
    BasicVector,
    Concatenate,
    ConstantVectorSource,
    Context,
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
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    StartMeshcat,
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
# CONFIG
# -----------------------------------------------------------------------------

N_TRIALS = 50  # Number of trials to run
TEST = 'A'

SCENARIO_PATH = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
ICP_CSV_PATH = 'icp_errors' + TEST + '.csv'
GOAL_CSV_PATH = 'goal_position_errors' + TEST + '.csv'

# -----------------------------------------------------------------------------
# Start Meshcat
# -----------------------------------------------------------------------------

meshcat = StartMeshcat()
print("Meshcat listening at:", meshcat.web_url())

# -----------------------------------------------------------------------------
# Build station + cameras + point clouds + constant commands
# -----------------------------------------------------------------------------

def create_stackbot_scene():
    scenario = LoadScenario(filename=SCENARIO_PATH)

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

# -----------------------------------------------------------------------------
# Put the cups randomly on the table (lying on their side)
# -----------------------------------------------------------------------------

def place_cups_randomly_nonoverlapping(
    plant,
    plant_context,
    bodies,
    min_distance=0.12,
    max_attempts=100,
):
    """
    Randomly place cups on the table such that they do NOT overlap in x–y.

    Args:
        plant: MultibodyPlant
        plant_context: context for plant
        bodies: list of cup Body objects
        min_distance: minimum allowed distance between cup centers (meters)
        max_attempts: retries before giving up
    """
    z = 0.15
    rpy = RollPitchYaw(np.deg2rad([270, 0, -90]))

    extrema = 0.25
    X_MIN, X_MAX = -extrema, extrema
    Y_MIN, Y_MAX = -extrema + 0.1, extrema + 0.1

    placed_positions = []

    for body in bodies:
        success = False
        for _ in range(max_attempts):
            x = np.random.uniform(X_MIN, X_MAX)
            y = np.random.uniform(Y_MIN, Y_MAX)
            candidate = np.array([x, y])

            # Check distance from previously placed cups
            if all(
                np.linalg.norm(candidate - prev[:2]) >= min_distance
                for prev in placed_positions
            ):
                X_WC = RigidTransform(rpy, [x, y, z])
                plant.SetFreeBodyPose(plant_context, body, X_WC)
                placed_positions.append(np.array([x, y, z]))
                success = True
                break

        if not success:
            raise RuntimeError(
                "Failed to place cup without overlap — "
                "try reducing min_distance or increasing workspace."
            )

    return placed_positions

# -----------------------------------------------------------------------------
# Controller and grasp pose design
# -----------------------------------------------------------------------------

class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)

        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)

        self.iiwa_start = self._plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end   = self.iiwa_start + self._plant.num_velocities(self._iiwa)

    def CalcOutput(self, context: Context, output: BasicVector):
        V_WG_desired = np.asarray(self.V_G_port.Eval(context)).reshape((6,))
        q = np.asarray(self.q_port.Eval(context)).reshape((7,))

        self._plant.SetPositions(self._plant_context, self._iiwa, q)

        J_full = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            np.zeros((3, 1)),
            self._W,
            self._W,
        )
        J_iiwa = J_full[:, self.iiwa_start:self.iiwa_end]
        v = np.linalg.pinv(J_iiwa) @ V_WG_desired
        output.SetFromVector(v)


def design_grasp_pose(X_WC: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
    """
    Design grasp pose for a cup given X_WC (world-from-cup-base_link).

    We put the gripper directly above the cup base_link origin in the
    cup frame C, with no lateral offset by default. This should make the
    gripper center over the cup; if it's still off by a small constant,
    you can add a tiny dx_C / dy_C tweak below.
    """
    # Gripper orientation relative to the cup frame C
    # (this is what you had: gripper y-axis along cup axis, etc.)
    R_CG = RollPitchYaw(0, np.pi/2, 0).ToRotationMatrix()

    # ---- Centered grasp in the cup frame (NOT NEEDED) ----
    dx_C = 0.00   # lateral offset in cup frame x (tune if needed)
    dy_C = 0.00   # lateral offset in cup frame y (tune if needed)
    z_C  = 0.00   # height above base_link origin in cup frame z

    p_CG = np.array([dx_C, dy_C, z_C])

    # Build transforms
    X_CG = RigidTransform(R_CG, p_CG)
    X_WG = X_WC @ X_CG

    return X_CG, X_WG

def design_pregrasp_pose(X_WG: RigidTransform, dz: float = 0.15):
    p = X_WG.translation().copy()
    p[2] += dz
    return RigidTransform(X_WG.rotation(), p)

def design_pregoal_pose(X_WG: RigidTransform, dz: float = 0.10):
    p = X_WG.translation().copy()
    p[2] += dz
    return RigidTransform(X_WG.rotation(), p)

def design_postgoal_pose(X_WG: RigidTransform, dz: float = 0.15):
    p = X_WG.translation().copy()
    p[2] += dz
    return RigidTransform(X_WG.rotation(), p)

def make_trajectory(X_Gs, finger_values, sample_times):
    robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
    robot_velocity_trajectory = robot_position_trajectory.MakeDerivative()
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return robot_velocity_trajectory, traj_wsg_command

def get_pyramid_positions(center_xy: np.ndarray,
                          z_base: float = 0.15,
                          z_top: float = 0.35,
                          base_spacing: float = 0.12):
    """
    Define positions for a 3-cup pyramid (2 base cups + 1 top cup),
    centered around center_xy = [x, y].

    Returns positions [left_base, right_base, top] in WORLD frame.
    """
    cx, cy = center_xy

    left_base  = np.array([cx - base_spacing / 2.0, cy, z_base])
    right_base = np.array([cx + base_spacing / 2.0, cy, z_base])
    top_pos    = np.array([cx, cy, z_top])

    return [left_base, right_base, top_pos]

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

def make_safe_pose_above_workspace(X_WG_ref, z_safe=0.5):
    return RigidTransform(X_WG_ref.rotation(), [0.0, 0.0, z_safe])

def make_above_pose(X_WG_ref, dz=0.15):
    p = X_WG_ref.translation().copy()
    p[2] += dz
    return RigidTransform(X_WG_ref.rotation(), p)

# -----------------------------------------------------------------------------
# SINGLE TRIAL
# -----------------------------------------------------------------------------

def run_single_trial(trial_idx: int, icp_writer: csv.writer, goal_writer: csv.writer,
                     icp_file, goal_file):
    print("\n" + "=" * 80)
    print(f"Starting trial {trial_idx}")
    print("=" * 80)

    # Build station + diagram for perception / ICP
    builder, station = create_stackbot_scene()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    plant = station.plant()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    world_frame = plant.world_frame()

    # Get cup models and bodies
    cup_left_model  = plant.GetModelInstanceByName("cup_lower_left")
    cup_right_model = plant.GetModelInstanceByName("cup_lower_right")
    cup_top_model   = plant.GetModelInstanceByName("cup_top")

    cup_left_body  = plant.GetBodyByName("base_link", cup_left_model)
    cup_right_body = plant.GetBodyByName("base_link", cup_right_model)
    cup_top_body   = plant.GetBodyByName("base_link", cup_top_model)

    # Randomize initial cup locations on the table
    cup_bodies_list = [cup_left_body, cup_right_body, cup_top_body]

    place_cups_randomly_nonoverlapping(
        plant,
        plant_context,
        cup_bodies_list,
        min_distance=0.12,   # ~ cup diameter
    )

    # Push state to Meshcat
    diagram.ForcedPublish(context)

    # -----------------------------------------------------------------------------
    # Point Clouds + ICP for each cup
    # -----------------------------------------------------------------------------

    N_SAMPLE_POINTS = 150
    mesh = trimesh.load("assets/cup.obj", force="mesh")
    model_points_O = mesh.sample(count=N_SAMPLE_POINTS).T  # 3 x N, object frame

    plant = station.plant()
    plant_context = diagram.GetSubsystemContext(plant, context)
    world_frame = plant.world_frame()

    cup_bodies = [cup_left_body, cup_right_body, cup_top_body]
    cup_names = ["left", "right", "top"]

    cropped_point_clouds = []

    # Crop around each cup and build per-cup point clouds
    for body, name in zip(cup_bodies, cup_names):
        X_WC = plant.CalcRelativeTransform(plant_context, world_frame, body.body_frame())

        lower = X_WC.translation() - np.array([0.07, 0.07, 0.15])
        upper = X_WC.translation() + np.array([0.07, 0.07, 0.01])

        pc0 = diagram.GetOutputPort("camera0_point_cloud").Eval(context)
        pc1 = diagram.GetOutputPort("camera1_point_cloud").Eval(context)
        pc2 = diagram.GetOutputPort("camera2_point_cloud").Eval(context)

        pc0_crop = pc0.Crop(lower_xyz=lower, upper_xyz=upper)
        pc1_crop = pc1.Crop(lower_xyz=lower, upper_xyz=upper)
        pc2_crop = pc2.Crop(lower_xyz=lower, upper_xyz=upper)

        cup_pc = Concatenate([pc0_crop, pc1_crop, pc2_crop])
        cup_pc = cup_pc.VoxelizedDownSample(0.005)
        cup_pc = remove_table_points(cup_pc)

        cropped_point_clouds.append(cup_pc)

    # -------------------------------------------------------------
    # ICP REGISTRATION OF EACH CUP (base_link pose X_WC_hat)
    # -------------------------------------------------------------

    MAX_ITERATIONS = 25
    icp_results = {}  # name -> (X_WC_hat, cost)

    for cup_pc, body, name in zip(cropped_point_clouds, cup_bodies, cup_names):
        scene_points_W = np.asarray(cup_pc.xyzs())
        if scene_points_W.shape[1] < 5:
            print(f"[ICP] Not enough points for cup '{name}', skipping.")
            continue

        centroid_W = scene_points_W.mean(axis=1)
        X_WO_initial = RigidTransform(RotationMatrix(), centroid_W)

        X_WO_hat, c_hat = IterativeClosestPoint(
            p_Om=model_points_O,
            p_Ws=scene_points_W,
            X_Ohat=X_WO_initial,
            meshcat=meshcat,
            #meshcat_scene_path=f"icp_trial_{trial_idx}/{name}",
            max_iterations=MAX_ITERATIONS,
        )

        final_cost = float(c_hat[-1] if hasattr(c_hat, "__len__") else c_hat)

        # Convert from mesh frame O to cup base_link frame C
        X_WC_true = plant.CalcRelativeTransform(
            plant_context,
            world_frame,
            body.body_frame(),
        )
        X_CO_est = X_WC_true.inverse().multiply(X_WO_hat)
        X_OC_est = X_CO_est.inverse()
        X_WC_hat = X_WO_hat.multiply(X_OC_est)

        icp_results[name] = (X_WC_hat, final_cost)
        print(f"[ICP] Cup '{name}' finished with cost {final_cost:.4f}")

    # Check ICP error (and log to CSV)
    np.set_printoptions(precision=3, suppress=True)
    for (body, name) in zip(cup_bodies, cup_names):
        if name not in icp_results:
            continue
        X_WC_hat, final_cost = icp_results[name]
        X_WC_true = plant.CalcRelativeTransform(
            plant_context,
            world_frame,
            body.body_frame(),
        )
        X_err = X_WC_hat.inverse().multiply(X_WC_true)
        rpy_err = RollPitchYaw(X_err.rotation()).vector()
        xyz_err = X_err.translation()
        pos_err_norm = np.linalg.norm(xyz_err)
        rpy_err_norm = np.linalg.norm(rpy_err)

        print(
            f"cup '{name}' error: rpy(rad)={rpy_err}, xyz(m)={xyz_err}, cost={final_cost:.4f}"
        )

        # Log ICP error to CSV
        icp_writer.writerow([
            trial_idx,
            name,
            rpy_err[0], rpy_err[1], rpy_err[2],
            xyz_err[0], xyz_err[1], xyz_err[2],
            rpy_err_norm,
            pos_err_norm,
            final_cost,
        ])

    # Make sure data is pushed to disk incrementally
    icp_file.flush()
    os.fsync(icp_file.fileno())

    # -----------------------------------------------------------------------------
    # Define goal pyramid poses based on *random* cup locations
    # -----------------------------------------------------------------------------

    # Compute pyramid center from current cup locations (average x, y)
    cup_order = ["left", "right", "top"]
    xy_list = []
    for name in cup_order:
        X_WC_hat, _ = icp_results[name]
        xy_list.append(X_WC_hat.translation()[:2])
    xy_list = np.array(xy_list)
    center_xy = xy_list.mean(axis=0)

    # Orientation for all cups in the pyramid (lying on their side)
    rpy_cup_goal = RollPitchYaw(np.deg2rad([270, 0, -90]))

    # Define goal positions for the pyramid around center_xy
    pyramid_positions = get_pyramid_positions(
        center_xy,
        z_base=0.15,
        z_top=0.35,
        base_spacing=0.12,
    )

    cup_goal_poses = {
        "left":  RigidTransform(rpy_cup_goal, pyramid_positions[0]),
        "right": RigidTransform(rpy_cup_goal, pyramid_positions[1]),
        "top":   RigidTransform(rpy_cup_goal, pyramid_positions[2]),
    }

    # Initial cup poses from ICP
    cup_initial_poses = {}
    for name in ["left", "right", "top"]:
        X_WC_hat, _ = icp_results[name]
        cup_initial_poses[name] = X_WC_hat

    # Also store the *true* positions from the first diagram to re-place in plant2
    cup_positions_true = {}
    for name, body in [("left", cup_left_body), ("right", cup_right_body), ("top", cup_top_body)]:
        X_WC = plant.CalcRelativeTransform(plant_context, world_frame, body.body_frame())
        cup_positions_true[name] = X_WC

    # -----------------------------------------------------------------------------
    # Build NEW Diagram for Multi-cup Pick and Place
    # -----------------------------------------------------------------------------

    builder2 = DiagramBuilder()
    scenario = LoadScenario(filename=SCENARIO_PATH)
    station2 = builder2.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
    plant2 = station2.GetSubsystemByName("plant")

    # Initial gripper pose in the new plant
    temp_context2 = station2.CreateDefaultContext()
    temp_plant_context2 = plant2.GetMyContextFromRoot(temp_context2)
    X_WGinitial = plant2.EvalBodyPoseInWorld(
        temp_plant_context2, plant2.GetBodyByName("body")
    )

    opened = 0.1
    closed = 0.05

    cup_order = ["left", "right", "top"]
    #dt = 2.5
    # REDUCED FOR FASTER TRIALS ERNIE
    dt = 1.5

    gripper_poses = []
    finger_cmds = []
    sample_times = []

    t = 0.0
    current_X_WG = X_WGinitial

    gripper_poses.append(current_X_WG)
    finger_cmds.append(opened)
    sample_times.append(t)

    for name in cup_order:
        print(f"Planning segment for cup: {name}")

        X_WC_initial = cup_initial_poses[name]   # ICP estimate
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
    traj_V_G, traj_wsg_command = make_trajectory(gripper_poses, finger_values, sample_times)

    # -----------------------------------------------------------------------------
    # Add controller + WSG sources and wire everything
    # -----------------------------------------------------------------------------

    V_G_source = builder2.AddSystem(TrajectorySource(traj_V_G))
    controller = builder2.AddSystem(PseudoInverseController(plant2))
    integrator = builder2.AddSystem(Integrator(7))
    wsg_source = builder2.AddSystem(TrajectorySource(traj_wsg_command))

    builder2.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))
    builder2.Connect(controller.get_output_port(), integrator.get_input_port())
    builder2.Connect(integrator.get_output_port(), station2.GetInputPort("iiwa.position"))
    builder2.Connect(
        station2.GetOutputPort("iiwa.position_measured"),
        controller.GetInputPort("iiwa.position"),
    )
    builder2.Connect(wsg_source.get_output_port(), station2.GetInputPort("wsg.position"))

    # -----------------------------------------------------------------------------
    # Initialize cup poses in plant2 using *true* initial poses
    # -----------------------------------------------------------------------------

    cup_models2 = {
        "left":  plant2.GetModelInstanceByName("cup_lower_left"),
        "right": plant2.GetModelInstanceByName("cup_lower_right"),
        "top":   plant2.GetModelInstanceByName("cup_top"),
    }

    for name, model_instance in cup_models2.items():
        body2 = plant2.GetBodyByName("base_link", model_instance)
        X_WC_init = cup_positions_true[name]
        plant2.SetFreeBodyPose(temp_plant_context2, body2, X_WC_init)

    # Build final diagram, set up simulator
    diagram2 = builder2.Build()
    simulator = Simulator(diagram2)
    context2 = simulator.get_mutable_context()

    plant_context2 = plant2.GetMyMutableContextFromRoot(context2)
    for name, model_instance in cup_models2.items():
        body2 = plant2.GetBodyByName("base_link", model_instance)
        X_WC_init = cup_positions_true[name]
        plant2.SetFreeBodyPose(plant_context2, body2, X_WC_init)

    integrator_context = integrator.GetMyMutableContextFromRoot(context2)
    current_iiwa_position = plant2.GetPositions(
        plant_context2, plant2.GetModelInstanceByName("iiwa")
    )
    integrator.set_integral_value(integrator_context, current_iiwa_position)

    diagram2.ForcedPublish(context2)

    print(f"Simulation will run for {traj_V_G.end_time()} seconds (trial {trial_idx})")

    meshcat.StartRecording()
    simulator.AdvanceTo(traj_V_G.end_time())
    meshcat.StopRecording()
    meshcat.PublishRecording()

    # -----------------------------------------------------------------------------
    # Log goal position error at end of simulation
    # -----------------------------------------------------------------------------

    for name, model_instance in cup_models2.items():
        body2 = plant2.GetBodyByName("base_link", model_instance)

        # Final cup pose in world (plant2)
        X_WC_final = plant2.EvalBodyPoseInWorld(plant_context2, body2)

        # Goal pose for this cup (constructed earlier from ICP + pyramid layout)
        X_WC_goal = cup_goal_poses[name]

        # Pose error from goal to final:
        #   X_err = (goal frame)^{-1} * (final pose in world)
        # This expresses the final pose *in the goal frame*.
        X_err = X_WC_goal.inverse().multiply(X_WC_final)

        # Translation error in the goal cup frame
        diff = X_err.translation()
        dist = np.linalg.norm(diff)

        # Log goal position error to CSV
        goal_writer.writerow([
            trial_idx,
            name,
            diff[0], diff[1], diff[2],
            dist,
        ])


    # Flush goal data as well
    goal_file.flush()
    os.fsync(goal_file.fileno())

    print("Multi-cup pyramid pick-and-place complete for trial", trial_idx)

# -----------------------------------------------------------------------------
# MAIN: RUN N TRIALS AND LOG CSVs
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure directories exist
    for path_str in [ICP_CSV_PATH, GOAL_CSV_PATH]:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    # Check if files already exist to decide whether to write headers
    icp_exists_before = os.path.isfile(ICP_CSV_PATH)
    goal_exists_before = os.path.isfile(GOAL_CSV_PATH)

    # Open in append mode so we *add* to existing logs; create if missing
    with open(ICP_CSV_PATH, "a", newline="") as icp_file, \
         open(GOAL_CSV_PATH, "a", newline="") as goal_file:

        icp_writer = csv.writer(icp_file)
        goal_writer = csv.writer(goal_file)

        # Write headers only if the file did not exist before
        if not icp_exists_before:
            icp_writer.writerow([
                "trial",
                "cup_name",
                "rpy_err_x",
                "rpy_err_y",
                "rpy_err_z",
                "xyz_err_x",
                "xyz_err_y",
                "xyz_err_z",
                "rpy_err_norm",
                "xyz_err_norm",
                "final_cost",
            ])
            icp_file.flush()
            os.fsync(icp_file.fileno())

        if not goal_exists_before:
            goal_writer.writerow([
                "trial",
                "cup_name",
                "goal_err_x",
                "goal_err_y",
                "goal_err_z",
                "goal_err_dist",
            ])
            goal_file.flush()
            os.fsync(goal_file.fileno())

        for trial_idx in range(N_TRIALS):
            run_single_trial(trial_idx, icp_writer, goal_writer, icp_file, goal_file)

    print("\nAll trials complete.")
    print(f"ICP error log saved to: {ICP_CSV_PATH}")
    print(f"Goal position error log saved to: {GOAL_CSV_PATH}")
