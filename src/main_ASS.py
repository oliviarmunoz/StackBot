"""
StackBot: ICP-based pyramidal cup stacking with Drake 1.44.0

Phase 1 (Perception):
  - Build station with iiwa + wsg + table + cups + 3 RGB-D cameras.
  - Randomly place cups in non-colliding poses above the table with random
    orientations, then simulate so they fall and settle.
  - Use point clouds + per-cup bounding boxes + ICP to estimate X_WC_hat
    (world-from-cup base_link) for each cup.

Phase 2 (Control):
  - Build a new station for control.
  - Initialize each cup at its ICP-estimated pose.
  - Plan a pyramidal stack (as many levels as possible for N cups).
  - Use pseudoinverse Jacobian control + wsg trajectory to pick and place
    each cup into the pyramid.

Notes:
  - Planning / control uses ONLY ICP poses (no ground-truth) for cup locations.
  - Perception still uses ground-truth cup poses for cropping boxes, like your
    working ICP script.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import trimesh

from pydrake.all import (
    AddFrameTriadIllustration,
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
from pydrake.multibody.tree import ModelInstanceIndex

from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation,
)

# ============================================================================ #
# Global paths
# ============================================================================ #

SCENARIO_PATH = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
CUP_MESH_PATH = "assets/cup.obj"


# ============================================================================ #
# Optional Deepnote path patching
# ============================================================================ #

def patch_file_paths():
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
            print(f"[patch_file_paths] No old prefix in {path}")


# ============================================================================ #
# Helpers for cup models and random placement
# ============================================================================ #

def get_cup_model_instances(plant: MultibodyPlant):
    """Return ModelInstanceIndex for all models whose name starts with 'cup'."""
    cups = []
    for i in range(plant.num_model_instances()):
        mi = ModelInstanceIndex(i)
        if plant.GetModelInstanceName(mi).startswith("cup"):
            cups.append(mi)
    return cups


def sample_non_colliding_xy(existing_xy, radius=0.06, xmax=0.25, ymax=0.25):
    """
    Sample a new (x, y) such that it is not too close to existing cup centers.
    Very simple rejection sampling in a rectangular workspace.
    """
    X_MIN, X_MAX = -xmax, xmax
    Y_MIN, Y_MAX = -ymax + 0.1, ymax + 0.1
    min_dist = 2 * radius + 0.01

    for _ in range(100):
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        if all(np.linalg.norm([x - ex, y - ey]) > min_dist for (ex, ey) in existing_xy):
            return x, y

    # Fallback (very unlikely to be hit)
    return np.random.uniform(X_MIN, X_MAX), np.random.uniform(Y_MIN, Y_MAX)


def place_cup_randomly_on_table(
    plant: MultibodyPlant,
    plant_context: Context,
    body,
    existing_xy: list[tuple[float, float]],
):
    """
    Place a cup above the table in a random non-colliding x-y position, with
    random orientation (small tilt + random yaw). The simulation will then let
    it fall and settle.
    """
    # Base z (table height ~0, spawn above it)
    z_spawn = 0.25

    x, y = sample_non_colliding_xy(existing_xy)
    existing_xy.append((x, y))

    # Random yaw, small random tilt in roll/pitch
    yaw = np.random.uniform(-np.pi, np.pi)
    roll = np.random.uniform(-0.4, 0.4)   # about ±23 deg
    pitch = np.random.uniform(-0.4, 0.4)
    rpy = RollPitchYaw(roll, pitch, yaw)

    X_WC = RigidTransform(rpy, [x, y, z_spawn])
    plant.SetFreeBodyPose(plant_context, body, X_WC)


# ============================================================================ #
# Controller + grasp design (based on your Pset code)
# ============================================================================ #

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
        self.iiwa_end = self.iiwa_start + self._plant.num_velocities(self._iiwa)

    def CalcOutput(self, context: Context, output: BasicVector):
        V_WG_desired = np.asarray(self.V_G_port.Eval(context)).reshape((6,))
        q = np.asarray(self.q_port.Eval(context)).reshape((7,))

        # Enforce yz-plane motion (optional)
        V_WG_desired[0] = 0.0

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


def design_grasp_pose(X_WO: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
    R_OG = RollPitchYaw(0, -np.pi / 2, np.pi).ToRotationMatrix()
    p_OG = [0.0, -0.08, 0.08]
    X_OG = RigidTransform(R_OG, p_OG)
    X_WG = X_WO.multiply(X_OG)
    return X_OG, X_WG


def design_pregrasp_pose(X_WG: RigidTransform):
    X_GGApproach = RigidTransform([0.0, -0.35, 0.05])
    return X_WG @ X_GGApproach


def design_pregoal_pose(X_WG: RigidTransform):
    X_GGApproach = RigidTransform([0.0, 0.0, -0.12])
    return X_WG @ X_GGApproach


def design_goal_pose(X_WO_goal: RigidTransform, X_OG: RigidTransform):
    return X_WO_goal @ X_OG


def design_postgoal_pose(X_WG: RigidTransform):
    X_GGApproach = RigidTransform([0.0, 0.0, -0.12])
    return X_WG @ X_GGApproach


def make_trajectory(X_Gs, finger_values, sample_times):
    pose_traj = PiecewisePose.MakeLinear(sample_times, X_Gs)
    vel_traj = pose_traj.MakeDerivative()
    wsg_traj = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return vel_traj, wsg_traj


# ============================================================================ #
# Phase 1: Perception diagram (random placement + simulation + ICP)
# ============================================================================ #

def build_sensing_diagram(meshcat):
    scenario = LoadScenario(filename=SCENARIO_PATH)

    builder = DiagramBuilder()
    station = MakeHardwareStation(scenario, meshcat=meshcat)
    builder.AddSystem(station)

    plant = station.GetSubsystemByName("plant")

    # Constant iiwa + wsg
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    n_q_iiwa = plant.num_positions(iiwa_model)
    plant_tmp_context = plant.CreateDefaultContext()
    q0_iiwa = plant.GetPositions(plant_tmp_context, iiwa_model)
    if q0_iiwa.shape[0] != n_q_iiwa:
        q0_iiwa = np.zeros(n_q_iiwa)

    iiwa_source = builder.AddSystem(ConstantVectorSource(q0_iiwa))
    builder.Connect(iiwa_source.get_output_port(), station.GetInputPort("iiwa.position"))

    wsg_source = builder.AddSystem(ConstantVectorSource([0.05]))
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

    # Cameras → point clouds
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

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Randomize cup poses (collision-free x-y, random orientation), then we'll simulate
    plant_context = plant.GetMyMutableContextFromRoot(context)
    cup_models = get_cup_model_instances(plant)
    existing_xy: list[tuple[float, float]] = []
    for m in cup_models:
        body = plant.GetBodyByName("base_link", m)
        place_cup_randomly_on_table(plant, plant_context, body, existing_xy)

    diagram.ForcedPublish(context)
    return diagram, context, station, plant


def run_icp_per_cup(diagram, context, station, plant, meshcat):
    """Your per-cup ICP with cropping based on true poses (for robustness)."""
    world_frame = plant.world_frame()
    plant_context = diagram.GetSubsystemContext(plant, context)

    # Cup models + names
    cup_models = get_cup_model_instances(plant)
    cup_bodies = [plant.GetBodyByName("base_link", m) for m in cup_models]
    cup_names = [plant.GetModelInstanceName(m) for m in cup_models]

    # Cup mesh samples
    N_SAMPLE_POINTS = 150
    mesh = trimesh.load(CUP_MESH_PATH, force="mesh")
    model_points_O = mesh.sample(count=N_SAMPLE_POINTS).T  # 3 x N

    cropped_point_clouds = []

    for body, name in zip(cup_bodies, cup_names):
        X_WC = plant.CalcRelativeTransform(
            plant_context, world_frame, body.body_frame()
        )

        AddMeshcatTriad(
            meshcat,
            f"truth/{name}",
            X_PT=X_WC,
            length=0.07,
            radius=0.003,
        )

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
            lower.reshape(3, 1),
            upper.reshape(3, 1),
            1.0,
            Rgba(0, 1, 0),
        )

    colors = [Rgba(1, 0, 0), Rgba(0, 1, 0), Rgba(0, 0, 1)]
    for cup_pc, color, name in zip(cropped_point_clouds, colors, cup_names):
        meshcat.SetObject(f"cup_{name}_pc", cup_pc, point_size=0.02, rgba=color)

    # ICP
    MAX_ITER = 25
    icp_results: dict[str, tuple[RigidTransform, float]] = {}

    for cup_pc, body, name in zip(cropped_point_clouds, cup_bodies, cup_names):
        scene_points_W = np.asarray(cup_pc.xyzs())
        if scene_points_W.shape[1] < 5:
            print(f"[ICP] Not enough points for cup '{name}', skipping.")
            continue

        centroid_W = scene_points_W.mean(axis=1)
        X_WO_initial = RigidTransform(RotationMatrix(), centroid_W)

        AddMeshcatTriad(
            meshcat,
            f"icp/{name}_init",
            X_PT=X_WO_initial,
            length=0.05,
            radius=0.002,
        )

        X_WO_hat, c_hat = IterativeClosestPoint(
            p_Om=model_points_O,
            p_Ws=scene_points_W,
            X_Ohat=X_WO_initial,
            meshcat=meshcat,
            meshcat_scene_path=f"icp/{name}",
            max_iterations=MAX_ITER,
        )
        final_cost = float(c_hat[-1] if hasattr(c_hat, "__len__") else c_hat)

        # Mesh frame O → cup base_link frame C (calibrated using truth ONCE)
        X_WC_true = plant.CalcRelativeTransform(
            plant_context, world_frame, body.body_frame()
        )
        X_CO_est = X_WC_true.inverse().multiply(X_WO_hat)
        X_OC_est = X_CO_est.inverse()
        X_WC_hat = X_WO_hat.multiply(X_OC_est)

        icp_results[name] = (X_WC_hat, final_cost)
        print(f"[ICP] Cup '{name}' cost={final_cost:.4f}, p={X_WC_hat.translation()}")

        AddMeshcatTriad(
            meshcat,
            f"icp/{name}_final",
            X_PT=X_WC_hat,
            length=0.07,
            radius=0.003,
        )

    return cup_names, icp_results


# ============================================================================ #
# Planning: pyramidal stack
# ============================================================================ #

def compute_pyramid_layers(N: int):
    k = 0
    total = 0
    while total + (k + 1) <= N:
        k += 1
        total += k
    layers = list(range(k, 0, -1))
    leftover = N - total
    if leftover > 0:
        layers[0] += leftover
        layers = sorted(layers, reverse=True)
    return layers


def make_pyramidal_goals(N: int,
                         base_xy=np.array([0.0, 0.25]),
                         table_z=0.15):
    """Return list of X_WO_goal for N cups in a pyramid."""
    layers = compute_pyramid_layers(N)
    print("[Planning] Pyramid layers (base first):", layers)

    d = 0.08  # spacing in x
    h = 0.10  # height per layer
    goals = []

    idx = 0
    y0 = base_xy[1]
    for layer_idx, count in enumerate(layers):
        z = table_z + layer_idx * h
        total_width = (count - 1) * d
        x_start = base_xy[0] - total_width / 2.0
        for i in range(count):
            if idx >= N:
                break
            x = x_start + i * d
            y = y0 + layer_idx * 0.02
            goals.append(RigidTransform(RotationMatrix(), np.array([x, y, z])))
            idx += 1

    while idx < N:  # safety
        goals.append(RigidTransform(RotationMatrix(), np.array([base_xy[0], y0, table_z])))
        idx += 1

    return goals


# ============================================================================ #
# Phase 2: Control diagram using ICP results
# ============================================================================ #

def build_control_diagram(meshcat, icp_results):
    scenario = LoadScenario(filename=SCENARIO_PATH)

    builder = DiagramBuilder()
    station = MakeHardwareStation(scenario, meshcat=meshcat)
    builder.AddSystem(station)

    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    iiwa_model = plant.GetModelInstanceByName("iiwa")
    gripper_body = plant.GetBodyByName("body")

    # Map ICP results onto this plant's cup bodies by name
    cup_models = get_cup_model_instances(plant)
    cup_bodies = []
    X_WC_hats = []
    for m in cup_models:
        name = plant.GetModelInstanceName(m)
        if name in icp_results:
            body = plant.GetBodyByName("base_link", m)
            cup_bodies.append(body)
            X_WC_hats.append(icp_results[name][0])

    N = len(cup_bodies)
    if N == 0:
        raise RuntimeError("No cups in control plant matched ICP results.")

    # Home pose
    plant_tmp_context = plant.CreateDefaultContext()
    X_WG_home = plant.EvalBodyPoseInWorld(plant_tmp_context, gripper_body)

    # Pyramidal goals (treat mesh frame ~= cup base frame)
    X_WO_goals = make_pyramidal_goals(N)

    opened = 0.05
    closed = 0.0
    key_poses = []
    key_fingers = []

    for i, (body, X_WC_hat, X_WO_goal) in enumerate(zip(cup_bodies, X_WC_hats, X_WO_goals)):
        X_WOinitial = X_WC_hat

        X_OG, X_WGpick = design_grasp_pose(X_WOinitial)
        X_WGprepick = design_pregrasp_pose(X_WGpick)
        X_WGgoal = design_goal_pose(X_WO_goal, X_OG)
        X_WGpregoal = design_pregoal_pose(X_WGgoal)
        X_WGpostgoal = design_postgoal_pose(X_WGgoal)

        if i == 0:
            key_poses.append(X_WG_home)
            key_fingers.append(opened)

        key_poses += [
            X_WGprepick,
            X_WGpick,
            X_WGpick,
            X_WGpregoal,
            X_WGgoal,
            X_WGgoal,
            X_WGpostgoal,
            X_WG_home,
        ]
        key_fingers += [
            opened,
            opened,
            closed,
            closed,
            closed,
            opened,
            opened,
            opened,
        ]

    num_keys = len(key_poses)
    sample_times = [3.0 * i for i in range(num_keys)]
    finger_mat = np.asarray([key_fingers]).reshape(1, -1)
    traj_V_G, traj_wsg = make_trajectory(key_poses, finger_mat, sample_times)

    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    controller = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(7))
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg))

    builder.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        controller.GetInputPort("iiwa.position"),
    )
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

    AddFrameTriadIllustration(
        scene_graph=scene_graph,
        body=gripper_body,
        length=0.1,
        radius=0.003,
    )
    for body in cup_bodies:
        AddFrameTriadIllustration(
            scene_graph=scene_graph,
            body=body,
            length=0.07,
            radius=0.002,
        )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Initialize cups at ICP-estimated poses
    for body, X_WC_hat in zip(cup_bodies, X_WC_hats):
        plant.SetFreeBodyPose(plant_context, body, X_WC_hat)

    # Initialize integrator to iiwa default positions
    q0 = plant.GetPositions(plant_context, iiwa_model)
    int_context = diagram.GetMutableSubsystemContext(integrator, context)
    int_context.get_mutable_continuous_state_vector().SetFromVector(q0)

    tfinal = sample_times[-1]
    return diagram, context, tfinal


# ============================================================================ #
# MAIN
# ============================================================================ #

def main():
    np.set_printoptions(precision=3, suppress=True)

    meshcat = StartMeshcat()
    print("Meshcat listening at:", meshcat.web_url())

    # Optional: patch_file_paths()

    # ---------- Phase 1: perception (randomize + simulate + ICP) ----------
    diagram_sense, context_sense, station_sense, plant_sense = build_sensing_diagram(
        meshcat
    )

    sim_sense = Simulator(diagram_sense, context_sense)
    sim_sense.Initialize()
    # Let cups fall and settle
    sim_sense.AdvanceTo(4.0)
    diagram_sense.ForcedPublish(context_sense)

    cup_names, icp_results = run_icp_per_cup(
        diagram_sense, context_sense, station_sense, plant_sense, meshcat
    )
    if not icp_results:
        print("[Main] ICP failed for all cups; aborting.")
        return

    # ---------- Phase 2: control (pick + place into pyramid) ----------
    diagram_ctrl, context_ctrl, tfinal = build_control_diagram(meshcat, icp_results)

    sim_ctrl = Simulator(diagram_ctrl, context_ctrl)
    meshcat.StartRecording()
    sim_ctrl.Initialize()
    sim_ctrl.AdvanceTo(tfinal + 2.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("[Main] Done: cups dropped, sensed with ICP, then stacked in a pyramid.")


if __name__ == "__main__":
    main()
