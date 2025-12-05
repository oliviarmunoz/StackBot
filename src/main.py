#!/usr/bin/env python3
"""
Pyramid-stacking robot with KUKA iiwa + WSG gripper + 3 depth cameras.

This script:
  1. Loads the StackBot scenario (iiwa + wsg + table + 3 cups + 3 cameras).
  2. Randomly places the three cups on the table in the workspace.
  3. Computes a set of pyramid target poses for the cups.
  4. Builds a gripper-frame trajectory:
       home -> (pregrasp, grasp, lift, preplace, place, release, retreat) for each cup -> home
  5. Uses a PseudoInverseController + Integrator to convert spatial gripper velocity
     into iiwa joint velocities and joint positions.
  6. Crucial fix: the integrator state is initialized to the *measured* iiwa position,
     so the arm doesn't fold or wander.

You can later replace the ground-truth cup poses with ICP-estimated poses.
"""

from pathlib import Path
import numpy as np

from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Concatenate,
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

from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation,
)

# -----------------------------------------------------------------------------
# 1. (Optional) Patch Deepnote-style file paths to current working directory
# -----------------------------------------------------------------------------

def patch_file_paths():
    """
    Original project files may reference file:///datasets/_deepnote_work.
    Replace that prefix with the current working directory so Drake can find:
      - table.sdf
      - cups
      - directives
      - scenario
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
            print(f"[patch_file_paths] No old prefix in {path}, leaving as-is")

# Uncomment if you actually need this
# patch_file_paths()

# -----------------------------------------------------------------------------
# 2. Start Meshcat
# -----------------------------------------------------------------------------

meshcat = StartMeshcat()
print("Meshcat URL:", meshcat.web_url())

# -----------------------------------------------------------------------------
# 3. Build station + cameras (same scenario you're already using)
# -----------------------------------------------------------------------------

def create_stackbot_scene():
    """
    Load the bimanual_IIWA14_with_table_and_cameras scenario and:
      - Build a station with iiwa + wsg + table + 3 cups + 3 cameras.
      - Export point cloud outputs (so cameras "exist" even if we don't use them for control).
    """
    scenario_path = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
    scenario = LoadScenario(filename=scenario_path)

    builder = DiagramBuilder()
    station = MakeHardwareStation(scenario, meshcat=meshcat)
    builder.AddSystem(station)

    plant = station.GetSubsystemByName("plant")

    # Add point clouds (optional for control, but matches your original setup).
    to_pc = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=meshcat,
    )
    builder.ExportOutput(to_pc["camera0"].get_output_port(), "camera0_point_cloud")
    builder.ExportOutput(to_pc["camera1"].get_output_port(), "camera1_point_cloud")
    builder.ExportOutput(to_pc["camera2"].get_output_port(), "camera2_point_cloud")

    return builder, station

# -----------------------------------------------------------------------------
# 4. Cup helpers: random placement + pyramid goals
# -----------------------------------------------------------------------------

def get_cup_bodies(plant: MultibodyPlant):
    """Return (name, body) for the three cups described in your directives."""
    model_left = plant.GetModelInstanceByName("cup_lower_left")
    model_right = plant.GetModelInstanceByName("cup_lower_right")
    model_top = plant.GetModelInstanceByName("cup_top")

    cup_left_body = plant.GetBodyByName("base_link", model_left)
    cup_right_body = plant.GetBodyByName("base_link", model_right)
    cup_top_body = plant.GetBodyByName("base_link", model_top)

    return [
        ("left", cup_left_body),
        ("right", cup_right_body),
        ("top", cup_top_body),
    ]


def place_cup_randomly_on_table(plant, plant_context, body):
    """
    Randomly place a cup above the table in the robot workspace.
    Keeps your original "lying on the side" orientation.
    """
    z = 0.15  # slightly above the table

    extrema = 0.25
    x_min, x_max = -extrema, extrema
    y_min, y_max = -extrema + 0.1, extrema + 0.1

    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)

    rpy = RollPitchYaw(np.deg2rad([270, 0, -90]))
    X_WC = RigidTransform(rpy, [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, X_WC)

    return X_WC


def compute_pyramid_world_poses(num_cups, base_center, horizontal_spacing=0.08, vertical_spacing=0.07):
    """
    Compute world-from-cup base_link poses forming a pyramid.

    base_center: [x, y, z] of the first layer's center.
    """
    poses = []
    remaining = num_cups
    layer = 0

    while remaining > 0:
        count = min(remaining, layer + 1)
        z = base_center[2] + layer * vertical_spacing
        y = base_center[1] + layer * 0.03  # slight forward shift per layer

        x0 = base_center[0] - 0.5 * (count - 1) * horizontal_spacing
        for i in range(count):
            x = x0 + i * horizontal_spacing
            poses.append(RigidTransform(RotationMatrix(), np.array([x, y, z])))
        remaining -= count
        layer += 1

    return poses

# -----------------------------------------------------------------------------
# 5. Gripper-frame pick-and-place design for cups
# -----------------------------------------------------------------------------

def design_cup_grasp_pose(X_WC: RigidTransform) -> RigidTransform:
    """
    Side grasp pose for a cup at world pose X_WC (world-from-cup base_link).
    Similar spirit to your design_grasp_pose but tuned for cups.
    """
    R_CG = RollPitchYaw(0, -np.pi / 2, np.pi).ToRotationMatrix()
    p_CG = np.array([0.0, 0.0, 0.0])  # pinch near the center of the cup
    X_CG = RigidTransform(R_CG, p_CG)
    return X_WC @ X_CG


def design_pregrasp_pose(X_WG: RigidTransform) -> RigidTransform:
    """
    Back off along -Y_G and slightly above, to approach the cup.
    """
    X_GG_approach = RigidTransform([0.0, -0.25, 0.08])
    return X_WG @ X_GG_approach


def design_pregoal_pose(X_WG_goal: RigidTransform) -> RigidTransform:
    """
    Hover above the goal pose before descending to place.
    """
    X_GG_approach = RigidTransform([0.0, 0.0, -0.12])
    return X_WG_goal @ X_GG_approach


def design_postgoal_pose(X_WG_goal: RigidTransform) -> RigidTransform:
    """
    Retract after placing the cup.
    """
    X_GG_approach = RigidTransform([0.0, 0.0, -0.12])
    return X_WG_goal @ X_GG_approach


def make_trajectory(X_Gs, finger_values, sample_times):
    """
    Build:
      - gripper spatial-velocity trajectory traj_V_G
      - WSG command trajectory traj_wsg_command

    This is your existing pattern: PiecewisePose -> derivative -> piecewise polynomial.
    """
    robot_pose_traj = PiecewisePose.MakeLinear(sample_times, X_Gs)
    robot_vel_traj = robot_pose_traj.MakeDerivative()
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return robot_vel_traj, traj_wsg_command


class PseudoInverseController(LeafSystem):
    """
    Velocity-level controller:
      input: desired V_WG (spatial velocity of gripper in world frame)
             current iiwa joint positions q
      output: joint velocities v
    """
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()  # end-effector frame
        self._W = plant.world_frame()

        # Ports
        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)

        # Velocity indices for iiwa
        self.iiwa_start = self._plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = self.iiwa_start + self._plant.num_velocities(self._iiwa)

    def CalcOutput(self, context: Context, output: BasicVector):
        V_WG_desired = np.asarray(self.V_G_port.Eval(context)).reshape((6,))
        q = np.asarray(self.q_port.Eval(context)).reshape((7,))

        # Optional: keep x-translation = 0 to limit weird motions
        V_WG_desired[0] = 0.0

        # Update plant context with current joint positions
        self._plant.SetPositions(self._plant_context, self._iiwa, q)

        # Full Jacobian
        J_full = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            np.zeros((3, 1)),
            self._W,
            self._W,
        )

        # Restrict to iiwa columns
        J_iiwa = J_full[:, self.iiwa_start:self.iiwa_end]

        # Pseudoinverse mapping
        v = np.linalg.pinv(J_iiwa) @ V_WG_desired
        output.SetFromVector(v)

# -----------------------------------------------------------------------------
# 7. Main: random cups -> pyramid trajectory -> simulate
# -----------------------------------------------------------------------------

def main():
    # Build station and plant
    builder, station = create_stackbot_scene()
    plant: MultibodyPlant = station.plant()

    # Temporary context to get initial poses
    plant_tmp_context = plant.CreateDefaultContext()

    iiwa_model = plant.GetModelInstanceByName("iiwa")
    gripper_body = plant.GetBodyByName("body")

    # Home configuration (default)
    q0_iiwa = plant.GetPositions(plant_tmp_context, iiwa_model)
    X_WG_initial = plant.EvalBodyPoseInWorld(plant_tmp_context, gripper_body)

    # Randomize cups & record their world poses
    cup_bodies = get_cup_bodies(plant)
    cup_initial_world_poses = []
    for name, body in cup_bodies:
        X_WC = place_cup_randomly_on_table(plant, plant_tmp_context, body)
        cup_initial_world_poses.append(X_WC)
        print(f"[init] cup '{name}' at:\n{X_WC}")

    num_cups = len(cup_bodies)

    # Compute pyramid goal poses (world-from-cup base_link)
    pyramid_base_center = np.array([0.0, 0.05, 0.20])  # tweak as needed
    cup_goal_world_poses = compute_pyramid_world_poses(
        num_cups=num_cups,
        base_center=pyramid_base_center,
        horizontal_spacing=0.08,
        vertical_spacing=0.07,
    )

    # ------------------------------------------------------------------
    # Build gripper keyframes for pick-and-place of all cups.
    # ------------------------------------------------------------------
    opened = 0.05
    closed = 0.0

    keyframes = []
    finger_states = []

    keyframes.append(X_WG_initial)
    finger_states.append(opened)

    for (name, body), X_WC_init, X_WC_goal in zip(
        cup_bodies, cup_initial_world_poses, cup_goal_world_poses
    ):
        X_WG_pick = design_cup_grasp_pose(X_WC_init)
        X_WG_prepick = design_pregrasp_pose(X_WG_pick)

        # For placement, align the cup upright at its goal pose.
        X_WG_goal = design_cup_grasp_pose(X_WC_goal)
        X_WG_pregoal = design_pregoal_pose(X_WG_goal)
        X_WG_postgoal = design_postgoal_pose(X_WG_goal)

        sequence = [
            (X_WG_prepick, opened),
            (X_WG_pick, opened),
            (X_WG_pick, closed),
            (X_WG_prepick, closed),
            (X_WG_pregoal, closed),
            (X_WG_goal, closed),
            (X_WG_goal, opened),
            (X_WG_postgoal, opened),
        ]

        for pose, fingers in sequence:
            keyframes.append(pose)
            finger_states.append(fingers)

    # Return home
    keyframes.append(X_WG_initial)
    finger_states.append(opened)

    dt = 2.0
    sample_times = [dt * i for i in range(len(keyframes))]

    finger_states_np = np.asarray(finger_states).reshape(1, -1)
    traj_V_G, traj_wsg_command = make_trajectory(
        X_Gs=keyframes,
        finger_values=finger_states_np,
        sample_times=sample_times,
    )

    # ------------------------------------------------------------------
    # Add control systems & connections
    # ------------------------------------------------------------------
    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    controller = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(7))
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))

    # V_WG -> controller
    builder.Connect(V_G_source.get_output_port(),
                    controller.GetInputPort("V_WG"))

    # controller qdot -> integrator -> iiwa.position command
    builder.Connect(controller.get_output_port(),
                    integrator.get_input_port())
    builder.Connect(integrator.get_output_port(),
                    station.GetInputPort("iiwa.position"))

    # feedback iiwa.position_measured -> controller
    builder.Connect(station.GetOutputPort("iiwa.position_measured"),
                    controller.GetInputPort("iiwa.position"))

    # wsg command
    builder.Connect(wsg_source.get_output_port(),
                    station.GetInputPort("wsg.position"))

    # Debug frame triads on gripper and cups
    scenegraph = station.GetSubsystemByName("scene_graph")
    AddFrameTriadIllustration(
        scene_graph=scenegraph,
        body=plant.GetBodyByName("body"),
        length=0.1,
    )
    for (name, body) in cup_bodies:
        AddFrameTriadIllustration(
            scene_graph=scenegraph,
            body=body,
            length=0.05,
        )

    # ------------------------------------------------------------------
    # Build diagram + context; **CRITICAL**: sync integrator state
    # ------------------------------------------------------------------
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Get the plant + station contexts from root
    plant_context = plant.GetMyMutableContextFromRoot(context)
    station_context = station.GetMyMutableContextFromRoot(context)

    # Set iiwa initial joint positions in the plant (home pose)
    plant.SetPositions(plant_context, iiwa_model, q0_iiwa)

    # Apply the randomized cup poses to the new plant_context
    for (name, body), X_WC in zip(cup_bodies, cup_initial_world_poses):
        plant.SetFreeBodyPose(plant_context, body, X_WC)

    # **KEY FIX**: initialize integrator state to the *measured* iiwa position,
    # so that commanded and actual positions are aligned at t=0.
    q_measured = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
    integrator_context = integrator.GetMyMutableContextFromRoot(context)
    integrator_context.SetContinuousState(q_measured)

    # Publish initial state to Meshcat
    diagram.ForcedPublish(context)

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    simulator = Simulator(diagram, context)
    simulator.Initialize()

    meshcat.StartRecording()
    T_final = sample_times[-1] + 2.0
    simulator.AdvanceTo(T_final)
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("Pyramid stacking sequence complete.")


if __name__ == "__main__":
    main()
