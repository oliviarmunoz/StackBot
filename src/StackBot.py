import numpy as np

from pydrake.all import (
    Concatenate,
    DiagramBuilder,
    Integrator,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)

from manipulation.icp import IterativeClosestPoint
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)

from setup.scene_setup import create_stackbot_scene
from setup.cup_placement import place_cup_randomly_on_table
from perception_module.point_clouds import make_point_clouds
from perception_module.icp import crop_point_clouds_run_icp
from control_module.PseudoInverseController import PseudoInverseController
from manipulation_module.poses import *

MAX_ITERATIONS = 25
N_SAMPLE_POINTS = 150

def stackbot():
    # -----------------------------------------------------------------------------
    # SCENE SETUP
    # -----------------------------------------------------------------------------
    meshcat = StartMeshcat()
    print(" ------------- Meshcat listening at:", meshcat.web_url(), "-------------")

    builder, station = create_stackbot_scene(meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    plant = station.plant()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    cup_left_model  = plant.GetModelInstanceByName("cup_lower_left")
    cup_right_model = plant.GetModelInstanceByName("cup_lower_right")
    cup_top_model   = plant.GetModelInstanceByName("cup_top")
    cup_left_body  = plant.GetBodyByName("base_link", cup_left_model)
    cup_right_body = plant.GetBodyByName("base_link", cup_right_model)
    cup_top_body   = plant.GetBodyByName("base_link", cup_top_model)

    cup_bodies = [cup_left_body, cup_right_body, cup_top_body]
    cup_names = ["left", "right", "top"]
    cup_models = [cup_left_model, cup_right_model, cup_top_model]

    # place cups randomly
    X_WC_left = place_cup_randomly_on_table(plant, plant_context, cup_left_body)
    X_WC_right = place_cup_randomly_on_table(plant, plant_context, cup_right_body)
    X_WC_top = place_cup_randomly_on_table(plant, plant_context, cup_top_body)
    all_X_WC = [X_WC_left, X_WC_right, X_WC_top]

    diagram.ForcedPublish(context)
    print("------------- Scene setup complete. -------------")

    # -----------------------------------------------------------------------------
    # PERCEPTION MODULE 
    # -----------------------------------------------------------------------------
    model_points_O, world_frame, scene_pcls = make_point_clouds(station, context, diagram, meshcat, cup_bodies, cup_names)
    cup_icp_results = crop_point_clouds_run_icp(cup_names, cup_bodies, plant_context, world_frame, plant, diagram, context, model_points_O, meshcat, scene_pcls)
    
    print("------------- Perception module complete. -------------")
    
    # -----------------------------------------------------------------------------
    # MANIPULATION MODULE 
    # -----------------------------------------------------------------------------
   
    # builder2 = DiagramBuilder()
    # scenario_path = (
    #     "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
    # )
    # scenario = LoadScenario(filename=scenario_path)

    # # Create new station
    # station = builder2.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
    # plant = station.GetSubsystemByName("plant")
    
    # === Step 4: Run ICP for all three cups ===
    
    # === Step 5: Build trajectory for all three cups ===
    builder, station = create_stackbot_scene(meshcat)    
    plant = station.GetSubsystemByName("plant")
    station_context = station.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(station_context)

    X_WGinitial = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))
    
    # goal positions
    pyramid_positions = get_pyramid_positions()

    # Keep same orientation for all cups (lying down)
    cup_orientation = RollPitchYaw(np.deg2rad([270, 0, -90])).ToRotationMatrix()

    # keyframes
    keyframes = [("initial", X_WGinitial, 0.1)]

    cup_names_ordered = ["left", "right", "top"]
    for i, name in enumerate(cup_names_ordered):
        X_WC_initial = cup_icp_results[name]
        goal_pos = pyramid_positions[i]
        
        # Get current gripper position (end of previous sequence)
        current_gripper = keyframes[-1][1]
        
        # Generate keyframes for this cup
        cup_keyframes = create_pick_place_trajectory(
            X_WC_initial, current_gripper, goal_pos, cup_orientation
        )
        
        keyframes.extend(cup_keyframes)

    # Extract poses and gripper commands
    gripper_poses = [kf[1] for kf in keyframes]
    finger_states = np.array([[kf[2] for kf in keyframes]])
    sample_times = [2.5 * i for i in range(len(keyframes))]
    
  
    # Create trajectories
    traj_V_G, traj_wsg_command = make_trajectory(gripper_poses, finger_states, sample_times)

    # === Step 5: Add controller systems ===
    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    controller = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(7))
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))

    builder.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), controller.GetInputPort("iiwa.position"))
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))
    

    diagram = builder.Build()

    # === Step 7: Create simulator and set initial cup poses ===
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    plant_context = plant.GetMyMutableContextFromRoot(context)
    for name, model_instance in zip(cup_names, cup_models):
        body = plant.GetBodyByName("base_link", model_instance)
        plant.SetFreeBodyPose(plant_context, body, cup_icp_results[name])
    
    station_context = station.GetMyMutableContextFromRoot(context)
    diagram.ForcedPublish(context)

    # cup_models = {
    #     "left": plant.GetModelInstanceByName("cup_lower_left"),
    #     "right": plant.GetModelInstanceByName("cup_lower_right"),
    #     "top": plant.GetModelInstanceByName("cup_top")
    # }


    # === Step 8: Run Simulation ===
    integrator_context = integrator.GetMyMutableContextFromRoot(context)
    current_iiwa_position = plant.GetPositions(plant_context, plant.GetModelInstanceByName("iiwa"))
    integrator.set_integral_value(integrator_context, current_iiwa_position)

    # Force publish to see initial state

    print(f"Simulation will run for {traj_V_G.end_time()} seconds")

    # Run simulation with recording
    meshcat.StartRecording()
    simulator.AdvanceTo(traj_V_G.end_time())
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("Pick and place simulation complete!")


if __name__ == "__main__":
    stackbot()