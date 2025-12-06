
from pydrake.all import (
    Integrator,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)

from setup.scene_setup import create_stackbot_scene
from setup.cup_placement import place_cups_randomly_nonoverlapping
from perception_module.point_clouds import make_point_clouds
from perception_module.icp import crop_point_clouds_run_icp
from control_module.PseudoInverseController import PseudoInverseController
from manipulation_module.poses import *
from manipulation_module.keyframes import make_keyframes

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

    # place cups
    place_cups_randomly_nonoverlapping(plant, plant_context, cup_bodies)

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

    builder, station = create_stackbot_scene(meshcat)    
    plant = station.GetSubsystemByName("plant")
    station_context = station.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(station_context)

    X_WGinitial = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))
    gripper_poses, finger_values, sample_times = make_keyframes(X_WGinitial, cup_icp_results)

    # Create trajectories
    traj_V_G, traj_wsg_command = make_trajectory(gripper_poses, finger_values, sample_times)
    
    print("------------- Manipulation module complete. -------------")

    # -----------------------------------------------------------------------------
    # CONTROL MODULE 
    # -----------------------------------------------------------------------------

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
    
    print("------------- Control module complete. -------------")
    
    # -----------------------------------------------------------------------------
    # SIMULATION
    # -----------------------------------------------------------------------------
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    plant_context = plant.GetMyMutableContextFromRoot(context)
    for name, model_instance in zip(cup_names, cup_models):
        body = plant.GetBodyByName("base_link", model_instance)
        plant.SetFreeBodyPose(plant_context, body, cup_icp_results[name])
    
    station_context = station.GetMyMutableContextFromRoot(context)
    diagram.ForcedPublish(context)

    integrator_context = integrator.GetMyMutableContextFromRoot(context)
    current_iiwa_position = plant.GetPositions(plant_context, plant.GetModelInstanceByName("iiwa"))
    integrator.set_integral_value(integrator_context, current_iiwa_position)

    print(f"Simulation will run for {traj_V_G.end_time()} seconds")

    meshcat.StartRecording()
    simulator.AdvanceTo(traj_V_G.end_time())
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("Completed!")


if __name__ == "__main__":
    stackbot()