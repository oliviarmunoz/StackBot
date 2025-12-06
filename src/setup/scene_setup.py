from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
)

from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation,
)


import numpy as np


def create_stackbot_scene(meshcat):
    scenario_path = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
    scenario = LoadScenario(filename=scenario_path)

    station = MakeHardwareStation(scenario, meshcat=meshcat)
    
    builder = DiagramBuilder()
    builder.AddSystem(station)

    # plant = station.GetSubsystemByName("plant")

    # ---- Constant commands so the iiwa/wsg just "sit there" ----
    # iiwa_model = plant.GetModelInstanceByName("iiwa")
    # n_q_iiwa = plant.num_positions(iiwa_model)

    # Use the default configuration as the constant command.
    # plant_tmp_context = plant.CreateDefaultContext()
    # q0_iiwa = plant.GetPositions(plant_tmp_context, iiwa_model)
    # if q0_iiwa.shape[0] != n_q_iiwa:
    #     q0_iiwa = np.zeros(n_q_iiwa)

    # iiwa_source = builder.AddSystem(ConstantVectorSource(q0_iiwa))
    # # builder.Connect(
    # #     iiwa_source.get_output_port(),
    # #     station.GetInputPort("iiwa.position"),
    # # )

    # # Gripper: slightly open
    # wsg_source = builder.AddSystem(ConstantVectorSource([0.05]))
    # # builder.Connect(
    # #     wsg_source.get_output_port(),
    # #     station.GetInputPort("wsg.position"),
    # # )

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


