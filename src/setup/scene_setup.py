from pydrake.all import DiagramBuilder

from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation,
)


def create_stackbot_scene(meshcat):
    scenario_path = "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
    scenario = LoadScenario(filename=scenario_path)

    station = MakeHardwareStation(scenario, meshcat=meshcat)
    
    builder = DiagramBuilder()
    builder.AddSystem(station)

    # add point cloud systems
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


