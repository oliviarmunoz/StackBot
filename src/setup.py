import os

import numpy as np
from pathlib import Path

from pydrake.all import (
    Diagram,
    DiagramBuilder,
    StartMeshcat,
    MeshcatVisualizer,
    Simulator,
    Cylinder,
    Box,
    Integrator,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.geometry import SceneGraph
from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation,
    RobotDiagram,
)
from manipulation.utils import RenderDiagram

# Start MeshCat server
meshcat = StartMeshcat()

assets_dir = Path.cwd() / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)

# Cup Model

cup_mesh_path = f"{Path.cwd()}/assets/cup.obj"
print(cup_mesh_path)

cup_sdf_xml = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="cup_template">
    <link name="base_link">
      <inertial>
        <mass>0.03</mass>
        <inertia>
          <ixx>0.00032</ixx>
          <iyy>0.00032</iyy>
          <izz>0.000024</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.04</radius>
            <length>0.095</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="cup_mesh">
        <geometry>
          <mesh>
            <uri>file://{cup_mesh_path}</uri>
            <scale>0.03 0.03 0.03</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>1.0 0.4 0.7 1.0</ambient>
          <diffuse>1.0 0.4 0.7 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

model_cup_sdf_path = assets_dir / "model_cup.sdf"
with open(model_cup_sdf_path, "w") as f:
    f.write(cup_sdf_xml)


# Add the directives for the bimanual IIWA arms, table, and initials


def generate_bimanual_IIWA14_with_assets_directives_file() -> (
    tuple[Diagram, RobotDiagram]
):
    table_sdf = f"{Path.cwd()}/assets/table.sdf"
    cup_sdf = f"{Path.cwd()}/assets/model_cup.sdf"

    directives_yaml = f"""directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
    X_PC:
        translation: [0, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
- add_model:
    name: table
    file: file://{table_sdf}
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
- add_model:
    name: cup_lower_left
    file: file://{cup_sdf}
- add_model:
    name: cup_lower_right
    file: file://{cup_sdf}
- add_model:
    name: cup_top
    file: file://{cup_sdf}
- add_weld:
    parent: table::table_link
    child: cup_lower_left::base_link
    X_PC:
        translation: [-0.065, 0.0, 0.15]
        rotation: !Rpy {{ deg: [180, 0, -90] }}
- add_weld:
    parent: table::table_link
    child: cup_lower_right::base_link
    X_PC:
        translation: [0.065, 0.0, 0.15]
        rotation: !Rpy {{ deg: [180, 0, -90] }}
- add_weld:
    parent: table::table_link
    child: cup_top::base_link
    X_PC:
        translation: [0, 0, 0.3125]
        rotation: !Rpy {{ deg: [180, 0, -90] }}


"""
    os.makedirs("directives", exist_ok=True)

    with open(
        "directives/bimanual_IIWA14_with_table.dmd.yaml", "w"
    ) as f:
        f.write(directives_yaml)


generate_bimanual_IIWA14_with_assets_directives_file()

def create_camera_directives() -> None:
    camera_directives_yaml = """
directives:
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: [-120.0, 0.0, 180.0]}
        translation: [0, 0.8, 0.5]

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera0_origin
    child: camera0::base

- add_frame:
    name: camera1_origin
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: [-125, 0.0, 90.0]}
        translation: [0.8, 0.1, 0.5]

- add_model:
    name: camera1
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera1_origin
    child: camera1::base

- add_frame:
    name: camera2_origin
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: [-120.0, 0.0, -90.0]}
        translation: [-0.8, 0.1, 0.5]

- add_model:
    name: camera2
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera2_origin
    child: camera2::base
"""
    with open("directives/camera_directives.dmd.yaml", "w") as f:
        f.write(camera_directives_yaml)


create_camera_directives()

def create_bimanual_IIWA14_with_assets_and_cameras_scenario() -> None:
    # TODO: create a scenario yaml with the directives added with `add_directives`

    directive_file = f"{Path.cwd()}/directives/bimanual_IIWA14_with_table.dmd.yaml"
    camera_directive_file = f"{Path.cwd()}/directives/camera_directives.dmd.yaml"

    scenario_yaml = f"""
    directives:
        - add_directives:
            file: file://{directive_file}
        - add_directives:
            file: file://{camera_directive_file}
            

    cameras:
        camera0:
            name: camera0
            depth: True
            X_PB:
                base_frame: camera0::base

        camera1:
            name: camera1
            depth: True
            X_PB:
                base_frame: camera1::base

        camera2:
            name: camera2
            depth: True
            X_PB:
                base_frame: camera2::base

    model_drivers:
        iiwa: !IiwaDriver
            control_mode: position_only
            hand_model_name: wsg
        wsg: !SchunkWsgDriver {{}}
    """
    # TODO: add the camera configs and iiwa drivers with `add_cameras` and `add_iiwa_drivers`

    os.makedirs("scenarios", exist_ok=True)

    with open(
        "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml",
        "w",
    ) as f:
        f.write(scenario_yaml)


create_bimanual_IIWA14_with_assets_and_cameras_scenario()

def create_bimanual_IIWA14_with_table_and_cameras() -> (
    tuple[DiagramBuilder, RobotDiagram]
):
    # TODO: Load the scenario created above into a Scenario object
    scenario_path = (
        "scenarios/bimanual_IIWA14_with_table_and_cameras.scenario.yaml"
    )
    scenario = LoadScenario(filename=scenario_path)
    # TODO: Create HardwareStation with the scenario and meshcat
    station = MakeHardwareStation(scenario, meshcat)

    # TODO: Make a DiagramBuilder, add the station, and build the diagram
    builder = DiagramBuilder()
    station = builder.AddSystem(station)

    # TODO: Add the point clouds to the diagram with AddPointClouds
    to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder, meshcat=meshcat)

    # TODO: export the point cloud outputs to the builder
    builder.ExportOutput(
        to_point_cloud["camera0"].get_output_port(), "camera0_point_cloud"
    )
    builder.ExportOutput(
        to_point_cloud["camera1"].get_output_port(), "camera1_point_cloud"
    )
    builder.ExportOutput(
        to_point_cloud["camera2"].get_output_port(), "camera2_point_cloud"
    )
    
    # TODO: Return the builder AND the station (notice that here we will need both)
    return builder, station


builder, station = (
    create_bimanual_IIWA14_with_table_and_cameras()
)

# in order to debug, we will build the diagram once here.
diagram = builder.Build()

# visualize the diagram
RenderDiagram(diagram, max_depth=1)

# publish the diagram with some default context
context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)

# simulator = Simulator(diagram)
# context = simulator.get_mutable_context()
# station_context = station.GetMyContextFromRoot(context)

# diagram.ForcedPublish(context)
