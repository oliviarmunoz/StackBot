import numpy as np

from pydrake.all import (
    RigidTransform,
    RollPitchYaw,
)

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
                break
    
    