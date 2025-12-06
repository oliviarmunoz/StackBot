import numpy as np

from pydrake.all import (
    RigidTransform,
    RollPitchYaw,
)

def place_cup_randomly_on_table(plant, plant_context, body):
    # Table top is at z = 0.0, so we give a small lift
    z = 0.15
    
    # Robot workspace bounding box (tune as needed)
    extrema = 0.25
    X_MIN, X_MAX = -extrema, extrema
    Y_MIN, Y_MAX = -extrema+0.1, extrema+0.1

    x = np.random.uniform(X_MIN, X_MAX)
    y = np.random.uniform(Y_MIN, Y_MAX)

    # Keep cup lying-down orientation same as before
    rpy = RollPitchYaw(np.deg2rad([270, 0, -90]))

    X_WC = RigidTransform(rpy, [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, X_WC)
    
    return X_WC
    
    