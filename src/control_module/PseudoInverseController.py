import numpy as np

from pydrake.all import (
    BasicVector,
    Context,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
)

class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()  # end-effector frame
        self._W = plant.world_frame()

        # Input/output ports
        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)

        # Velocity indices for IIWA
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
            np.zeros((3,1)),
            self._W,
            self._W
        )

        J_iiwa = J_full[:, self.iiwa_start:self.iiwa_end]

        v = np.linalg.pinv(J_iiwa) @ V_WG_desired

        output.SetFromVector(v)
