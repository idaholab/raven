import math
from pythonfmu.fmi2slave import Fmi2Slave, Fmi2Causality, Real
from localmodule import get_amplitude, get_time_constant


class PythonSlaveWithDep(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.realIn = 22.0
        self.realOut = 0.0
        self.register_variable(Real("realIn", causality=Fmi2Causality.input))
        self.register_variable(Real("realOut", causality=Fmi2Causality.output))

    def do_step(self, current_time, step_size):
        self.realOut = self.realIn * get_amplitude() * math.exp((current_time + step_size) / get_time_constant())
        return True
