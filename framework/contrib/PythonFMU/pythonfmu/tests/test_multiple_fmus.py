import math
import pytest

from pythonfmu.builder import FmuBuilder

pytestmark = pytest.mark.skipif(
    not FmuBuilder.has_binary(), reason="No binary available for the current platform."
)
pyfmi = pytest.importorskip(
    "pyfmi", reason="pyfmi is required for testing the produced FMU"
)
# fmpy = pytest.importorskip(
#     "fmpy", reason="fmpy is not available for testing the produced FMU"
# )


@pytest.mark.integration
def test_integration_multiple_fmus_pyfmi(tmp_path):
    slave1_code = """import math
from pythonfmu.fmi2slave import Fmi2Slave, Fmi2Causality, Integer, Real, Boolean, String


class Slave1(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.realIn = 22.0
        self.realOut = 0.0
        self.register_variable(Real("realIn", causality=Fmi2Causality.input))
        self.register_variable(Real("realOut", causality=Fmi2Causality.output))

    def do_step(self, current_time, step_size):
        self.log("Do step on Slave1.")
        self.realOut = self.realIn * 5.0 * (1.0 - math.exp(-1.0 * (current_time + step_size) / 0.1))
        return True
"""

    slave2_code = """from pythonfmu.fmi2slave import Fmi2Slave, Fmi2Causality, Integer, Real, Boolean, String


class Slave2(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.realIn = 22.0
        self.realOut = 0.0
        self.register_variable(Real("realIn", causality=Fmi2Causality.input))
        self.register_variable(Real("realOut", causality=Fmi2Causality.output))

    def do_step(self, current_time, step_size):
        self.log("Do step on Slave2.")
        self.realOut = -2.0 * self.realIn
        return True
"""

    script1_file = tmp_path / "orig" / "slave1.py"
    script1_file.parent.mkdir(parents=True, exist_ok=True)
    script1_file.write_text(slave1_code)

    fmu1 = FmuBuilder.build_FMU(
        script1_file,
        dest=tmp_path,
        needsExecutionTool="false"
    )
    assert fmu1.exists()

    script2_file = tmp_path / "orig" / "slave2.py"
    script2_file.write_text(slave2_code)

    fmu2 = FmuBuilder.build_FMU(
        script2_file,
        dest=tmp_path,
        needsExecutionTool="false"
    )
    assert fmu2.exists()

    model1 = pyfmi.load_fmu(str(fmu1), log_level=7)
    model2 = pyfmi.load_fmu(str(fmu2), log_level=7)

    connections = [(model1, "realOut", model2, "realIn")]
    sim = pyfmi.Master([model1, model2], connections)

    res = sim.simulate(final_time=0.1, options={'step_size': 0.025})

    res1 = res[model1]
    assert res1["realOut"][-1] == pytest.approx(
        22.0 * 5.0 * (1.0 - math.exp(-1.0 * res1["time"][-1] / 0.1)), rel=1e-7
    )
    res2 = res[model2]
    assert res2["realIn"][-1] == pytest.approx(res1["realOut"][-1])
    # pyfmi master algorithm seems all fmus at once with the output from the previous time step
    assert res2["realOut"][-1] == pytest.approx(-2.0 * res2["realIn"][-2])
