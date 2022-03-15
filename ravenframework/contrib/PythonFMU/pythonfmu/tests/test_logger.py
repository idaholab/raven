import itertools
import pytest
from unittest.mock import call, MagicMock

from pythonfmu.builder import FmuBuilder
from pythonfmu.enums import Fmi2Status
from pythonfmu.fmi2slave import Fmi2Slave

fmpy = pytest.importorskip(
    "fmpy", reason="fmpy is not available for testing the produced FMU"
)
pytestmark = pytest.mark.skipif(
    not FmuBuilder.has_binary(), reason="No binary available for the current platform."
)


@pytest.mark.integration
@pytest.mark.parametrize("debug_logging", [True, False])
def test_logger(tmp_path, debug_logging):
    name = "PythonSlaveWithDebugLogger" if debug_logging else "PythonSlaveWithLogger"
    category = "category"
    message = "log message"

    log_calls = [
        (
            f"{status.name.upper()} - {debug} - {message}", 
            status, 
            category, 
            debug
        ) for debug, status in itertools.product([True, False], Fmi2Status)
    ]

    fmu_calls = "\n".join([
        '        self.log("{}", Fmi2Status.{}, "{}", {})'.format(c[0], c[1].name, c[2], c[3]) for c in log_calls
    ])

    slave_code = f"""from pythonfmu.fmi2slave import Fmi2Slave, Fmi2Status, Fmi2Causality, Integer, Real, Boolean, String


class {name}(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.realIn = 22.0
        self.realOut = 0.0
        self.register_variable(Real("realIn", causality=Fmi2Causality.input))
        self.register_variable(Real("realOut", causality=Fmi2Causality.output))


    def do_step(self, current_time, step_size):
{fmu_calls}
        return True
"""

    script_file = tmp_path / "orig" / f"{name.lower()}.py"
    script_file.parent.mkdir(parents=True, exist_ok=True)
    script_file.write_text(slave_code)

    fmu = FmuBuilder.build_FMU(script_file, dest=tmp_path)
    assert fmu.exists()

    logger = MagicMock()

    fmpy.simulate_fmu(
        str(fmu),
        stop_time=1e-3,
        output_interval=1e-3,
        logger=logger,
        debug_logging=debug_logging
    )

    expected_calls = [
        call(
            logger.call_args[0][0],  # Don't test the first argument
            bytes(name, encoding="utf-8"),
            int(c[1]),
            bytes(c[2], encoding="utf-8"),
            bytes(c[0], encoding="utf-8")
        ) for c in filter(lambda c: debug_logging or not c[3], log_calls)
    ]
    
    assert logger.call_count == len(Fmi2Status) * (1 + int(debug_logging))
    logger.assert_has_calls(expected_calls)


@pytest.mark.integration
@pytest.mark.parametrize("debug_logging", [True, False])
@pytest.mark.parametrize("categories", [(), ("logStatusError", "logStatusFatal")])
def test_log_categories(tmp_path, debug_logging, categories):
    name = "PythonSlaveDebugCategories" if debug_logging else "PythonSlaveCategories"
    message = "log message"

    log_calls = [
        (
            f"{status.name.upper()} - {debug} - {message}", 
            status,
            debug
        ) for debug, status in itertools.product([True, False], Fmi2Status)
    ]

    fmu_calls = "\n".join([
        '        self.log("{}", Fmi2Status.{}, None, {})'.format(c[0], c[1].name, c[2]) for c in log_calls
    ])

    slave_code = f"""from pythonfmu.fmi2slave import Fmi2Slave, Fmi2Status, Fmi2Causality, Integer, Real, Boolean, String


class {name}(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.realIn = 22.0
        self.realOut = 0.0
        self.register_variable(Real("realIn", causality=Fmi2Causality.input))
        self.register_variable(Real("realOut", causality=Fmi2Causality.output))


    def do_step(self, current_time, step_size):
{fmu_calls}
        return True
"""

    script_file = tmp_path / "orig" / f"{name.lower()}.py"
    script_file.parent.mkdir(parents=True, exist_ok=True)
    script_file.write_text(slave_code)

    fmu = FmuBuilder.build_FMU(script_file, dest=tmp_path)
    assert fmu.exists()

    logger = MagicMock()

    # Load the model
    callbacks = fmpy.fmi2.fmi2CallbackFunctions()
    callbacks.logger = fmpy.fmi2.fmi2CallbackLoggerTYPE(logger)
    callbacks.allocateMemory = fmpy.fmi2.fmi2CallbackAllocateMemoryTYPE(fmpy.fmi2.allocateMemory)
    callbacks.freeMemory = fmpy.fmi2.fmi2CallbackFreeMemoryTYPE(fmpy.fmi2.freeMemory)

    model_description = fmpy.read_model_description(fmu)
    unzip_dir = fmpy.extract(fmu)

    model = fmpy.fmi2.FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzip_dir,
        modelIdentifier=model_description.coSimulation.modelIdentifier,
        instanceName='instance1')
    # Instantiate the model
    model.instantiate(callbacks=callbacks)
    model.setDebugLogging(debug_logging, categories)
    model.setupExperiment()
    model.enterInitializationMode()
    model.exitInitializationMode()
    # Execute the model
    model.doStep(0., 0.1)
    # Clean the model
    model.terminate()

    expected_calls = []
    for c in filter(lambda c: debug_logging or not c[2], log_calls):
        category = f"logStatus{c[1].name.capitalize()}"
        if category not in Fmi2Slave.log_categories:
            category = "logAll"
        if len(categories) == 0 or category in categories:
            expected_calls.append(call(
                logger.call_args[0][0],  # Don't test the first argument
                b'instance1',
                int(c[1]),
                bytes(category, encoding="utf-8"),
                bytes(c[0], encoding="utf-8")
            ))

    n_calls = len(Fmi2Status) if len(categories) == 0 else len(categories)

    assert logger.call_count == n_calls * (1 + int(debug_logging))
    logger.assert_has_calls(expected_calls)
