import math
from pathlib import Path

import pytest

from pythonfmu.builder import FmuBuilder

pytestmark = pytest.mark.skipif(
    not FmuBuilder.has_binary(), reason="No binary available for the current platform."
)
pyfmi = pytest.importorskip(
    "pyfmi", reason="pyfmi is required for testing the produced FMU"
)

@pytest.mark.integration
def test_integration_demo(tmp_path):
    script_file = Path(__file__).parent / "slaves/pythonslave.py"
    fmu = FmuBuilder.build_FMU(script_file, dest=tmp_path, needsExecutionTool="false")
    assert fmu.exists()
    model = pyfmi.load_fmu(str(fmu))
    res = model.simulate(final_time=0.5)

    assert res["realOut"][-1] == pytest.approx(res["time"][-1], rel=1e-7)


@pytest.mark.integration
def test_integration_reset(tmp_path):
    script_file = Path(__file__).parent / "slaves/pythonslave.py"
    fmu = FmuBuilder.build_FMU(script_file, dest=tmp_path, needsExecutionTool="false")
    assert fmu.exists()

    vr = 5  # realOut
    dt = 0.1
    model = pyfmi.load_fmu(str(fmu))
    initial_value = model.get_real([vr])[0]
    assert initial_value == pytest.approx(3.0, rel=1e-7)
    model.do_step(0.0, dt, True)
    read = model.get_real([vr])[0]
    assert read == pytest.approx(dt, rel=1e-7)
    model.reset()
    read = model.get_real([vr])[0]
    assert read == pytest.approx(initial_value, rel=1e-7)


@pytest.mark.integration
def test_integration_get_state(tmp_path):
    script_file = Path(__file__).parent / "slaves/pythonslave.py"
    fmu = FmuBuilder.build_FMU(
        script_file,
        dest=tmp_path,
        needsExecutionTool="false",
        canGetAndSetFMUstate="true")
    assert fmu.exists()

    model = pyfmi.load_fmu(str(fmu))

    vr = model.get_model_variables()["realOut"].value_reference
    dt = 0.1
    t = 0.0

    def step_model():
        nonlocal t
        model.do_step(t, dt, True)
        t += dt

    model.initialize()
    step_model()
    state = model.get_fmu_state()
    assert model.get_real([vr])[0] == pytest.approx(dt, rel=1e-7)
    step_model()
    assert model.get_real([vr])[0] == pytest.approx(dt * 2, rel=1e-7)
    model.set_fmu_state(state)
    assert model.get_real([vr])[0] == pytest.approx(dt, rel=1e-7)
    step_model()
    assert model.get_real([vr])[0] == pytest.approx(dt * 3, rel=1e-7)
    model.free_fmu_state(state)


@pytest.mark.integration
def test_integration_get_serialize_state(tmp_path):
    fmpy = pytest.importorskip(
        "fmpy", reason="fmpy is not available for testing the produced FMU"
    )

    script_file = Path(__file__).parent / "slaves/pythonslave.py"
    fmu = FmuBuilder.build_FMU(
        script_file,
        dest=tmp_path,
        canGetAndSetFMUstate="true",
        canSerializeFMUstate="true")
    assert fmu.exists()

    model_description = fmpy.read_model_description(fmu)
    unzip_dir = fmpy.extract(fmu)

    model = fmpy.fmi2.FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzip_dir,
        modelIdentifier=model_description.coSimulation.modelIdentifier,
        instanceName='instance1')

    realOut = filter(
        lambda var: var.name == "realOut", model_description.modelVariables
    )
    vrs = list(map(lambda var: var.valueReference, realOut))
    t = 0.0
    dt = 0.1

    def step_model():
        nonlocal t
        model.doStep(t, dt)
        t += dt

    model.instantiate()
    model.setupExperiment()
    model.enterInitializationMode()
    model.exitInitializationMode()

    step_model()
    state = model.getFMUstate()
    assert model.getReal(vrs)[0] == pytest.approx(dt, rel=1e-7)
    step_model()
    assert model.getReal(vrs)[0] == pytest.approx(dt * 2, rel=1e-7)
    model.setFMUstate(state)
    assert model.getReal(vrs)[0] == pytest.approx(dt, rel=1e-7)
    step_model()
    assert model.getReal(vrs)[0] == pytest.approx(dt * 3, rel=1e-7)

    serialize_fmu_state = model.serializeFMUstate(state)
    model.freeFMUstate(state)
    de_serialize_fmu_state = model.deSerializeFMUstate(serialize_fmu_state)
    model.setFMUstate(de_serialize_fmu_state)
    assert model.getReal(vrs)[0] == pytest.approx(dt, rel=1e-7)

    model.freeFMUstate(de_serialize_fmu_state)
    model.terminate()


@pytest.mark.integration
def test_integration_get(tmp_path):
    script_file = Path(__file__).parent / "slaves/pythonslave.py"
    fmu = FmuBuilder.build_FMU(script_file, dest=tmp_path, needsExecutionTool="false")
    assert fmu.exists()
    model = pyfmi.load_fmu(str(fmu))

    to_test = {
        "intParam": 42,
        "intOut": 23,
        "realOut": 3.0,
        "booleanVariable": True,
        "stringVariable": "Hello World!",
        "realIn": 2.0 / 3.0,
        "booleanParameter": False,
        "stringParameter": "dog",
        "container.someReal": 99.0,
        "container.subContainer.someInteger": -15
    }

    model_value = None
    variables = model.get_model_variables()
    for key, value in to_test.items():
        var = variables[key]
        if var.type == pyfmi.fmi.FMI2_INTEGER:
            model_value = model.get_integer([var.value_reference])[0]
        elif var.type == pyfmi.fmi.FMI2_REAL:
            model_value = model.get_real([var.value_reference])[0]
        elif var.type == pyfmi.fmi.FMI2_BOOLEAN:
            model_value = model.get_boolean([var.value_reference])[0]
        elif var.type == pyfmi.fmi.FMI2_STRING:
            model_value = model.get_string([var.value_reference])[0]
        else:
            pytest.xfail("Unsupported type")

        assert model_value == value


@pytest.mark.integration
def test_integration_read_from_file(tmp_path):
    script_file = Path(__file__).parent / "slaves/pythonslave_read_file.py"
    project_file = Path(__file__).parent / "data/hello.txt"
    fmu = FmuBuilder.build_FMU(script_file, project_files=[project_file], dest=tmp_path, needsExecutionTool="false")
    assert fmu.exists()
    model = pyfmi.load_fmu(str(fmu))

    variables = model.get_model_variables()
    var = variables["file_content"]
    model_value = model.get_string([var.value_reference])[0]

    with (open(project_file, 'r')) as file:
        data = file.read()

    assert model_value == data


@pytest.mark.integration
def test_integration_set(tmp_path):
    script_file = Path(__file__).parent / "slaves/pythonslave.py"
    fmu = FmuBuilder.build_FMU(script_file, dest=tmp_path, needsExecutionTool="false")
    assert fmu.exists()
    model = pyfmi.load_fmu(str(fmu))

    to_test = {
        "intParam": 20,
        "realIn": 1.0 / 3.0,
        "booleanParameter": True,
        "stringParameter": "cat",
        "container.someReal": 42.0,
        "container.subContainer.someInteger": 421
    }

    model_value = None
    variables = model.get_model_variables()
    for key, value in to_test.items():
        var = variables[key]
        if var.type == pyfmi.fmi.FMI2_INTEGER:
            model.set_integer([var.value_reference], [value])
            model_value = model.get_integer([var.value_reference])[0]
        elif var.type == pyfmi.fmi.FMI2_REAL:
            model.set_real([var.value_reference], [value])
            model_value = model.get_real([var.value_reference])[0]
        elif var.type == pyfmi.fmi.FMI2_BOOLEAN:
            model.set_boolean([var.value_reference], [value])
            model_value = model.get_boolean([var.value_reference])[0]
        elif var.type == pyfmi.fmi.FMI2_STRING:
            model.set_string([var.value_reference], [value])
            model_value = model.get_string([var.value_reference])[0]
        else:
            pytest.xfail("Unsupported type")

        assert model_value == value


@pytest.mark.integration
def test_simple_integration_fmpy(tmp_path):
    fmpy = pytest.importorskip(
        "fmpy", reason="fmpy is not available for testing the produced FMU"
    )

    script_file = Path(__file__).parent / "slaves/pythonslave.py"
    fmu = FmuBuilder.build_FMU(script_file, dest=tmp_path)
    assert fmu.exists()
    res = fmpy.simulate_fmu(str(fmu), stop_time=2.0)

    assert res["realOut"][-1] == pytest.approx(res["time"][-1], rel=1e-7)


@pytest.mark.integration
def test_integration_has_local_dep(tmp_path):

    script_file = Path(__file__).parent / "slaves/slavewithdep.py"
    local_file = Path(__file__).parent / "slaves/localmodule.py"

    fmu = FmuBuilder.build_FMU(
        script_file,
        dest=tmp_path,
        project_files=[local_file],
        needsExecutionTool="false",
    )
    assert fmu.exists()
    model = pyfmi.load_fmu(str(fmu))
    res = model.simulate(final_time=0.5)

    assert res["realOut"][-1] == pytest.approx(
        22.0 * 5.0 * math.exp(res["time"][-1] / 0.1), rel=1e-7
    )


@pytest.mark.integration
def test_integration_throw_py_error(tmp_path):
    fmpy = pytest.importorskip(
        "fmpy", reason="fmpy is not available for testing the produced FMU"
    )

    script_file = Path(__file__).parent / "slaves/PythonSlaveWithException.py"
    fmu = FmuBuilder.build_FMU(script_file, dest=tmp_path)
    assert fmu.exists()

    with pytest.raises(Exception):
        fmpy.simulate_fmu(str(fmu), stop_time=1.0)
