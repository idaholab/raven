import pytest
from pathlib import Path

from pythonfmu.csvbuilder import CsvFmuBuilder

EPS = 1e-7


def test_csvslave(tmp_path):
    fmpy = pytest.importorskip(
        "fmpy", reason="fmpy is not available for testing the produced FMU"
    )

    csv_file = Path(__file__).parent / "data/csvdemo.csv"

    fmu = CsvFmuBuilder.build_FMU(csv_file, dest=tmp_path)
    assert fmu.exists()

    model_description = fmpy.read_model_description(fmu)
    unzip_dir = fmpy.extract(fmu)

    model = fmpy.fmi2.FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzip_dir,
        modelIdentifier=model_description.coSimulation.modelIdentifier,
        instanceName='instance1')

    interpolate_var = list(filter(
        lambda var: var.name == "interpolate", model_description.modelVariables
    ))[0]

    t = 0.0
    dt = 0.1

    def init_model(interpolate=True):
        model.instantiate()
        if not interpolate:
            model.setBoolean([interpolate_var.valueReference], [False])
        model.setupExperiment()
        model.enterInitializationMode()
        model.exitInitializationMode()

    def step_model():
        nonlocal t
        model.doStep(t, dt)
        t += dt

    init_model()

    for i in range(1, 6):
        assert model.getReal([0])[0] == pytest.approx(-1, rel=EPS)
        assert model.getInteger([1])[0] == i
        assert model.getReal([2])[0] == pytest.approx(pow(2, i), rel=EPS)
        assert model.getBoolean([3])[0] == i % 2
        assert model.getString([4])[0].decode("utf-8") == str(i)
        step_model()

    model.reset()
    init_model()

    t = 0.0
    dt = 0.05

    actual_ints = []
    actual_reals = []
    actual_bools = []
    actual_strings = []

    expected_ints = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
    expected_reals = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
    expected_bools = [True, True, False, False, True, True, False, False, True, True, False]
    expected_strings = list(map(lambda i: str(i), expected_ints))

    for i in range(0, 11):
        actual_ints.append(model.getInteger([1])[0])
        actual_reals.append(model.getReal([2])[0])
        actual_bools.append(model.getBoolean([3])[0])
        actual_strings.append(model.getString([4])[0].decode("utf-8"))
        step_model()

    assert actual_ints == expected_ints
    for i in range(0, len(actual_reals)):
        assert actual_reals[i] == pytest.approx(expected_reals[i], rel=EPS)
    assert actual_bools == expected_bools
    assert actual_strings == expected_strings

    model.reset()
    init_model(False)

    t = 0.0
    actual_ints.clear()
    actual_reals.clear()
    actual_bools.clear()
    actual_strings.clear()
    expected_reals = [2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0, 32.0, 32.0, 64.0]
    for i in range(0, 11):
        actual_ints.append(model.getInteger([1])[0])
        actual_reals.append(model.getReal([2])[0])
        actual_bools.append(model.getBoolean([3])[0])
        actual_strings.append(model.getString([4])[0].decode("utf-8"))
        step_model()

    assert actual_ints == expected_ints
    for i in range(0, len(actual_reals)):
        assert actual_reals[i] == pytest.approx(expected_reals[i], rel=EPS)
    assert actual_bools == expected_bools
    assert actual_strings == expected_strings
