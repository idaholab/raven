from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from pythonfmu.builder import FmuBuilder
from pythonfmu.deploy import deploy

PYTHON_SLAVE = Path(__file__).parent / "slaves/pythonslave.py"

@pytest.mark.parametrize("test_manager", [None, "pip", "conda"])
@pytest.mark.parametrize(
    "requirements, test_requirements, expected", [
        (None, None, ValueError),
        (None, "requirements.txt", ValueError),
        (None, "environment.yaml", ValueError),
        ("requirements.txt", None, "pip"),
        ("environment.yaml", None, "conda"),
        ("environment.yml", None, "conda"),
        ("req.txt", "req.txt", "pip"),
        ("env.yml", "env.yml", "conda"),
    ]
)
def test_deploy(tmp_path, test_manager, requirements, test_requirements, expected):

    dummy_requirements = """numpy=1.16
scipy
"""
    if requirements is not None:
        with TemporaryDirectory() as tempd:
            requirements_file = Path(tempd) / requirements
            requirements_file.write_text(dummy_requirements)
            fmu = FmuBuilder.build_FMU(PYTHON_SLAVE, dest=tmp_path, project_files=[requirements_file, ])
    else:
        fmu = FmuBuilder.build_FMU(PYTHON_SLAVE, dest=tmp_path)
    assert fmu.exists()

    with patch("subprocess.run") as run:
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                deploy(fmu, environment=test_requirements, package_manager=test_manager)
            run.assert_not_called()
        else:
            deploy(fmu, environment=test_requirements, package_manager=test_manager)

            run.assert_called_once()
            assert (test_manager or expected) in " ".join(run.call_args[0][0])
