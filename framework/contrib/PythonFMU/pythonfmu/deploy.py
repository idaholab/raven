"""CLI command to deploy a FMU."""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

from .enums import PackageManager


# Dictionary of ("default file name", "associated package manager")
ENVIRONMENT_FILES = {
    "requirements.txt": PackageManager.pip,
    "environment.yaml": PackageManager.conda,
    "environment.yml": PackageManager.conda
}


def deploy(
    fmu: Union[str, Path],
    environment: Union[str, Path, None] = None,
    package_manager: Union[str, PackageManager, None] = None
) -> None:
    """Install Python dependency packages from requirement file shipped within the FMU.
    
    Args:
        fmu (str or pathlib.Path) : FMU file path
        environment (str or pathlib.Path) : optional, requirements file within the `resources` folder of the FMU
        package_manager (str) : optional, Python package manager
    """
    fmu = Path(fmu)
    manager = None
    if package_manager is not None:
        manager = PackageManager(package_manager)

    env_content = None
    environment_file = None
    with zipfile.ZipFile(fmu) as files:
        names = files.namelist()

        environment_file = None
        if environment is None:
            for spec in ENVIRONMENT_FILES:
                test = Path("resources") / spec
                if test.as_posix() in names:
                    environment_file = test
                    manager = manager or ENVIRONMENT_FILES[spec]
                    break
            if environment_file is None:
                raise ValueError("Unable to find requirement file in the FMU resources folder.")
        else:
            environment_file = Path("resources") / environment
            if environment_file.as_posix() not in names:
                raise ValueError(f"Unable to find requirement file {environment!s} in the FMU resources folder.")

            if manager is None:
                if environment in ENVIRONMENT_FILES:
                    manager = ENVIRONMENT_FILES[environment]
                elif environment.endswith(".yaml") or environment.endswith(".yml"):
                    manager = PackageManager.conda
                else:
                    manager = PackageManager.pip

        with files.open(environment_file.as_posix(), mode="r") as env_file:
            env_content = env_file.read()

    with TemporaryDirectory() as tmp:
        tempd = Path(tmp)

        copy_env = tempd / environment_file.name
        copy_env.write_bytes(env_content)

        if manager == PackageManager.pip:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", f"{copy_env!s}", "--progress-bar", "off"],
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True
            )
        elif manager == PackageManager.conda:
            conda_exe = os.environ.get("CONDA_EXE", "conda")
            subprocess.run(
                [conda_exe, "env", "update", f"--file={copy_env!s}", "--quiet"],
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True
            )


def create_command_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-f",
        "--file",
        dest="fmu",
        help="Path to the Python FMU.",
        required=True
    )

    parser.add_argument(
        "-e",
        "--env",
        dest="environment",
        help="Requirements or environment file.",
        default=None
    )

    parser.add_argument(
        choices=["pip", "conda"],
        dest="package_manager",
        nargs='?',
        help="Python packages manager"
    )

    parser.set_defaults(execute=deploy)
