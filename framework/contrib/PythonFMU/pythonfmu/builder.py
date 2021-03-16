"""Python FMU builder"""
import argparse
import importlib
import itertools
import logging
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring
from .osutil import get_lib_extension, get_platform
from .fmi2slave import FMI2_MODEL_OPTIONS, Fmi2Slave

FilePath = Union[str, Path]
HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


def get_class_name(file_name: Path) -> str:
    with open(str(file_name), 'r') as file:
        data = file.read()
        return re.search(r'class (\w+)\(\s*Fmi2Slave\s*\)\s*:', data).group(1)


def get_model_description(filepath: Path, module_name: str) -> Tuple[str, Element]:
    """Extract the FMU model description as XML.

    Args:
        filepath (pathlib.Path) : script file path
        module_name (str) : python module to load

    Returns:
        Tuple[str, xml.etree.TreeElement.Element] : FMU model name, model description
    """
    # Add current folder to handle local dependencies
    sys.path.insert(0, str(filepath.parent))
    try:
        # Import the user interface
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        fmu_interface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fmu_interface)
        # Instantiate the interface
        class_name = get_class_name(filepath)
        instance = getattr(fmu_interface, class_name)(instance_name="dummyInstance", resources=str(filepath.parent))
    finally:
        sys.path.remove(str(filepath.parent))  # remove inserted temporary path

    if not isinstance(instance, Fmi2Slave):
        raise TypeError(
            f"The provided class '{class_name}' does not inherit from {Fmi2Slave.__qualname__}"
        )
    # Produce the xml
    return instance.modelName, instance.to_xml()


class FmuBuilder:

    @staticmethod
    def build_FMU(
        script_file: FilePath,
        dest: FilePath = ".",
        project_files: Iterable[FilePath] = set(),
        documentation_folder: Optional[FilePath] = None,
        **options,
    ) -> Path:
        script_file = Path(script_file)
        if not script_file.exists():
            raise ValueError(f"No such file {script_file!s}")
        if not script_file.suffix.endswith(".py"):
            raise ValueError(f"File {script_file!s} must have extension '.py'!")

        dest = Path(dest)
        if not dest.exists():
            dest.mkdir(parents=True)
        project_files = set(map(Path, project_files))

        if documentation_folder is not None:
            documentation_folder = Path(documentation_folder)
            if not documentation_folder.exists():
                raise ValueError(
                    f"The documentation folder does not exists {documentation_folder!s}"
                )

        module_name = script_file.stem

        with tempfile.TemporaryDirectory(prefix="pythonfmu_") as tempd:
            temp_dir = Path(tempd)
            shutil.copy2(script_file, temp_dir)

            # Embed pythonfmu in the FMU so it does not need to be included
            dep_folder = temp_dir / "pythonfmu"
            dep_folder.mkdir()
            for dep in HERE.glob('*.py'):  # Find all python files at the same level as this one
                shutil.copy2(dep, dep_folder)
            for file_ in project_files:
                if file_ == script_file.parent:
                    new_folder = temp_dir / file_.name
                    new_folder.mkdir()
                    for f in file_.iterdir():
                        if f.is_dir():
                            temp_dest = new_folder / f.name
                            shutil.copytree(f, temp_dest)
                        elif f.name != script_file.name:
                            shutil.copy2(f, new_folder)
                        else:
                            logger.debug(
                                "Skip file with the same name as the script found in project file."
                            )
                else:
                    if file_.is_dir():
                        temp_dest = temp_dir / file_.name
                        shutil.copytree(file_, temp_dest)
                    else:
                        shutil.copy2(file_, temp_dir)

            model_identifier, xml = get_model_description(
                temp_dir.absolute() / script_file.name, module_name
            )

            dest_file = dest / f"{model_identifier}.fmu"

            type_node = xml.find("CoSimulation")
            option_names = [opt.name for opt in FMI2_MODEL_OPTIONS]
            for option, value in options.items():
                if option in option_names:
                    type_node.set(option, str(value).lower())

            with zipfile.ZipFile(dest_file, "w") as zip_fmu:

                resource = Path("resources")

                # Add files copied in temporary directory
                for f in temp_dir.rglob("*"):
                    if f.is_file() and f.parent.name != "__pycache__":
                        relative_f = f.relative_to(temp_dir)
                        zip_fmu.write(f, arcname=(resource / relative_f))

                # Add information for the Python loader
                zip_fmu.writestr(str(resource.joinpath("slavemodule.txt")), module_name)

                # Add FMI API wrapping Python class source
                source_node = SubElement(type_node, "SourceFiles")
                sources = Path("sources")
                src = HERE / "pythonfmu-export"
                for f in itertools.chain(
                    src.rglob("*.hpp"), src.rglob("*.cpp"), src.rglob("CMakeLists.txt")
                ):
                    relative_f = f.relative_to(src)
                    SubElement(
                        source_node, "File", attrib={"name": relative_f.as_posix()}
                    )
                    zip_fmu.write(f, arcname=(sources / relative_f))

                # Add FMI API wrapping Python class library
                binaries = Path("binaries")
                src_binaries = HERE / "resources" / "binaries"
                for f in itertools.chain(
                    src_binaries.rglob("*.dll"),
                    src_binaries.rglob("*.so"),
                    src_binaries.rglob("*.dylib"),
                ):
                    relative_f = f.relative_to(src_binaries)
                    arcname = (
                        binaries
                        / relative_f.parent
                        / f"{model_identifier}{relative_f.suffix}"
                    )
                    zip_fmu.write(f, arcname=arcname)

                # Add the documentation folder
                if documentation_folder is not None:
                    documentation = Path("documentation")
                    for f in documentation_folder.rglob("*"):
                        if f.is_file():
                            relative_f = f.relative_to(documentation_folder)
                            zip_fmu.write(f, arcname=(documentation / relative_f))

                # Add the model description
                xml_str = parseString(tostring(xml, "UTF-8"))
                zip_fmu.writestr(
                    "modelDescription.xml", xml_str.toprettyxml(encoding="UTF-8")
                )

            return dest_file

    @staticmethod
    def has_binary() -> bool:
        """Does the binary for this platform exits?"""
        binary_folder = get_platform()
        lib_ext = get_lib_extension() or "*"  # if library extension is unknown, it will look for '*.*' in src_binaries
        src_binaries = HERE / "resources" / "binaries" / binary_folder
        return src_binaries.exists() and len(list(src_binaries.glob(f"*.{lib_ext}"))) >= 1


def create_command_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-f",
        "--file",
        dest="script_file",
        help="Path to the Python script.",
        required=True
    )

    parser.add_argument(
        "-d", "--dest", dest="dest", help="Where to save the FMU.", default="."
    )

    parser.add_argument(
        "--doc",
        dest="documentation_folder",
        help="Documentation folder to include in the FMU.",
        default=None
    )

    for option in FMI2_MODEL_OPTIONS:
        action = "store_false" if option.value else "store_true"
        parser.add_argument(
            f"--{option.cli}",
            dest=option.name,
            help=f"If given, {option.name}={action[6:]}",
            action=action
        )

    parser.add_argument(
        "project_files",
        metavar="Project files",
        nargs="*",
        help="Additional project files required by the Python script.",
        default=set()
    )

    parser.set_defaults(execute=FmuBuilder.build_FMU)
