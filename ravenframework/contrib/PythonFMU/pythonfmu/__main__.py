import argparse

from pythonfmu import builder, csvbuilder, deploy
from ._version import __version__


def cli_main():
    parser = argparse.ArgumentParser(prog="pythonfmu")

    parser.add_argument("-V", "--version", action="version", version=__version__)

    def default_execution(**kwargs):
        print("A subcommand must be provided.\n")
        parser.print_help()

    parser.set_defaults(execute=default_execution)

    subparsers = parser.add_subparsers(
        title="Subcommands", description="Call `pythonfmu _command_ -h` to get more help."
    )

    build_parser = subparsers.add_parser(
        "build",
        description="Build an FMU from a Python script.",
        help="Build an FMU from a Python script."
    )
    builder.create_command_parser(build_parser)

    csv_parser = subparsers.add_parser(
        "buildcsv",
        description="Build an FMU from a CSV file.",
        help="Build an FMU from a CSV file."
    )
    csvbuilder.create_command_parser(csv_parser)

    deploy_parser = subparsers.add_parser(
        "deploy",
        description="""Deploy a Python FMU.

        The command will look in the `resources` folder for one of the following files:
        `requirements.txt` or `environment.yml`.

        If you specify a environment file but no package manager, `conda` will be selected
        for `.yaml` and `.yml` otherwise `pip` will be used.

        The tool assume the Python environment in which the FMU should be executed
        is the current one.
        """,
        help="Install Python FMU dependencies."
    )
    deploy.create_command_parser(deploy_parser)

    options = vars(parser.parse_args())
    execute = options.pop("execute")
    execute(**options)


if __name__ == "__main__":
    cli_main()
