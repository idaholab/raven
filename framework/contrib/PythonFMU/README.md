# PythonFMU

> A lightweight framework that enables the packaging of Python 3 code or CSV files as co-simulation FMUs (following FMI version 2.0).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/NTNU-IHB/PythonFMU/issues)

[![CI](https://github.com/NTNU-IHB/PythonFMU/workflows/CI/badge.svg)](https://github.com/NTNU-IHB/PythonFMU/actions?query=workflow%3ACI)
[![PyPI](https://img.shields.io/pypi/v/pythonfmu)](https://pypi.org/project/pythonfmu/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pythonfmu.svg)](https://anaconda.org/conda-forge/pythonfmu)

[![Gitter](https://badges.gitter.im/NTNU-IHB/FMI4j.svg)](https://gitter.im/NTNU-IHB/PythonFMU?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

### How do I build an FMU from python code?

1. Install `pythonfmu` package:

```bash
pip install pythonfmu
```

2. Create a new class extending the `Fmi2Slave` class declared in the `pythonfmu.fmi2slave` module (see below for an example).
3. Run `pythonfmu build` to create the fmu.

```
usage: pythonfmu build [-h] -f SCRIPT_FILE [-d DEST] [--doc DOCUMENTATION_FOLDER] [--no-external-tool]
                       [--no-variable-step] [--interpolate-inputs] [--only-one-per-process] [--handle-state]
                       [--serialize-state] [--use-memory-management]
                       [Project files [Project files ...]]

Build an FMU from a Python script.

positional arguments:
  Project files         Additional project files required by the Python script.

optional arguments:
  -h, --help            show this help message and exit
  -f SCRIPT_FILE, --file SCRIPT_FILE
                        Path to the Python script.
  -d DEST, --dest DEST  Where to save the FMU.
  --doc DOCUMENTATION_FOLDER
                        Documentation folder to include in the FMU.
  --no-external-tool    If given, needsExecutionTool=false
  --no-variable-step    If given, canHandleVariableCommunicationStepSize=false
  --interpolate-inputs  If given, canInterpolateInputs=true
  --only-one-per-process
                        If given, canBeInstantiatedOnlyOncePerProcess=true
  --handle-state        If given, canGetAndSetFMUstate=true
  --serialize-state     If given, canSerializeFMUstate=true
```

### How do I build an FMU from python code with third-party dependencies?

Often, Python scripts depends on non-builtin libraries like `numpy`, `scipy`, etc.
_PythonFMU_ does not package a full environment within the FMU.
However, you can package a `requirements.txt` or `environment.yml` file within your FMU following these steps:

1. Install _pythonfmu_ package: `pip install pythonfmu`
2. Create a new class extending the `Fmi2Slave` class declared in the `pythonfmu.fmi2slave` module (see below for an example).
3. Create a `requirements.txt` file (to use _pip_ manager) and/or a `environment.yml` file (to use _conda_ manager) that defines your dependencies.
4. Run `pythonfmu build -f myscript.py requirements.txt` to create the fmu including the dependencies file.

And using `pythonfmu deploy`, end users will be able to update their local Python environment. The steps to achieve that:

1. Install _pythonfmu_ package: `pip install pythonfmu`
2. Be sure to be in the Python environment to be updated. Then execute `pythonfmu deploy -f my.fmu`

```
usage: pythonfmu deploy [-h] -f FMU [-e ENVIRONMENT] [{pip,conda}]

Deploy a Python FMU. The command will look in the `resources` folder for one of the following files:
`requirements.txt` or `environment.yml`. If you specify a environment file but no package manager, `conda` will be selected for `.yaml` and `.yml` otherwise `pip` will be used. The tool assume the Python environment in which the FMU should be executed is the current one.

positional arguments:
  {pip,conda}           Python packages manager

optional arguments:
  -h, --help            show this help message and exit
  -f FMU, --file FMU    Path to the Python FMU.
  -e ENVIRONMENT, --env ENVIRONMENT
                        Requirements or environment file.
```

### Example:

#### Write the script

```python

from pythonfmu import Fmi2Causality, Fmi2Slave, Boolean, Integer, Real, String


class PythonSlave(Fmi2Slave):

    author = "John Doe"
    description = "A simple description"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.intOut = 1
        self.realOut = 3.0
        self.booleanVariable = True
        self.stringVariable = "Hello World!"
        self.register_variable(Integer("intOut", causality=Fmi2Causality.output))
        self.register_variable(Real("realOut", causality=Fmi2Causality.output))
        self.register_variable(Boolean("booleanVariable", causality=Fmi2Causality.local))
        self.register_variable(String("stringVariable", causality=Fmi2Causality.local))
        
        # Note:
        # it is also possible to explicitly define getters and setters as lambdas in case the variable is not backed by a Python field.
        # self.register_variable(Real("myReal", causality=Fmi2Causality.output, getter=lambda: self.realOut, setter=lambda v: set_real_out(v))

    def do_step(self, current_time, step_size):
        return True

```

#### Create the FMU

```
pythonfmu build -f pythonslave.py myproject
```

In this example a python class named `PythonSlave` that extends `Fmi2Slave` is declared in a file named `pythonslave.py`,
where `myproject` is an optional folder containing additional project files required by the python script.
Project folders such as this will be recursively copied into the FMU. Multiple project files/folders may be added.

### Test it online

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NTNU-IHB/PythonFMU/master?urlpath=lab/tree/examples/demo.ipynb)

### Note

PythonFMU does not bundle Python, which makes it a tool coupling solution.
This means that you can not expect the generated FMU to work on a different system (The system would need a compatible Python version and libraries).
But to ease its usage the wrapper uses the limited Python API, making the pre-built binaries for Linux and Windows 64-bits
compatible with any Python 3 environment. If you need to compile the wrapper for a specific configuration,
you will need CMake and a C++ compiler. The commands for building the wrapper on Linux and on Windows can be seen in
the [GitHub workflow](./.github/workflows/main.yml).

PythonFMU does not automatically resolve 3rd party dependencies. If your code includes e.g. `numpy`, the target system also needs to have `numpy` installed.

---

Would you rather build FMUs in Java? Check out [FMI4j](https://github.com/NTNU-IHB/FMI4j)!  
Need to distribute your FMUs? [FMU-proxy](https://github.com/NTNU-IHB/FMU-proxy) to the rescue!

### Publications

[Hatledal, Lars Ivar, Frédéric Collonval, and Houxiang Zhang. "Enabling python driven co-simulation models with PythonFMU." Proceedings of the 34th International ECMS-Conference on Modelling and Simulation-ECMS 2020. ECMS European Council for Modelling and Simulation, 2020.](https://doi.org/10.7148/2020-0235)

### Credits

This work has been possible thanks to the contributions of: <br> 
@markaren from NTNU-IHB <br>
@fcollonval from Safran SA.
