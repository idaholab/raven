# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Various utilities to find the correct library versions.
"""
from __future__ import print_function
import os
import sys
import subprocess
import distutils.version
import importlib

def check_versions():
  """
    Returns true if versions should be checked.
    @ In, None
    @ Out, check_versions, bool, if true versions should be checked.
  """
  return os.environ.get("RAVEN_IGNORE_VERSIONS", "0") != "1"

def in_python_3():
  """
    returns true if raven should be using python3
    @ In, None
    @ Out, in_python_3, boolean, True if should be using python3.
  """
  if sys.version_info.major > 2:
    return True
  return False

#This list is made of (module, how to check the version, minimum version,
# quality assurance module version, maximum version)
# Deep learning requires Scikit-Learn version at least 0.18
## working Conda 4.5.4, May 2018
modules_to_try = [("h5py", 'h5py.__version__', "2.9.0", "2.9.0", None), # 2.9.0
                  ("numpy", 'numpy.__version__', "1.16.4", "1.16.4", None),
                  ("scipy", 'scipy.__version__', "1.2.1", "1.2.1", None),
                  ("sklearn", 'sklearn.__version__', "0.21.2", "0.21.2", None),
                  ("pandas", 'pandas.__version__', "0.24.2", "0.24.2", None),
                  ("xarray", 'xarray.__version__', "0.12.1", "0.12.1", None),
                  ("netCDF4", 'netCDF4.__version__', "1.4.2", "1.4.2", None),
                  ("statsmodels", 'statsmodels.__version__', "0.9.0", "0.9.0", None),
                  ("matplotlib", 'matplotlib.__version__', "3.0.3", "3.0.3", None),
                  ("cloudpickle", 'cloudpickle.__version__', "1.1.1", "1.2.1", None)]

optional_test_libraries = [('pillow', 'PIL.__version__', "6.0.0", "6.0.0", None),
                           # On Windows conda, there are no Python 2.7-compatible
                           ## versions of TensorFlow, although
                           ## these exist on Mac and Linux condas.  Darn.
                           ("tensorflow", 'tensorflow.__version__', "1.13.1", "1.13.1", None)]

def __lookup_preferred_version(name, optional=False):
  """
    Look up the preferred version in the modules.
    @In, name, string, the name of the module
    @Out, result, string, returns the version as a string or "" if unknown
  """
  index = modules_to_try if not optional else optional_test_libraries
  for  mod_name, _, _, qa_ver, _ in index:
    if name == mod_name:
      return qa_ver
  return ""
# libraries to install with Conda
__condaList = [("h5py", __lookup_preferred_version("h5py")),
               ("numpy", __lookup_preferred_version("numpy")),
               ("scipy", __lookup_preferred_version("scipy")),
               ("scikit-learn", __lookup_preferred_version("sklearn")),
               ("pandas", __lookup_preferred_version("pandas")),
               ("xarray", __lookup_preferred_version("xarray")),
               ("netcdf4", __lookup_preferred_version("netCDF4")),
               ("matplotlib", __lookup_preferred_version("matplotlib")),
               ("statsmodels", __lookup_preferred_version("statsmodels")),
               ("tensorflow", __lookup_preferred_version("tensorflow", optional=True)),
               ("cloudpickle", __lookup_preferred_version("cloudpickle", optional=True)),
               ("python", "3"),
               ("hdf5", "1.10.4"),
               ("swig", ""),
               ("pylint", ""),
               ("coverage", ""),
               ("lxml", ""),
               ("psutil", "")]
# libraries to install with conda-forge
__condaForgeList = [("pyside2", ""),]
# optional conda libraries
__condaOptional = [('pillow', __lookup_preferred_version("pillow"))]

__pipList = [("numpy", __lookup_preferred_version("numpy")),
             ("cloudpickle", __lookup_preferred_version("cloudpickle")),
             ("h5py", __lookup_preferred_version("h5py")),
             ("scipy", __lookup_preferred_version("scipy")),
             ("scikit-learn", __lookup_preferred_version("sklearn")),
             ("matplotlib", __lookup_preferred_version("matplotlib")),
             ("xarray", __lookup_preferred_version("xarray")),
             ("netCDF4", __lookup_preferred_version("netCDF4")),
             ("statsmodels", __lookup_preferred_version("statsmodels")),
             ("tensorflow", __lookup_preferred_version("tensorflow", optional=True)),
             ("pandas", __lookup_preferred_version("pandas")),
             ("pylint", ""),
             ("psutil", ""),
             ("coverage", ""),
             ("lxml", "")]

def module_report(module, version=''):
  """
    Checks if the module exists.
    @ In, module, string, module name
    @ In, version, string, optional, if here use this to get the version.
    @ Out, (found_boolean,message,version)
     The found_boolean is true if the module is found.
     The message will be the result of print(module)
     The version is the version number or "NA" if not known.
  """
  if in_python_3():
    if os.name == "nt":
      #Command is python on windows in conda and Python.org install
      python = 'python'
    else:
      python = 'python3'
  else:
    python = 'python'
  try:
    command = 'import '+module+';print('+module+')'
    output = subprocess.check_output([python, '-c', command],
                                     universal_newlines=True)
    if len(version) > 0:
      try:
        command = 'import '+module+';print('+version+')'
        found_version = subprocess.check_output([python, '-c', command],
                                                universal_newlines=True).strip()
      except Exception:
        found_version = "NA"
    else:
      found_version = "NA"
    return (True, output, found_version)
  except Exception:
    return (False, 'Failed to find module '+module, "NA")

def __import_report(module):
  """
    Directly checks the module version.
    @ In, module, string, the module name
    @ Out,(found_boolean,message,version)
     The found_boolean is true if the module is found.
     The message will be the result of print(module)
     The version is the version number or "NA" if not known.
  """
  try:
    loaded = importlib.import_module(module)
    found_version = loaded.__version__
    output = str(loaded)
    return (True, output, found_version)
  except ImportError:
    return (False, 'Failed to find module '+module, "NA")

def modules_report():
  """
    Return a report on the modules.
    @ In, None
    @ Out, modules_report, a list of [(module_name,found_boolean,message,version)]
  """
  report_list = []
  for mod_name, find_ver, min_ver, qa_ver, max_ver in modules_to_try:
    found, message, version = module_report(mod_name, find_ver)
    if found:
      _, out_of_range, not_qa = __check_version(mod_name, min_ver, qa_ver, max_ver, found, version)
      if len(out_of_range) > 0:
        message += " ".join(out_of_range)
      elif len(not_qa) > 0:
        message += " ".join(not_qa)
    report_list.append((mod_name, found, message, version))
  return report_list

def __check_version(mod_name, min_ver, qa_ver, max_ver, found, version):
  """
    Checks that the version found is new enough, and also if it matches the
    tested version
    @In, mod_name, string, module name
    @In, min_ver, string, minimum version
    @In, qa_ver, string, tested version
    @In, max_ver, string, maximum allowed version
    @In, found, bool, if true module was found
    @In, version, string, found version
    @Out, result, tuple, returns (missing, out_of_range, not_qa)
  """
  missing = []
  out_of_range = []
  not_qa = []
  if not found:
    missing.append(mod_name)
  elif distutils.version.LooseVersion(version) < distutils.version.LooseVersion(min_ver):
    out_of_range.append(mod_name+" should be at least version "+min_ver+" but is "+version)
  elif max_ver is not None and distutils.version.LooseVersion(version) > \
       distutils.version.LooseVersion(max_ver):
    out_of_range.append(mod_name+" should not be more than version "+max_ver+" but is "+version)
  else:
    try:
      if distutils.version.StrictVersion(version) != distutils.version.StrictVersion(qa_ver):
        not_qa.append(mod_name + " has version " + version + " but tested version is " + qa_ver)
    except ValueError:
      not_qa.append(mod_name + " has version " + version +
                    " but tested version is " + qa_ver +
                    " and unable to parse version")
  return missing, out_of_range, not_qa

def check_for_missing_modules(subprocess_check=True):
  """
  Looks for a list of modules, and the version numbers.
  returns (missing, out_of_range, not_qa) where if they are all found is
  ([], [], []), but if they are missing or too old or too new  will put a
  error message in the result for each missing or too old module or not on
  the quality assurance version.
  @In, subprocess_check, bool, if true use a subprocess to check
  @Out, result, tuple, returns (missing, out_of_range, not_qa)
  """
  missing = []
  out_of_range = []
  not_qa = []
  for mod_name, find_ver, min_ver, qa_ver, max_ver in modules_to_try:
    if subprocess_check:
      found, _, version = module_report(mod_name, find_ver)
    else:
      found, _, version = __import_report(mod_name)
    module_missing, module_out_of_range, module_not_qa = \
      __check_version(mod_name, min_ver, qa_ver, max_ver, found, version)
    missing.extend(module_missing)
    out_of_range.extend(module_out_of_range)
    not_qa.extend(module_not_qa)
  return missing, out_of_range, not_qa

def __conda_string(include_optionals=False, op_sys=None):
  """
    Generates a string with version ids that can be passed to conda.
    returns s, string, a list of packages for conda to install
    @ In, include_optionals, bool, optional, if True then add optional testing
       libraries to install list
    @ In, op_sys, string, optional, if included then parse conda list according to request
    @ Out, conda_s, str, message that could be pasted into a bash command to install libraries
  """
  conda_s = ""
  lib_list = __condaList + ([] if not include_optionals else __condaOptional)
  if op_sys is not None:
    lib_list = parse_conda_for_os(lib_list, op_sys)
  for name, version in lib_list:
    # no conda-forge
    if len(version) == 0:
      conda_s += name+" "
    else:
      conda_s += name+"="+version+" "
  return conda_s

def __conda_forge_string(op_sys=None):
  """
    Generates a string with version ids that can be passed to conda (conda-forge).
    returns cfs, string, a list of packages for conda to install
    @ In, op_sys, string, optional, if included then parse conda list according to request
    @ Out, cfs, str, message that could be pasted into a bash command to install libraries
  """
  cfs = ""
  lib_list = __condaForgeList
  if op_sys is not None:
    lib_list = parse_conda_for_os(lib_list, op_sys)
  for name, version in lib_list:
    # conda-forge
    if len(version) == 0:
      cfs += name+" "
    else:
      cfs += name+"="+version+" "
  return cfs

def parse_conda_for_os(libs, op_sys):
  """
    Modifies the list of Conda for a particular operating system.
    @ In, libs, list, list of libraries as (lib,version)
    @ In, op_sys, string, name of operating system (mac, windows, linux)
    @ Out, libs, updated libs list
  """
  if op_sys == 'windows':
    pass # nothing special to do currently
  elif op_sys == 'mac':
    pass # nothing special to do currently
  elif op_sys == 'linux':
    # add noMKL libraries to prevent Intel crash errors
    libs.append(('nomkl', ''))
    libs.append(('numexpr', ''))
  return libs

if __name__ == '__main__':
  # allow the operating system to be specified
  op_sys_arg = None
  condaForge = False
  if '--windows' in sys.argv:
    op_sys_arg = 'windows'
  elif '--mac' in sys.argv:
    op_sys_arg = 'mac'
  elif '--linux' in sys.argv:
    op_sys_arg = 'linux'
  if '--conda-forge' in sys.argv:
    # just install command is generated
    condaForge = True
  # check for environemnt definition of raven libs
  libName = os.getenv('RAVEN_LIBS_NAME', 'raven_libraries')
  # what did the caller ask to do?
  if '--conda-create' in sys.argv and not condaForge:
    print("conda create --name {} -y ".format(libName), end="")
    print(__conda_string(include_optionals=('--optional' in sys.argv), op_sys=op_sys_arg))
  elif '--conda-install' in sys.argv:
    print("conda install --name {} -y ".format(libName), end=" ")
    if not condaForge:
      print(__conda_string(include_optionals=('--optional' in sys.argv), op_sys=op_sys_arg))
    else:
      print("-c conda-forge "+ __conda_forge_string(op_sys=op_sys_arg))
  elif '--pip-install' in sys.argv:
    print("pip3 install", end=" ")
    for k, qa_version in __pipList:
      if len(qa_version.strip()) > 0:
        print(k+"=="+qa_version, end=" ")
      else:
        print(k, end=" ")
    print()
  elif '--manual-list' in sys.argv:
    print('\\begin{itemize}')
    for k, qa_version in __condaList+__condaForgeList+__condaOptional:
      print("  \\item", k+ (("-"+qa_version) if len(qa_version.strip()) > 0 else ""))
    print("\\end{itemize}")
