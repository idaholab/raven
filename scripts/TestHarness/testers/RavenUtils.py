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
from __future__ import print_function
import os, sys
import subprocess
import distutils.version
import importlib

def checkVersions():
  """
  Returns true if versions should be checked.
  @Out, checkVersions, bool, if true versions should be checked.
  """
  return not os.environ.get("RAVEN_IGNORE_VERSIONS","0") == "1"

def inPython3():
  """returns true if raven should be using python3
  """
  return os.environ.get("CHECK_PYTHON3","0") == "1"

#This list is made of (module, how to check the version, minimum version,
# quality assurance module version, maximum version)
# Deep learning requires Scikit-Learn version at least 0.18

## working Conda 4.5.4, May 2018
modules_to_try = [("h5py"      ,'h5py.__version__'      ,'2.4.0' ,'2.7.1' ,None   ), # 2.6.0
                  ("numpy"     ,'numpy.__version__'     ,"1.8.0" ,"1.12.1",None   ),
                  ("scipy"     ,'scipy.__version__'     ,"1.1.0" ,"1.1.0" ,None   ),
                  ("sklearn"   ,'sklearn.__version__'   ,"0.19.1","0.19.1",None   ),
                  ("pandas"    ,'pandas.__version__'    ,"0.20.0","0.20.3",None   ),
                  ("xarray"    ,'xarray.__version__'    ,"0.9.5" ,"0.10.3",None   ),
                  ("netCDF4"   ,'netCDF4.__version__'   ,"1.2.3" ,"1.4.0" ,None   ), # 1.2.4
                  ## NOTE there is a known bug in netCDF4 prior to 1.3.1 where having a path length
                  # of exactly 88 characters can create a seg fault.  However, h5py has no new libraries
                  # after 2.7 and is not compatible with hdf5 greater than 1.8.17, while netCDF4 requires
                  # hdf5 of at least 1.10.1.  For now, we avoid using netCDF until we transition from
                  # HDF5 databases and drop them like hot rocks.
                  #
                  #("tensorflow",'tensorflow.__version__',"1.1.0" ,"1.1.0" ,None   ),
                  # On Windows conda, there are no Python 2.7-compatible versions of TensorFlow, although
                  ## these exist on Mac and Linux condas.  Darn.
                  ("statsmodels",'statsmodels.__version__',"0.8.0","0.8.0" ,None   ),
                  ("matplotlib",'matplotlib.__version__'  ,"1.3.1","2.1.1" ,None   )]

optional_test_libraries = [ ('pillow','PIL.__version__',"5.0.0","5.1.0",None) ]

def __lookUpPreferredVersion(name,optional=False):
  """
    Look up the preferred version in the modules.
    @In, name, string, the name of the module
    @Out, result, string, returns the version as a string or "" if unknown
  """
  index = modules_to_try if not optional else optional_test_libraries
  for  i,fv,ev,qa,mv in index:
    if name == i:
      return qa
  return ""
# libraries to install with Conda
__condaList = [("h5py"        ,__lookUpPreferredVersion("h5py"      )),
               ("numpy"       ,__lookUpPreferredVersion("numpy"     )),
               ("scipy"       ,__lookUpPreferredVersion("scipy"     )),
               ("scikit-learn",__lookUpPreferredVersion("sklearn"   )),
               ("pandas"      ,__lookUpPreferredVersion("pandas"    )),
               ("xarray"      ,__lookUpPreferredVersion("xarray"    )),
               ("netcdf4"     ,__lookUpPreferredVersion("netCDF4"   )),
               ("matplotlib"  ,__lookUpPreferredVersion("matplotlib")),
               ("statsmodels" ,__lookUpPreferredVersion("statsmodels")),
               #("tensorflow"  ,__lookUpPreferredVersion("tensorflow")),
               ("python"      ,"2.7"),
               ("hdf5"        ,"1.8.18"),
               ("swig"        ,""),
               ("pylint"      ,""),
               ("coverage"    ,""),
               ("lxml"        ,""),
               #("nomkl"       ,""),
               #("numexpr"     ,"")
               ]
# libraries to install with conda-forge
__condaForgeList = [("pyside",""),
                   ]
# optional conda libraries
__condaOptional = [ ('pillow',__lookUpPreferredVersion("pillow")) ]


__pipList = [("numpy",__lookUpPreferredVersion("numpy")),
             ("h5py",__lookUpPreferredVersion("h5py")),
             ("scipy",__lookUpPreferredVersion("scipy")),
             ("scikit-learn",__lookUpPreferredVersion("sklearn")),
             ("matplotlib",__lookUpPreferredVersion("matplotlib")),
             ("xarray",__lookUpPreferredVersion("xarray")),
             ("netCDF4",__lookUpPreferredVersion("netCDF4")),
             ("statsmodels",__lookUpPreferredVersion("statsmodels")),
             #("tensorflow",__lookUpPreferredVersion("tensorflow")),
             ("pandas",__lookUpPreferredVersion("pandas")) ]

def moduleReport(module,version=''):
  """Checks if the module exists.
  Returns (found_boolean,message,version)
  The found_boolean is true if the module is found.
  The message will be the result of print(module)
  The version is the version number or "NA" if not known.
  """
  if inPython3():
    python = 'python3'
  else:
    python = 'python'
  try:
    command = 'import '+module+';print('+module+')'
    output = subprocess.check_output([python,'-c',command],
                                     universal_newlines=True)
    if len(version) > 0:
      try:
        command =  'import '+module+';print('+version+')'
        foundVersion = subprocess.check_output([python,'-c',command],
                                               universal_newlines=True).strip()
      except:
        foundVersion = "NA"
    else:
      foundVersion = "NA"
    return (True,output,foundVersion)
  except:
    return (False,'Failed to find module '+module,"NA")

def __importReport(module):
  """ Directly checks the module version.
  Returns (found_boolean,message,version)
  The found_boolean is true if the module is found.
  The message will be the result of print(module)
  The version is the version number or "NA" if not known.
  """
  try:
    loaded = importlib.import_module(module)
    foundVersion = loaded.__version__
    output = str(loaded)
    return (True, output, foundVersion)
  except ImportError:
    return (False, 'Failed to find module '+module, "NA")

def modulesReport():
  """Return a report on the modules.
  Returns a list of [(module_name,found_boolean,message,version)]
  """
  report_list = []
  for i,fv,ev,qa,mv in modules_to_try:
    found, message, version = moduleReport(i,fv)
    if found:
      missing, outOfRange, notQA = __checkVersion(i, ev, qa, mv, found, version)
      if len(outOfRange) > 0:
        message += " ".join(outOfRange)
      elif len(notQA) > 0:
        message += " ".join(notQA)
    report_list.append((i,found,message, version))
  return report_list

def __checkVersion(i, ev, qa, mv, found, version):
  """
    Checks that the version found is new enough, and also if it matches the
    tested version
    @In, i, string, module name
    @In, ev, string, minimum version
    @In, qa, string, tested version
    @In, mv, string, maximum allowed version
    @In, found, bool, if true module was found
    @In, version, string, found version
    @Out, result, tuple, returns (missing, outOfRange, notQA)
  """
  missing = []
  outOfRange = []
  notQA = []
  if not found:
    missing.append(i)
  elif distutils.version.LooseVersion(version) < distutils.version.LooseVersion(ev):
    outOfRange.append(i+" should be at least version "+ev+" but is "+version)
  elif mv is not None and distutils.version.LooseVersion(version) > distutils.version.LooseVersion(mv):
    outOfRange.append(i+" should not be more than version "+mv+" but is "+version)
  else:
    try:
      if distutils.version.StrictVersion(version) != distutils.version.StrictVersion(qa):
        notQA.append(i + " has version " + version + " but tested version is " + qa)
    except ValueError:
      notQA.append(i + " has version " + version + " but tested version is " + qa + " and unable to parse version")
  return missing, outOfRange, notQA

def checkForMissingModules(subprocessCheck = True):
  """
  Looks for a list of modules, and the version numbers.
  returns (missing, outOfRange, notQA) where if they are all found is
  ([], [], []), but if they are missing or too old or too new  will put a
  error message in the result for each missing or too old module or not on
  the quality assurance version.
  @In, subprocessCheck, bool, if true use a subprocess to check
  @Out, result, tuple, returns (missing, outOfRange, notQA)
  """
  missing = []
  outOfRange = []
  notQA = []
  for i,fv,ev, qa, mv in modules_to_try:
    if subprocessCheck:
      found, message, version = moduleReport(i, fv)
    else:
      found, message, version = __importReport(i)
    moduleMissing, moduleOutOfRange, moduleNotQA = __checkVersion(i, ev, qa, mv, found, version)
    missing.extend(moduleMissing)
    outOfRange.extend(moduleOutOfRange)
    notQA.extend(moduleNotQA)
  return missing, outOfRange, notQA

def __condaString(includeOptionals=False,opSys=None):
  """
    Generates a string with version ids that can be passed to conda.
    returns s, string, a list of packages for conda to install
    @ In, includeOptionals, bool, optional, if True then add optional testing libraries to install list
    @ In, opSys, string, optional, if included then parse conda list according to request
    @ Out, s, str, message that could be pasted into a bash command to install libraries
  """
  s = ""
  libList = __condaList + ([] if not includeOptionals else __condaOptional)
  if opSys is not None:
    libList = parseCondaForOS(libList,opSys)
  for name, version in libList:
    # no conda-forge
    if len(version) == 0:
      s += name+" "
    else:
      s += name+"="+version+" "
  return s

def __condaForgeString(opSys=None):
  """
    Generates a string with version ids that can be passed to conda (conda-forge).
    returns s, string, a list of packages for conda to install
    @ In, opSys, string, optional, if included then parse conda list according to request
    @ Out, s, str, message that could be pasted into a bash command to install libraries
  """
  s = ""
  libList = __condaForgeList
  if opSys is not None:
    libList = parseCondaForOS(libList,opSys)
  for name, version in libList:
    # conda-forge
    if len(version) == 0:
      s += name+" "
    else:
      s += name+"="+version+" "
  return s

def parseCondaForOS(libs,opSys):
  """
    Modifies the list of Conda for a particular operating system.
    @ In, libs, list, list of libraries as (lib,version)
    @ In, opSys, string, name of operating system (mac, windows, linux)
    @ Out, libs, updated libs list
  """
  if opSys == 'windows':
    pass # nothing special to do currently
  elif opSys == 'mac':
    pass # nothing special to do currently
  elif opSys == 'linux':
    # add noMKL libraries to prevent Intel crash errors
    libs.append( ('nomkl'  ,'') )
    libs.append( ('numexpr','') )
  return libs


if __name__ == '__main__':
  # allow the operating system to be specified
  opSys = None
  condaForge = False
  if '--windows' in sys.argv:
    opSys = 'windows'
  elif '--mac' in sys.argv:
    opSys = 'mac'
  elif '--linux' in sys.argv:
    opSys = 'linux'
  if '--conda-forge' in sys.argv:
    # just install command is generated
    condaForge = True
  if '--py3' in sys.argv:
    pythonIndex = __condaList.index(("python","2.7"))
    __condaList[pythonIndex] = ("python","3")
    __condaForgeList.append(("matplotlib",""))

  # check for environemnt definition of raven libs
  libName = os.getenv('RAVEN_LIBS_NAME','raven_libraries')
  # what did the caller ask to do?
  if '--conda-create' in sys.argv and not condaForge:
    print("conda create --name {} -y ".format(libName), end="")
    print(__condaString(includeOptionals = ('--optional' in sys.argv), opSys = opSys))
  elif '--conda-install' in sys.argv:
    print("conda install --name {} -y ".format(libName), end=" ")
    if not condaForge:
      print(__condaString(includeOptionals = ('--optional' in sys.argv), opSys = opSys))
    else:
      print("-c conda-forge "+ __condaForgeString(opSys = opSys))
  elif '--pip-install' in sys.argv:
    print("pip install",end=" ")
    for i,qa in __pipList:
      print(i+"=="+qa,end=" ")
    print()
  elif '--manual-list' in sys.argv:
    print('\\begin{itemize}')
    for i,qa in __condaList+__condaForgeList+__condaOptional:
      print("  \item",i+ (("-"+qa) if len(qa.strip())>0 else ""))
    print("\end{itemize}")
