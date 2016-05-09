from __future__ import print_function
import os, sys
import subprocess
import distutils.version


def inPython3():
  """returns true if raven should be using python3
  """
  return os.environ.get("CHECK_PYTHON3","0") == "1"

#This list is made of (module, how to check the version, minimum version,
# quality assurance module)
modules_to_try = [("numpy",'numpy.version.version',"1.8.0","1.8.0"),
                  ("h5py",'h5py.__version__','2.2.1','2.2.1'),
                  ("scipy",'scipy.__version__',"0.13.3","0.13.3"),
                  ("sklearn",'sklearn.__version__',"0.14.1","0.14.1"),
                  ("matplotlib",'matplotlib.__version__',"1.3.1","1.3.1")]

def __lookUpPreferredVersion(name):
  """
    Look up the preferred version in the modules.
    name: string, the name of the module
    returns the version as a string or "" if unknown
  """
  for  i,fv,ev,qa in modules_to_try:
    if name == i:
      return qa
  return ""

__condaList = [("numpy",__lookUpPreferredVersion("numpy")),
              ("h5py",__lookUpPreferredVersion("h5py")),
              ("scipy",__lookUpPreferredVersion("scipy")),
              ("scikit-learn",__lookUpPreferredVersion("sklearn")),
              ("matplotlib",__lookUpPreferredVersion("matplotlib")),
              ("hdf5",""),
              ("swig","")]


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
    output = subprocess.check_output([python,'-c',command])
    if len(version) > 0:
      try:
        command =  'import '+module+';print('+version+')'
        foundVersion = subprocess.check_output([python,'-c',command]).strip()
      except:
        foundVersion = "NA"
    else:
      foundVersion = "NA"
    return (True,output,foundVersion)
  except:
    return (False,'Failed to find module '+module,"NA")

def modulesReport():
  """Return a report on the modules.
  Returns a list of [(module_name,found_boolean,message,version)]
  """
  report_list = []
  for i,fv,ev,qa in modules_to_try:
    found, message, version = moduleReport(i,fv)
    if found:
      missing, tooOld, notQA = __checkVersion(i, ev, qa, found, version)
      if len(tooOld) > 0:
        message += " ".join(tooOld)
      elif len(notQA) > 0:
        message += " ".join(notQA)
    report_list.append((i,found,message, version))
  return report_list

def __checkVersion(i, ev, qa, found, version):
  """
    Checks that the version found is new enough, and also if it matches the
    tested version
    i: string, module name
    ev: string, minimum version
    qa: string, tested version
    found: bool, if true module was found
    version: string, found version
    returns (missing, tooOld, notQA)
  """
  missing = []
  tooOld = []
  notQA = []
  if not found:
    missing.append(i)
  elif distutils.version.LooseVersion(version) < distutils.version.LooseVersion(ev):
    tooOld.append(i+" should be at least version "+ev+" but is "+version)
  elif distutils.version.StrictVersion(version) != distutils.version.StrictVersion(qa):
    notQA.append(i + " has version " + version + " but tested version is " + qa)
  return missing, tooOld, notQA

def checkForMissingModules():
  """
  Looks for a list of modules, and the version numbers.
  returns (missing, tooOld, notQA) where if they are all found is ([], [], []),
  but if they are missing or too old will put a error message in the result for
  each missing or too old module or not on the quality assurance version.
  """
  missing = []
  tooOld = []
  notQA = []
  for i,fv,ev, qa in modules_to_try:
    found, message, version = moduleReport(i, fv)
    moduleMissing, moduleTooOld, moduleNotQA = __checkVersion(i, ev, qa, found, version)
    missing.extend(moduleMissing)
    tooOld.extend(moduleTooOld)
    notQA.extend(moduleNotQA)
    #moduleMissing, moduleTooOld = checkForMissingModule(i, fv, ev)
    #missing.extend(moduleMissing)
    #tooOld.extend(moduleTooOld)
  return missing, tooOld, notQA

if __name__ == '__main__':
  if '--conda-create' in sys.argv:
    print("conda create --name raven_libraries -y ",end="")
    for name, version in __condaList:
      if len(version) == 0:
        print(name,end=" ")
      else:
        print(name+"="+version,end=" ")
    print()
