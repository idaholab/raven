import os
import subprocess

def inPython3():
  return os.environ.get("CHECK_PYTHON3","0") == "1"

def checkForMissingModule(i,fv,ev):
  """Checks to see if python can import the module
  it uses subprocess.call to try and import it.
  i: the module to try importing
  fv: a function to try and find the version
  ev: expected version of the module as a string.  If less than, will
  stop because it is too old.
  returns (missing, too_old) where if they are found is ([], []), but
  if they are missing or too old will put a error message in the result
  """
  missing = []
  too_old = []
  if len(fv) > 0:
    check = ';import sys; sys.exit(not '+fv+' >= "'+ev+'")'
  else:
    check = ''
  if inPython3():
    python = 'python3'
  else:
    python = 'python'
  result = subprocess.call([python,'-c','import '+i])
  if result != 0:
    missing.append(i)
  else:
    result = subprocess.call([python,'-c','import '+i+check])
    if result != 0:
      too_old.append(i+" should be at least version "+ev)
  return missing, too_old


def checkForMissingModules():
  """
  Looks for a list of modules, and the version numbers.
  returns (missing, too_old) where if they are all found is ([], []), but
  if they are missing or too old will put a error message in the result for
  each missing or too old module.
  """
  missing = []
  too_old = []
  to_try = [("numpy",'numpy.version.version',"1.7"),
            ("h5py",'',''),
            ("scipy",'scipy.__version__',"0.12"),
            ("sklearn",'sklearn.__version__',"0.14"),
            ("matplotlib",'matplotlib.__version__',"1.3")]
  for i,fv,ev in to_try:
    module_missing, module_too_old = checkForMissingModule(i, fv, ev)
    missing.extend(module_missing)
    too_old.extend(module_too_old)
  return missing,too_old
