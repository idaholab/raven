import os
import subprocess

def inPython3():
  """returns true if raven should be using python3
  """
  return os.environ.get("CHECK_PYTHON3","0") == "1"

modules_to_try = [("numpy",'numpy.version.version',"1.7"),
                  ("h5py",'',''),
                  ("scipy",'scipy.__version__',"0.12"),
                  ("sklearn",'sklearn.__version__',"0.14"),
                  ("matplotlib",'matplotlib.__version__',"1.3")]

def moduleReport(module):
  """Checks if the module exists.
  Returns (found_boolean,message)
  The found_boolean is true if the module is found.
  The message will be the result of print(module)
  """
  if inPython3():
    python = 'python3'
  else:
    python = 'python'
  try:
    command = 'import '+module+';print('+module+')'
    output = subprocess.check_output([python,'-c',command])
    return (True,output)
  except:
    return (False,'Failed to find module '+module)

def modulesReport():
  """Return a report on the modules.
  Returns a list of [(module_name,found_boolean,message)]
  """
  report_list = []
  for i,fv,ev in modules_to_try:
    found, message = moduleReport(i)
    if found:
      missing, too_old = checkForMissingModule(i,fv,ev)
    if len(too_old) > 0:
      message += " ".join(too_old)
    report_list.append((i,found,message))
  return report_list

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
  for i,fv,ev in modules_to_try:
    module_missing, module_too_old = checkForMissingModule(i, fv, ev)
    missing.extend(module_missing)
    too_old.extend(module_too_old)
  return missing,too_old
