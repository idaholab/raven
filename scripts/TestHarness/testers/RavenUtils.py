import os
import subprocess

def inPython3():
  """returns true if raven should be using python3
  """
  return os.environ.get("CHECK_PYTHON3","0") == "1"

modules_to_try = [("numpy",'numpy.version.version',"1.7"),
                  ("h5py",'h5py.__version__','1.8.12'),
                  ("scipy",'scipy.__version__',"0.12"),
                  ("sklearn",'sklearn.__version__',"0.14"),
                  ("matplotlib",'matplotlib.__version__',"1.3")]

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
  for i,fv,ev in modules_to_try:
    found, message, version = moduleReport(i,fv)
    if found:
      missing, tooOld = checkForMissingModule(i,fv,ev)
    if len(tooOld) > 0:
      message += " ".join(tooOld)
    report_list.append((i,found,message, version))
  return report_list

def checkForMissingModule(i,fv,ev):
  """Checks to see if python can import the module
  it uses subprocess.call to try and import it.
  i: the module to try importing
  fv: a function to try and find the version
  ev: expected version of the module as a string.  If less than, will
  stop because it is too old.
  returns (missing, tooOld) where if they are found is ([], []), but
  if they are missing or too old will put a error message in the result
  """
  missing = []
  tooOld = []
  if len(fv) > 0:
    check = ';import sys; import distutils.version; sys.exit(not distutils.version.LooseVersion('+fv+') >= distutils.version.LooseVersion("'+ev+'"))'
  else:
    check = ''
  if inPython3():
    python = 'python3'
  else:
    python = 'python'

  suppressedOutput = open(os.devnull, 'w')
  result = subprocess.call([python,'-c','import '+i],stdout=suppressedOutput,stderr=suppressedOutput)

  if result != 0:
    missing.append(i)
  else:
    result = subprocess.call([python,'-c','import '+i+check],stdout=suppressedOutput,stderr=suppressedOutput)
    if result != 0:
      tooOld.append(i+" should be at least version "+ev)
  suppressedOutput.close()
  return missing, tooOld


def checkForMissingModules():
  """
  Looks for a list of modules, and the version numbers.
  returns (missing, tooOld) where if they are all found is ([], []), but
  if they are missing or too old will put a error message in the result for
  each missing or too old module.
  """
  missing = []
  tooOld = []
  for i,fv,ev in modules_to_try:
    moduleMissing, moduleTooOld = checkForMissingModule(i, fv, ev)
    missing.extend(moduleMissing)
    tooOld.extend(moduleTooOld)
  return missing,tooOld
