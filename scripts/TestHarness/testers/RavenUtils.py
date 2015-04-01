import os
import subprocess

def inPython3():
  return os.environ.get("CHECK_PYTHON3","0") == "1"

def checkForMissingModules():
  missing = []
  too_old = []
  to_try = [("numpy",'numpy.version.version',"1.7"),
            ("h5py",'',''),
            ("scipy",'scipy.__version__',"0.12"),
            ("sklearn",'sklearn.__version__',"0.14"),
            ("matplotlib",'matplotlib.__version__',"1.3")]
  for i,fv,ev in to_try:
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
  return missing,too_old
