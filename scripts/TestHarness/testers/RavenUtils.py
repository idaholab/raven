import os
import subprocess

def inPython3():
  return os.environ.get("CHECK_PYTHON3","0") == "1"

def checkForMissingModules():
  missing = []
  to_try = ["numpy","h5py","scipy","sklearn","matplotlib"]
  for i in to_try:
    if inPython3():
      result = subprocess.call(['python3','-c','import '+i])
    else:
      result = subprocess.call(['python','-c','import '+i])
    if result != 0:
      missing.append(i)
  return missing
