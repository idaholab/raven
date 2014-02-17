#!/usr/bin/env python3
import sys, os, subprocess

try:
  if os.environ.get("CHECK_PYTHON3","0") == "1":
    raven_libs_dir = subprocess.check_output("ls -d $HOME/raven_libs/pylibs3/*/python*/site-packages",shell=True)
  else:
    raven_libs_dir = subprocess.check_output("ls -d $HOME/raven_libs/pylibs/*/python*/site-packages",shell=True)
  raven_libs_dir = raven_libs_dir.decode().strip()
  os.environ["PYTHONPATH"] = raven_libs_dir +":"+os.environ.get("PYTHONPATH","")
  sys.path.append(raven_libs_dir)
  #print("PYTHONPATH="+os.environ["PYTHONPATH"])
except:
  pass
  #print("No raven_libs found")


import numpy 
np_include_dir = numpy.get_include()
print(np_include_dir)
