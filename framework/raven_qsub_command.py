#!/usr/bin/env python

import os,subprocess,sys

moduleCommand = False
for possibleModuleCommand in ["/apps/local/modules/bin/modulecmd","/usr/bin/modulecmd"]:
  if os.path.exists(possibleModuleCommand):
    moduleCommand = possibleModuleCommand
    break

if not moduleCommand:
  sys.stderr.write("Could not find a modulecmd")
  sys.exit(-1)

availModules = subprocess.Popen([moduleCommand,"python","avail"],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[1]

if "raven-devel-gcc" in availModules:
  ravenAndPbsEval = subprocess.Popen([moduleCommand,"python","load","pbs","raven-devel-gcc"],stdout=subprocess.PIPE).communicate()[0]
  exec(ravenAndPbsEval)
else:
  pythonAndPbsEval = subprocess.Popen([moduleCommand,"python","load","pbs","python/2.7"],stdout=subprocess.PIPE).communicate()[0]
  exec(pythonAndPbsEval)

  #sys.stdout.write(str(subprocess.Popen(["env"],stdout=subprocess.PIPE).communicate()[0]))

  newModuleFiles = '/apps/projects/moose/modulefiles'
  if "MODULEPATH" in os.environ:
    oldModulepath = os.environ["MODULEPATH"]
    if newModuleFiles not in oldModulepath:
      os.environ["MODULEPATH"] = oldModulepath +":"+newModuleFiles
  else:
    os.environ["MODULEPATH"] = newModuleFiles

  mooseDevAndPython3Eval = subprocess.Popen([moduleCommand,"python","load","moose-dev-gcc","python/3.2"],stdout=subprocess.PIPE).communicate()[0]
  exec(mooseDevAndPython3Eval)

  os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.path.join(os.path.expanduser("~"),"raven_libs","pylibs","lib","python2.7","site-packages")


if "PBS_O_WORKDIR" in os.environ:
  os.chdir(os.environ["PBS_O_WORKDIR"])

sys.stdout.write(os.environ["COMMAND"])
subprocess.call(os.environ["COMMAND"],shell=True)
