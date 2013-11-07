#!/usr/bin/env python

import os,subprocess,sys

python_and_pbs_eval = subprocess.Popen(["/apps/local/modules/bin/modulecmd","python","load","pbs","python/2.7"],stdout=subprocess.PIPE).communicate()[0]
exec(python_and_pbs_eval)

#sys.stdout.write(str(subprocess.Popen(["env"],stdout=subprocess.PIPE).communicate()[0]))

if "PBS_O_WORKDIR" in os.environ:
  os.chdir(os.environ["PBS_O_WORKDIR"])

new_module_files = '/apps/projects/moose/modulefiles'
if "MODULEPATH" in os.environ:
  old_modulepath = os.environ["MODULEPATH"]
  if new_module_files not in old_modulepath:
    os.environ["MODULEPATH"] = old_modulepath +":"+new_module_files
else:
  os.environ["MODULEPATH"] = new_module_files

moose_dev_and_python3_eval = subprocess.Popen(["/apps/local/modules/bin/modulecmd","python","load","moose-dev-gcc","python/3.2"],stdout=subprocess.PIPE).communicate()[0]
exec(moose_dev_and_python3_eval)

os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.path.join(os.path.expanduser("~"),"raven_libs","pylibs","lib","python2.7","site-packages")

sys.stdout.write(os.environ["COMMAND"])
subprocess.call(os.environ["COMMAND"],shell=True)
