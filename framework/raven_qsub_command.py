#!/usr/bin/env python

import os,subprocess,sys

module_command = False
for possible_module_command in ["/apps/local/modules/bin/modulecmd","/usr/bin/modulecmd"]:
  if os.path.exists(possible_module_command):
    module_command = possible_module_command
    break

if not module_command:
  sys.stderr.write("Could not find a modulecmd")
  sys.exit(-1)

avail_modules = subprocess.Popen([module_command,"python","avail"],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[1]

if "raven-devel-gcc" in avail_modules:
  raven_and_pbs_eval = subprocess.Popen([module_command,"python","load","pbs","raven-devel-gcc"],stdout=subprocess.PIPE).communicate()[0]
  exec(raven_and_pbs_eval)
else:
  python_and_pbs_eval = subprocess.Popen([module_command,"python","load","pbs","python/2.7"],stdout=subprocess.PIPE).communicate()[0]
  exec(python_and_pbs_eval)

  #sys.stdout.write(str(subprocess.Popen(["env"],stdout=subprocess.PIPE).communicate()[0]))

  new_module_files = '/apps/projects/moose/modulefiles'
  if "MODULEPATH" in os.environ:
    old_modulepath = os.environ["MODULEPATH"]
    if new_module_files not in old_modulepath:
      os.environ["MODULEPATH"] = old_modulepath +":"+new_module_files
  else:
    os.environ["MODULEPATH"] = new_module_files

  moose_dev_and_python3_eval = subprocess.Popen([module_command,"python","load","use.moose","moose-dev-gcc","python/3.2"],stdout=subprocess.PIPE).communicate()[0]
  exec(moose_dev_and_python3_eval)

  os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.path.join(os.path.expanduser("~"),"raven_libs","pylibs","lib","python2.7","site-packages")


if "PBS_O_WORKDIR" in os.environ:
  os.chdir(os.environ["PBS_O_WORKDIR"])

sys.stdout.write(os.environ["COMMAND"])
subprocess.call(os.environ["COMMAND"],shell=True)
