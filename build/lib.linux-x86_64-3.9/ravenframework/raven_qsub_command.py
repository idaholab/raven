#!/usr/bin/env python
# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Created on November 13, 2013

@author: cogljj

This module is used to employ QSUB commands when the PBS protocol is available
"""
from __future__ import division, print_function , unicode_literals, absolute_import

#External Modules------------------------------------------------------------------------------------
import os,subprocess,sys
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
#Internal Modules End--------------------------------------------------------------------------------


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

  mooseDevAndPython3Eval = subprocess.Popen([moduleCommand,"python","load","use.moose","moose-dev-gcc","python/3.2"],stdout=subprocess.PIPE).communicate()[0]
  exec(mooseDevAndPython3Eval)
  os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.path.join(os.path.expanduser("~"),"raven_libs","pylibs","lib","python2.7","site-packages")

if "PBS_O_WORKDIR" in os.environ:
  os.chdir(os.environ["PBS_O_WORKDIR"])

if "COMMAND" in os.environ:
  sys.stdout.write(os.environ["COMMAND"])
  subprocess.call(os.environ["COMMAND"],shell=True)
