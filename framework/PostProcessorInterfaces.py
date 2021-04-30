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
Created on December 1, 2015

"""
#External Modules------------------------------------------------------------------------------------
import os
from glob import glob
import inspect
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from EntityFactoryBase import EntityFactory
from utils import utils
#Internal Modules End--------------------------------------------------------------------------------

__moduleInterfaceList = []
startDir = os.path.join(os.path.dirname(__file__),'PostProcessorFunctions')
for dirr,_,_ in os.walk(startDir):
  __moduleInterfaceList.extend(glob(os.path.join(dirr,"*.py")))
  utils.add_path(dirr)
__moduleImportedList = []


factory = EntityFactory('InterfacedPostProcessor')
for moduleIndex in range(len(__moduleInterfaceList)):
  if 'class' in open(__moduleInterfaceList[moduleIndex]).read():
    __moduleImportedList.append(utils.importFromPath(__moduleInterfaceList[moduleIndex],False))
    for key,modClass in inspect.getmembers(__moduleImportedList[-1], inspect.isclass):
      # in this way we can get all the class methods
      classMethods = [method for method in dir(modClass) if callable(getattr(modClass, method))]
      if 'run' in classMethods:
        factory.registerType(key, modClass)

def interfaceClasses():
  """
    This returns the classes available
    @ In, None
    @ Out, interfaceClasses, list of classes available
  """
  return list(factory._registeredTypes.values())
