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
 This module is aimed to provide APIs for the lazy_import module
 Most of the imports of heavy dependencies should be performed using
 the utilities in this module
 created on 09/13/2019
 @author: alfoa
"""
#----- python 2 - 3 compatibility
from __future__ import division, print_function, absolute_import
#----- end python 2 - 3 compatibility

#External Modules------------------------------------------------------------------------------------
from ..contrib.lazy import lazy_loader
from importlib import util as imutil
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
#Internal Modules End--------------------------------------------------------------------------------

# filled by isLibAvail: store if libraries are available or not e.g. {'numpy':True/False, etc.}.
__moduleAvailability = {}
def importModuleLazy(moduleString, parentGlobals=None):
  """
    This method is aimed to import a module with lazy_import
    @ In, moduleString, str, the module to import (e.g. numpy or scipy.stats, etc.)
    @ In, parentGlobals, dict, the globals to put the name in (should be a call to
       globals() )
    @ Out, mod, Object, the imported module (lazy)
  """
  name = moduleString.strip()
  if parentGlobals is None:
    parentGlobals = globals()
  return lazy_loader.LazyLoader(name, parentGlobals, name)

def importModuleLazyRenamed(localName, parentGlobals, name):
  """
    This method is aimed to import a module with lazy_import
    @ In, localName, str, the name to use for the module
    @ In, parentGlobals, dict, the globals to put the name in (should be a call to
       globals() )
    @ In, name, str, the module to import (e.g. numpy or scipy.stats, etc.)
    @ Out, mod, Object, the imported module (lazy)
  """
  return lazy_loader.LazyLoader(localName, parentGlobals, name)

def isLibAvail(moduleString):
  """
    This method is aimed to check if a certain library is available in the system
    @ In, moduleString, str, the module to look for (e.g.numpy, scipy, etc.)
    @ Out, isLibAvail, bool, is it available?
  """
  global __moduleAvailability
  if moduleString not in __moduleAvailability:
    __moduleAvailability[moduleString] = False if imutil.find_spec(moduleString) is None else True
  return __moduleAvailability.get(moduleString)
