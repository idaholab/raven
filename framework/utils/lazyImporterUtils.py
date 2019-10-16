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
import warnings
warnings.simplefilter('default',DeprecationWarning)
#----- end python 2 - 3 compatibility

#External Modules------------------------------------------------------------------------------------
import lazy_import
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
#Internal Modules End--------------------------------------------------------------------------------

def import_module_lazy(moduleString):
  """
    This method is aimed to import a module with lazy_import
    @ In, moduleString, str, the module to import (e.g. numpy or scipy.stats, etc.)
    @ Out, mod, Object, the imported module (lazy)
  """
  return lazy_import.lazy_module(moduleString.strip())

def import_collable_lazy(collableString):
  """
    This method is aimed to import a collable method or attribute
    within a module that needs to be lazy imported
    @ In, moduleString, str, the collable to point (e.g. numpy.arange, numpy.ndarray, etc.)
    @ Out, callable, Object_Pointer, the collable (lazy)
  """
  return lazy_import.lazy_callable(collableString.strip())
