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
Created on August 30, 2018

@author: alfoa

This module is in charge to list and load with a Lazy import strategy
all the heavy RAVEN dependencies. In this way, the modules are loaded
only when they are actually used for the first time. This reduces
the start up time of a good amount
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#External Modules------------------------------------------------------------------------------------
import lazy_import
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
#Internal Modules End--------------------------------------------------------------------------------

# Lazy import of all the main dependencies - BEGIN
print(lazy_import.lazy_module("sklearn"))
lazy_import.lazy_module("statsmodels")
lazy_import.lazy_module("matplotlib")
lazy_import.lazy_module("cloudpickle")
lazy_import.lazy_module("tensorflow")
# Lazy import of all the main dependencies - END



