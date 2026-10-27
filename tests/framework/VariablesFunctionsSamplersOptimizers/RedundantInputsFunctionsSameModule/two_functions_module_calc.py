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
import importlib.util
import os
import sys

def _loadSource(name, path):
  """
    Loads a module from a file path (replacement for imp.load_source, removed in python 3.12)
    @ In, name, str, the module name
    @ In, path, str, the path of the file
    @ Out, module, module, the loaded module
  """
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module

# the functions below are imported from the respective files
# this is aimed to show how multiple functions can be coded in the same file

os.path.join(os.path.dirname(os.path.abspath(__file__)),'../RedundantInputs/d_calc.py')
d_calc = _loadSource('d_calc', os.path.join(os.path.dirname(os.path.abspath(__file__)),'../RedundantInputs/d_calc.py'))
e_calc = _loadSource('e_calc', os.path.join(os.path.dirname(os.path.abspath(__file__)),'../RedundantInputs/e_calc.py'))

def raven_d_calc(ravenContainer):
  return d_calc.evaluate(ravenContainer)

def raven_e_calc(ravenContainer):
  return e_calc.evaluate(ravenContainer)
