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
This runs tests written in python that use the pytest module.
"""

from RavenPython import RavenPython

try:
  import pytest
  pytest_found = True
except ImportError:
  pytest_found = False

class PyTest(RavenPython):
  """
  This class runs a pytest module through rook. The test passes when pytest exits with 0,
  i.e. every collected test passed; a failing, erroring, or empty (no tests collected)
  module fails the test.
  """

  @staticmethod
  def get_valid_params():
    """
      Return a list of valid parameters and their descriptions for this type
      of test.
      @ In, None
      @ Out, params, _ValidParameters, the parameters for this class.
    """
    params = RavenPython.get_valid_params()
    params.add_param('pytest_args', '', "Arguments to the pytest module")
    return params

  def __init__(self, name, params):
    """
      Initializer for the class. Takes a String name and a dictionary params
      @ In, name, string, name of the test.
      @ In, params, dictionary, parameters for the class
      @ Out, None.
    """
    RavenPython.__init__(self, name, params)

  def check_runnable(self):
    """
      Checks if this test can be run.
      @ In, None
      @ Out, check_runnable, boolean, If True can run this test.
    """
    if not pytest_found:
      self.set_skip('skipped (required pytest module is not found)')
      return False

    return RavenPython.check_runnable(self)

  def get_command(self):
    """
      returns the command used by this tester.
      @ In, None
      @ Out, get_command, string, command to run.
    """
    # -p no:cacheprovider keeps pytest from writing .pytest_cache into the test directory
    args = ['-m pytest -p no:cacheprovider', self.specs["pytest_args"], self.specs["input"]]
    # If the test command has been specified, use it
    if (command := self._get_test_command()) is not None:
      return ' '.join([command] + args)

    # Otherwise, if the python command has been specified, use it
    if len(self.specs["python_command"]) == 0:
      pythonCommand = self._get_python_command()
    else:
      pythonCommand = self.specs["python_command"]
    return ' '.join([pythonCommand] + args)
