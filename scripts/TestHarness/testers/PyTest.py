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
  Tester that drives pytest collections within the RAVEN TestHarness.
"""
from __future__ import absolute_import
import shlex

from RavenPython import RavenPython


class PyTest(RavenPython):
  """
    Executes pytest for the provided targets.
  """

  @staticmethod
  def get_valid_params():
    """
      Return allowed parameters for the PyTest tester.
    """
    params = RavenPython.get_valid_params()
    params.add_param('pytest_args', '', "Additional command line arguments passed to pytest.")
    return params

  def __init__(self, name, params):
    """
      Constructor.
      @ In, name, string, test name
      @ In, params, dict, tester parameters
    """
    RavenPython.__init__(self, name, params)
    self._pytest_args = self.specs.get('pytest_args', '')

  def check_runnable(self):
    """
      Verify pytest is importable in addition to the standard RavenPython checks.
    """
    if not RavenPython.check_runnable(self):
      return False
    try:
      import pytest  # pylint: disable=unused-import, import-outside-toplevel
    except ImportError:
      self.set_skip('skipped (Unable to import pytest)')
      return False
    return True

  def get_command(self):
    """
      Construct the command line that invokes pytest as a module.
    """
    if (command := self._get_test_command()) is not None:
      pieces = shlex.split(command)
    else:
      pieces = [self._get_python_command(), "-m", "pytest"]

    if self._pytest_args:
      pieces.extend(shlex.split(self._pytest_args))

    if self.specs['input']:
      pieces.extend(shlex.split(self.specs['input']))

    return ' '.join(shlex.quote(part) for part in pieces)
