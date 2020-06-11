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
RavenFramework is a tool to test raven inputs.
"""
from __future__ import absolute_import
import os
import sys
import platform

# need to use RavenFramework as a base
try:
  from RavenFramework import RavenFramework
except ModuleNotFoundError:
  sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '..', '..', '..', '..',
                               'scripts', 'TestHarness', 'testers'))
  from RavenFramework import RavenFramework
  sys.path.pop()

class ExampleIntegration(RavenFramework):
  """
    Example class for integration tests in ExamplePlugin
  """
  def get_command(self):
    """
      Gets command to run this test.
      @ In, None
      @ Out, get_command, string, command to run
    """
    # same as RAVEN, but for demonstration add a little print
    command = RavenFramework.get_command(self)
    command += '&& echo ExamplePlugin Integration Test Complete!'
    return command
