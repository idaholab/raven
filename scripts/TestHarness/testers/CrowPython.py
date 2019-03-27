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
#from util import *
"""
Tests by running a python program.
"""
import subprocess
from Tester import Tester

class CrowPython(Tester):
  """ A python test interface for Crow """
  try:
    output_swig = subprocess.Popen(["swig", "-version"], stdout=subprocess.PIPE,
                                   universal_newlines=True).communicate()[0]
  except OSError:
    output_swig = "Failed"

  has_swig2 = "Version 2.0" in output_swig or "Version 3.0" in output_swig

  @staticmethod
  def get_valid_params():
    """
      Return a list of valid parameters and their descriptions for this type
      of test.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Tester.get_valid_params()
    params.add_required_param('input', "The python file to use for this test.")
    params.add_param('python_command', '', 'The command to use to run python')
    params.add_param('requires_swig2', False, "Requires swig2 for test")
    return params

  def get_command(self):
    """
      Return the command this test will run.
      @ In, None
      @ Out, get_command, string, string command to use.
    """
    if len(self.specs["python_command"]) == 0:
      python_command = self._get_python_command()
    else:
      python_command = self.specs["python_command"]
    return python_command+" "+self.specs["input"]

  def __init__(self, name, params):
    """ Constructor that will setup this test with a name and a list of
        parameters.
        @ In, name: the name of this test case.
        @ In, params, a dictionary of parameters and their values to use.
    """
    Tester.__init__(self, name, params)
    self.specs['scale_refine'] = False

  def check_runnable(self):
    """
      Checks if a test case is capable of being run on the current system.
      @ In, None
      @ Out, check_runnable, boolean, True if this test can run.
    """
    if self.specs['requires_swig2'] and not CrowPython.has_swig2:
      self.set_skip('skipped (No swig 2.0 found)')
      return False
    return True

  def process_results(self, output):
    """ Handle the results of test case.
        @ In, moose_dir: the root directory where MOOSE resides on the current
                         system.
        @ In, options: options (unused)
        @ In, output: the output from the test case.
        @ Out: a tuple with the error return code and the output passed in.
    """
    #check_exit_code fails test if != 0 so passes
    self.set_success()
    return output
