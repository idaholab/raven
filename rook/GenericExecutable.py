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
Tests by running an executable.
"""
import os
import subprocess
from Tester import Tester

class GenericExecutable(Tester):
  """ A generic executable test interface. """

  @staticmethod
  def get_valid_params():
    """ Return a list of valid parameters and their descriptions for this type
        of test.
    """
    params = Tester.get_valid_params()
    params.add_required_param('executable', "The executable to use")
    params.add_param('parameters', '', "arguments to the executable")
    return params

  def get_command(self):
    """ Return the command this test will run. """
    return self.specs["executable"]+" "+self.specs["parameters"]

  def __init__(self, name, params):
    """ Constructor that will setup this test with a name and a list of
        parameters.
        @ In, name: the name of this test case.
        @ In, params, a dictionary of parameters and their values to use.
    """
    Tester.__init__(self, name, params)

  def check_runnable(self):
    """ Checks if a test case is capable of being run on the current system. """
    return True

  def process_results(self, output):
    """ Handle the results of test case.
        @ In, output: the output from the test case.
        @ Out: a tuple with the error return code and the output passed in.
    """
    if self.results.exit_code != 0:
      self.set_status(str(self.results.exit_code), self.bucket_fail)
      return output
    self.set_success()
    return output
