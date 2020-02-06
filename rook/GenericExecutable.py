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
from Tester import Tester

class GenericExecutable(Tester):
  """
  A generic executable test interface.
  """

  @staticmethod
  def get_valid_params():
    """
      Return a list of valid parameters and their descriptions for this type
      of test.
      @ In, None
      @ Out, params, _ValidParameters, the parameters for this class.
    """
    params = Tester.get_valid_params()
    params.add_required_param('executable', "The executable to use")
    params.add_param('parameters', '', "arguments to the executable")
    return params

  def get_command(self):
    """
      Return the command this test will run.
      @ In, None
      @ Out, get_command, string, command to run
    """
    return self.specs["executable"]+" "+self.specs["parameters"]

  def __init__(self, name, params):
    """
        Constructor that will setup this test with a name and a list of
        parameters.
        @ In, name: the name of this test case.
        @ In, params, a dictionary of parameters and their values to use.
    """
    Tester.__init__(self, name, params)

  def process_results(self, _):
    """ Handle the results of test case.
        @ In, ignored, string, the output from the test case.
        @ Out, None
    """
    #If the exit code != 0 then check_exit_code will fail the test.
    self.set_success()
