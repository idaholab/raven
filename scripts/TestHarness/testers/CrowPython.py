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
from Tester import Tester
import os, subprocess

class CrowPython(Tester):
  """ A python test interface for Crow """
  try:
    output_swig = subprocess.Popen(["swig","-version"],stdout=subprocess.PIPE,
                                   universal_newlines=True).communicate()[0]
  except OSError:
    output_swig = "Failed"

  has_swig2 = "Version 2.0" in output_swig or "Version 3.0" in output_swig

  @staticmethod
  def validParams():
    """ Return a list of valid parameters and their descriptions for this type
        of test.
    """
    params = Tester.validParams()
    params.add_required_param('input',"The python file to use for this test.")
    if os.environ.get("CHECK_PYTHON3","0") == "1":
      params.add_param('python_command','python3','The command to use to run python')
    else:
      params.add_param('python_command','python','The command to use to run python')
    params.add_param('requires_swig2', False, "Requires swig2 for test")
    return params

  def getCommand(self, options):
    """ Return the command this test will run. """
    return self.specs["python_command"]+" "+self.specs["input"]

  def __init__(self, name, params):
    """ Constructor that will setup this test with a name and a list of
        parameters.
        @ In, name: the name of this test case.
        @ In, params, a dictionary of parameters and their values to use.
    """
    Tester.__init__(self, name, params)
    self.specs['scale_refine'] = False

  def __init__(self, name, params):
    """ Constructor that will setup this test with a name and a list of
        parameters.
        @ In, name: the name of this test case.
        @ In, params, a dictionary of parameters and their values to use.
    """
    Tester.__init__(self, name, params)
    self.specs['scale_refine'] = False

  def checkRunnable(self, option):
    """ Checks if a test case is capable of being run on the current system. """
    if self.specs['requires_swig2'] and not CrowPython.has_swig2:
      self.setStatus('skipped (No swig 2.0 found)', self.bucket_skip)
      return False
    return True

  def processResults(self, moose_dir, options, output):
    """ Handle the results of test case.
        @ In, moose_dir: the root directory where MOOSE resides on the current
                         system.
        @ In, options: options (unused)
        @ In, output: the output from the test case.
        @ Out: a tuple with the error return code and the output passed in.
    """
    if self.results.exit_code != 0:
      self.setStatus(str(self.results.exit_code), self.bucket_fail)
      return output
    self.set_success()
    return output
