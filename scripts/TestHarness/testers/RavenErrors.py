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
This tests for expected errors in a program
"""
import os
import subprocess
import platform
from Tester import Tester
import RavenUtils

fileDir = os.path.dirname(os.path.realpath(__file__))
raven = os.path.abspath(os.path.join(fileDir, '..', '..', '..', 'framework',
                                     'Driver.py'))

class RavenErrors(Tester):
  """
  This class tests if the expected error messages are generated or not.
  """
  @staticmethod
  def get_valid_params():
    """
      This method add defines the valid parameters for the tester.
      The expected error message shuld be unique...
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Tester.get_valid_params()
    params.add_required_param('input', "The input file to use for this test.")
    params.add_required_param('expect_err',
                              "All or part of the expected error message (unique keyword)")
    params.add_param('required_executable', '', 'Skip test if this executable is not found')
    params.add_param('required_libraries', '', 'Skip test if any of these libraries are not found')
    params.add_param('skip_if_env', '', 'Skip test if this environmental variable is defined')
    params.add_param('test_interface_only', 'False',
                     'Test the interface only (without running the driven code')
    params.add_param('python3_only', False, 'if true, then only use with Python3')
    return params

  def get_command(self):
    """
      This method returns the command to execute for the test
      @ In, None
      @ Out, getCommand, string, the command to run
    """
    ravenflag = ''
    if self.specs['test_interface_only'].lower() == 'true':
      ravenflag = 'interfaceCheck '
    return ' '.join([self._get_python_command(), raven, ravenflag,
                     self.specs["input"]])


  def __init__(self, name, params):
    """
      Initializer for the class. Takes a String name and a dictionary params
      @ In, name, string, name of the test.
      @ In, params, dictionary, parameters for the class
      @ Out, None.
    """
    Tester.__init__(self, name, params)
    self.required_libraries = self.specs['required_libraries'].split(' ')  \
      if len(self.specs['required_libraries']) > 0 else []
    self.required_executable = self.specs['required_executable']
    self.required_executable = self.required_executable.replace("%METHOD%",
                                                                os.environ.get("METHOD", "opt"))
    self.specs['scale_refine'] = False

  def check_runnable(self):
    """
      This method checks if the the test is runnable within the current settings
      @ In, None
      @ Out, check_runnable, boolean, If True this test can run.
    """
    missing, too_old, _ = RavenUtils.check_for_missing_modules()
    if len(missing) > 0:
      self.set_skip('skipped (Missing python modules: '+" ".join(missing)+
                    " PYTHONPATH="+os.environ.get("PYTHONPATH", "")+')')
      return False
    if len(too_old) > 0:
      self.set_skip('skipped (Old version python modules: '+" ".join(too_old)+
                    " PYTHONPATH="+os.environ.get("PYTHONPATH", "")+')')
      return False
    for lib in self.required_libraries:
      if platform.system() == 'Windows':
        lib += '.pyd'
      else:
        lib += '.so'
      if not os.path.exists(lib):
        self.set_skip('skipped (Missing library: "'+lib+'")')
        return False
    if self.specs['python3_only'] and not RavenUtils.in_python_3():
      self.set_skip('Python 3 only')
      return False
    if len(self.required_executable) > 0 and \
       not os.path.exists(self.required_executable):
      self.set_skip('skipped (Missing executable: "'+self.required_executable+'")')
      return False
    try:
      if len(self.required_executable) > 0 and \
         subprocess.call([self.required_executable], stdout=subprocess.PIPE) != 0:
        self.set_skip('skipped (Failing executable: "'+self.required_executable+'")')
        return False
    except Exception:
      self.set_skip('skipped (Error when trying executable: "'+self.required_executable+'")')
      return False
    if len(self.specs['skip_if_env']) > 0:
      env_var = self.specs['skip_if_env']
      if env_var in os.environ:
        self.set_skip('skipped (found environmental variable "'+env_var+'")')
        return False
    return True

  def process_results(self, output):
    """
      This method processes results.
      It checks if the expected error messgae keyword exists in the output stream.
      @ In, output, string, output of the run.
      @ Out, None
    """
    for line in output.split('\n'):
      if self.specs['expect_err'] in line:
        self.set_success()
        return
    self.set_fail('The expected Error: ' +self.specs['expect_err']+' is not raised!')

  def check_exit_code(self, _):
    """
      Allow any exit code (but this could be extended to have an expected exit
      code in the parameters at some point)
      @ In, exit_code, int, the exit code of the test command.
      @ Out, check_exit_code, bool, always True since errors are expected
    """
    return True
