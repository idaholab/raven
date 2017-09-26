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
from Tester import Tester
from CSVDiffer import CSVDiffer
import RavenUtils
import os
import subprocess
import platform

fileDir = os.path.dirname(os.path.realpath(__file__))
raven = os.path.abspath(os.path.join(fileDir,'..','..','..','framework','Driver.py'))

class RavenErrors(Tester):
  """
  This class tests if the expected error messages are generated or not.
  """
  @staticmethod
  def validParams():
    """This method add defines the valid parameters for the tester. The expected error message shuld be unique..."""
    params = Tester.validParams()
    params.addRequiredParam('input',"The input file to use for this test.")
    params.addRequiredParam('expect_err',"All or part of the expected error message (unique keyword)")
    params.addParam('required_executable','','Skip test if this executable is not found')
    params.addParam('required_libraries','','Skip test if any of these libraries are not found')
    params.addParam('skip_if_env','','Skip test if this environmental variable is defined')
    params.addParam('test_interface_only','False','Test the interface only (without running the driven code')
    return params

  def getCommand(self, options):
    """
      This method returns the command to execute for the test
      @ In, options, argparse.Namespace, functionally a dictionary of options from the input
      @ Out, getCommand, string, the command to run
    """
    ravenflag = ''
    if self.specs['test_interface_only'].lower() == 'true': ravenflag = 'interfaceCheck '
    if RavenUtils.inPython3():
      return ' '.join(["python3",raven,ravenflag,self.specs["input"]])
    else:
      return ' '.join(["python",raven,ravenflag,self.specs["input"]])


  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.required_libraries = self.specs['required_libraries'].split(' ')  if len(self.specs['required_libraries']) > 0 else []
    self.required_executable = self.specs['required_executable']
    self.required_executable = self.required_executable.replace("%METHOD%",os.environ.get("METHOD","opt"))
    self.specs['scale_refine'] = False

  def checkRunnable(self, option):
    """This method checks if the the test is runnable within the current settings"""
    missing,too_old,notQA = RavenUtils.checkForMissingModules()
    if len(missing) > 0:
      self.setStatus('skipped (Missing python modules: '+" ".join(missing)+
                     " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')',
                     self.bucket_skip)
      return False
    if len(too_old) > 0:
      self.setStatus('skipped (Old version python modules: '+" ".join(too_old)+
                     " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')',
                     self.bucket_skip)
      return False
    for lib in self.required_libraries:
      if platform.system() == 'Windows':
        lib += '.pyd'
      else:
        lib += '.so'
      if not os.path.exists(lib):
        self.setStatus('skipped (Missing library: "'+lib+'")', self.bucket_skip)
        return False
    if len(self.required_executable) > 0 and \
       not os.path.exists(self.required_executable):
      self.setStatus('skipped (Missing executable: "'+self.required_executable+'")',
                     self.bucket_skip)
      return False
    try:
      if len(self.required_executable) > 0 and \
         subprocess.call([self.required_executable],stdout=subprocess.PIPE) != 0:
        self.setStatus('skipped (Failing executable: "'+self.required_executable+'")',
                       self.bucket_skip)
        return False
    except:
      self.setStatus('skipped (Error when trying executable: "'+self.required_executable+'")',
                     self.bucket_skip)
      return False
    if len(self.specs['skip_if_env']) > 0:
      env_var = self.specs['skip_if_env']
      if env_var in os.environ:
        self.setStatus('skipped (found environmental variable "'+env_var+'")',
                       self.bucket_skip)
        return False
    return True

  def processResults(self, moose_dir, options, output):
    """This method processes results. It checks if the expected error messgae keyword exists in the output stream."""
    for line in output.split('\n'):
      if self.specs['expect_err'] in line:
        self.setStatus(self.success_message, self.bucket_success)
        return output
    self.setStatus('The expected Error: ' +self.specs['expect_err']+' is not raised!', self.bucket_fail)
    return output

