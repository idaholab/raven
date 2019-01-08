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
This tests programs by running a python command.
"""
import os
import subprocess
import distutils.version
from Tester import Tester
import RavenUtils

class RavenPython(Tester):
  """
  This class runs a python program to test something.
  """
  try:
    output_swig = subprocess.Popen(["swig", "-version"], stdout=subprocess.PIPE,
                                   universal_newlines=True).communicate()[0]
  except OSError:
    output_swig = "Failed"

  has_swig2 = "Version 2.0" in output_swig or "Version 3.0" in output_swig


  @staticmethod
  def get_valid_params():
    """
      Returns the valid parameters.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Tester.get_valid_params()
    params.add_required_param('input', "The python file to use for this test.")
    params.add_param('output', '', "List of output files that this input should create.")
    if os.environ.get("CHECK_PYTHON3", "0") == "1":
      params.add_param('python_command', 'python3', 'The command to use to run python')
    else:
      params.add_param('python_command', 'python', 'The command to use to run python')
    params.add_param('requires_swig2', False, "Requires swig2 for test")
    params.add_param('required_executable', '', 'Skip test if this executable is not found')
    params.add_param('required_libraries', '', 'Skip test if any of these libraries are not found')
    params.add_param('required_executable_check_flags', '', 'Flags to add to '+
                     'the required executable to make sure it runs without '+
                     'fail when testing its existence on the machine')
    params.add_param('minimum_library_versions', '', 'Skip test if the library'+
                     ' listed is below the supplied version (e.g. '+
                     'minimum_library_versions = \"name1 version1 name2 version2\")')
    params.add_param('python3_only', False, 'if true, then only use with Python3')
    return params

  def prepare(self):
    """
      Copied from RavenFramework since we should still clean out test files
      before running an external tester, though we will not test if they
      are created later (for now), so it may behoove us to not save
      check_files for later use.
      @ In, None
      @ Out, None
    """
    for filename in self.check_files:
      if os.path.exists(filename):
        os.remove(filename)

  def get_command(self):
    """
      returns the command used by this tester.
      @ In, None
      @ Out, get_command, string, command to run.
    """
    return self.specs["python_command"]+" "+self.specs["input"]

  def __init__(self, name, params):
    """
      Initializer for the class. Takes a String name and a dictionary params
      @ In, name, string, name of the test.
      @ In, params, dictionary, parameters for the class
      @ Out, None.
    """
    Tester.__init__(self, name, params)
    self.specs['scale_refine'] = False
    self.required_executable = self.specs['required_executable']
    self.required_executable = self.required_executable.replace("%METHOD%",
                                                                os.environ.get("METHOD", "opt"))
    self.required_libraries = self.specs['required_libraries'].split(' ') \
      if len(self.specs['required_libraries']) > 0 else []
    self.required_executable_check_flags = self.specs['required_executable_check_flags'].split(' ')
    self.minimum_libraries = self.specs['minimum_library_versions'].split(' ')\
      if len(self.specs['minimum_library_versions']) > 0 else []
    if self.specs['output'].strip() != '':
      self.check_files = [os.path.join(self.specs['test_dir'], filename)
                          for filename in self.specs['output'].split(" ")]
    else:
      self.check_files = []

  def check_runnable(self):
    """
      Checks if this test can be run.
      @ In, None
      @ Out, check_runnable, boolean, If True can run this test.
    """
    i = 0
    if len(self.minimum_libraries) % 2:
      self.set_status('skipped (libraries are not matched to versions numbers: '
                      +str(self.minimum_libraries)+')', self.bucket_skip)
      return False
    while i < len(self.minimum_libraries):
      library_name = self.minimum_libraries[i]
      library_version = self.minimum_libraries[i+1]
      found, _, actual_version = RavenUtils.module_report(library_name, library_name+'.__version__')
      if not found:
        self.set_status('skipped (Unable to import library: "'+library_name+'")',
                        self.bucket_skip)
        return False
      if distutils.version.LooseVersion(actual_version) < \
         distutils.version.LooseVersion(library_version):
        self.set_status('skipped (Outdated library: "'+library_name+'")',
                        self.bucket_skip)
        return False
      i += 2

    if len(self.required_executable) > 0:
      try:
        args_list = [self.required_executable]
        args_list.extend(self.required_executable_check_flags)
        ret_value = subprocess.call(args_list, stdout=subprocess.PIPE)
        if ret_value != 0:
          self.set_status('skipped (Failing executable: "'
                          +self.required_executable+'")', self.bucket_skip)
          return False
      except Exception:
        self.set_status('skipped (Error when trying executable: "'
                        +self.required_executable+'")', self.bucket_skip)
        return False

    if self.specs['requires_swig2'] and not RavenPython.has_swig2:
      self.set_status('skipped (No swig 2.0 found)', self.bucket_skip)
      return False
    missing, too_old, _ = RavenUtils.check_for_missing_modules()
    if len(missing) > 0:
      self.set_status('skipped (Missing python modules: '+" ".join(missing)+
                      " PYTHONPATH="+os.environ.get("PYTHONPATH", "")+')',
                      self.bucket_skip)
      return False
    if len(too_old) > 0 and RavenUtils.check_versions():
      self.set_status('skipped (Old version python modules: '+" ".join(too_old)+
                      " PYTHONPATH="+os.environ.get("PYTHONPATH", "")+')',
                      self.bucket_skip)
      return False
    for lib in self.required_libraries:
      found, _, _ = RavenUtils.module_report(lib, '')
      if not found:
        self.set_status('skipped (Unable to import library: "'+lib+'")',
                        self.bucket_skip)
        return False
    if self.specs['python3_only'] and not RavenUtils.in_python_3():
      self.set_status('Python 3 only',
                      self.bucket_skip)
      return False

    return True

  def process_results(self, output):
    """
      Sets the status of this test.
      @ In, output, string, output of running the test.
      @ Out, None
    """
    if self.results.exit_code != 0:
      self.set_status(str(self.results.exit_code), self.bucket_fail)
      return
    self.set_success()
