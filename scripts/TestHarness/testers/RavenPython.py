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
from util import *
from Tester import Tester
from CSVDiffer import CSVDiffer
import RavenUtils
import os, subprocess
import distutils.version

class RavenPython(Tester):
  try:
    output_swig = subprocess.Popen(["swig","-version"],stdout=subprocess.PIPE).communicate()[0]
  except OSError:
    output_swig = "Failed"

  has_swig2 = "Version 2.0" in output_swig or "Version 3.0" in output_swig


  @staticmethod
  def validParams():
    params = Tester.validParams()
    params.addRequiredParam('input',"The python file to use for this test.")
    params.addParam('output','',"List of output files that this input should create.")
    if os.environ.get("CHECK_PYTHON3","0") == "1":
      params.addParam('python_command','python3','The command to use to run python')
    else:
      params.addParam('python_command','python','The command to use to run python')
    params.addParam('requires_swig2', False, "Requires swig2 for test")
    params.addParam('required_executable','','Skip test if this executable is not found')
    params.addParam('required_libraries','','Skip test if any of these libraries are not found')
    params.addParam('required_executable_check_flags','','Flags to add to the required executable to make sure it runs without fail when testing its existence on the machine')
    params.addParam('minimum_library_versions','','Skip test if the library listed is below the supplied version (e.g. minimum_library_versions = \"name1 version1 name2 version2\")')

    return params

  def prepare(self, options = None):
    """
      Copied from RavenFramework since we should still clean out test files
      before running an external tester, though we will not test if they
      are created later (for now), so it may behoove us to not save
      check_files for later use.
    """
    if self.specs['output'].strip() != '':
      self.check_files = [os.path.join(self.specs['test_dir'],filename)  for filename in self.specs['output'].split(" ")]
    else:
      self.check_files = []
    for filename in self.check_files:
      if os.path.exists(filename):
        os.remove(filename)

  def getCommand(self, options):
    return self.specs["python_command"]+" "+self.specs["input"]

  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.specs['scale_refine'] = False
    self.required_executable = self.specs['required_executable']
    self.required_executable = self.required_executable.replace("%METHOD%",os.environ.get("METHOD","opt"))
    self.required_libraries = self.specs['required_libraries'].split(' ')  if len(self.specs['required_libraries']) > 0 else []
    self.required_executable_check_flags = self.specs['required_executable_check_flags'].split(' ')
    self.minimum_libraries = self.specs['minimum_library_versions'].split(' ')  if len(self.specs['minimum_library_versions']) > 0 else []

  def checkRunnable(self, option):
    i = 0
    if len(self.minimum_libraries) % 2:
      self.addCaveats('skipped (libraries are not matched to versions numbers: '+str(self.minimum_libraries)+')')
      self.setStatus(self.skip)
      return False
    while i < len(self.minimum_libraries):
      libraryName = self.minimum_libraries[i]
      libraryVersion = self.minimum_libraries[i+1]
      found, message, actualVersion = RavenUtils.moduleReport(libraryName,libraryName+'.__version__')
      if not found:
        self.addCaveats('skipped (Unable to import library: "'+libraryName+'")')
        self.setStatus(self.skip)
        return False
      if distutils.version.LooseVersion(actualVersion) < distutils.version.LooseVersion(libraryVersion):
        self.addCaveats('skipped (Outdated library: "'+libraryName+'")')
        self.setStatus(self.skip)
        return False
      i+=2

    if len(self.required_executable) > 0:
      try:
        argsList = [self.required_executable]
        argsList.extend(self.required_executable_check_flags)
        retValue = subprocess.call(argsList,stdout=subprocess.PIPE)
        if retValue != 0:
          self.addCaveats('skipped (Failing executable: "'+self.required_executable+'")')
          self.setStatus(self.skip)
          return False
      except:
        self.addCaveats('skipped (Error when trying executable: "'+self.required_executable+'")')
        self.setStatus(self.skip)
        return False

    if self.specs['requires_swig2'] and not RavenPython.has_swig2:
      self.addCaveats('skipped (No swig 2.0 found)')
      self.setStatus(self.skip)
      return False
    missing,too_old, notQA = RavenUtils.checkForMissingModules()
    if len(missing) > 0:
      self.addCaveats('skipped (Missing python modules: '+" ".join(missing)+
                     " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
      self.setStatus(self.skip)
      return False
    if len(too_old) > 0 and RavenUtils.checkVersions():
      self.addCaveats('skipped (Old version python modules: '+" ".join(too_old)+
                     " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
      self.setStatus(self.skip)
      return False
    for lib in self.required_libraries:
      found, message, version =  RavenUtils.moduleReport(lib,'')
      if not found:
        self.addCaveats('skipped (Unable to import library: "'+lib+'")')
        self.setStatus(self.skip)
        return False

    return True

  def processResults(self, moose_dir, options, output):
    if self.exit_code != 0:
      self.setStatus(self.fail, str(self.exit_code))
      return output
    self.setStatus(self.success)
    return output
