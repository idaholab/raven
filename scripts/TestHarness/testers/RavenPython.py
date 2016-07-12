from util import *
from Tester import Tester
from CSVDiffer import CSVDiffer
import RavenUtils
import os, subprocess

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

    return params

  def prepare(self):
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

  def checkRunnable(self, option):
    if len(self.required_executable) > 0:
      try:
        argsList = [self.required_executable]
        argsList.extend(self.required_executable_check_flags)
        retValue = subprocess.call(argsList,stdout=subprocess.PIPE)
        if retValue != 0:
          return (False,'skipped (Failing executable: "'+self.required_executable+'")')
      except:
        return (False,'skipped (Error when trying executable: "'+self.required_executable+'")')

    if self.specs['requires_swig2'] and not RavenPython.has_swig2:
      return (False, 'skipped (No swig 2.0 found)')
    missing,too_old, notQA = RavenUtils.checkForMissingModules()
    if len(missing) > 0:
      return (False,'skipped (Missing python modules: '+" ".join(missing)+
              " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
    if len(too_old) > 0:
      return (False,'skipped (Old version python modules: '+" ".join(too_old)+
              " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
    for lib in self.required_libraries:
      found, message, version =  RavenUtils.moduleReport(lib,'')
      if not found:
        return (False,'skipped (Unable to import library: "'+lib+'")')

    return (True, '')

  def processResults(self, moose_dir,retcode, options, output):
    if retcode != 0:
      return (str(retcode),output)
    return ('',output)
