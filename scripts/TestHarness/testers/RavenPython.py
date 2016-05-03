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
    if os.environ.get("CHECK_PYTHON3","0") == "1":
      params.addParam('python_command','python3','The command to use to run python')
    else:
      params.addParam('python_command','python','The command to use to run python')
    params.addParam('requires_swig2', False, "Requires swig2 for test")
    params.addParam('required_executable','','Skip test if this executable is not found')
    params.addParam('required_executable_check_flags','','Flags to add to the required executable to make sure it runs without fail when testing its existence on the machine')

    return params

  def getCommand(self, options):
    return self.specs["python_command"]+" "+self.specs["input"]

  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.specs['scale_refine'] = False
    self.required_executable = self.specs['required_executable']
    self.required_executable = self.required_executable.replace("%METHOD%",os.environ.get("METHOD","opt"))
    self.required_executable_check_flags = self.specs['required_executable_check_flags'].split(' ')

  def checkRunnable(self, option):
    try:
      argsList = [self.required_executable]
      argsList.extend(self.required_executable_check_flags)
      retValue = subprocess.call(argsList,stdout=subprocess.PIPE)
      if len(self.required_executable) > 0 and retValue != 0:
        return (False,'skipped (Failing executable: "'+self.required_executable+'")')
    except:
      return (False,'skipped (Error when trying executable: "'+self.required_executable+'")')

    if self.specs['requires_swig2'] and not RavenPython.has_swig2:
      return (False, 'skipped (No swig 2.0 found)')
    missing,too_old = RavenUtils.checkForMissingModules()
    if len(missing) > 0:
      return (False,'skipped (Missing python modules: '+" ".join(missing)+
              " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
    if len(too_old) > 0:
      return (False,'skipped (Old version python modules: '+" ".join(too_old)+
              " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
    return (True, '')

  def processResults(self, moose_dir,retcode, options, output):
    if retcode != 0:
      return (str(retcode),output)
    return ('',output)
