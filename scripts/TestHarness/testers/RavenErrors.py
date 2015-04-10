from Tester import Tester
from CSVDiffer import CSVDiffer
import RavenUtils
import os
import subprocess

class RavenErrors(Tester):

  @staticmethod
  def validParams():
    params = Tester.validParams()
    params.addRequiredParam('input',"The input file to use for this test.")
    params.addParam('required_executable','','Skip test if this executable is not found')
    params.addParam('skip_if_env','','Skip test if this environmental variable is defined')
    params.addParam('test_interface_only','False','Test the interface only (without running the driven code')
    params.addParam('expect_err',"The expected error message (keyword)")
    return params

  def getCommand(self, options):
    ravenflag = ''
    if self.specs['test_interface_only'].lower() == 'true': ravenflag = 'interfaceCheck '
    if RavenUtils.inPython3():
      return "python3 ../../framework/Driver.py " + ravenflag + self.specs["input"]
    else:
      return "python ../../framework/Driver.py " + ravenflag + self.specs["input"]


  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.required_executable = self.specs['required_executable']
    self.required_executable = self.required_executable.replace("%METHOD%",os.environ.get("METHOD","opt"))
    self.specs['scale_refine'] = False

  def checkRunnable(self, option):
    missing,too_old = RavenUtils.checkForMissingModules()
    if len(missing) > 0:
      return (False,'skipped (Missing python modules: '+" ".join(missing)+
              " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
    if len(too_old) > 0:
      return (False,'skipped (Old version python modules: '+" ".join(too_old)+
              " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
    if len(self.required_executable) > 0 and \
       not os.path.exists(self.required_executable):
      return (False,'skipped (Missing executable: "'+self.required_executable+'")')
    try:
      if len(self.required_executable) > 0 and \
         subprocess.call([self.required_executable],stdout=subprocess.PIPE) != 0:
        return (False,'skipped (Failing executable: "'+self.required_executable+'")')
    except:
      return (False,'skipped (Error when trying executable: "'+self.required_executable+'")')
    if len(self.specs['skip_if_env']) > 0:
      env_var = self.specs['skip_if_env']
      if env_var in os.environ:
        return (False,'skipped (found environmental variable "'+env_var+'")')
    return (True, '')

  def processResults(self, moose_dir,retcode, options, output):
    for line in output.split('\n'):
      if self.specs['expect_err'] in line:
        return ('',output)
    return ('The expected Error: ' +self.specs['expect_err']+' is not raised!' , output)

