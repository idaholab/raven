from util import *
from Tester import Tester
from CSVDiffer import CSVDiffer
import os

class RavenFramework(Tester):

  def getValidParams():
    params = Tester.getValidParams()
    params.addRequiredParam('input',"The input file to use for this test.")
    params.addParam('output','',"List of output files that the input should create.")
    params.addParam('csv','',"List of csv files to check")
    params.addParam('rel_err','','Relative Error for csv files')
    params.addParam('required_executable','','Skip test if this executable is not found')
    return params
  getValidParams = staticmethod(getValidParams)

  def getCommand(self, options):
    if os.environ.get("CHECK_PYTHON3","0") == "1":
      return "python3 ../../framework/Driver.py "+self.specs["input"]  
    else:
      return "python ../../framework/Driver.py "+self.specs["input"]
    

  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.check_files = [os.path.join(self.specs['test_dir'],filename)  for filename in self.specs['output'].split(" ")]
    self.csv_files = self.specs['csv'].split(" ") if len(self.specs['csv']) > 0 else []
    for filename in self.check_files+self.csv_files:# + [os.path.join(self.specs['test_dir'],filename)  for filename in self.csv_files]:
      if os.path.exists(filename):
        os.remove(filename)
    self.required_executable = self.specs['required_executable']
    self.required_executable = self.required_executable.replace("%METHOD%",os.environ.get("METHOD","opt"))
    self.specs['scale_refine'] = False

  def checkRunnable(self, option):
    missing = []
    try:
      import h5py
    except:
      missing.append('h5py')
    try:
      import numpy
    except:
      missing.append('numpy')
    try:
      import scipy
    except:
      missing.append('scipy')
    try:
      import sklearn
    except:
      missing.append('sklearn')
    try:
      import matplotlib
    except:
      missing.append('matplotlib')
    if len(missing) > 0:
      return (False,'skipped (Missing python modules: '+" ".join(missing)+
              " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')')
    if len(self.required_executable) > 0 and \
       not os.path.exists(self.required_executable):
      return (False,'skipped (Missing executable: "'+self.required_executable+'")')
    return (True, '')

  def processResults(self, moose_dir,retcode, options, output):
    missing = []
    for filename in self.check_files:
      if not os.path.exists(filename):
        missing.append(filename)
    
    if len(missing) > 0:
      return ('CWD '+os.getcwd()+' METHOD '+os.environ.get("METHOD","?")+' Expected files not created '+" ".join(missing),output)
    if len(self.specs["rel_err"]) > 0:
      csv_diff = CSVDiffer(self.specs['test_dir'],self.csv_files,relative_error=float(self.specs["rel_err"]))
    else:
      csv_diff = CSVDiffer(self.specs['test_dir'],self.csv_files)
    message = csv_diff.diff()
    if csv_diff.getNumErrors() > 0:
      return (message,output)
    return ('',output)
