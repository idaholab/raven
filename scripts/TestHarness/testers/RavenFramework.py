from util import *
from Tester import Tester
import os

class RavenFramework(Tester):

  def getValidParams():
    params = Tester.getValidParams()
    params.addRequiredParam('input',"The input file to use for this test.")
    params.addRequiredParam('output',"List of output files that the input should create.")
    return params
  getValidParams = staticmethod(getValidParams)

  def getCommand(self, options):
    return "python ../../framework/Driver.py "+self.specs["input"]

  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.check_files = [os.path.join(self.specs['test_dir'],filename)  for filename in self.specs['output'].split(" ")]
    for filename in self.check_files:
      if os.path.exists(filename):
        os.remove(filename)
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
    return (True, '')

  def processResults(self, moose_dir,retcode, options, output):
    missing = []
    for filename in self.check_files:
      if not os.path.exists(filename):
        missing.append(filename)
    if len(missing) > 0:
      return ('CWD '+os.getcwd()+' Expected files not created '+" ".join(missing),output)
    return ('',output)
