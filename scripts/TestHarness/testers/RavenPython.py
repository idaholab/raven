from util import *
from Tester import Tester
from CSVDiffer import CSVDiffer
import os

class RavenPython(Tester):

  def getValidParams():
    params = Tester.getValidParams()
    params.addRequiredParam('input',"The python file to use for this test.")
    params.addParam('python_command','python','The command to use to run python')
    return params
  getValidParams = staticmethod(getValidParams)

  def getCommand(self, options):
    return self.specs["python_command"]+" "+self.specs["input"]

  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.specs['scale_refine'] = False

  def checkRunnable(self, option):
    return (True, '')

  def processResults(self, moose_dir,retcode, options, output):
    if retcode != 0:
      return (str(retcode),output)
    return ('',output)
