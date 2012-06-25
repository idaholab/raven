import os, sys, re, inspect, types, errno, pprint
from socket import gethostname
from optparse import OptionParser, OptionGroup
#from optparse import OptionG
from timeit import default_timer as clock

from tools import TestHarness
from r7options import *
from util import *
from RunParallel import RunParallel
from CSVDiffer import CSVDiffer


## Called by ./run_tests in an application directory
def runTests(argv, app_name, moose_dir):
  host_name = gethostname()
  if host_name == 'service0' or host_name == 'service1':
    print 'Testing not supported on Icestorm head node'
    sys.exit(0)

  harness = R7TestHarness(argv, app_name, moose_dir)
  harness.findAndRunTests()


class R7TestHarness(TestHarness):

  def __init__(self, argv, app_name, moose_dir):
    TestHarness.__init__(self, argv, app_name, moose_dir)

  # R7 Extended the TestHarness by adding the "simulation" flag
  # The createCommand function is the only function that needs to be overriden
  def createCommand(self, test):
    if test[MIN_PARALLEL] > 1:
      return 'mpiexec -n ' + test[MIN_PARALLEL] + ' ' + self.executable + ' -i ' + test[INPUT] + ' ' +  ' '.join(test[CLI_ARGS])
    elif test[SIMUL]:
      return self.executable + ' -i ' + test[SIMUL] + ' ' + ' '.join(test[CLI_ARGS])
    else:
      return self.executable + ' -s ' + test[INPUT] + ' ' + ' '.join(test[CLI_ARGS])

