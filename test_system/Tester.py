
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import subprocess
import sys
import time

class _Parameter:

  def __init__(self, name, help_text, default=None):
    self.name = name
    self.help_text = help_text
    self.default = default

class _ValidParameters:

  def __init__(self):
    self.__parameters = {}

  def addParam(self, name, default, help_text):
    self.__parameters[name] = _Parameter(name, help_text, default)

  def addRequiredParam(self, name, help_text):
    self.__parameters[name] = _Parameter(name, help_text)

  def get_filled_dict(self, partial_dict):
    """
    Returns a dictionary where default values are filled in for everything
    that is not in the partial_dict
    """
    ret_dict = dict(partial_dict)
    for param in self.__parameters.values():
      if param.default is not None and param.name not in ret_dict:
        ret_dict[param.name] = param.default
    return ret_dict

  def check_for_required(self, check_dict):
    """
    Returns True if all the required parameters are present
    """
    for param in self.__parameters.values():
      if param.default is None and param.name not in check_dict:
        print("Missing:", param.name)
        return False
    return True

  def check_for_all_known(self, check_dict):
    """
    Returns True if all the parameters are known
    """
    for param_name in check_dict:
      if param_name not in self.__parameters:
        print("Unknown:", param_name)
        return False
    return True

class TestResult:
  """
  Class to store results of the test data
  """

  def __init__(self):
    self.bucket = Tester.bucket_not_set
    self.exit_code = None
    self.message = None
    self.output = None
    self.runtime = None

class Tester:

  #Various possible status buckets.
  bucket_skip = 0
  bucket_fail = 1
  bucket_diff = 2
  bucket_success = 3
  bucket_not_set = 4

  success_message = "SUCCESS"

  @staticmethod
  def validParams():
    params = _ValidParameters()
    params.addRequiredParam('type', 'The type of this test')
    params.addParam('skip', False, 'If true skip test')
    params.addParam('prereq', '', 'list of tests to run before running this one')
    params.addParam('max_time', 300, 'Maximum time that test is allowed to run')
    params.addParam('method', False, 'Method is ignored, but kept for compatibility')
    params.addParam('heavy', False, 'If true, run only with heavy tests')
    params.addParam('output', '', 'Output of the test')
    return params

  def __init__(self, name, params):
    """
    Initializer for the class.  Takes a String name and a dictionary params
    """
    self.__name = name
    valid_params = self.validParams()
    self.specs = valid_params.get_filled_dict(params)
    self.results = TestResult()

  def getTestDir(self):
    """
    Returns the test directory
    """
    return self.specs['test_dir']

  def didPass(self):
    """
    Returns true if this test passed
    """
    return self.results.bucket == self.bucket_success

  def run(self, data):
    """
    Runs this tester.
    """
    options = None
    if self.specs['skip'] is not False:
      self.results.bucket = self.bucket_skip
      self.results.message = self.specs['skip']
      return self.results
    if self.specs['heavy'] is not False:
      self.results.bucket = self.bucket_skip
      self.results.message = "SKIPPED (Heavy)"
      return self.results
    if not self.checkRunnable(options):
      return self.results

    self.prepare()

    command = self.getCommand(options)

    timeout = int(self.specs['max_time'])
    directory = self.specs['test_dir']
    start_time = time.time() #Change to monotonic when min python raised to 3.3
    try:
      process = subprocess.Popen(command, shell=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 cwd=directory,
                                 universal_newlines=True)
    except IOError as ioe:
      self.results.bucket = self.bucket_fail
      self.results.message = "FAILED "+str(ioe)
      return self.results
    if sys.version_info >= (3,3):
      #New timeout interface available starting in Python 3.3
      try:
        output = process.communicate(timeout=timeout)[0]
      except subprocess.TimeoutExpired:
        process.kill()
        output = process.communicate()[0]
    else:
      output = process.communicate()[0]
    end_time = time.time()
    process_time = end_time - start_time
    self.results.exit_code = process.returncode
    self.results.runtime = process_time
    self.processResults(None, options, output)
    self.results.output = output
    return self.results

  @staticmethod
  def get_bucket_name(bucket):
    """
    Returns the name of this bucket
    """
    names = ["SKIPPED", "FAILED", "DIFF", "SUCCESS", "NOT_SET"]
    if 0 <= bucket < len(names):
      return names[bucket]
    return "UNKNOWN BUCKET"

  def checkRunnable(self, options):
    """
    Checks if this test case can run
    """
    return True

  def set_success(self):
    """
    Called by subclasses if this was a success.
    """
    self.results.bucket = self.bucket_success
    self.results.message = Tester.get_bucket_name(self.results.bucket)

  def setStatus(self, message, bucket):
    """
    Sets the message string and the bucket type
    """
    self.results.message = message
    self.results.bucket = bucket

  def processResults(self, moose_dir, options, output):
    """
    Handle the results of the test case.
    moose_dir: unused
    options: unused
    output: the output of the test case.
    """
    assert False, "processResults not implemented"

  def getCommand(self, options):
    """
    returns the command used to run the test
    """
    assert False, "getCommand not implemented"

  def prepare(self, options = None):
    """
    gets the test ready to run.
    """
    return
