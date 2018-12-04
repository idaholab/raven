
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import subprocess

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

  def getTestDir(self):
    """
    Returns the test directory
    """
    return self.specs['test_dir']

  def didPass(self):
    """
    Returns true if this test passed
    """
    return self.__bucket == self.bucket_success

  def run(self, data):
    """
    Runs this tester.
    """
    options = None
    self.__bucket = self.bucket_not_set
    if self.specs['skip'] is not False:
      self.__bucket = self.bucket_skip
      return (self.__bucket, "SKIPPED", self.specs['skip'])
    if self.specs['heavy'] is not False:
      self.__bucket = self.bucket_skip
      return (self.__bucket, "SKIPPED (Heavy)", self.specs['skip'])
    if not self.checkRunnable(options):
      return (self.__bucket, "Not Run", self.__message)

    self.prepare()

    command = self.getCommand(options)

    directory = self.specs['test_dir']
    try:
      process = subprocess.Popen(command, shell=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 cwd=directory,
                                 universal_newlines=True)
    except IOError as ioe:
      self.__bucket = self.bucket_fail
      return (self.__bucket, "FAILED", str(ioe))
    output = process.communicate()[0]
    self.exit_code = process.returncode
    self.processResults(None, options, output)
    return (self.__bucket,
            self.get_bucket_name(self.__bucket),
            output)

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

  def setStatus(self, message, bucket):
    """
    Sets the message string and the bucket type
    """
    self.__message = message
    self.__bucket = bucket

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
