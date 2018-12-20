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
"""
This module implements classes for running tests.
"""
from __future__ import division, print_function, absolute_import
import warnings

import subprocess
import sys
import os
import time
import threading
from distutils import spawn

warnings.simplefilter('default', DeprecationWarning)

class _Parameter:

  def __init__(self, name, help_text, default=None):
    self.name = name
    self.help_text = help_text
    self.default = default

class _ValidParameters:

  def __init__(self):
    self.__parameters = {}

  def add_param(self, name, default, help_text):
    """
    Adds an optional parameter.
    name: string parameter name
    default: string the default value for the parameter
    help_text: Description of the parameter
    """
    self.__parameters[name] = _Parameter(name, help_text, default)

  def add_required_param(self, name, help_text):
    """
    Adds a mandatory parameter.
    name: string parameter name
    help_text: Description of the parameter
    """
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


class Differ:
  """
  Subclass are intended to check something, such as that some
  files exist and match the gold files.
  """

  @staticmethod
  def get_valid_params():
    """
    Generates the allowed parameters for this class.
    """
    params = _ValidParameters()
    params.add_required_param('type', 'The type of this differ')
    params.add_required_param('output', 'Output of to check')
    params.add_param('gold_files', '', 'Gold filenames')
    return params

  def __init__(self, name, params, test_dir):
    """
    Initializer for the class.
    name: String name of class
    params: dictionary of parameters
    test_dir: String path to test directory
    """
    self.__name = name
    self.__test_dir = test_dir
    valid_params = self.get_valid_params()
    self.specs = valid_params.get_filled_dict(params)
    self.__output_files = self.specs['output'].split()

  def get_remove_files(self):
    """
    Returns a list of files to remove before running test.
    returns List(Strings)
    """
    return self._get_test_files()

  def _get_test_files(self):
    """
    returns a list of the full path of the test files
    """
    return [os.path.join(self.__test_dir, f) for f in self.__output_files]

  def _get_gold_files(self):
    """
    returns a list of the full path to the gold files
    """
    if len(self.specs['gold_files']) > 0:
      gold_files = self.specs['gold_files'].split()
      return [os.path.join(self.__test_dir, f) for f in gold_files]
    return [os.path.join(self.__test_dir, "gold", f) for f in self.__output_files]

  def check_output(self):
    """
    Checks that the output matches the gold.
    Should return (same, message) where same is true if the
    test passes, or false if the test failes.  message should
    give a human readable explaination of the differences.
    """
    assert False, "Must override check_output"

class _TimeoutThread(threading.Thread):
  """
  This class will kill a process after a certain amount of time
  """

  def __init__(self, process, timeout):
    """
    process: A process that can be killed
    timeout: float time in seconds to wait before killing the process.
    """
    self.__process = process
    self.__timeout = timeout
    self.__killed = False
    threading.Thread.__init__(self)

  def run(self):
    """
    Runs and waits for timeout, then kills process
    """
    start = time.time()
    end = start + self.__timeout
    while True:
      if self.__process.poll() is not None:
        #Process finished
        break
      if time.time() > end:
        #Time over
        if os.name == "nt" and spawn.find_executable("taskkill"):
          subprocess.call(['taskkill', '/f', '/t', '/pid', str(self.__process.pid)])
        else:
          self.__process.kill()
        self.__killed = True
        break
      time.sleep(1.0)

  def killed(self):
    """
    Returns if the process was killed.  Notice this will be false at the
    start.
    """
    return self.__killed

class Tester:
  """
  This is the base class for something that can run tests.
  """

  #Various possible status buckets.
  bucket_skip = 0
  bucket_fail = 1
  bucket_diff = 2
  bucket_success = 3
  bucket_timed_out = 4
  bucket_not_set = 5

  success_message = "SUCCESS"

  @staticmethod
  def get_valid_params():
    """
    This generates the parameters for this class.
    """
    params = _ValidParameters()
    params.add_required_param('type', 'The type of this test')
    params.add_param('skip', False, 'If true skip test')
    params.add_param('prereq', '', 'list of tests to run before running this one')
    params.add_param('max_time', 300, 'Maximum time that test is allowed to run')
    params.add_param('method', False, 'Method is ignored, but kept for compatibility')
    params.add_param('heavy', False, 'If true, run only with heavy tests')
    params.add_param('output', '', 'Output of the test')
    params.add_param('expected_fail', False,
                     'if true, then the test should fails, and if it passes, it fails.')
    return params

  def __init__(self, name, params):
    """
    Initializer for the class.  Takes a String name and a dictionary params
    """
    self.__name = name
    valid_params = self.get_valid_params()
    self.specs = valid_params.get_filled_dict(params)
    self.results = TestResult()
    self.__differs = []
    self.__run_heavy = False

  def get_differ_remove_files(self, test_dir):
    """
    Returns the files that need to be removed for testing.
    returns List(String)
    """
    remove_files = []
    for differ in self.__differs:
      remove_files.extend(differ.get_remove_files())
    return remove_files

  def add_differ(self, differ):
    """
    Adds a differ to run after the test completes.
    differ: A subclass of Differ that tests a file produced by the run.
    """
    self.__differs.append(differ)

  def get_test_dir(self):
    """
    Returns the test directory
    """
    return self.specs['test_dir']

  def didPass(self):
    """
    Returns true if this test passed
    """
    return self.results.bucket == self.bucket_success

  def run_heavy(self):
    """
    If called, run the heavy tests and not the light
    """
    self.__run_heavy = True


  def run(self, data):
    """
    Runs the tester.
    """
    expectedFail = bool(self.specs['expected_fail'])
    results = self.run_backend(data)
    if not expectedFail:
      return results
    else:
      if results.bucket == self.bucket_success:
        results.bucket = self.bucket_fail
        results.message = "Unexpected Success"
      else:
        results.bucket = self.bucket_success
      return results

  def run_backend(self, data):
    """
    Runs this tester.  This does the main work,
    but is separate to allow run to invert the result if expected_fail
    """
    options = None
    if self.specs['skip'] is not False:
      self.results.bucket = self.bucket_skip
      self.results.message = self.specs['skip']
      return self.results
    if self.specs['heavy'] is not False and not self.__run_heavy:
      self.results.bucket = self.bucket_skip
      self.results.message = "SKIPPED (Heavy)"
      return self.results
    if self.specs['heavy'] is False and self.__run_heavy:
      self.results.bucket = self.bucket_skip
      self.results.message = "SKIPPED (not Heavy)"
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
    timed_out = False
    if sys.version_info >= (3, 3):
      #New timeout interface available starting in Python 3.3
      try:
        output = process.communicate(timeout=timeout)[0]
      except subprocess.TimeoutExpired:
        process.kill()
        output = process.communicate()[0]
        timed_out = True
    else:
      timeout_killer = _TimeoutThread(process, timeout)
      timeout_killer.start()
      output = process.communicate()[0]
      if timeout_killer.killed():
        timed_out = True
    end_time = time.time()
    process_time = end_time - start_time
    self.results.exit_code = process.returncode
    self.results.runtime = process_time
    self.results.output = output
    if timed_out:
      self.results.bucket = self.bucket_timed_out
      self.results.message = "Timed Out"
      return self.results
    self.processResults(None, options, output)
    for differ in self.__differs:
      same, message = differ.check_output()
      if not same:
        if self.results.bucket == self.bucket_success:
          self.results.bucket = self.bucket_diff
          self.results.message = "" #remove success message.
        self.results.message += "\n" + message
    return self.results

  @staticmethod
  def get_bucket_name(bucket):
    """
    Returns the name of this bucket
    """
    names = ["Skipped", "Failed", "Diff", "Success", "Timeout", "NOT_SET"]
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

  def prepare(self, options=None):
    """
    gets the test ready to run.
    """
    return
