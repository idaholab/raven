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
import shutil
import time
import threading
import platform

warnings.simplefilter('default', DeprecationWarning)

class _Parameter:
  """
  Stores a single parameter for the input.
  """

  def __init__(self, name, help_text, default=None):
    """
      Initializes the class
      @ In, name, string, the name of the parameter
      @ In, help_text, string, some help text for the user
      @ In, default, optional, if not None this is the default value
      @ Out, None
    """
    self.name = name
    self.help_text = help_text
    self.default = default

  def __str__(self):
    """
      Converts the class to a string.
      @ In, None
      @ Out, str, string, the class as a string.
    """
    if self.default is None:
      required = "Required"
    else:
      required = "Optional with default: "+str(self.default)
    return self.name + " " +\
      required + "\n" +\
      self.help_text

class _ValidParameters:
  """
    Contains the valid parameters that a tester or a differ can use.
  """

  def __init__(self):
    """
      Initializes the valid parameters class.
      @ In, None
      @ Out, None
    """
    self.__parameters = {}

  def __str__(self):
    """
      Converts the class to a string.
      @ In, None
      @ Out, str, string, the class as a string.
    """
    return "\n\n".join([str(x) for x in self.__parameters.values()])

  def add_param(self, name, default, help_text):
    """
      Adds an optional parameter.
      @ In, name, string, parameter name
      @ In, default, the default value for the parameter
      @ In, help_text, string, Description of the parameter
      @ Out, None
    """
    self.__parameters[name] = _Parameter(name, help_text, default)

  def add_required_param(self, name, help_text):
    """
      Adds a mandatory parameter.
      @ In, name, string parameter name
      @ In, help_text, string, Description of the parameter
      @ Out, None
    """
    self.__parameters[name] = _Parameter(name, help_text)

  def get_filled_dict(self, partial_dict):
    """
      Returns a dictionary where default values are filled in for everything
      that is not in the partial_dict
      @ In, partial_dict, dictionary, a dictionary with some parameters.
      @ Out, ret_dict, dictionary, a dictionary where all the parameters that
        are not in partial_dict but have default values have been added.
    """
    ret_dict = dict(partial_dict)
    for param in self.__parameters.values():
      if param.default is not None and param.name not in ret_dict:
        ret_dict[param.name] = param.default
    return ret_dict

  def check_for_required(self, check_dict):
    """
      Returns True if all the required parameters are present
      @ In, check_dict, dictionary, dictionary to check
      @ Out, all_required, boolean, True if all required parameters are in
        check_dict.
    """
    all_required = True
    for param in self.__parameters.values():
      if param.default is None and param.name not in check_dict:
        print("Missing:", param.name)
        all_required = False
    return all_required

  def check_for_all_known(self, check_dict):
    """
      Returns True if all the parameters are known
      @ In, check_dict, dictionary, dictionary to check
      @ Out, no_unknown, boolean, True if all the parameters are known.
    """
    no_unknown = True
    for param_name in check_dict:
      if param_name not in self.__parameters:
        print("Unknown:", param_name)
        no_unknown = False
    return no_unknown

class TestResult:
  """
  Class to store results of the test data
  """

  def __init__(self):
    """
      Initializes the class
      @ In, None
      @ Out, None
    """
    self.group = Tester.group_not_set
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
      @ In, None
      @ Out, params, _ValidParameters, the allowed parameters for this class.
    """
    params = _ValidParameters()
    params.add_required_param('type', 'The type of this differ')
    params.add_required_param('output', 'Output files to check')
    params.add_param('windows_gold', '', 'Paths to Windows specific gold files,'+
                     ' relative to gold directory')
    params.add_param('mac_gold', '', 'Paths to Mac specific gold files, relative to gold directory')
    params.add_param('linux_gold', '', 'Paths to Linux specific gold files,'+
                     ' relative to gold directory')
    params.add_param('gold_files', '', 'Paths to gold files, relative to gold directory')
    return params

  def __init__(self, _name, params, test_dir):
    """
      Initializer for the class.
      @ In, _name, string, name of class (currently unused)
      @ In, params, dictionary, dictionary of parameters
      @ In, test_dir, string, path to test directory
      @ Out, None
    """
    self.__test_dir = test_dir
    valid_params = self.get_valid_params()
    self.specs = valid_params.get_filled_dict(params)
    self.__output_files = self.specs['output'].split()

  def get_remove_files(self):
    """
      Returns a list of files to remove before running test.
      @ In, None
      @ Out, get_remove_files, returns List(Strings)
    """
    return self._get_test_files()

  def check_if_test_files_exist(self):
    """
      Returns true if all the test files exist.
      @ In, None
      @ Out, all_test_files, bool, true if all the test files exist
    """
    all_test_files = True
    for filename in self._get_test_files():
      if not os.path.exists(filename):
        all_test_files = False
    return all_test_files

  def _get_test_files(self):
    """
      returns a list of the full path of the test files
      @ In, None
      @ Out, _get_test_files, List(Strings), the path of the test files.
    """
    return [os.path.join(self.__test_dir, f) for f in self.__output_files]

  def _get_gold_files(self):
    """
      returns a list of the full path to the gold files
      @ In, None
      @ Out, paths, List(Strings), the paths of the gold files.
    """
    this_os = platform.system().lower()
    available_os = ['windows', 'mac', 'linux'] # list of OS with specific gold file options

    # replace "darwin" with "mac"
    if this_os == 'darwin':
      this_os = 'mac'

    # check if OS specific gold files should be used
    if (this_os in available_os) and (len(self.specs[f'{this_os}_gold']) > 0):
      gold_files = self.specs[f'{this_os}_gold'].split()
      paths = [os.path.join(self.__test_dir, "gold", f) for f in gold_files]
    # if OS specific gold files are not given, are specific gold files given?
    elif len(self.specs['gold_files']) > 0:
      gold_files = self.specs['gold_files'].split()
      paths = [os.path.join(self.__test_dir, "gold", f) for f in gold_files]
    # otherwise, use output files
    else:
      paths = [os.path.join(self.__test_dir, "gold", f) for f in self.__output_files]

    return paths

  def check_output(self):
    """
      Checks that the output matches the gold.
      Should return (same, message) where same is true if the
      test passes, or false if the test failes.  message should
      give a human readable explaination of the differences.
      @ In, None
      @ Out, check_output, (same, message),  same is bool, message is str,
         same is True if checks pass.
    """
    assert False, "Must override check_output "+str(self)

class _TimeoutThread(threading.Thread):
  """
  This class will kill a process after a certain amount of time
  """

  def __init__(self, process, timeout):
    """
      Initializes this class.
      @ In, process, process, A process that can be killed
      @ In, timeout, float, time in seconds to wait before killing the process.
      @ Out, None
    """
    self.__process = process
    self.__timeout = timeout
    self.__killed = False
    threading.Thread.__init__(self)

  def run(self):
    """
      Runs and waits for timeout, then kills process
      @ In, None
      @ Out, None
    """
    start = time.time()
    end = start + self.__timeout
    while True:
      if self.__process.poll() is not None:
        #Process finished
        break
      if time.time() > end:
        #Time over
        #If we are on windows, process.kill() is insufficient, so using
        # taskkill instead.
        if os.name == "nt" and shutil.which("taskkill"):
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
      @ In, None
      @ Out, __killed, boolean, true if process killed.
    """
    return self.__killed

class Tester:
  """
  This is the base class for something that can run tests.
  """

  #Various possible status groups.
  group_skip = 0
  group_fail = 1
  group_diff = 2
  group_success = 3
  group_timed_out = 4
  group_not_set = 5

  success_message = "SUCCESS"

  __default_run_type_set = set(["normal"])
  __non_default_run_type_set = set()
  __base_current_run_type = None

  @classmethod
  def add_default_run_type(cls, run_type):
    """
      This adds a new default run type.  These are used to decide
      which tests to run. These types run automatically.
      @ In, run_type, string, the default run type to add
      @ Out, None
    """
    cls.__default_run_type_set.add(run_type)
    assert run_type not in cls.__non_default_run_type_set

  @classmethod
  def add_non_default_run_type(cls, run_type):
    """
      This adds a new non default run type.  These are used to decide
      which tests to run. These types have to be requested to run.
      @ In, run_type, string, the non default run type to add
      @ Out, None
    """
    cls.__non_default_run_type_set.add(run_type)
    assert run_type not in cls.__default_run_type_set

  @classmethod
  def initialize_current_run_type(cls):
    """
      This initializes the current run type from the default run type.
      It should be called after the last call to add_default_run_type and
      add_non_default_run_type.  This will be automatically called the
      first time Tester is initialized.
      @ In, None
      @ Out, None
    """
    cls.__base_current_run_type = cls.__default_run_type_set

  @staticmethod
  def get_valid_params():
    """
      This generates the parameters for this class.
      @ In, None
      @ Out, params, _ValidParameters, the parameters for this class.
    """
    params = _ValidParameters()
    params.add_required_param('type', 'The type of this test')
    params.add_param('skip', False, 'If true skip test')
    params.add_param('prereq', '', 'list of tests to run before running this one')
    params.add_param('max_time', 300, 'Maximum time that test is allowed to run')
    params.add_param('os_max_time', '', 'Maximum time by os. '+
                     ' Example: Linux 20 Windows 300 OpenVMS 1000')
    params.add_param('method', False, 'Method is ignored, but kept for compatibility')
    params.add_param('heavy', False, 'If true, run only with heavy tests')
    params.add_param('output', '', 'Output of the test')
    params.add_param('expected_fail', False,
                     'if true, then the test should fails, and if it passes, it fails.')
    params.add_param('run_types', 'normal', 'The run types that this test is')
    params.add_param('output_wait_time', '-1', 'Number of seconds to wait for output')
    params.add_param('min_python_version', 'none',
                     'The Minimum python version required for this test.'+
                     ' Example 3.8 (note, format is major.minor)')
    params.add_param('needed_executable', '',
                     'Only run test if needed executable is on path.')
    params.add_param('skip_if_OS', '', 'Skip test if the operating system defined')
    return params

  def __init__(self, _name, params):
    """
      Initializer for the class.  Takes a String name and a dictionary params
      @ In, _name, string, name of the class (currently unused)
      @ In, params, dictionary, the parameters for this class to use.
      @ Out, None
    """
    valid_params = self.get_valid_params()
    self.specs = valid_params.get_filled_dict(params)
    self.results = TestResult()
    self._needed_executable = self.specs['needed_executable']
    self.__command_prefix = ""
    self.__python_command = sys.executable
    if os.name == "nt":
      #Command is python on windows in conda and Python.org install
      self.__python_command = "python"
    self.__differs = []
    if self.__base_current_run_type is None:
      self.initialize_current_run_type()
    self.__test_run_type = set(self.specs['run_types'].split())
    if self.specs['heavy'] is not False:
      self.__test_run_type.add("heavy")
      #--heavy makes the run type set(["heavy"]) so "normal" needs to be removed
      if "normal" in self.__test_run_type:
        self.__test_run_type.remove("normal")

  @classmethod
  def add_run_types(cls, run_types):
    """
      Adds run types to be run.  In general these are non default run types.
      @ In, run_types, set, run types to be added to the current run set.
      @ Out, None
    """
    assert run_types.issubset(set.union(cls.__default_run_type_set,
                                        cls.__non_default_run_type_set))
    cls.__base_current_run_type.update(run_types)

  @classmethod
  def set_only_run_types(cls, run_types):
    """
      Sets the run types to only the provided ones.
      @ In, run_types, set, run types to set the current set to.
      @ Out, None
    """
    assert run_types.issubset(set.union(cls.__default_run_type_set,
                                        cls.__non_default_run_type_set))
    cls.__base_current_run_type = set(run_types)

  def get_differ_remove_files(self):
    """
      Returns the files that need to be removed for testing.
      @ In, None
      @ Out, remove_files, List(String), files to be removed
    """
    remove_files = []
    for differ in self.__differs:
      remove_files.extend(differ.get_remove_files())
    return remove_files

  def add_differ(self, differ):
    """
      Adds a differ to run after the test completes.
      @ In, differ, Differ, A subclass of Differ that tests a file produced by the run.
      @ Out, None
    """
    self.__differs.append(differ)

  def get_test_dir(self):
    """
      Returns the test directory
      @ In, None
      @ Out, test_dir, string, the path to the test directory.
    """
    return self.specs['test_dir']

  def run_heavy(self):
    """
      If called, run the heavy tests and not the light.  Note that
      run still needs to be called.
      @ In, None
      @ Out, None
    """
    self.set_only_run_types(set(["heavy"]))

  def set_command_prefix(self, command_prefix):
    """
      Sets the command prefix.  This is prefixed to the front of the test
      command.
      @ In, command_prefix, string, the prefix for the command
      @ Out, None
    """
    self.__command_prefix = command_prefix

  def set_python_command(self, python_command):
    """
      Sets the python command.  This is used to run python commands.
      See alse _get_python_command
      @ In, python_command, string, the python command (including arguments)
      @ Out, None.
    """
    self.__python_command = python_command

  def run(self, data):
    """
      Runs the tester.
      @ In, data, ignored, but required by pool.MultiRun
      @ Out, results, TestResult, the results of the test.
    """
    expected_fail = bool(self.specs['expected_fail'])
    results = self._run_backend(data)
    if not expected_fail:
      return results
    if results.group == self.group_success:
      results.group = self.group_fail
      results.message = "Unexpected Success"
    else:
      results.group = self.group_success
    return results

  def __get_timeout(self):
    """
      Returns the timeout
      @ In, None
      @ Out, timeout, int, The maximum time for the test.
    """
    timeout = int(self.specs['max_time'])
    if len(self.specs['os_max_time']) > 0:
      time_list = self.specs['os_max_time'].lower().split()
      system = platform.system().lower()
      if system in time_list:
        timeout = int(time_list[time_list.index(system)+1])
    return timeout

  def _run_backend(self, _):
    """
      Runs this tester.  This does the main work,
      but is separate to allow run to invert the result if expected_fail
      @ In, None
      @ Out, results, TestResult, the results of the test.
    """
    if self.specs['skip'] is not False:
      self.results.group = self.group_skip
      self.results.message = self.specs['skip']
      return self.results
    ## OS
    if len(self.specs['skip_if_OS']) > 0:
      skip_os = [x.strip().lower() for x in self.specs['skip_if_OS'].split(',')]
      # get simple-name platform (options are Linux, Windows, Darwin, or SunOS that I've seen)
      current_os = platform.system().lower()
      # replace Darwin with more expected "mac"
      if current_os == 'darwin':
        current_os = 'mac'
      if current_os in skip_os:
        self.set_skip('skipped (OS is "{}")'.format(current_os))
        return self.results

    if self.specs['min_python_version'].strip().lower() != 'none':
      major, minor = self.specs['min_python_version'].strip().split(".")
      #check to see if current version of python too old.
      if int(major) > sys.version_info.major or \
         (int(major) == sys.version_info.major and int(minor) > sys.version_info.minor):
        self.results.group = self.group_skip
        self.results.message = "skipped because need python version "\
          +self.specs['min_python_version'].strip()+" but have "+sys.version
        return self.results
    if not self.__test_run_type.issubset(self.__base_current_run_type):
      self.results.group = self.group_skip
      self.results.message = "SKIPPED ("+str(self.__test_run_type)+\
        " is not a subset of "+str(self.__base_current_run_type)+")"
      return self.results
    if len(self._needed_executable) > 0 and \
       shutil.which(self._needed_executable) is None:
      self.set_skip('skipped (Missing executable: "'+self._needed_executable+'")')
      return self.results
    if not self.check_runnable():
      return self.results

    self.prepare()

    command = self.__command_prefix + self.get_command()

    timeout = self.__get_timeout()
    directory = self.specs['test_dir']
    start_time = time.time() #Change to monotonic when min python raised to 3.3
    try:
      process = subprocess.Popen(command, shell=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 cwd=directory,
                                 universal_newlines=True)
    except IOError as ioe:
      self.results.group = self.group_fail
      self.results.message = "FAILED "+str(ioe)
      return self.results
    timed_out = False
    if sys.version_info >= (3, 3) and os.name != "nt":
      #New timeout interface available starting in Python 3.3
      # But doesn't seem to fully work in Windows.
      try:
        output = process.communicate(timeout=timeout)[0]
      except subprocess.TimeoutExpired:
        process.kill()
        try:
          #wait 20 seconds, then give up trying to get output
          output = process.communicate(timeout=20)[0]
        except subprocess.TimeoutExpired as error:
          output = "Getting output timed out: " + repr(error)
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
      self.results.group = self.group_timed_out
      self.results.message = "Timed Out"
      return self.results
    if not self.check_exit_code(self.results.exit_code):
      return self.results
    self.process_results(output)
    self._wait_for_all_written()
    for differ in self.__differs:
      same, message = differ.check_output()
      if not same:
        if self.results.group == self.group_success:
          self.results.group = self.group_diff
          self.results.message = "" #remove success message.
        self.results.message += "\n" + message
    return self.results

  def _wait_for_all_written(self):
    """
      Waits until all the files for the differ have been written
      @ In, None
      @ Out, None
    """
    start_wait = time.time()
    wait_time = float(self.specs['output_wait_time'])
    all_written = False
    while start_wait + wait_time > time.time() and not all_written:
      all_written = True
      for differ in self.__differs:
        if not differ.check_if_test_files_exist():
          all_written = False
      if not all_written:
        time.sleep(5.0)
        print("waiting for files...")

  @staticmethod
  def get_group_name(group):
    """
      Returns the name of this group
      @ In, group, int, group constant
      @ Out, get_group_name, string, name of group constant
    """
    names = ["Skipped", "Failed", "Diff", "Success", "Timeout", "NOT_SET"]
    if 0 <= group < len(names):
      return names[group]
    return "UNKNOWN GROUP"

  def check_runnable(self):
    """
      Checks if this test case can run
      @ In, None
      @ Out, check_runnable, boolean, True if this can run.
    """
    return True

  def set_success(self):
    """
      Called by subclasses if this was a success.
      @ In, None
      @ Out, None
    """
    self.results.group = self.group_success
    self.results.message = Tester.get_group_name(self.results.group)

  def set_fail(self, message):
    """
      Sets the message string when failing
      @ In, message, string, string description of the failure
      @ Out, None
    """
    self.results.message = message
    self.results.group = self.group_fail

  def set_skip(self, message):
    """
      Sets the message string when skipping
      @ In, message, string, string description of the reason to skip
      @ Out, None
    """
    self.results.message = message
    self.results.group = self.group_skip

  def set_diff(self, message):
    """
      Sets the message string when failing for a diff
      @ In, message, string, string description of the difference
      @ Out, None
    """
    self.results.message = message
    self.results.group = self.group_diff

  def check_exit_code(self, exit_code):
    """
      Lets the subclasses decide if the exit code fails the test.
      @ In, exit_code, int, the exit code of the test command.
      @ Out, check_exit_code, bool, if true the exit code is acceptable.
      If false the tester should use set_fail or other methods to set the
      status and message.
    """
    if exit_code != 0:
      self.set_fail("Running test failed with exit code "+str(exit_code))
      return False
    return True

  def process_results(self, output):
    """
      Handle the results of the test case.
      @ In, output, string, the output of the test case.
      @ Out, None
    """
    assert False, "process_results not implemented "+output

  def _get_python_command(self):
    """
      returns a command that can run a python program.  So a possible return
      would be "python".  Another possibility is "coverage --append"
      @ In, None
      @ Out, __python_command, string, string command to run a python program
    """
    return self.__python_command

  def get_command(self):
    """
      returns the command used to run the test
      @ In, None
      @ Out, get_command, string, string command to run.
    """
    assert False, "getCommand not implemented"
    return "none"

  def prepare(self):
    """
      gets the test ready to run.
      @ In, None,
      @ Out, None
    """
    return
