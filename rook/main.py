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
This module is the main program for running tests.
"""
from __future__ import division, print_function, absolute_import
import warnings

import os
import sys
import argparse
try:
  import configparser
except ImportError:
  import ConfigParser as configparser
import re
import inspect
import time
import threading
import signal
import traceback

try:
  import psutil
  psutil_avail = True
except ImportError:
  psutil_avail = False
import pool
import trees.TreeStructure
from Tester import Tester, Differ

warnings.simplefilter('default', DeprecationWarning)

# set up colors
class UseColors:
  """
    Container for color definitions when using colors.
  """
  norm = '\033[0m'  #reset color
  skip = '\033[90m' #dark grey
  fail = '\033[91m' #red
  okay = '\033[92m' #green
  name = '\033[93m' #yellow
  time = '\033[94m' #blue

class NoColors:
  """
    Container for color definitions when not using colors.
  """
  norm = ''
  skip = ''
  fail = ''
  okay = ''
  name = ''
  time = ''

parser = argparse.ArgumentParser(description="Test Runner")
parser.add_argument('-j', '--jobs', dest='number_jobs', type=int, default=1,
                    help='Specifies number of tests to run simultaneously (default: 1)')
parser.add_argument('--re', dest='test_re_raw', default='.*',
                    help='Only tests with this regular expression inside will be run')
parser.add_argument('-l', dest='load_average', type=float, default=-1.0,
                    help='wait until load average is below the number before starting a new test')
parser.add_argument('-t', action='store_true',
                    help='argument added by civet recipes but unused in rook')
parser.add_argument('--heavy', action='store_true',
                    help='Run only heavy tests')
parser.add_argument('--no-color', action='store_true',
                    help='disable ANSI escape colors')

parser.add_argument('--list-testers', action='store_true', dest='list_testers',
                    help='Print out the possible testers')
parser.add_argument('--test-dir', dest='test_dir',
                    help='specify where the tests are located')

parser.add_argument('--testers-dir', dest='testers_dirs',
                    help='specify where the scripts are located. May be comma-separated.')

parser.add_argument('--run-types', dest='add_run_types',
                    help='add run types to the ones to be run')
parser.add_argument('--only-run-types', dest='only_run_types',
                    help='only run the listed types')
parser.add_argument('--add-non-default-run-types',
                    dest='add_non_default_run_types',
                    help='add a run type that is not run by default')

parser.add_argument('--command-prefix', dest='command_prefix',
                    help='prefix for the test commands')

parser.add_argument('--python-command', dest='python_command',
                    help='command to run python')

parser.add_argument('--config-file', dest='config_file',
                    help='Configuration file location')

parser.add_argument('--unkillable', action='store_true',
                    help='Ignore SIGTERM so test running is harder to be killed')

parser.add_argument('--add-path', dest='add_path',
                    help='additional paths that need be added in PYTHON PATH (sys.path)')

parser.add_argument('--update-or-add-env-variables', dest='update_or_add_env_variables',
                    help='comma separated list of environment variables to update or add. ' +
                    'The syntax is at follows: NAME=NEW_VALUE (if a new env variable needs ' +
                    'to be created or updated), NAME>NEW_VALUE (if an env variable needs to' +
                    ' be updated appending NEW_VALUE to it).')

args = parser.parse_args()

if args.config_file is not None:
  config = configparser.ConfigParser()
  print('rook: loading init file "{}"'.format(args.config_file))
  config.read(args.config_file)

  if 'rook' in config:
    for key in config['rook']:
      if key in args and getattr(args, key) is None:
        value = config['rook'][key]
        print('rook: ... loaded setting "{} = {}"'.format(key, value))
        setattr(args, key, value)
  else:
    print("No section [rook] in config file ", args.config_file)


class LoadClass(threading.Thread):
  """
    This class keeps track of the one second load average
  """

  def __init__(self):
    """
      Initialize this class
      @ In, None
      @ Out, None
    """
    threading.Thread.__init__(self)
    self.__load_avg = psutil.cpu_percent(1.0)*psutil.cpu_count()/100.0
    self.__smooth_avg = self.__load_avg
    self.__load_lock = threading.Lock()
    self.daemon = True #Exit even if this thread is running.

  def run(self):
    """
      Run forever, grabbing the load average every second
      @ In, None
      @ Out, None
    """
    while True:
      load_avg = psutil.cpu_percent(1.0)*psutil.cpu_count()/100.0
      with self.__load_lock:
        self.__load_avg = load_avg
        self.__smooth_avg = 0.9*self.__smooth_avg + 0.1*load_avg

  def get_load_average(self):
    """
      Get the most recent load average
      @ In, None,
      @ Out, float value for load average (average number of processors running)
    """
    load_avg = -1
    with self.__load_lock:
      load_avg = self.__load_avg
    return load_avg

  def get_smooth_average(self):
    """
      Get the most recent smooth average
      @ In, None,
      @ Out, float value for smooth average (average number of processors running) but smoothed
    """
    smooth_avg = -1
    with self.__load_lock:
      smooth_avg = self.__smooth_avg
    return smooth_avg

if args.load_average > 0:
  if not psutil_avail:
    print("No module named 'psutil' and load average specified")
    sys.exit(-1)
  load = LoadClass()
  load.start()

def load_average_adapter(function):
  """
    Adapts function to not start until load average is low enough
    @ In, function, function, function to call
    @ Out, new_func, function, function that checks load average before running
  """
  def new_func(data):
    """
      function that waits until load average is lower.
      @ In, data, Any, data to pass to function
      @ Out, result, result of running function on data
    """
    #basically get the load average for 0.1 seconds:
    while load.get_smooth_average() > args.load_average:
      time.sleep(1.0)
    return function(data)
  return new_func

def get_test_lists(directories):
  """
    Returns a list of all the files named tests under the directory
    @ In, directory, string, the directory to start at
    @ Out, dir_test_list, list, the files named tests
  """
  dir_test_list = []
  found = 0
  for directory in directories:
    for root, _, files in os.walk(directory):
      if 'tests' in files:
        dir_test_list.append((root, os.path.join(root, 'tests')))
    print('rook: found {} test dirs under "{}" ...'.format(len(dir_test_list) - found, directory))
    found = len(dir_test_list)
  return dir_test_list

def get_testers_and_differs(directory):
  """
    imports all the testers and differs in a directory
    @ In, directory, string, directory to search
    @ Out, (tester_dict, differ_dict), tuple of dictionaries
      returns dictionaries with all the subclasses of Tester and Differ.
  """
  # if no testers added, that's fine
  if not os.path.isdir(directory):
    print("invalid tester directory: "+directory)
    return {}, {}
  tester_dict = {}
  differ_dict = {}
  os.sys.path.append(directory)
  for filename in os.listdir(directory):
    if filename.endswith(".py") and not filename.startswith("__"):
      try:
        module = __import__(filename[:-3]) #[:-3] to remove .py
        for name, val in module.__dict__.items():
          if inspect.isclass(val) and val is not Tester\
             and issubclass(val, Tester):
            tester_dict[name] = val
          if inspect.isclass(val) and val is not Differ\
             and issubclass(val, Differ):
            differ_dict[name] = val
      except Exception as ex:
        print("Failed loading",filename,"with exception:",ex)
        traceback.print_exc()

  return tester_dict, differ_dict

def sec_format(runtime):
  """
    Formats the runtime into a string of the number seconds.
    If runtime is none, format as None!
    @ In, runtime, float or None, runtime to be formated.
    @ Out, sec_format, string of runtime.
  """
  if isinstance(runtime, float):
    return "{:6.2f}sec".format(runtime)
  return "  None!  "

def process_result(index, _input_data, output_data):
  """
    This is a callback function that Processes the result of a test.
    @ In, index, int, Index into functions list.
    @ In, _input_data, ignored, the input data passed to the function
    @ In, output_data, Tester.TestResult the output data passed to the function
    @ Out, None
  """
  group = output_data.group
  process_test_name = test_name_list[index]
  if group == Tester.group_success:
    results["pass"] += 1
    for postreq in function_postreq.get(process_test_name, []):
      if postreq in name_to_id:
        job_id = name_to_id[postreq]
        print("Enabling", postreq, job_id)
        run_pool.enable_job(job_id)
    okaycolor = Colors.okay
  elif group == Tester.group_skip:
    results["skipped"] += 1
    print(output_data.message)
    okaycolor = Colors.skip
  else:
    results["fail"] += 1
    failed_list.append(Tester.get_group_name(group)+" "+process_test_name)
    print("Output of'"+process_test_name+"':")
    print(output_data.output)
    print(output_data.message)
    okaycolor = Colors.fail
  number_done = sum(results.values())
  if results["fail"] > 0:
    done = "{0}F{1}".format(number_done,results["fail"])
  else:
    done = number_done
  print(' '.join(["({done}/{togo})",
                  "{statcolor}{status:7s}{normcolor}"
                  "({timecolor}{time}{normcolor})"
                  "{namecolor}{test}{normcolor}"])
        .format(done=done,
                togo=len(function_list),
                statcolor=okaycolor,
                normcolor=Colors.norm,
                namecolor=Colors.name,
                timecolor=Colors.time,
                status=Tester.get_group_name(group),
                time=sec_format(output_data.runtime),
                test=process_test_name))

if __name__ == "__main__":
  if args.unkillable:
    def term_handler(signum, _):
      """
        Ignores the termination signal
        @ In, signum, integer, the signal sent in
        @ Out, None
      """
      print("termination signal("+str(signum)+") ignored")

    signal.signal(signal.SIGTERM, term_handler)

  if args.no_color:
    Colors = NoColors
  else:
    Colors = UseColors

  if args.add_path:
    # add additional paths
    for new_path in args.add_path.split(","):
      print('rook: added new path "{}" in sys.path.'.format(new_path.strip()))
      sys.path.append(new_path.strip())
  if args.update_or_add_env_variables:
    # update enviroment variable
    for new_env_var in args.update_or_add_env_variables.split(","):
      sep = "=" if "=" in new_env_var else ">"
      if sep not in new_env_var:
        raise IOError('Syntax for enviroment variable setting must be ENV_VAR=VALUE ' +
                      '(for replacement) or ENV_VAR>VALUE (for update)')
      env_var_name, env_var_value = new_env_var.split(sep)
      cur_env_var = os.environ.get(env_var_name.strip(), "None")
      if sep == ">":
        env_var_value = cur_env_var + env_var_value if cur_env_var != "None" else env_var_value
      print('rook: update enviroment variable "{}" from "{}" to "{}".'.format(env_var_name,
                                                                              cur_env_var,
                                                                              env_var_value))
      os.environ[env_var_name] = env_var_value

  test_re = re.compile(args.test_re_raw)

  this_dir = os.path.abspath(os.path.dirname(__file__))
  up_one_dir = os.path.dirname(this_dir)
  if args.test_dir is None:
    #XXX fixme to find a better way to the tests directory

    base_test_dir = [os.path.join(up_one_dir, "tests")]
  else:
    base_test_dir = [x.strip() for x in args.test_dir.split(',')]

  test_list = get_test_lists(base_test_dir)

  base_testers, base_differs = get_testers_and_differs(this_dir)
  if not args.testers_dirs:
    testers_dirs = [os.path.join(up_one_dir, "scripts", "TestHarness", "testers")]
  else:
    testers_dirs = args.testers_dirs.split(',')
  testers = {}
  differs = {}
  for testers_dir in testers_dirs:
    new_testers, new_differs = get_testers_and_differs(testers_dir)
    testers.update(new_testers)
    differs.update(new_differs)
  testers.update(base_testers)
  differs.update(base_differs)
  Tester.add_non_default_run_type("heavy")
  if args.add_non_default_run_types is not None:
    non_default_run_types = args.add_non_default_run_types.split(",")
    for ndrt in non_default_run_types:
      Tester.add_non_default_run_type(ndrt)

  if args.list_testers:
    print("Testers:")
    for tester_name, tester in testers.items():
      print("Tester:", tester_name)
      print(tester.get_valid_params())
      print()
    print("Differs:")
    for differ_name, differ in differs.items():
      print("Differ:", differ_name)
      print(differ.get_valid_params())
      print()

  tester_params = {}
  for tester_key, tester_value in testers.items():
    #Note as a side effect, testers can add run types to
    # the tester.
    tester_params[tester_key] = tester_value.get_valid_params()

  Tester.initialize_current_run_type()
  if args.add_run_types is not None:
    Tester.add_run_types(set(el.strip() for el in args.add_run_types.split(",")))

  if args.only_run_types is not None:
    Tester.set_only_run_types(set(el.strip() for el in args.only_run_types.split(",")))

  function_list = [] #Store the data for the pool runner
  test_name_list = []
  ready_to_run = []
  function_postreq = {} #If this is non-empty for a key, enable the postreq's
  name_to_id = {}

  for test_dir, test_file in test_list:
    #print(test_file)
    tree = trees.TreeStructure.getpot_to_input_node(open(test_file, 'r'))
    for node in tree:
      #print(node.tag)
      #print(node.attrib)
      param_handler = tester_params[node.attrib['type']]
      if not param_handler.check_for_required(node.attrib):
        raise IOError("Missing Parameters in: " + node.tag + " for Tester: " + node.attrib['type'])
      if not param_handler.check_for_all_known(node.attrib):
        print("Unknown Parameters in:", node.tag, test_file)
      rel_test_dir = test_dir#[len(base_test_dir)+1:]
      test_name = rel_test_dir+os.sep+node.tag
      if "prereq" in node.attrib:
        prereq = node.attrib['prereq']
        prereq_name = rel_test_dir+os.sep+prereq
        l = function_postreq.get(prereq_name, [])
        l.append(test_name)
        function_postreq[prereq_name] = l
        has_prereq = True
      else:
        has_prereq = False
      if test_re.search(test_name):
        params = dict(node.attrib)
        params['test_dir'] = test_dir
        tester = testers[node.attrib['type']](test_name, params)
        if args.command_prefix is not None:
          tester.set_command_prefix(args.command_prefix)
        if args.python_command is not None:
          tester.set_python_command(args.python_command)
        if args.heavy:
          tester.run_heavy()
        for child in node.children:
          #print(test_name,"child",child)
          child_type = child.attrib['type']
          child_param_handler = differs[child_type].get_valid_params()
          if not child_param_handler.check_for_required(child.attrib):
            raise IOError("Missing Parameters in: " +  child.tag + "/" + node.tag +
                          " for Differ: " + child_type + " in test file: "+ test_file)
          if not child_param_handler.check_for_all_known(child.attrib):
            print("Unknown Parameters in:", child.tag, node.tag, test_file)
          differ = differs[child_type](child.tag, dict(child.attrib), test_dir)
          tester.add_differ(differ)
        id_num = len(function_list)
        func = tester.run
        if args.load_average > 0:
          func = load_average_adapter(func)
        function_list.append((func, (test_dir)))
        test_name_list.append(test_name)
        ready_to_run.append(not has_prereq)
        name_to_id[test_name] = id_num

  run_pool = pool.MultiRun(function_list, args.number_jobs, ready_to_run)

  run_pool.run()

  results = {"pass":0, "fail":0, "skipped":0}
  failed_list = []

  output_list = run_pool.process_results(process_result)
  run_pool.wait()

  if results["fail"] > 0:
    print("{}FAILED:".format(Colors.fail))
  for path in failed_list:
    print(path)
  print(Colors.norm)

  with open("test_report.csv", "w") as csv_report:
    csv_report.write(",".join(["name", "passed", "group", "time"])+"\n")
    for result, test_name in zip(output_list, test_name_list):
      if result is not None:
        group_name = Tester.get_group_name(result.group)
        out_line = ",".join([test_name, str(result.group == Tester.group_success),
                             group_name, str(result.runtime)])
      else:
        out_line = ",".join([test_name, str(False), "NO_PREREQ", str(0.0)])
      csv_report.write(out_line+"\n")
    csv_report.close()

  print("PASSED: {}{}{}".format(Colors.okay, results["pass"], Colors.norm))
  print("SKIPPED: {}{}{}".format(Colors.skip, results["skipped"], Colors.norm))
  print("FAILED: {}{}{}".format(Colors.fail, results["fail"], Colors.norm))
  sys.exit(results["fail"])
