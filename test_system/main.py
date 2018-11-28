
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)


import os
import sys
import subprocess
import argparse
import re

import pool
import trees.TreeStructure

parser = argparse.ArgumentParser(description="Test Runner")
parser.add_argument('-j', '--jobs', dest='number_jobs', type=int, default=1,
                    help='Specifies number of tests to run simultaneously (default: 1)')
parser.add_argument('--re', dest='test_re_raw', default='.*',
                    help='Only tests with this regular expression inside will be run')
args = parser.parse_args()

test_re = re.compile(args.test_re_raw)

#XXX fixme to find a better way to the tests directory

this_dir = os.path.abspath(os.path.dirname(__file__))
base_test_dir = os.path.join(os.path.dirname(this_dir),"tests")

print(this_dir,base_test_dir)

def get_test_lists(directory):
  """
  Returns a list of all the files named tests under the directory
  directory: the directory to start at
  """
  test_list = []
  for root, dirs, files in os.walk(directory):
    if 'tests' in files:
      test_list.append((root,os.path.join(root,'tests')))
  return test_list

test_list = get_test_lists(base_test_dir)

def run_python_test(data):
  """
  runs a python test and if the return code is 0, it passes
  returns (passed,short_comment,long_comment) where pass is a boolean
  that is True if the test passes, and short_comment and long comment are
  comments on why it fails.
  data: (directory, code.py) Runs code.py in directory
  """
  directory, code_filename = data
  command = ["python3", code_filename]
  process = subprocess.Popen(command, shell=False,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             cwd=directory,
                             universal_newlines=True)
  output = process.communicate()[0]
  retcode = process.returncode
  passed = (retcode == 0)
  if passed:
    short = "Success"
  else:
    short = "Failed"
  return (passed, short, output)

function_list = [] #Store the data for the pool runner
test_name_list = []
for test_dir, test_file in test_list:
  #print(test_file)
  tree = trees.TreeStructure.parse(test_file, 'getpot')
  for node in tree.getroot():
    #print(node.tag)
    #print(node.attrib)
    if node.attrib['type'] in ['RavenPython','CrowPython']:
      input_filename = node.attrib['input']
      rel_test_dir = test_dir[len(base_test_dir)+1:]
      test_name = rel_test_dir+os.sep+node.tag
      if test_re.search(test_name):
        function_list.append((run_python_test, (test_dir, input_filename)))
        test_name_list.append(test_name)

run_pool = pool.MultiRun(function_list, args.number_jobs)

run_pool.run()

results = {"pass":0,"fail":0}
failed_list = []

def process_result(index, input_data, output_data):
  test_dir, input_filename = input_data
  passed, short_comment, long_comment = output_data
  test_name = test_name_list[index]
  print(test_name)
  print(passed, short_comment)
  if not passed:
    results["fail"] += 1
    failed_list.append(test_name)
    print(long_comment)
  else:
    results["pass"] += 1

output_list = run_pool.process_results(process_result)
run_pool.wait()

if results["fail"] > 0:
  print("FAILED:")
for path in failed_list:
  print(path)

print(results)
sys.exit(results["fail"])
