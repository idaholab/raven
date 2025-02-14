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
import os
import argparse

ravenDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

parser = argparse.ArgumentParser(description="Find tests")
parser.add_argument('--tests-dir', dest='tests_dir', type=str, default=os.path.join(ravenDir, 'tests'),
                    help="Test directory in which to search for tests")
parser.add_argument('--get-test-names', action='store_true', dest='find_test_names',
                    help='Find all test names')
parser.add_argument('--get-test-input-filenames', action='store_true', dest='find_input_filenames',
                    help='Find all test filenames')
parser.add_argument('--skip-fails', action='store_true', dest='skip_fails',
                    help='Skip the expected failed tests')
parser.add_argument('--get-python-tests', action='store_true', dest='get_python_tests',
                    help='Find python (unit) tests only')
parser.add_argument('--get-interface-check-tests', action='store_true', dest='get_interface_check_tests',
                    help='Find interface tests only')
parser.add_argument('--get-all-tests', action='store_true', dest='get_all_tests',
                    help='Find all tests')
args = parser.parse_args()

def getRegressionTests(whichTests=1, skipExpectedFails=True, groupBy='directory'):
  """
    Collects all relevant tests into a dictionary keyed as specified.
    Must be run from this directory or another that is two directories below RAVEN.
    @ In, whichTests, integer, optional, the test type:
                                       - 1 => xml test files,
                                       - 2 => python tests,
                                       - 3 => interfaceCheck tests,
                                       - 4 => all tests
                                       default 1 => xml test files
    @ In, skipExpectedFails, optional, bool, if True skips framework/ErrorCheck directory
    @ In, groupBy, optional, str, how to sort the test info:
                                       - "directory" => output dict keyed by directories with
                                                        values of test input files
                                       - "test_name" => output dict keyed by test names with
                                                        values of dicts containing test info
                                       default "directory" => output dict keyed by directories
    @ Out, dict, dict[dir] = list(filenames) OR dict[test_name] = dict[test_info]
  """
  if (groupBy != 'directory') and (groupBy != 'test_name'):
    print("Unrecognized input for groupBy: ", groupBy)
    print("Allowed values for groupBy input are 'directory' and 'test_name'. Defaulting to 'directory'.")
    groupBy = 'directory'
  testsFilenames = []

  # Search for all the 'tests' files
  for root, _, files in os.walk(args.tests_dir):
    if skipExpectedFails and 'ErrorChecks' in root.split(os.sep):
      continue
    if 'tests' in files:
      testsFilenames.append((root, os.path.join(root, 'tests')))

  # Read all "input" node files from "tests" files
  doTests = {}
  for root, testFilename in testsFilenames:
    testsFile = open(testFilename, 'r')
    # Collect the test specs in a dictionary
    testList = []
    testSpecs = {}
    startReading = False
    collectSpecs = False # Flag to save specs and move to the next test
    depth = 0 # Current location in hierarchy
    for line in testsFile:
      if line.strip().startswith("#"):
        continue

      if line.strip().startswith("[../]"):
        depth -= 1 # Going up a level
        if depth == 0: # That's all for this test
          startReading = False
          collectSpecs = True
        else: # That's all for this differ
          startReading = True
          collectSpecs = False

      if line.strip().startswith("[./"):
        if depth == 0: # This line contains the test name
          testSpecs['test_name'] = os.path.normpath(os.path.join(root, line.strip()[3:-1]))
          testSpecs['test_directory'] = root
          startReading = True
          collectSpecs = False
        else: # This line is the start of a differ
          startReading = False
          collectSpecs = False
        depth += 1 # Going down a level

      if startReading:
        splitted = line.strip().split('=', 1)
        if len(splitted) == 2:
          testSpecs[splitted[0].strip()] = splitted[1].replace("'", "").replace('"', '').strip()

      if collectSpecs:
        # Collect specs
        testList.append(testSpecs)
        collectSpecs = False
        testSpecs = {}
    # Now we have all the specs collected

    if groupBy == 'directory' and root not in doTests.keys():
      doTests[root] = []

    for spec in testList:
      # Check if test is skipped or an executable is required
      if "required_executable" in spec or "skip" in spec:
        continue
      if "input" not in spec:
        continue
      testType = spec.get('type', "notfound").strip()

      if whichTests == 4:
        addTest = True
      else:
        addTest = False
        testInterfaceOnly = False
        if 'test_interface_only' in spec:
          testInterfaceOnly = True if spec['test_interface_only'].lower() == 'true' else False
        if spec['input'].endswith('.xml') and '.py' not in spec['input']:
          if whichTests == 3 and testInterfaceOnly:
            addTest = True
          elif whichTests == 1 and not testInterfaceOnly:
            addTest = True
        else:
          if whichTests == 2 and testType.lower() != 'crowpython':
            addTest = True

      if addTest:
        if groupBy == 'directory':
          for newTestFile in spec['input'].split(' '):
            if whichTests == 4 and (newTestFile.endswith('.py') or
                                    (newTestFile.endswith('.xml') and '.py' not in spec['input'])):
              doTests[root].append(newTestFile)
            elif whichTests in [1, 3] and newTestFile.endswith('.xml'):
              doTests[root].append(newTestFile)
            elif whichTests == 2 and newTestFile.endswith('.py'):
              doTests[root].append(newTestFile)
        else:
          for newTestFile in spec['input'].split(' '):
            if whichTests == 4 and (newTestFile.endswith('.py') or
                                    (newTestFile.endswith('.xml') and '.py' not in spec['input'])):
              doTests[spec['test_name']] = spec
            elif whichTests in [1, 3] and newTestFile.endswith('.xml'):
              doTests[spec['test_name']] = spec
            elif whichTests == 2 and newTestFile.endswith('.py'):
              doTests[spec['test_name']] = spec

  return doTests

if __name__ == '__main__':
  if args.find_test_names:
    # Test name flag has priority over input filename flag
    searchTarget = 'test_names'
  elif args.find_input_filenames:
    searchTarget = 'input_filenames'
  else:
    searchTarget = 'input_filenames'

  # Skip the expected failed tests
  skipFails = True if args.skip_fails else False

  if args.get_python_tests:
    # Unit tests flag has priority over interface check and all tests
    which = 2
  elif args.get_interface_check_tests:
    which = 3
  elif args.get_all_tests:
    which = 4
  else:
    which = 1
  if searchTarget == 'input_filenames':
    keysType = 'directory'
  else:
    keysType = 'test_name'

  tests = getRegressionTests(which, skipExpectedFails=skipFails, groupBy=keysType)

  targetList = []
  for key in tests:
    if searchTarget == 'input_filenames':
      # Keys are directories, values are input filenames
      targetList.extend([os.path.join(key, l) for l in tests[key]])
    elif searchTarget == 'test_names':
      # Keys are test names, values are dicts with test specs
      targetList.append(key)

  print(' '.join(targetList))
