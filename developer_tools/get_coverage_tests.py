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
from __future__ import print_function, unicode_literals
import os
import sys

ravenDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
testDir = os.path.join(ravenDir, 'tests')

def getRegressionTests(whichTests=1, skipExpectedFails=True):
  """
    Collects all the RAVEN regression tests into a dictionary keyed by directory.
    Must be run from this directory or another that is two directories below RAVEN.
    @ In, whichTests, integer, optional, the test type:
                                       - 1 => xml test files,
                                       - 2 => python tests,
                                       - 3 => interfaceCheck tests
                                       default 1 => xml test files
    @ In, skipExpectedFails, optional, bool, if True skips framework/ErrorCheck directory
    @ Out, dict, dict[dir] = list(filenames)
  """
  testsFilenames = []
  #search for all the 'tests' files
  for root, _, files in os.walk(testDir):
    if skipExpectedFails and 'ErrorChecks' in root.split(os.sep):
      continue
    if 'tests' in files:
      testsFilenames.append((root, os.path.join(root, 'tests')))
  suffix = ".xml" if whichTests in [1, 3] else ".py"
  #read all "input" node files from "tests" files
  doTests = {}
  for root, testFilename in testsFilenames:
    testsFile = open(testFilename, 'r')
    # collect the test specs in a dictionary
    testFileList = []
    testSpecs = {}
    startReading = False
    collectSpecs = False
    for line in testsFile:
      if line.strip().startswith("#"):
        continue
      if line.strip().startswith("[../]"):
        collectSpecs = True
        startReading = False
      if startReading:
        splitted = line.strip().split('=')
        if len(splitted) == 2:
          testSpecs[splitted[0].strip()] = splitted[1].replace("'", "").replace('"', '').strip()
      if line.strip().startswith("[./"):
        startReading = True
        collectSpecs = False
      if collectSpecs:
        # collect specs
        testFileList.append(testSpecs)
        collectSpecs = False
        testSpecs = {}
    if root not in doTests.keys():
      doTests[root] = []
    # now we have all the specs collected
    for spec in testFileList:
      # check if test is skipped or an executable is required
      if "required_executable" in spec or "skip" in spec:
        continue
      if "input" not in spec:
        continue
      testType = spec.get('type', "notfound").strip()
      newTest = spec['input'].split()[0]
      testInterfaceOnly = False
      if 'test_interface_only' in spec:
        testInterfaceOnly = True if spec['test_interface_only'].lower() == 'true' else False
      if whichTests in [1, 3]:
        if newTest.endswith(suffix) and testType.lower() not in 'ravenpython':
          if whichTests == 3 and testInterfaceOnly:
            doTests[root].append(newTest)
          elif whichTests == 1 and not testInterfaceOnly:
            doTests[root].append(newTest)
      else:
        if newTest.endswith(suffix) and testType.lower() in 'ravenpython':
          doTests[root].append(newTest)
  return doTests

if __name__ == '__main__':
  # skip the expected failed tests
  skipFails = True if '--skip-fails' in sys.argv else  False
  if '--get-python-tests' in sys.argv:
    # unit tests flag has priority over interface check
    which = 2
  elif '--get-interface-check-tests' in sys.argv:
    which = 3

  tests = getRegressionTests(which, skipExpectedFails=skipFails)
  #print doTests
  testFiles = []
  for key in tests:
    testFiles.extend([os.path.join(key, l) for l in tests[key]])
  print(' '.join(testFiles))
