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
import sys

ravenDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
testDir = os.path.join(ravenDir,'tests')

def getRegressionTests(skipThese=[],skipExpectedFails=True):
  """
    Collects all the RAVEN regression tests into a dictionary keyed by directory.
    Must be run from this directory or another that is two directories below RAVEN.
    @ In, skipThese, list(str), filenames to skip in check
    @ In, skipExpectedFails, bool, if True skips framework/ErrorCheck directory
    @ Out, dict, dict[dir] = list(filenames)
  """
  testsFilenames = []
  #search for all the 'tests' files
  for root, dirs, files in os.walk(testDir):
    if skipExpectedFails and 'ErrorChecks' in root.split(os.sep):
      continue
    if 'tests' in files:
      testsFilenames.append((root,os.path.join(root, 'tests')))
  #read all "input" node files from "tests" files
  doTests = {}
  for root,testFilename in testsFilenames:
    testsFile = file(testFilename,'r')
    testType = "notfound"
    for line in testsFile:
      if line.strip().startswith('type'):
        testType = line.strip().split('=')[1].replace("'","").replace('"','').strip()
      if line.strip().startswith('input'):
        newtest = line.split('=')[1].strip().strip("'")
        if newtest not in skipThese and newtest.endswith('.xml') and testType.lower() not in 'ravenpython':
          if root not in doTests.keys(): doTests[root]=[]
          doTests[root].append(newtest)
          testType = "notfound"
  return doTests

def getRegressionList(skipThese=[],skipExpectedFails=True):
  """
    Collects all the RAVEN regression tests into a list
    Must be run from this directory or another that is two directories below RAVEN.
    @ In, skipThese, list(str), filenames to skip in check
    @ In, skipExpectedFails, bool, if True skips framework/ErrorCheck directory
    @ Out, list(str), list of full paths to files
  """
  fileDict = getRegressionTests()
  filelist = []
  for dir,files in fileDict.items():
    for f in files:
      filelist.append(os.path.join(dir,f))
  return filelist

if __name__ == '__main__':
  if '--skip-fails' in sys.argv: skipFails = True
  else: skipFails = False
  skipThese = ['test_rom_trainer.xml','../../framework/TestDistributions.py']
  doTests = getRegressionTests(skipThese,skipExpectedFails = skipFails)
  #print doTests
  xmlTests = []
  for key in doTests:
    xmlTests.extend([os.path.join(key,l) for l in doTests[key]])
  print ' '.join(xmlTests)
