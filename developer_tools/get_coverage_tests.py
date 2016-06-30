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
    for line in testsFile:
      if line.strip().startswith('input'):
        newtest = line.split('=')[1].strip().strip("'")
        if newtest not in skipThese and newtest.endswith('.xml'):
          if root not in doTests.keys(): doTests[root]=[]
          doTests[root].append(newtest)
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
