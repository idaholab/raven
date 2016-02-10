import os
import sys

raven_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
test_dir = os.path.join(raven_dir,'tests')
print 'DEBUG raven_dir:',raven_dir

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
  for root, dirs, files in os.walk(test_dir):
    if skipExpectedFails and 'ErrorChecks' in root.split(os.sep):
      continue
    if 'tests' in files:
      testsFilenames.append((root,os.path.join(root, 'tests')))
  #read all "input" node files from "tests" files
  do_tests = {}
  for root,testFilename in testsFilenames:
    testsFile = file(testFilename,'r')
    for line in testsFile:
      if line.strip().startswith('input'):
        newtest = line.split('=')[1].strip().strip("'")
        if newtest not in skipThese and newtest.endswith('.xml'):
          if root not in do_tests.keys(): do_tests[root]=[]
          do_tests[root].append(newtest)
  return do_tests

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
  do_tests = getRegressionTests(skipThese,skipExpectedFails = skipFails)
  print ' '.join(do_tests)
