import os
import sys

raven_dir = os.getcwd().split('raven')[0]+'raven'
test_dir = os.path.join(raven_dir,'tests')

skip_these = ['test_rom_trainer.xml','../../framework/TestDistributions.py']
do_tests = []
testsFilenames = []

for root, dirs, files in os.walk(test_dir):
  for filename in files:
    if filename == "tests":
      testsFilenames.append((root,os.path.join(root, filename)))
for root,testFilename in testsFilenames:
  testsFile = file(testFilename,'r')
  for line in testsFile:
    if line.strip().startswith('input'):
      newtest = line.split('=')[1].strip().strip("'")
      if newtest not in skip_these and newtest.endswith('.xml'): do_tests.append(os.path.join(root,newtest))

print ' '.join(do_tests)

def returnTests():
  """
    Provides a python way to obtain test list instead of screen print
    @ In, None
    @ Out, None
  """
  return do_tests
