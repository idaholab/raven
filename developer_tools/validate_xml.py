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
import subprocess
import shutil
import get_coverage_tests
import xmlschema

scriptDir = os.path.dirname(os.path.abspath(__file__))
conversionDir = os.path.join(scriptDir, '..', 'scripts', 'conversionScripts')
tlen = shutil.get_terminal_size((80, 20))[0]
maxlen = min(160, tlen)

def validateTests():
  """
    Runs validation tests on regression tests and displays results.
    @ In, None
    @ Out, int, number of failed tests.
  """
  print('Beginning test validation...')
  tests = get_coverage_tests.getRegressionTests(skipExpectedFails=True)
  schema = os.path.join(scriptDir, 'XSDSchemas', 'raven.xsd')
  mySchema = xmlschema.XMLSchema11(schema)
  res = [0, 0, 0] #run, pass, fail
  failed = {}
  devnull = open(os.devnull, "wb")
  for dr, files in tests.items():
    #print directory being checked'
    checkmsg = ' Directory: ' + dr
    print(colors.neutral + checkmsg.rjust(maxlen, '-'))
    #check files in directory
    for f in files:
      fullpath = os.path.join(dr, f)
      # check if input exists; if not, it probably gets created by something else, and can't get checked here
      if not os.path.isfile(fullpath):
        print('File "{}" does not exist; skipping validation test ...'.format(fullpath))
        continue
      res[0] += 1
      startmsg = 'Validating ' + f
      #expand external XML nodes
      # - first, though, check if the backup file already exists
      if os.path.isfile(fullpath + '.bak'):
        print(colors.neutral + 'Could not check for ExternalXML since a backup file exists! Please remove it to validate.')
      # Since directing output of shell commands to file /dev/null is problematic on Windows, this equivalent form is used
      #   which is better supported on all platforms.
      cmd = 'python ' + os.path.join(conversionDir, 'externalXMLNode.py') + ' ' + fullpath
      result = subprocess.call(cmd, shell=True, stdout=devnull)
      err = 'Error running ' + cmd

      screenmsg = colors.neutral + startmsg + ' '
      if result == 0:
        #run xmlschema library
        result = mySchema.is_valid(fullpath)
      if result:# == 0: #success
        res[1] += 1
        endmsg = 'validated'
        endcolor = colors.ok
        postprint = ''
      else:
        try:
          err = mySchema.validate(fullpath)
        except xmlschema.validators.exceptions.XMLSchemaValidationError as ae:
          err = " {}".format(str(ae))
        res[2] += 1
        endmsg = 'FAILED'
        endcolor = colors.fail
        postprint = colors.fail+str(err)
        if dr not in failed.keys():
          failed[dr] = []
        failed[dr].append(f)
      screenmsg += colors.neutral + ''.rjust(maxlen - len(startmsg) - len(endmsg) - 1, '.')
      screenmsg += endcolor + endmsg + colors.neutral

      if len(postprint):
        screenmsg += postprint + colors.neutral
      print(screenmsg)
      #return externalNode xmls
      os.system('mv {orig}.bak {orig}'.format(orig=fullpath))
  print(colors.neutral + '')
  print('-----------------------------------------------------------------------')
  print('                           Failure Summary')
  print('-----------------------------------------------------------------------')
  for dr, files in failed.items():
    for f in files:
      print(colors.fail + os.path.join(dr, f))
  print(colors.neutral + '')
  print('Validated: {ok} Failed: {fail} Run: {run}'.format(ok=colors.ok + str(res[1]) + colors.neutral,
                                                           fail=colors.fail + str(res[2]) + colors.neutral,
                                                           run=res[0]))
  return res[2]

class colors:
  ok       = '\033[92m'
  fail     = '\033[91m'
  neutral  = '\033[0m'

if __name__=='__main__':
  sys.exit(validateTests())
