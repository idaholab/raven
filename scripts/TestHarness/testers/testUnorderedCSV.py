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
  Unit test for the UnorderedCSV differ
"""

import sys
from UnorderedCSVDiffer import UnorderedCSVDiffer as UCSV

def checkSame(comment,first,second,msg,results=None):
  if results is None:
    results = {'pass':0,'fail':0}
  if first == second:
    results['pass'] += 1
  else:
    results['fail'] += 1
    print 'FAILED '+comment
    print msg
    print ''
  return results

def testAFile(fname):
  differ = UCSV('.',[fname])
  differ.diff()
  return differ.__dict__['_UnorderedCSVDiffer__same'], differ.__dict__['_UnorderedCSVDiffer__message']

if __name__ == '__main__':
  results = {'pass':0,'fail':0}
  # passes
  ok,msg = testAFile('okay.csv')
  checkSame('Okay',ok,True,msg,results)
  # mismatch
  ok,msg = testAFile('mismatch.csv')
  checkSame('Mismatch',ok,False,msg,results)
  # matching with inf, nan
  ok,msg = testAFile('inf.csv')
  checkSame('Infinity',ok,True,msg,results)




  print 'Passed:',results['pass'],'| Failed:',results['fail']
  sys.exit(results['fail'])
