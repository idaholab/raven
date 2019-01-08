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
from __future__ import division, print_function, absolute_import
import warnings

import sys
from UnorderedCSVDiffer import UnorderedCSVDiffer as UCSV

warnings.simplefilter('default', DeprecationWarning)

def check_same(comment, first, second, local_msg, local_results):
  """
    checks that the first and second are the same.
    @ In, comment, string, comment if failed.
    @ In, first, Any, first thing to compare
    @ In, second, Any, second thing to compare
    @ In, local_msg, printable, extra thing to print if failed
    @ In, local_results, dictionary, dictionary of results
  """
  if first == second:
    local_results['pass'] += 1
  else:
    local_results['fail'] += 1
    print('FAILED '+comment)
    print(local_msg)
    print('')

def test_a_file(fname):
  """
    Tests the file
    @ In, fname, string, filename string
    @ Out, test_a_file, (same, message), (bool, str) result of test.
  """
  differ = UCSV('.', [fname], zeroThreshold=5e-14)
  differ.diff()
  return differ.__dict__['_UnorderedCSVDiffer__same'],\
    differ.__dict__['_UnorderedCSVDiffer__message']

if __name__ == '__main__':
  results = {'pass':0, 'fail':0}
  # passes
  ok, msg = test_a_file('okay.csv')
  check_same('Okay', ok, True, msg, results)
  # mismatch
  ok, msg = test_a_file('mismatch.csv')
  check_same('Mismatch', ok, False, msg, results)
  # matching with inf, nan
  ok, msg = test_a_file('inf.csv')
  check_same('Infinity', ok, True, msg, results)
  # zero threshold
  ok, msg = test_a_file('nearzero.csv')
  check_same('Near zero', ok, True, msg, results)
  # sorting
  ok, msg = test_a_file('sort.csv')
  check_same('sort', ok, True, msg, results)




  print('Passed:', results['pass'], '| Failed:', results['fail'])
  sys.exit(results['fail'])
