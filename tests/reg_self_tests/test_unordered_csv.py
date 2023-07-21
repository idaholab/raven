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

import os
import sys

test_system_dir = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "TestHarness", "testers"
        )
    )
print(test_system_dir)
sys.path.append(test_system_dir)
from UnorderedCSVDiffer import UnorderedCSVDiffer as UCSV


def check_same(comment, first, second, localMsg, localResults):
  """
    checks that the first and second are the same.
    @ In, comment, string, comment if failed.
    @ In, first, Any, first thing to compare
    @ In, second, Any, second thing to compare
    @ In, localMsg, printable, extra thing to print if failed
    @ In, localResults, dictionary, dictionary of results
  """
  if first == second:
    localResults['pass'] += 1
  else:
    localResults['fail'] += 1
    print('FAILED '+comment)
    print(localMsg)
    print('')

def test_a_file(fname):
  """
    Tests the file
    @ In, fname, string, filename string
    @ Out, test_a_file, (same, message), (bool, str) result of test.
  """
  differ = UCSV([fname], [f'gold/{fname}'], zeroThreshold=5e-14)
  differ.diff()
  return differ._same, differ._message

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
