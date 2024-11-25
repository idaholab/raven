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
  This Module performs Unit Tests for the BatchRealization objects.
"""

import os
import sys
import unittest

# find location of crow, message handler
ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(ravenDir)

from ravenframework import Realizations

class TestBatchRealization(unittest.TestCase):
  """
    Unit tests for the Realization object
  """

  def setUp(self):
    """
      Named unittest method to set up tests
      Runs before each test_* method
      @ In, None
      @ Out, None
    """
    self.batch = Realizations.RealizationBatch(2)
    self.batch[0]['a'] = 2
    self.batch[0]['b'] = 41
    self.batch[0]['pi'] = 2.14159
    self.batch[0][5] = 'b'
    self.batch[1]['a'] = 3
    self.batch[1]['b'] = 42
    self.batch[1]['pi'] = 3.14159
    self.batch[1][5] = 'c'

  def test_getitem(self):
    """ tests default indexed getter """
    r0a = self.batch[0]['a']
    r0b = self.batch[0]['b']
    r1a = self.batch[1]['a']
    r1b = self.batch[1]['b']
    self.assertEqual(r0a, 2, 'incorrect index 0 "a" value')
    self.assertEqual(r0b, 41, 'incorrect index 0 "b" value')
    self.assertEqual(r1a, 3, 'incorrect index 1 "a" value')
    self.assertEqual(r1b, 42, 'incorrect index 1 "b" value')

  def test_setitem(self):
    """ tests error on setting object """
    with self.assertRaises(IndexError, msg="setting item") as cm:
      self.batch[0] = 1.618

  def test_len(self):
    """ tests length builtin """
    self.assertEqual(len(self.batch), 2, 'incorrect "len"')

  def test_iter(self):
    """ tests iter builtin """
    for i, rlz in enumerate(self.batch):
      self.assertEqual(rlz['a'], 2 + i, f'iter index "{i}" key "a"')
      self.assertEqual(rlz['b'], 41 + i, f'iter index "{i}" key "b"')

  def test_pop(self):
    """ tests pop method """
    rlz = self.batch[1]
    self.assertTrue(rlz in self.batch, 'membership by realization object')
    popped = self.batch.pop()
    self.assertEqual(popped['b'], 42, 'value check in "pop" rlz')
    self.assertFalse(rlz in self.batch, 'membership after pop')


if __name__ == '__main__':
  unittest.main()

  # <TestInfo>
  #   <name>framework.test_realization</name>
  #   <author>talbpaul</author>
  #   <created>2024-10-23</created>
  #   <classesTested>BatchRealization</classesTested>
  #   <description>
  #      This test is a Unit Test for the BatchRealization class.
  #   </description>
  # </TestInfo>
