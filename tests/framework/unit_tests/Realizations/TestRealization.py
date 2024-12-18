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
  This Module performs Unit Tests for the Realization objects.
"""

import os
import sys
import unittest

# find location of crow, message handler
ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(ravenDir)

from ravenframework import Realizations

class TestRealization(unittest.TestCase):
  """
    Unit tests for Realization object
  """
  def setUp(self):
    """
      Named unittest method to set up tests
      Also indirectly tests setting variable values in realization
      Runs before each test_* method
      @ In, None
      @ Out, None
    """
    self.rlz = Realizations.Realization()
    # var-value pairs to use in testing
    self.a = 3
    self.b = 42
    self.pi = 3.14159
    self.rlz['a'] = self.a   # str var, integer vals
    self.rlz['b'] = self.b   #
    self.rlz['pi'] = self.pi # float val
    self.rlz[5] = 'c'        # integer var, string val


  def test_membership(self):
    """ tests checking for membership """
    for member in ['a', 'b', 'pi', 5]:
      self.assertTrue(member in self.rlz, f'member "{member}" not found in realization')
    for nonmember in ['d', 2, 1.618, '_values']:
      self.assertTrue(nonmember not in self.rlz, f'nonmember "{nonmember}" found in realization')

  def test_getitem(self):
    """ tests accessing variables and values """
    self.assertEqual(self.rlz['b'], 42, 'incorrect stored value of "b"')
    self.assertEqual(self.rlz['a'], 3, 'incorrect stored value of "a"')
    self.assertEqual(self.rlz['pi'], 3.14159, 'incorrect stored value of "pi"')
    self.assertEqual(self.rlz[5], 'c', 'incorrect stored value of "5"')

  def test_get(self):
    """ tests accessing variables and default values """
    self.assertEqual(self.rlz.get('a'), 3, 'incorrect "get" for "a"')
    self.assertEqual(self.rlz.get('d', 15), 15, 'incorrect "get" default value')

  def test_len(self):
    """ tests measuring vector length """
    self.assertEqual(len(self.rlz), 4, 'incorrect length')

  def test_del(self):
    """ tests removing an entry """
    del self.rlz['b']
    self.assertTrue('b' not in self.rlz, '"b" still in rlz despite removal')

  def test_iter(self):
    """ tests iterating over entries """
    expected = ['a', 'b', 'pi', 5]
    for i, k in enumerate(self.rlz):
      self.assertEqual(expected[i], k, f'unexpected iter key "{i}"')

  def test_keys(self):
    """ tests iterating over keys """
    expected = ['a', 'b', 'pi', 5]
    for i, k in enumerate(self.rlz.keys()):
      self.assertEqual(expected[i], k, f'unexpected "keys" key "{i}"')

  def test_values(self):
    """ tests iterating over values """
    expected = [self.a, self.b, self.pi, 'c']
    for i, k in enumerate(self.rlz.values()):
      self.assertEqual(expected[i], k, f'unexpected "values" value "{i}"')

  def test_items(self):
    """ tests iterating over key-value pairs """
    expectKeys = ['a', 'b', 'pi', 5]
    expectValues = [self.a, self.b, self.pi, 'c']
    for i, (k, v) in enumerate(self.rlz.items()):
      self.assertEqual(expectKeys[i], k, f'unexpected "items" key "{i}"')
      self.assertEqual(expectValues[i], v, f'unexpected "items" value "{i}"')

  def test_update(self):
    """ tests update method """
    new = {'a': 30,    # update old entry
          'b': 420,    # add back old entry in new position
          5: 'c2',     # update old entry
          'new': 372}  # new entry
    self.rlz.update(new)
    expectKeys =   ['a', 'b',   'pi', 5,  'new']
    expectValues = [30, 420, 3.14159, 'c2', 372]
    for i, (k, v) in enumerate(self.rlz.items()):
      self.assertEqual(expectKeys[i], k, f'unexpected "update" key "{i}"')
      self.assertEqual(expectValues[i], v, f'unexpected "update" value "{i}"')

  def test_pop(self):
    """ tests pop method """
    val = self.rlz.pop(5)
    self.assertEqual(val, 'c', 'incorrect "pop" value')
    self.assertFalse(5 in self.rlz, 'member present after "pop"')


if __name__ == '__main__': # Not run when unittest called from command line or Unittest tester is used
  unittest.main()

  # <TestInfo>
  #   <name>framework.test_realization</name>
  #   <author>talbpaul</author>
  #   <created>2024-10-23</created>
  #   <classesTested>Realization</classesTested>
  #   <description>
  #      This test is a Unit Test for the Realization class.
  #   </description>
  # </TestInfo>
