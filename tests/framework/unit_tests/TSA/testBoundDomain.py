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
  This Module performs Unit Tests for the TSA.BoundDomain class.
  It cannot be considered part of the active code but of the regression test system
"""
import os
import sys
import unittest
import numpy as np
from numpy.testing import assert_array_equal

# add RAVEN to path
ravenDir =  os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4)))
frameworkDir = os.path.join(ravenDir, 'framework')
if ravenDir not in sys.path:
  sys.path.append(ravenDir)

from ravenframework.utils import xmlUtils
from ravenframework.TSA import BoundDomain

print('Modules undergoing testing:')
print(BoundDomain)
print('')


class TestBoundDomain(unittest.TestCase):
  """ Tests for the BoundDomain TSA transformer class """

  def setUp(self):
    """
      Sets up transformer object and input/output test data
      @ In, None
      @ Out, None
    """
    self.targets = ["signal"]
    self.signal = np.array([-np.inf, np.inf, np.nan, -1, 0, 1, 10]).reshape(-1, 1)
    self.lowerBound = -5.0
    self.upperBound = 5.0
    self.boundedSignal = np.array([-5, 5, np.nan, -1, 0, 1, 5]).reshape(-1, 1)
    self.pivots = np.arange(self.signal.shape[0])

    self.transformer = BoundDomain()
    xml = xmlUtils.newNode("bounddomain", attrib={"target": ",".join(self.targets)})
    xml.append(xmlUtils.newNode("lowerBound", text=str(self.lowerBound)))
    xml.append(xmlUtils.newNode("upperBound", text=str(self.upperBound)))
    spec = self.transformer.getInputSpecification()()
    spec.parseNode(xml)
    self.settings = self.transformer.handleInput(spec)

  def testSettings(self):
    """
      Test that settings are correctly read from XML
      @ In, None
      @ Out, None
    """
    self.assertDictContainsSubset({"lowerBound": self.lowerBound, "upperBound": self.upperBound}, self.settings)

  def testFit(self):
    """
      Test the `fit` method
      @ In, None
      @ Out, None
    """
    params = self.transformer.fit(self.signal, self.pivots, self.targets, self.settings)
    # params should have an entry for each fitted target
    for target in self.targets:
      self.assertIn(target, params)

  def testGetResidual(self):
    """
      Test the `getResidual` method
      @ In, None
      @ Out, None
    """
    params = self.transformer.fit(self.signal, self.pivots, self.targets, self.settings)
    residual = self.transformer.getResidual(self.signal, params, self.pivots, self.settings)
    assert_array_equal(residual, self.boundedSignal)

  def testGetComposite(self):
    """
      Test the `getComposite` method
      @ In, None
      @ Out, None
    """
    params = self.transformer.fit(self.signal, self.pivots, self.targets, self.settings)
    composite = self.transformer.getComposite(self.signal, params, self.pivots, self.settings)
    assert_array_equal(composite, self.boundedSignal)
