import os
import sys
import unittest
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from numpy.testing import assert_array_equal

# add RAVEN to path
ravenDir =  os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4)))
frameworkDir = os.path.join(ravenDir, 'framework')
if ravenDir not in sys.path:
  sys.path.append(ravenDir)

from ravenframework.TSA import BoundDomain
from ravenframework.utils import xmlUtils


class TestBoundDomain(unittest.TestCase):
  """ Unit tests for BoundDomain transformer """
  def setUp(self):
    """
      Set up test objects
      @ In, None
      @ Out, None
    """
    self.targets = ["signal"]
    self.signals = np.array([-np.inf, -5, -3, 0, 2, 5, np.inf, np.nan], dtype=float).reshape(-1, 1)
    self.signalsTransformed = np.array([-3, -3, -3, 0, 2, 2, 2, np.nan], dtype=float).reshape(-1, 1)
    self.pivots = np.arange(self.signals.shape[0])

    xml = xmlUtils.newNode("bounddomain", attrib={"target": "signal"})
    xml.append(xmlUtils.newNode("lowerBound", "-3"))
    xml.append(xmlUtils.newNode("upperBound", "2"))

    self.boundDomain = BoundDomain()
    inputSpec = self.boundDomain.getInputSpecification()()
    inputSpec.parseNode(xml)
    self.settings = self.boundDomain.handleInput(inputSpec)

  def testHandleInput(self):
    """
      Tests to see if the input values were handled correctly
      @ In, None
      @ Out, None
    """
    self.assertIn("lowerBound", self.settings)
    self.assertEqual(-3.0, self.settings["lowerBound"])
    self.assertIn("upperBound", self.settings)
    self.assertEqual(2.0, self.settings["upperBound"])

  def testFit(self):
    """
      Tests the fit method.
      @ In, None
      @ Out, None
    """
    params = self.boundDomain.fit(self.signals, self.pivots, self.targets, {})
    self.assertSetEqual(set(params.keys()), set(self.targets))
    for target, targetParams in params.items():
      self.assertIn("model", targetParams)
      self.assertIsInstance(targetParams["model"], FunctionTransformer)

  def testGetResidual(self):
    """
      Tests the getResidual method
      @ In, None
      @ Out, None
    """
    params = self.boundDomain.fit(self.signals, self.pivots, self.targets, {})
    residual = self.boundDomain.getResidual(self.signals, params, self.pivots, {})
    assert_array_equal(residual, self.signalsTransformed)

  def testGetComposite(self):
    """
      Tests the getComposite method
      @ In, None
      @ Out, None
    """
    params = self.boundDomain.fit(self.signals, self.pivots, self.targets, {})
    composite = self.boundDomain.getComposite(self.signals, params, self.pivots, {})
    assert_array_equal(composite, self.signalsTransformed)
