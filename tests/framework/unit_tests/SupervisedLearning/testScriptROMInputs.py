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
  Perform Unit Tests for running RAVEN in Python workflows.
  SQA Note: unittest requires all test methods start with test_, so we excercise SQA exception
  SQA Note: ALL tests take no input and provide no output, so these are omitted in the
            test_* method descriptions.
"""

import os
import sys
import unittest
import pickle
import xml.etree.ElementTree as ET
import numpy as np

# ROM needs to be aware of raven directory
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(frameworkDir)

class TestScriptROMInputs(unittest.TestCase):
  """
    Defines unit tests for checking ROM inputs in Python scripts
    ROM used is a DMDc ROM created with pickle_ROM.xml
  """

  def setUp(self):
    """
      Includes all necessary set up for tests
      @ In, None
      @ Out, None
    """
    # load the pickled ROM
    dirnameForFiles = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirnameForFiles, 'ROM.pk'), mode='rb') as tmp:
      self.rom = pickle.load(tmp)

    # load the A, B, C matrices from DMDcCxCoeff.xml
    tree = ET.parse(os.path.join(dirnameForFiles, 'DMDcCxCoeff.xml'))
    root = tree.getroot()
    model = root.find('./DMDcROM/DMDcModel')
    matricesList = ['Atilde', 'Btilde', 'Ctilde']
    for matrix in matricesList:
      node = model.find(matrix)
      # find the shape of the array
      matrixShape = node.find('./realization/matrixShape')
      shapeList = [int(x) for x in matrixShape.text.split(',')]
      # get the values of the array
      real = node.find('./realization/real')
      valueList = [float(x) for x in real.text.split(' ')]
      array = np.asarray(valueList)
      # reshape the array to the proper shape
      array = array.reshape([x for x in shapeList])
      # store as attribute
      setattr(self, matrix, array)

    # use the same u1 and u2 arrays each time
    self.u1 = np.array([2.692102, 0])
    self.u2 = np.array([10.0641, 0])


  def test_floatInput(self):
    """ Tests passing in floats """
    # initial state value
    x1_init = 0.036
    # run ROM and calculate comparison with matrices
    rom, calc = self.runDMDc(x1_init)
    # make comparisons between methods
    self.makeComparison(rom, calc)

  def test_intInput(self):
    """ Tests passing in integer """
    # initial state value
    x1_init = 1
    # run ROM and calculate comparison with matrices
    rom, calc = self.runDMDc(x1_init)
    # make comparisons between methods
    self.makeComparison(rom, calc)

  def test_numpyFloatInput(self):
    """ Tests passing in a numpy float """
    # initial state value
    x1_init = np.array([0.036])[0]
    # run ROM and calculate comparison with matrices
    rom, calc = self.runDMDc(x1_init)
    # make comparisons between methods
    self.makeComparison(rom, calc)

  def test_numpyIntegerInput(self):
    """ Tests passing in a numpy integer """
    # initial state value
    x1_init = np.array([1])[0]
    # run ROM and calculate comparison with matrices
    rom, calc = self.runDMDc(x1_init)
    # make comparisons between methods
    self.makeComparison(rom, calc)

  def runDMDc(self, x1_init):
    """
      Runs DMDc ROM with given inputs and calculates using matrices
      @ In, x1_init, float/int initial state value
      @ Out, romResult, dict, result dictionary from running ROM
      @ Out, calcResult, dict, result dictionary from calculating DMDc
    """
    # get result from running ROM
    inputs = {'x1_init': x1_init,
              'u1': self.u1,
              'u2': self.u2}
    romResult = self.rom.evaluate(inputs)

    # get result from calculations
    x = np.array([[x1_init]])
    u = np.vstack((self.u1, self.u2))
    xReturn = np.zeros((1, u.shape[-1]))
    yReturn = np.zeros((1, u.shape[-1]))
    xReturn[:, 0] = x
    yReturn[:, 0] = self.Ctilde.dot(xReturn[:, 0])
    for i in range(self.u1.shape[-1]-1):
      xReturn[:, i+1] = self.Atilde.dot(x[:, i]) + self.Btilde.dot(u[:, i])
      yReturn[:, i+1] = self.Ctilde.dot(xReturn[:, i+1])
    calcResult = {'x': xReturn.reshape(-1), 'y': yReturn.reshape(-1)}

    return romResult, calcResult

  def makeComparison(self, rom, calc):
    """
      Compares results from DMDc ROM and calculation with A, B, C matrices
      @ In, rom, dict, result dictionary from running ROM
      @ In, calc, dict, result dictionary from calculating DMDc
      @ Out, None
    """
    for i in range(len(rom['u1'])):
      # compare x
      romX = float(rom['x1'][i])
      calcX = float(calc['x'][i])
      self.assertAlmostEqual(romX, calcX, places=5)
      # compare y
      romY = float(rom['y1'][i])
      calcY = float(calc['y'][i])
      self.assertAlmostEqual(romY, calcY, places=5)


if __name__ == '__main__':
  unittest.main()
  # note: return code is 1 if any tests fail/crash

"""
  <TestInfo>
    <name>framework.test_script_ROM_inputs</name>
    <author>dgarrett622</author>
    <created>2022-09-16</created>
    <classesTested>SupervisedLearning</classesTested>
    <description>
       This test is a Unit Test for ROM inputs when used in a Python script.
    </description>
  </TestInfo>
"""
