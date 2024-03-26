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

frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(frameworkDir)

frameworkTestDir = os.path.abspath(os.path.join(frameworkDir, 'tests', 'framework'))
targetWorkflow = os.path.join(frameworkTestDir, 'basic.xml')

from ravenframework import Raven


class TestPythonRaven(unittest.TestCase):
  """
    Define unit tests for PythonRaven
  """
  # ********************
  # Utils
  #
  def setUp(self):
    """
      Set up tests.
      Runs before each test_* method
      @ In, None
      @ Out, None
    """
    self.raven = Raven()

  # ********************
  # Finding Files
  #
  def test_findFileAbs(self):
    """ Find file given absolute path """
    target = self.raven._findFile(targetWorkflow)
    self.assertEqual(targetWorkflow, target)

  def test_findFileCWD(self):
    """ Find file given path relative to current working directory """
    fname = os.path.join(*(['..']*2), 'basic.xml')
    target = self.raven._findFile(fname)
    self.assertEqual(targetWorkflow, target)

  def test_findFileFramework(self):
    """ Find file given path relative to RAVEN framework """
    # If ravenframework is installed in any way other than the git installation, such as
    # a pip installation or using an executable, looking for a file from the framework
    # directory doesn't make sense because those installations don't ship with the
    # ravenframework tests. We'll skip this test in those cases.
    if os.path.abspath(os.path.join('..', 'tests', 'framework')) != frameworkTestDir:
      self.skipTest('Skipping test_findFileFramework because the framework tests directory is not where' + \
                    ' we expect it to be based on the location of the current file. Perhaps the installation' + \
                    ' of ravenframework is not the git installation?')
    fname = os.path.join('..', 'tests', 'framework', 'basic.xml')
    target = self.raven._findFile(fname)
    self.assertEqual(targetWorkflow, target)

  # ********************
  # Workflows
  #
  def test_loadXMLFile(self):
    """ Load an XML Workflow. """
    self.raven.loadWorkflowFromFile(targetWorkflow)

  def test_runXML(self):
    """ Run an XML Workflow. """
    self.raven.loadWorkflowFromFile(targetWorkflow)
    code = self.raven.runWorkflow()
    self.assertEqual(code, 0)

if __name__ == '__main__':
  unittest.main()
  # note: return code is 1 if any tests fail/crash

"""
  <TestInfo>
    <name>framework.test_python_raven</name>
    <author>talbpaul</author>
    <created>2021-10-14</created>
    <classesTested>PythonRaven</classesTested>
    <description>
       This test is a Unit Test for running RAVEN as part of a Python workflow.
    </description>
  </TestInfo>
"""
