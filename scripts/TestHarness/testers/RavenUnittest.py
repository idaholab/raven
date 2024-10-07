import os
import sys

from RavenPython import RavenPython

try:
  import unittest
  unittest_found = True
except ModuleNotFoundError or ImportError:
  unittest_found = False

class Unittest(RavenPython):
  """
  This class simplifies use of the unittest module for running unit tests through rook.
  """

  @staticmethod
  def get_valid_params():
    """
      Return a list of valid parameters and their descriptions for this type
      of test.
      @ In, None
      @ Out, params, _ValidParameters, the parameters for this class.
    """
    params = RavenPython.get_valid_params()
    # 'input' param can be test case or test suite; unittest will handle either when called
    params.add_param('unittest_args', '', "Arguments to the unittest module")
    return params

  def __init__(self, name, params):
    """
      Initializer for the class. Takes a String name and a dictionary params
      @ In, name, string, name of the test.
      @ In, params, dictionary, parameters for the class
      @ Out, None.
    """
    RavenPython.__init__(self, name, params)

  def check_runnable(self):
    """
      Checks if this test can be run.
      @ In, None
      @ Out, check_runnable, boolean, If True can run this test.
    """
    if not unittest_found:
      self.set_skip('skipped (required unittest module is not found)')
      return False

    return RavenPython.check_runnable(self)

  def get_command(self):
    """
      returns the command used by this tester.
      @ In, None
      @ Out, get_command, string, command to run.
    """
    # If the test command has been specified, use it
    if (command := self._get_test_command()) is not None:
      return ' '.join([command, '-m unittest', self.specs["unittest_args"], self.specs["input"]])

    # Otherwise, if the python command has been specified, use it
    if len(self.specs["python_command"]) == 0:
      pythonCommand = self._get_python_command()
    else:
      pythonCommand = self.specs["python_command"]
    return ' '.join([pythonCommand, '-m unittest', self.specs["unittest_args"], self.specs["input"]])
