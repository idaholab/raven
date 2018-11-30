
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

class _Parameter:

  def __init__(self, name, help_text, default=None):
    self.name = name
    self.help_text = help_text
    self.default = default

class _ValidParameters:

  def __init__(self):
    self.__parameters = {}

  def addParam(self, name, default, help_text):
    self.__parameters[name] = _Parameter(name, help_text, default)

  def addRequiredParam(self, name, help_text):
    self.__parameters[name] = _Parameter(name, help_text)

  def get_filled_dict(self, partial_dict):
    """
    Returns a dictionary where default values are filled in for everything
    that is not in the partial_dict
    """
    ret_dict = dict(partial_dict)
    for param in self.__parameters.values():
      if param.default is not None and param.name not in ret_dict:
        ret_dict[param.name] = param.default
    return ret_dict

  def check_for_required(self, check_dict):
    """
    Returns True if all the required parameters are present
    """
    for param in self.__parameters.values():
      if param.default is None and param.name not in check_dict:
        print("Missing:", param.name)
        return False
    return True

  def check_for_all_known(self, check_dict):
    """
    Returns True if all the parameters are known
    """
    for param_name in check_dict:
      if param_name not in self.__parameters:
        print("Unknown:", param_name)
        return False
    return True

class Tester:

  @staticmethod
  def validParams():
    params = _ValidParameters()
    params.addRequiredParam('type', 'The type of this test')
    params.addParam('skip', False, 'If true skip test')
    params.addParam('prereq', '', 'list of tests to run before running this one')
    params.addParam('max_time', 300, 'Maximum time that test is allowed to run')
    params.addParam('method', False, 'Method is ignored, but kept for compatibility')
    params.addParam('heavy', False, 'If true, run only with heavy tests')
    return params
