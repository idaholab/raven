import sys

import pytest


class _ResultCollector:
  """
    Pytest plugin used to count passed and failed tests when modules are run as scripts.
  """

  def __init__(self):
    self.results = {'pass': 0, 'fail': 0}

  def pytest_runtest_logreport(self, report):
    """
      Hook called by pytest after each test phase.
    """
    if report.when != 'call':
      return
    if report.passed:
      self.results['pass'] += 1
    elif report.failed:
      self.results['fail'] += 1


def run_module(module_file, extra_args=None):
  """
    Execute pytest for the provided module and print a legacy-style summary.
    @ In, module_file, str, path to the module to execute.
    @ In, extra_args, list(str), optional, additional pytest arguments.
    @ Out, None, exits the interpreter with pytest's return code.
  """
  collector = _ResultCollector()
  args = [module_file]
  if extra_args:
    args.extend(extra_args)
  exit_code = pytest.main(args, plugins=[collector])
  print('Results:', collector.results)
  sys.exit(exit_code)
