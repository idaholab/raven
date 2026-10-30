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
  Helper that runs a pytest module as a script and prints a RAVEN-style pass/fail summary.
"""
import sys

import pytest


class _ResultCollector:
  """
    Pytest plugin used to count passed and failed tests when modules are run as scripts.
  """

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.results = {'pass': 0, 'fail': 0}

  def pytest_runtest_logreport(self, report):
    """
      Hook called by pytest after each test phase.
      @ In, report, pytest.TestReport, the phase report
      @ Out, None
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
