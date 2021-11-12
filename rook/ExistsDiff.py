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
This implements a test to check if a file exists.
"""

from __future__ import division, print_function, absolute_import

import os

from Tester import Differ

class Exists(Differ):
  """
  This is the class to check if a file exists.
  """

  @staticmethod
  def get_valid_params():
    """
      Return the valid parameters for this class.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Differ.get_valid_params()
    return params

  def __init__(self, name, params, test_dir):
    """
      Initializer for the class. Takes a String name and a dictionary params
      @ In, name, string, name of the test.
      @ In, params, dictionary, parameters for the class
      @ In, test_dir, string, path to the test.
      @ Out, None.
    """
    Differ.__init__(self, name, params, test_dir)

  def check_output(self):
    """
      Checks that the output matches the gold.
      returns (same, message) where same is true if the
      test passes, or false if the test failes.  message should
      gives a human readable explaination of the differences.
      @ In, None
      @ Out, (same, message), same is bool, message is str,
         same is true if the tests passes.
    """
    exists_files = self._get_test_files()
    all_exist = True
    non_existing = []
    for filename in exists_files:
      if not os.path.exists(filename):
        all_exist = False
        non_existing.append(filename)
    if len(non_existing) > 0:
      message = "Files not created: "+" ".join(non_existing)
    else:
      message = "All files exist"
    return (all_exist, message)
