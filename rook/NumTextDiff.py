# Copyright 2019 Battelle Energy Alliance, LLC
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
This implements a text differ that can process the numbers inside the text.
"""

from __future__ import division, print_function,  absolute_import
import os
import sys

from Tester import Differ
import DiffUtils as DU

class NumericText(Differ):
  """
  This class is used for comparing text blocks with numbers in the text.
  """
  @staticmethod
  def get_valid_params():
    """
      Return the valid parameters for this class.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Differ.get_valid_params()
    params.add_param('zero_threshold', sys.float_info.min*4.0, 'it represents '
                     +'the value below which a float is considered zero')
    params.add_param('remove_whitespace', False,
                     'Removes whitespace before comparing text if True')
    params.add_param('rel_err', '', 'Relative Error for floats')
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
    self.__text_opts = {}
    if len(self.specs["rel_err"]) > 0:
      self.__text_opts['rel_err'] = float(self.specs["rel_err"])
    else:
      self.__text_opts['rel_err'] = None
    self.__text_opts['zero_threshold'] = float(self.specs["zero_threshold"])
    self.__text_opts['remove_whitespace'] = bool(self.specs['remove_whitespace'])

  def check_output(self):
    """
      Checks that the output matches the gold.
      returns (same, message) where same is true if the
      test passes, or false if the test failes.  message should
      gives a human readable explaination of the differences.
      @ In, None
      @ Out, (same, message), same is true if the tests passes.
    """
    test_files = self._get_test_files()
    gold_files = self._get_gold_files()
    self.__same = True
    self.__message = ""
    for test_filename, gold_filename in zip(test_files, gold_files):
      # local "same" and message list
      same = True
      msg = []
      # load test file
      try:
        test_file = open(test_filename, 'r')
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Test file does not exist!')
        same = False
      # load gold file
      try:
        gold_file = open(gold_filename, 'r')
        # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Gold file does not exist!')
        same = False
      # if either file did not exist, clean up and go to next outfile
      if not same:
        self.finalize_message(same, msg, test_filename)
        continue
      cswf = DU.compare_strings_with_floats
      same, message = cswf(test_file.read(),
                           gold_file.read(),
                           zero_threshold=self.__text_opts['zero_threshold'],
                           remove_whitespace=self.__text_opts['remove_whitespace'],
                           rel_err=self.__text_opts['rel_err'])
      if not same:
        msg.append(message)
      self.finalize_message(same, msg, test_filename)
    return self.__same, self.__message


  def finalize_message(self, same, msg, filename):
    """
      Compiles useful messages to print, prepending with file paths.
      @ In, same, bool, True if files are the same
      @ In, msg, list(str), messages that explain differences
      @ In, filename, str, test filename/path
      @ Out, None
    """
    if not same:
      self.__same = False
      self.__message += '\nDIFF in {}: \n  {}'.format(filename, '\n  '.join(msg))
