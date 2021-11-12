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
This implements a text differ.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import difflib

from Tester import Differ

class TextDiff:
  """ TextDiff is used for comparing a bunch of xml files.
  """
  def __init__(self, out_files, gold_files, **kwargs):
    """
      Create a TextDiff class
      @ In, out_files, string list, the files to be compared.
      @ In, gold_files, string list, the gold files to compare to the outfiles
      @ In, kwargs, dictionary, other arguments that may be included:
        - 'comment': indicates the character or string that should be used to denote a comment line
      @ Out, None
    """
    assert len(out_files) == len(gold_files)
    self.__out_files = out_files
    self.__gold_files = gold_files
    self.__messages = ""
    self.__same = True
    self.__options = kwargs

  def diff(self):
    """
      Run the comparison.
      @ In, None
      @ Out, (same,messages), (boolean, string), where same is true if all
          the txt files are the same, and messages is a string with all
          the differences.
    """
    # read in files
    comment_symbol = self.__options['comment']
    for test_filename, gold_filename in zip(self.__out_files, self.__gold_files):
      if not os.path.exists(test_filename):
        self.__same = False
        self.__messages += 'Test file does not exist: '+test_filename
      elif not os.path.exists(gold_filename):
        self.__same = False
        self.__messages += 'Gold file does not exist: '+gold_filename
      else:
        files_read = True
        try:
          with open(test_filename) as test_file:
            test_lines = [line.split(comment_symbol, 1)[0].strip()
                          if len(comment_symbol) > 0
                          else line for line in test_file]
            test_lines = [line for line in test_lines if len(line) > 0]
            test_file.close()
        except Exception as exp:
          self.__same = False
          self.__messages += "Error reading " + test_filename + ":" + str(exp) + " "
          files_read = False
        try:
          with open(gold_filename) as gold_file:
            gold_lines = [line.split(comment_symbol, 1)[0].strip()
                          if len(comment_symbol) > 0
                          else line for line in gold_file]
            gold_lines = [line for line in gold_lines if len(line) > 0]
            gold_file.close()
        except Exception as exp:
          self.__same = False
          self.__messages += "Error reading " + gold_filename + ":" + str(exp) + " "
          files_read = False

        if files_read:
          diff = list(difflib.unified_diff(test_lines, gold_lines))
          # deletions = [ line for line in diff if line.startswith('-')]
          # additions = [ line for line in diff if line.startswith('+')]
          if len(diff):
            self.__same = False
            separator = "\n"+" "*4
            self.__messages += "Mismatch between "+test_filename+" and "+gold_filename+separator
            #truncation prevents too much output
            self.__messages += separator.join(diff[2:8]) + separator+'...' + "\n"
    if '[' in self.__messages or ']' in self.__messages:
      self.__messages = self.__messages.replace('[', '(')
      self.__messages = self.__messages.replace(']', ')')
    return (self.__same, self.__messages)

class Text(Differ):
  """
  This is the class to use for handling the Text block.
  """
  @staticmethod
  def get_valid_params():
    """
      Returns the parameters that this class can use.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Differ.get_valid_params()
    params.add_param('comment', '-20021986', "Character or string denoting "+
                     "comments, all text to the right of the symbol will be "+
                     "ignored in the diff of text files")
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
    self.__text_opts = {'comment': self.specs['comment']}
    #self.__text_files = self.specs['output'].split()

  def check_output(self):
    """
      Checks that the output matches the gold.
      returns (same, message) where same is true if the
      test passes, or false if the test failes.  message should
      gives a human readable explaination of the differences.
      @ In, None
      @ Out, (same, message), same is true if the tests passes.
    """
    text_files = self._get_test_files()
    gold_files = self._get_gold_files()
    text_diff = TextDiff(text_files, gold_files, **self.__text_opts)
    return text_diff.diff()
