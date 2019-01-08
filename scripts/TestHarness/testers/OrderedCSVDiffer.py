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
This implements comparing CSV files in order.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import csv

from Tester import Differ

whoAmI = False # enable to show test dir and out files
debug = False # enable to increase printing

def to_float(s_val):
  """
    Converts s to a float if possible.
    @ In, s_val, any, value to convert
    @ Out, s_val, float or any
  """
  try:
    return float(s_val)
  except ValueError:
    return s_val

class OrderedCSVDiffer:
  """
    Used for comparing two CSV files without regard for column, row orders
  """
  def __init__(self, out_files, gold_files, relative_error=1e-10,
               absolute_check=False, zero_threshold=None, ignore_sign=False):
    """
      Create an UnorderedCSVDiffer class
      Note naming conventions are out of our control due to MOOSE test harness standards.
      @ In, out_files, the files to be compared.  They will be in test_dir + out_files
      @ In, gold_files, the files to be compared to the out_files.
      @ In, relative_error, float, optional, relative error
      @ In, absolute_check, bool, optional, if True then check absolute
         differences in the values instead of relative differences
      @ In, ignore_sign, bool, optional, if True then the sign will be ignored during the comparison
      @ Out, None.
    """
    assert len(out_files) == len(gold_files)
    self.__out_files = out_files
    self.__gold_files = gold_files
    self.__message = ""
    self.__same = True
    self.__check_absolute_values = absolute_check
    self.__rel_err = relative_error
    self.__zero_threshold = float(zero_threshold) if zero_threshold is not None else 0.0
    self.__ignore_sign = ignore_sign
    if debug or whoAmI:
      print('gold files:', self.__gold_files)
      print('out files:', self.__out_files)
    if debug:
      print('err      :', self.__rel_err)
      print('abs check:', self.__check_absolute_values)
      print('zero thr :', self.__zero_threshold)

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

  def matches(self, a_obj, b_obj, is_number, tol):
    """
      Determines if two objects match within tolerance.
      @ In, a_obj, object, first object ("measured")
      @ In, b_obj, object, second object ("actual")
      @ In, is_number, bool, if True then treat as float with tolerance (else check equivalence)
      @ In, tol, float, tolerance at which to hold match (if float)
      @ Out, matches, bool, True if matching
    """
    if not is_number:
      return a_obj == b_obj
    if self.__ignore_sign:
      a_obj = abs(a_obj)
      b_obj = abs(b_obj)
    if abs(a_obj) < self.__zero_threshold:
      a_obj = 0.0
    if abs(b_obj) < self.__zero_threshold:
      b_obj = 0.0
    if self.__check_absolute_values:
      return abs(a_obj-b_obj) < tol
    # otherwise, relative error
    scale = abs(b_obj) if b_obj != 0 else 1.0
    return abs(a_obj-b_obj) < scale*tol

  def diff(self):
    """
      Run the comparison.
      @ In, None
      @ Out, same, bool, if True then files are the same
      @ Out, messages, str, messages to print on fail
    """
    # read in files
    for test_filename, gold_filename in zip(self.__out_files, self.__gold_files):
      # local "same" and message list
      same = True
      msg = []
      # load test file
      try:
        try:
          test_csv_file = open(test_filename, newline='')
        except TypeError:
          #We must be in Python 2
          test_csv_file = open(test_filename, 'rb')
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Test file does not exist!')
        same = False
      # load gold file
      try:
        try:
          gold_csv_file = open(gold_filename, newline='')
        except TypeError:
          #We must be in Python 2
          gold_csv_file = open(gold_filename, 'rb')
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Gold file does not exist!')
        same = False
      # if either file did not exist, clean up and go to next outfile
      if not same:
        self.finalize_message(same, msg, test_filename)
        continue
      # at this point, we've loaded both files (even if they're empty), so compare them.
      ## first, cover the case when both files are empty.
      gold_rows = list(csv.reader(gold_csv_file))
      test_rows = list(csv.reader(test_csv_file))
      ## at this point, both files have data loaded
      ## check columns using symmetric difference
      gold_headers = gold_rows[0]
      test_headers = test_rows[0]
      diff_columns = set(gold_headers)^set(test_headers)
      if len(diff_columns) > 0:
        same = False
        msg.append('Columns are not the same! Different: {}'.format(', '.join(diff_columns)))
        self.finalize_message(same, msg, test_filename)
        continue
      ## check index length
      if len(gold_rows) != len(test_rows):
        same = False
        msg.append('Different number of entires in Gold ({}) versus Test ({})!'
                   .format(len(gold_rows), len(test_rows)))
        self.finalize_message(same, msg, test_filename)
        continue
      ## at this point both CSVs have the same shape, with the same header contents.
      ## figure out column indexs
      if gold_headers == test_headers:
        test_indexes = range(len(gold_headers))
      else:
        test_indexes = [test_headers.index(s) for s in gold_headers]
      # So now for a test row:
      #  gold_row[x][y] should match test_row[x][test_indexes[y]]
      ## check for matching rows
      for idx in range(1, len(gold_rows)):
        gold_row = gold_rows[idx]
        test_row = test_rows[idx]
        for column in range(len(gold_row)):
          gold_value = to_float(gold_row[column])
          test_value = to_float(test_row[test_indexes[column]])
          match_is_number = isinstance(gold_value, float)
          val_is_number = isinstance(test_value, float)
          if match_is_number != val_is_number:
            same = False
            msg.append("Different types in "+gold_headers[column]+" for "
                       +str(gold_value)+" and "
                       +str(test_value))
          else:
            if not self.matches(gold_value, test_value, match_is_number,
                                self.__rel_err):
              same = False
              msg.append("Different values in "+gold_headers[column]+" for "
                         +str(gold_value)+" and "
                         +str(test_value))
      self.finalize_message(same, msg, test_filename)
    return self.__same, self.__message


class OrderedCSV(Differ):
  """
  This is the class to use for handling the parameters block.
  """

  @staticmethod
  def get_valid_params():
    """
      Returns the valid parameters for this class.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Differ.get_valid_params()
    params.add_param('rel_err', '', 'Relative Error for csv files')
    params.add_param('zero_threshold', sys.float_info.min*4.0, 'it represents '+
                     'the value below which a float is considered zero (XML comparison only)')
    params.add_param('ignore_sign', False, 'if true, then only compare the absolute values')
    params.add_param('check_absolute_value', False, 'if true the values are '+
                     'compared to the tolerance directectly, instead of relatively.')
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
    self.__zero_threshold = self.specs['zero_threshold']
    self.__ignore_sign = bool(self.specs['ignore_sign'])
    if len(self.specs['rel_err']) > 0:
      self.__rel_err = float(self.specs['rel_err'])
    else:
      self.__rel_err = 1e-10
    self.__check_absolute_value = self.specs["check_absolute_value"]

  def check_output(self):
    """
      Checks that the output matches the gold.
      returns (same, message) where same is true if the
      test passes, or false if the test failes.  message should
      gives a human readable explaination of the differences.
      @ In, None
      @ Out, (same, message), same is true if the tests passes.
    """
    csv_files = self._get_test_files()
    gold_files = self._get_gold_files()
    csv_diff = OrderedCSVDiffer(csv_files,
                                gold_files,
                                relative_error=self.__rel_err,
                                zero_threshold=self.__zero_threshold,
                                ignore_sign=self.__ignore_sign,
                                absolute_check=self.__check_absolute_value)
    return csv_diff.diff()
