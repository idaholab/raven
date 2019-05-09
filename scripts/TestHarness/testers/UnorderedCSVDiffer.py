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
This module implements a unordered csv differ.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import numpy as np
import pandas as pd

from Tester import Differ

# get access to math tools from RAVEN
try:
  from utils import utils
except ImportError:
  new = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', '..',
                                      '..', '..', 'framework'))
  sys.path.append(new)
  from utils import utils

whoAmI = False # enable to show test dir and out files
debug = False # enable to increase printing

class UnorderedCSVDiffer:
  """
    Used for comparing two CSV files without regard for column, row orders
  """
  def __init__(self, out_files, gold_files, relative_error=1e-10,
               absolute_check=False, zero_threshold=None, ignore_sign=False):
    """
      Create an UnorderedCSVDiffer class
      Note naming conventions are out of our control due to MOOSE test harness standards.
      @ In, test_dir, the directory where the test takes place
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
      print('out files:', self.__out_files)
      print('gold files:', self.__gold_files)
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

  def find_row(self, row, csv):
    """
      Searches for "row" in "csv"
      @ In, row, pd.Series, row of data
      @ In, csv, pd.Dataframe, dataframe to look in
      @ Out, match, pd.Dataframe or list, matching row of data (or empty list if none found)
    """
    if debug:
      print('')
      print('Looking for:\n', row)
      print('Looking in:\n', csv)
    match = csv.copy()
    # TODO can I do this as a single search, using binomial on floats +- rel_err?
    for idx, val in row.iteritems():
      if debug:
        print('  checking index', idx, 'value', val)
      # Due to relative matches in floats, we may not be sorted with respect to this index.
      ## In an ideal world with perfect matches, we would be.  Unfortunately, we have to sort again.
      match = match.sort_values(idx)
      # check type consistency
      ## get a sample from the matching CSV column
      ### TODO could check indices ONCE and re-use instead of checking each time
      match_val = match[idx].values.item(0) if match[idx].values.shape[0] != 0 else None
      ## find out if match[idx] and/or "val" are numbers
      match_is_number = utils.isAFloatOrInt(match_val)
      val_is_number = utils.isAFloatOrInt(val)
      ## if one is a number and the other is not, consider it a non-match.
      if match_is_number != val_is_number:
        if debug:
          print('  Not same type (number)! lfor: "{}" lin: "{}"'
                .format(val_is_number, match_is_number))
        return []
      # find index of lowest and highest possible matches
      ## if values are floats, then matches could be as low as val(1-rel_err)
      ## and as high as val(1+rel_err)
      if match_is_number:
        pval = abs(val) if self.__ignore_sign else val
        pmatch = abs(match[idx].values) if self.__ignore_sign else match[idx].values
        # adjust for negative values
        sign = np.sign(pval)
        lowest = np.searchsorted(pmatch, pval*(1.0-sign*self.__rel_err))
        highest = np.searchsorted(pmatch, pval*(1.0+sign*self.__rel_err), side='right')-1
      ## if not floats, then check exact matches
      else:
        lowest = np.searchsorted(match[idx].values, val)
        highest = np.searchsorted(match[idx].values, val, side='right')-1
      if debug:
        print('  low/hi match index:', lowest, highest)
      ## if lowest is past end of array, no match found
      if lowest == len(match[idx]):
        if debug:
          print('  Match is past end of sort list!')
        return []
      ## if entry at lowest index doesn't match entry, then it's not to be found
      if not self.matches(match[idx].values[lowest], val, match_is_number, self.__rel_err):
        if debug:
          print('  Match is not equal to insert point!')
        return []
      ## otherwise, we have some range of matches
      match = match[slice(lowest, highest+1)]
      if debug:
        print('  After searching for {}={}, remaining matches:\n'.format(idx, val), match)
    return match

  def matches(self, a_obj, b_obj, is_number, tol):
    """
      Determines if two objects match within tolerance.
      @ In, a, object, first object ("measured")
      @ In, b, object, second object ("actual")
      @ In, is_number, bool, if True then treat as float with tolerance (else check equivalence)
      @ In, tol, float, tolerance at which to hold match (if float)
      @ Out, matches, bool, True if matching
    """
    if not is_number:
      return a_obj == b_obj
    if self.__ignore_sign:
      a_obj = abs(a_obj)
      b_obj = abs(b_obj)
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
        test_csv = pd.read_csv(test_filename, sep=',')
      # if file is empty, we can check that's consistent, too
      except pd.errors.EmptyDataError:
        test_csv = None
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Test file does not exist!')
        same = False
      # load gold file
      try:
        gold_csv = pd.read_csv(gold_filename, sep=',')
      # if file is empty, we can check that's consistent, too
      except pd.errors.EmptyDataError:
        gold_csv = None
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
      if test_csv is None or gold_csv is None:
        if not (test_csv is None and gold_csv is None):
          same = False
          if test_csv is None:
            msg.append('Test file is empty, but Gold is not!')
          else:
            msg.append('Gold file is empty, but Test is not!')
        # either way, move on to the next file, as no more comparison is needed
        self.finalize_message(same, msg, test_filename)
        continue
      ## at this point, both files have data loaded
      ## check columns using symmetric difference
      diff_columns = set(gold_csv.columns)^set(test_csv.columns)
      if len(diff_columns) > 0:
        same = False
        msg.append('Columns are not the same! Different: {}'.format(', '.join(diff_columns)))
        self.finalize_message(same, msg, test_filename)
        continue
      ## check index length
      if len(gold_csv.index) != len(test_csv.index):
        same = False
        msg.append(('Different number of entires in Gold ({}) versus'+
                    ' Test ({})!').format(len(gold_csv.index), len(test_csv.index)))
        self.finalize_message(same, msg, test_filename)
        continue
      ## at this point both CSVs have the same shape, with the same header contents.
      ## align columns
      test_csv = test_csv[gold_csv.columns.tolist()]
      ## set marginal values to zero, fix infinites
      test_csv = self.prep_data_frame(test_csv, self.__zero_threshold)
      gold_csv = self.prep_data_frame(gold_csv, self.__zero_threshold)
      ## check for matching rows
      for idx in gold_csv.index:
        find = gold_csv.iloc[idx].rename(None)
        match = self.find_row(find, test_csv)
        if len(match) == 0:
          same = False
          msg.append(('Could not find match for row "{}" in '+
                      'Gold:\n{}').format(idx+1, find)) #+1 because of header row
          msg.append('The Test output csv is:')
          msg.append(str(test_csv))
          # stop looking once a mismatch is found
          break
      self.finalize_message(same, msg, test_filename)
    return self.__same, self.__message

  def prep_data_frame(self, csv, tol):
    """
      Does several prep actions:
        - For any columns that contain numbers, drop near-zero numbers to zero
        - replace infs and nans with symbolic values
      @ In, csv, pd.DataFrame, contents to reduce
      @ In, tol, float, tolerance sufficently near zero
      @ Out, csv, converted dataframe
    """
    # use absolute or relative?
    key = {'atol':tol} if self.__check_absolute_values else {'rtol':tol}
    # take care of infinites
    csv = csv.replace(np.inf, -sys.float_info.max)
    csv = csv.replace(np.nan, sys.float_info.max)
    for col in csv.columns:
      example = csv[col].values.item(0) if csv[col].values.shape[0] != 0 else None
      # skip columns that aren't numbers TODO might skip float columns with "None" early on
      if not utils.isAFloatOrInt(example):
        continue
      # flatten near-zeros
      csv[col].values[np.isclose(csv[col].values, 0, **key)] = 0
    # TODO would like to sort here, but due to relative errors it doesn't do
    #  enough good.  Instead, sort in find_row.
    return csv

class UnorderedCSV(Differ):
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
    params.add_param('zero_threshold', sys.float_info.min*4.0,
                     'it represents the value below which a float is '+
                     'considered zero (XML comparison only)')
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
    csv_diff = UnorderedCSVDiffer(csv_files,
                                  gold_files,
                                  relative_error=self.__rel_err,
                                  zero_threshold=self.__zero_threshold,
                                  ignore_sign=self.__ignore_sign,
                                  absolute_check=self.__check_absolute_value)
    return csv_diff.diff()
