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
  from utils import mathUtils
except ImportError:
  new = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', '..',
                                      '..', '..', 'framework'))
  sys.path.append(new)
  from utils import mathUtils

whoAmI = False # enable to show test dir and out files
debug = False # enable to increase printing

class UnorderedCSVDiffer:
  """
    Used for comparing two CSV files without regard for column, row orders
  """
  def __init__(self, outFiles, goldFiles, relativeError=1e-10,
               absoluteCheck=False, zeroThreshold=None, ignoreSign=False):
    """
      Create an UnorderedCSVDiffer class
      @ In, outFiles, the files to be compared.  They will be in testDir + outFiles
      @ In, goldFiles, the files to be compared to the outFiles.
      @ In, relativeError, float, optional, relative error
      @ In, absoluteCheck, bool, optional, if True then check absolute
         differences in the values instead of relative differences
      @ In, zeroThreshold, float, optional, if a number is less equal then
                                             abs(zeroThreshold), it will be considered 0
      @ In, ignoreSign, bool, optional, if True then the sign will be ignored during the comparison
      @ Out, None.
    """
    assert len(outFiles) == len(goldFiles)
    self._out_files = outFiles
    self._gold_files = goldFiles
    self._message = ""
    self._same = True
    self._check_absolute_values = absoluteCheck
    self._rel_err = relativeError
    self._zero_threshold = float(zeroThreshold) if zeroThreshold is not None else 0.0
    self._ignore_sign = ignoreSign
    if debug or whoAmI:
      print('out files:', self._out_files)
      print('gold files:', self._gold_files)
    if debug:
      print('err      :', self._rel_err)
      print('abs check:', self._check_absolute_values)
      print('zero thr :', self._zero_threshold)

  def finalizeMessage(self, same, msg, filename):
    """
      Compiles useful messages to print, prepending with file paths.
      @ In, same, bool, True if files are the same
      @ In, msg, list(str), messages that explain differences
      @ In, filename, str, test filename/path
      @ Out, None
    """
    if not same:
      self._same = False
      self._message += '\n'+'*'*20+'\nDIFF in {}: \n  {}'.format(filename, '\n  '.join(msg))

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
    # TODO can I do this as a single search, using binomial on floats +- relErr?
    for idx, val in row.iteritems():
      if debug:
        print('  checking index', idx, 'value', val)
      # Due to relative matches in floats, we may not be sorted with respect to this index.
      ## In an ideal world with perfect matches, we would be.  Unfortunately, we have to sort again.
      match = match.sort_values(idx)
      # check type consistency
      ## get a sample from the matching CSV column
      ### TODO could check indices ONCE and re-use instead of checking each time
      matchVal = match[idx].values.item(0) if match[idx].values.shape[0] != 0 else None
      ## find out if match[idx] and/or "val" are numbers
      matchIsNumber = mathUtils.isAFloatOrInt(matchVal)
      valIsNumber = mathUtils.isAFloatOrInt(val)
      ## if one is a number and the other is not, consider it a non-match.
      if matchIsNumber != valIsNumber:
        if debug:
          print('  Not same type (number)! lfor: "{}" lin: "{}"'
                .format(valIsNumber, matchIsNumber))
        return []
      # find index of lowest and highest possible matches
      ## if values are floats, then matches could be as low as val(1-relErr)
      ## and as high as val(1+relErr)
      if matchIsNumber:
        pval = abs(val) if self._ignore_sign else val
        pmatch = abs(match[idx].values) if self._ignore_sign else match[idx].values
        # adjust for negative values
        sign = np.sign(pval)
        lowest = np.searchsorted(pmatch, pval*(1.0-sign*self._rel_err))
        highest = np.searchsorted(pmatch, pval*(1.0+sign*self._rel_err), side='right')-1
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
      if not self.matches(match[idx].values[lowest], val, matchIsNumber, self._rel_err):
        if debug:
          print('  Match is not equal to insert point!')
        return []
      ## otherwise, we have some range of matches
      match = match[slice(lowest, highest+1)]
      if debug:
        print('  After searching for {}={}, remaining matches:\n'.format(idx, val), match)
    return match

  def matches(self, aObj, bObj, isNumber, tol):
    """
      Determines if two objects match within tolerance.
      @ In, a, object, first object ("measured")
      @ In, b, object, second object ("actual")
      @ In, isNumber, bool, if True then treat as float with tolerance (else check equivalence)
      @ In, tol, float, tolerance at which to hold match (if float)
      @ Out, matches, bool, True if matching
    """
    if not isNumber:
      return aObj == bObj
    if self._ignore_sign:
      aObj = abs(aObj)
      bObj = abs(bObj)
    if self._check_absolute_values:
      return abs(aObj-bObj) < tol
    # otherwise, relative error
    scale = abs(bObj) if bObj != 0 else 1.0
    return abs(aObj-bObj) < scale*tol

  def diff(self):
    """
      Run the comparison.
      @ In, None
      @ Out, same, bool, if True then files are the same
      @ Out, messages, str, messages to print on fail
    """
    # read in files
    for testFilename, goldFilename in zip(self._out_files, self._gold_files):
      # local "same" and message list
      same = True
      msg = []
      # load test file
      try:
        testCsv = pd.read_csv(testFilename, sep=',')
      # if file is empty, we can check that's consistent, too
      except pd.errors.EmptyDataError:
        testCsv = None
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Test file does not exist!')
        same = False
      # load gold file
      try:
        goldCsv = pd.read_csv(goldFilename, sep=',')
      # if file is empty, we can check that's consistent, too
      except pd.errors.EmptyDataError:
        goldCsv = None
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Gold file does not exist!')
        same = False
      # if either file did not exist, clean up and go to next outfile
      if not same:
        self.finalizeMessage(same, msg, testFilename)
        continue
      # at this point, we've loaded both files (even if they're empty), so compare them.
      ## first, cover the case when both files are empty.
      if testCsv is None or goldCsv is None:
        if not (testCsv is None and goldCsv is None):
          same = False
          if testCsv is None:
            msg.append('Test file is empty, but Gold is not!')
          else:
            msg.append('Gold file is empty, but Test is not!')
        # either way, move on to the next file, as no more comparison is needed
        self.finalizeMessage(same, msg, testFilename)
        continue
      ## at this point, both files have data loaded
      ## check columns using symmetric difference
      diffColumns = set(goldCsv.columns)^set(testCsv.columns)
      if len(diffColumns) > 0:
        same = False
        msg.append('Columns are not the same! Different: {}'.format(', '.join(diffColumns)))
        self.finalizeMessage(same, msg, testFilename)
        continue
      ## check index length
      if len(goldCsv.index) != len(testCsv.index):
        same = False
        msg.append(('Different number of entires in Gold ({}) versus'+
                    ' Test ({})!').format(len(goldCsv.index), len(testCsv.index)))
        self.finalizeMessage(same, msg, testFilename)
        continue
      ## at this point both CSVs have the same shape, with the same header contents.
      ## align columns
      testCsv = testCsv[goldCsv.columns.tolist()]
      ## set marginal values to zero, fix infinites
      testCsv = self.prep_data_frame(testCsv, self._zero_threshold)
      goldCsv = self.prep_data_frame(goldCsv, self._zero_threshold)
      ## check for matching rows
      for idx in goldCsv.index:
        find = goldCsv.iloc[idx].rename(None)
        match = self.find_row(find, testCsv)
        if len(match) == 0:
          same = False
          msg.append(('Could not find match for row "{}" in '+
                      'Gold:\n{}').format(idx+1, find)) #+1 because of header row
          msg.append('The Test output csv is:')
          msg.append(str(testCsv))
          # stop looking once a mismatch is found
          break
      self.finalizeMessage(same, msg, testFilename)
    return self._same, self._message

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
    key = {'atol':tol} if self._check_absolute_values else {'rtol':tol}
    # take care of infinites
    csv = csv.replace(np.inf, -sys.float_info.max)
    csv = csv.replace(np.nan, sys.float_info.max)
    for col in csv.columns:
      example = csv[col].values.item(0) if csv[col].values.shape[0] != 0 else None
      # skip columns that aren't numbers TODO might skip float columns with "None" early on
      if not mathUtils.isAFloatOrInt(example):
        continue
      # flatten near-zeros
      csv[col].values[np.isclose(csv[col].values, 0, **key)] = 0
    # TODO would like to sort here, but due to relative errors it doesn't do
    #  enough good.  Instead, sort in findRow.
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

  def __init__(self, name, params, testDir):
    """
      Initializer for the class. Takes a String name and a dictionary params
      @ In, name, string, name of the test.
      @ In, params, dictionary, parameters for the class
      @ In, testDir, string, path to the test.
      @ Out, None.
    """
    Differ.__init__(self, name, params, testDir)
    self._zero_threshold = self.specs['zero_threshold']
    self._ignore_sign = bool(self.specs['ignore_sign'])
    if len(self.specs['rel_err']) > 0:
      self._rel_err = float(self.specs['rel_err'])
    else:
      self._rel_err = 1e-10
    self._check_absolute_value = self.specs["check_absolute_value"]

  def check_output(self):
    """
      Checks that the output matches the gold.
      returns (same, message) where same is true if the
      test passes, or false if the test failes.  message should
      gives a human readable explaination of the differences.
      @ In, None
      @ Out, (same, message), same is true if the tests passes.
    """
    csvFiles = self._get_test_files()
    goldFiles = self._get_gold_files()
    csvDiff = UnorderedCSVDiffer(csvFiles,
                                  goldFiles,
                                  relativeError=self._rel_err,
                                  zeroThreshold=self._zero_threshold,
                                  ignoreSign=self._ignore_sign,
                                  absoluteCheck=self._check_absolute_value)
    return csvDiff.diff()
