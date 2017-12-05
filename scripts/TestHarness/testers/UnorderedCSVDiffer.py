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
from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os
import pandas as pd

class UnorderedCSVDiffer:
  """
    Used for comparing two CSV files without regard for column, row orders
  """
  def __init__(self, test_dir, out_files,relative_error=1e-10,absolute_check=False):
    """
      Create an UnorderedCSVDiffer class
      Note naming conventions are out of our control due to MOOSE test harness standards.
      @ In, test_dir, the directory where the test takes place
      @ In, out_files, the files to be compared.  They will be in test_dir + out_files
      @ In, relative_error, float, optional, relative error
      @ In, absolute_check, bool, optional, if True then check absolute values instead of values
      @ Out, None.
    """
    self.__out_files = out_files
    self.__message = ""
    self.__same = True
    self.__test_dir = test_dir
    self.__check_absolute_values = absolute_check
    self.__rel_err = relative_error

  def findRow(self,row,csv):
    """
      Searches for "row" in "csv"
      @ In, row, TODO, row of data
      @ In, csv, pd.Dataframe, dataframe to look in
      @ Out, match, TODO, matching row of data
    """
    match = csv.copy()
    for idx, val in row.iteritems():
      try:
        # try float/int first
        match = match[abs(match[idx] - val) < self.__rel_err]
      except TypeError:
        # otherwise, use exact matching
        match = match[match[idx] == val]
    return match

  def diff(self):
    """
      Run the comparison.
      @ In, None
      @ Out, same, bool, if True then files are the same
      @ Out, messages, str, messages to print on fail
    """
    # read in files
    for outFile in self.__out_files:
      # load test file
      testFilename = os.path.join(self.__test_dir,outFile)
      try:
        testCSV = pd.read_csv(testFilename)
      # if file is empty, we can check that's consistent, too
      except pd.errors.EmptyDataError:
        testCSV = None
      # if file doesn't exist, that's another problem
      except IOError:
        self.__same = False
        self.__message += '\nTest file does not exist: '+testFilename
        continue
      # load gold file
      goldFilename = os.path.join(self.__test_dir, 'gold', outFile)
      try:
        goldCSV = pd.read_csv(goldFilename)
      # if file is empty, we can check that's consistent, too
      except pd.errors.EmptyDataError:
        goldCSV = None
      # if file doesn't exist, that's another problem
      except IOError:
        self.__same = False
        self.__message += '\nGold file does not exist: '+goldFilename
        continue
      # at this point, we've loaded both files (even if they're empty), so compare them.
      ## first, cover the case when both files are empty.
      if testCSV is None or goldCSV is None:
        if not (testCSV is None and goldCSV is None):
          self.__same = False
          if testCSV is None:
            self.__message += '\nTest file is empty, but Gold is not!'
          else:
            self.__message += '\nGold file is empty, but Test is not!'
        # either way, move on to the next file, as no more comparison is needed
        continue
      ## at this point, both files have data loaded
      ## check columns using symmetric difference
      diffColumns = set(goldCSV.columns)^set(testCSV.columns)
      if len(diffColumns) > 0:
        self.__same = False
        self.__message += ('\nColumns are not the same! Different: {}'.format(', '.join(diffColumns)))
        continue
      ## check index length
      if len(goldCSV.index) != len(testCSV.index):
        self.__same = False
        self.__message += 'Different number of entires in Gold ({}) versus Test ({})!'.format(len(goldCSV.index),len(testCSV.index))
        continue
      ## at this point both CSVs have the same shape, with the same header contents.
      ## align columns
      testCSV = testCSV[goldCSV.columns.tolist()]
      ## check for matching rows
      for idx in goldCSV.index:
        find = goldCSV.iloc[idx].rename(None)
        match = self.findRow(find,testCSV)
        if not len(match) > 0:
          self.__same = False
          self.__message += '\nCould not find match for row "{}" in Gold:\n{}'.format(idx+2,find) #+2 because of header row
          # stop looking once a mismatch is found
          break
    return self.__same, self.__message
