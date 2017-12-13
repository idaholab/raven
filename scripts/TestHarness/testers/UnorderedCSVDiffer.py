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
import numpy as np
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

  def finalizeMessage(self,same,msg,filename):
    """
      Compiles useful messages to print, prepending with file paths.
      @ In, same, bool, True if files are the same
      @ In, msg, list(str), messages that explain differences
      @ In, filename, str, test filename/path
      @ Out, None
    """
    if not same:
      self.__same = False
      self.__message += '\nDIFF in {}: \n  {}'.format(filename,'\n  '.join(msg))

  def findRow(self,row,csv):
    """
      Searches for "row" in "csv"
      @ In, row, TODO, row of data
      @ In, csv, pd.Dataframe, dataframe to look in
      @ Out, match, TODO, matching row of data
    """
    match = csv.copy()
    match = match.replace(np.inf,-sys.maxint)
    match = match.replace(np.nan,sys.maxint)
    # mask inf as -sys.max and nan as +sys.max
    for idx, val in row.iteritems():
      if val == np.inf:
        val = -sys.maxint
      elif pd.isnull(val):
        val = sys.maxint
      try:
        # try float/int first
        match = match[(abs(match[idx] - val) < self.__rel_err)]
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
      # local "same" and message list
      same = True
      msg = []
      # load test file
      testFilename = os.path.join(self.__test_dir,outFile)
      try:
        testCSV = pd.read_csv(testFilename)
      # if file is empty, we can check that's consistent, too
      except pd.errors.EmptyDataError:
        testCSV = None
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Test file does not exist!')
        same = False
      # load gold file
      goldFilename = os.path.join(self.__test_dir, 'gold', outFile)
      try:
        goldCSV = pd.read_csv(goldFilename)
      # if file is empty, we can check that's consistent, too
      except pd.errors.EmptyDataError:
        goldCSV = None
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Gold file does not exist!')
        same = False
      # if either file did not exist, clean up and go to next outfile
      if not same:
        self.finalizeMessage(same,msg,testFilename)
        continue
      # at this point, we've loaded both files (even if they're empty), so compare them.
      ## first, cover the case when both files are empty.
      if testCSV is None or goldCSV is None:
        if not (testCSV is None and goldCSV is None):
          same = False
          if testCSV is None:
            msg.append('Test file is empty, but Gold is not!')
          else:
            msg.append('Gold file is empty, but Test is not!')
        # either way, move on to the next file, as no more comparison is needed
        self.finalizeMessage(same,msg,testFilename)
        continue
      ## at this point, both files have data loaded
      ## check columns using symmetric difference
      diffColumns = set(goldCSV.columns)^set(testCSV.columns)
      if len(diffColumns) > 0:
        same = False
        msg.append('Columns are not the same! Different: {}'.format(', '.join(diffColumns)))
        self.finalizeMessage(same,msg,testFilename)
        continue
      ## check index length
      if len(goldCSV.index) != len(testCSV.index):
        same = False
        msg.append('Different number of entires in Gold ({}) versus Test ({})!'.format(len(goldCSV.index),len(testCSV.index)))
        self.finalizeMessage(same,msg,testFilename)
        continue
      ## at this point both CSVs have the same shape, with the same header contents.
      ## align columns
      testCSV = testCSV[goldCSV.columns.tolist()]
      ## check for matching rows
      for idx in goldCSV.index:
        find = goldCSV.iloc[idx].rename(None)
        match = self.findRow(find,testCSV)
        if not len(match) > 0:
          same = False
          msg.append('Could not find match for row "{}" in Gold:\n{}'.format(idx+1,find)) #+1 because of header row
          # stop looking once a mismatch is found
          break
      self.finalizeMessage(same,msg,testFilename)
    return self.__same, self.__message

  ## accessors

