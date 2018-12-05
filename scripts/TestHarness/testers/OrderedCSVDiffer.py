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
import csv

whoAmI = False # enable to show test dir and out files
debug = False # enable to increase printing

def to_float(s):
  """
  Converts s to a float if possible.
  """
  try:
    return float(s)
  except ValueError:
    return s

class OrderedCSVDiffer:
  """
    Used for comparing two CSV files without regard for column, row orders
  """
  def __init__(self, test_dir, out_files,relative_error=1e-10,absolute_check=False,zeroThreshold=None,ignore_sign=False):
    """
      Create an UnorderedCSVDiffer class
      Note naming conventions are out of our control due to MOOSE test harness standards.
      @ In, test_dir, the directory where the test takes place
      @ In, out_files, the files to be compared.  They will be in test_dir + out_files
      @ In, relative_error, float, optional, relative error
      @ In, absolute_check, bool, optional, if True then check absolute differences in the values instead of relative differences
      @ In, ignore_sign, bool, optional, if True then the sign will be ignored during the comparison
      @ Out, None.
    """
    self.__out_files = out_files
    self.__message = ""
    self.__same = True
    self.__test_dir = test_dir
    self.__check_absolute_values = absolute_check
    self.__rel_err = relative_error
    self.__zero_threshold = float(zeroThreshold) if zeroThreshold is not None else 0.0
    self.__ignore_sign = ignore_sign
    if debug or whoAmI:
      print('test dir :',self.__test_dir)
      print('out files:',self.__out_files)
    if debug:
      print('err      :',self.__rel_err)
      print('abs check:',self.__check_absolute_values)
      print('zero thr :',self.__zero_threshold)

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

  def matches(self,a,b,isNumber,tol):
    """
      Determines if two objects match within tolerance.
      @ In, a, object, first object ("measured")
      @ In, b, object, second object ("actual")
      @ In, isNumber, bool, if True then treat as float with tolerance (else check equivalence)
      @ In, tol, float, tolerance at which to hold match (if float)
      @ Out, matches, bool, True if matching
    """
    if not isNumber:
      return a == b
    if self.__ignore_sign:
      a = abs(a)
      b = abs(b)
    if abs(a) < self.__zero_threshold:
      a = 0.0
    if abs(b) < self.__zero_threshold:
      b = 0.0
    if self.__check_absolute_values:
      return abs(a-b) < tol
    # otherwise, relative error
    scale = abs(b) if b != 0 else 1.0
    return abs(a-b) < scale*tol

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
        try:
          testCSV_file = open(testFilename, newline='')
        except TypeError:
          #We must be in Python 2
          testCSV_file = open(testFilename, 'rb')
      # if file doesn't exist, that's another problem
      except IOError:
        msg.append('Test file does not exist!')
        same = False
      # load gold file
      goldFilename = os.path.join(self.__test_dir, 'gold', outFile)
      try:
        try:
          goldCSV_file = open(goldFilename, newline='')
        except TypeError:
          #We must be in Python 2
          goldCSV_file = open(goldFilename, 'rb')
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
      goldRows = list(csv.reader(goldCSV_file))
      testRows = list(csv.reader(testCSV_file))
      ## at this point, both files have data loaded
      ## check columns using symmetric difference
      gold_headers = goldRows[0]
      test_headers = testRows[0]
      diffColumns = set(gold_headers)^set(test_headers)
      if len(diffColumns) > 0:
        same = False
        msg.append('Columns are not the same! Different: {}'.format(', '.join(diffColumns)))
        self.finalizeMessage(same,msg,testFilename)
        continue
      ## check index length
      if len(goldRows) != len(testRows):
        same = False
        msg.append('Different number of entires in Gold ({}) versus Test ({})!'.format(len(goldRows),len(testRows)))
        self.finalizeMessage(same,msg,testFilename)
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
      for idx in range(1,len(goldRows)):
        goldRow = goldRows[idx]
        testRow = testRows[idx]
        for column in range(len(goldRow)):
          goldValue = to_float(goldRow[column])
          testValue = to_float(testRow[test_indexes[column]])
          matchIsNumber = type(goldValue) == type(0.0)
          valIsNumber = type(testValue) == type(0.0)
          if matchIsNumber != valIsNumber:
            same = False
            msg.append("Different types in "+gold_headers[column]+" for "
                       +str(goldValue)+" and "
                       +str(testValue))
          else:
            if not self.matches(goldValue, testValue, matchIsNumber,
                               self.__rel_err):
              same = False
              msg.append("Different values in "+gold_headers[column]+" for "
                         +str(goldValue)+" and "
                       +str(testValue))
      self.finalizeMessage(same,msg,testFilename)
    return self.__same, self.__message


