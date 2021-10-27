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
This module implements a NetCDF database differ, assuming xarray structure.
"""
import sys
import os
import numpy as np
import xarray as xr

from Tester import Differ
from UnorderedCSVDiffer import UnorderedCSVDiffer, UnorderedCSV

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

class NetCDFDiffer(UnorderedCSVDiffer):
  """
    Used for comparing two NetCDF databases
  """
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
        testDS = xr.load_dataset(testFilename)
      # if file doesn't exist, that's a problem
      except IOError:
        msg.append('Test file "{}" does not exist!'.format(testFilename))
        same = False
      # load gold file
      try:
        goldDS = xr.load_dataset(goldFilename)
        goldCsv = None
      # if file doesn't exist, that's a problem
      except IOError:
        msg.append('Gold file "{}" does not exist!'.format(goldFilename))
        same = False
      # if either file did not exist, clean up and go to next outfile
      if not same:
        self.finalizeMessage(same, msg, testFilename)
        continue
      # at this point, we've loaded both files, so compare them.
      ## compare data contents
      # TODO zero threshold
      if self._check_absolute_values:
        kwargs = {'atol': self._rel_err}
      else:
        kwargs = {'rtol': self._rel_err}
      try:
        xr.testing.assert_allclose(testDS, goldDS, **kwargs)
      except AssertionError as e:
        same = False
        msg.append('Dataset diff detected ("left" is test, "right" is gold):\n{}'.format(str(e)))
      self.finalizeMessage(same, msg, testFilename)
    return self._same, self._message

class NetCDF(UnorderedCSV):
  """
  This is the class to use for handling the parameters block.
  """
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
    diff = NetCDFDiffer(csvFiles,
                        goldFiles,
                        relativeError=self._rel_err,
                        zeroThreshold=self._zero_threshold,
                        ignoreSign=self._ignore_sign,
                        absoluteCheck=self._check_absolute_value)
    return diff.diff()
