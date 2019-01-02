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
This tests images against a expected image.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os

try:
  from scipy.misc import imread
  correctImport = True
except ImportError:
  import scipy
  correctImport = False

import DiffUtils as DU

class ImageDiff:
  """
    ImageDiff is used for comparing two image files.
  """

  def __init__(self, testDir, outFile, **kwargs):
    """
      Create an XMLDiff class
      testDir: the directory where the test takes place
      outFile: the files to be compared.  They will be in testDir + outFile
               and testDir + gold + outFile
      args: other arguments that may be included:
    """
    self.__out_file = outFile
    self.__messages = ""
    self.__same = True
    self.__test_dir = testDir
    self.__options = kwargs

  def diff(self):
    """
      Run the comparison.
      returns (same,messages) where same is true if the
      image files are the same, and messages is a string with all the
      differences.
      In, None
      Out, None
    """
    # read in files
    files_read = False
    for outfile in self.__out_file:
      test_filename = os.path.join(self.__test_dir, outfile)
      gold_filename = os.path.join(self.__test_dir, 'gold', outfile)
      if not os.path.exists(test_filename):
        self.__same = False
        self.__messages += 'Test file does not exist: '+test_filename
      elif not os.path.exists(gold_filename):
        self.__same = False
        self.__messages += 'Gold file does not exist: '+gold_filename
      else:
        files_read = True
    #read in files
    if files_read:
      if not correctImport:
        self.__messages += 'ImageDiff cannot run with scipy version less '+\
          'than 0.15.0, and requires the PIL installed; scipy version is '+\
          str(scipy.__version__)
        self.__same = False
        return(self.__same, self.__messages)
      try:
        # RAK - The original line...
        # test_image = imread(open(test_filename,'r'))
        # ...didn't work on Windows Python because it couldn't sense the file type
        test_image = imread(test_filename)
      except IOError:
        self.__messages += 'Unrecognized file type for test image in scipy.imread: '+test_filename
        files_read = False
        return (False, self.__messages)
      try:
        # RAK - The original line...
        # gold_image = imread(open(gold_filename,'r'))
        # ...didn't work on Windows Python because it couldn't sense the file type
        gold_image = imread(gold_filename)
      except IOError:
        files_read = False
        self.__messages += 'Unrecognized file type for test image in scipy.imread: '+gold_filename
        return (False, self.__messages)
      #first check dimensionality
      if gold_image.shape != test_image.shape:
        self.__messages += 'Gold and test image are not the same shape: '+\
          str(gold_image.shape)+', '+str(test_image.shape)
        self.__same = False
        return (self.__same, self.__messages)
      #set default options
      DU.set_default_options(self.__options)
      #pixelwise comparison
      #TODO in the future we can add greyscale, normalized coloring, etc.
      # For now just do raw comparison of right/wrong pixels
      diff = gold_image - test_image
      only_diffs = diff[abs(diff) > self.__options['zero_threshold']]
      pct_num_diff = only_diffs.size/float(diff.size)
      if pct_num_diff > self.__options['rel_err']:
        self.__messages += 'Difference between images is too large:'+\
          ' %2.2f pct (allowable: %2.2f)' %(100*pct_num_diff,
                                            100*self.__options['rel_err'])
        self.__same = False
    return (self.__same, self.__messages)
