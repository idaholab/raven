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
import os, sys

try:
  from scipy.misc import imread
  correctImport = True
except ImportError:
  import scipy
  correctImport = False

from Tester import Differ

class ImageDiff:
  """
    ImageDiff is used for comparing two image files.
  """

  def __init__(self, out_files, gold_files, relative_error=1e-10, zero_threshold=None):
    """
      Create an ImageDiff class
      @ In, out_files, the files to be compared.
      @ In, gold_files, the files to be compared to the out_files.
      @ In, relative_error, float, optional, relative error
      @ In, zero_threshold, float, optional, if a number <= abs(zero_threshold) it will be considered 0
      @ Out, None.
    """
    #assert len(out_files) == len(gold_files)
    self.__out_files = out_files
    self.__gold_files = gold_files
    self.__message = ""
    self.__same = True
    self.__rel_err = relative_error
    self.__zero_threshold = float(zero_threshold) if zero_threshold is not None else 0.0

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
    for test_filename, gold_filename in zip(self.__out_files, self.__gold_files):
      if not os.path.exists(test_filename):
        self.__same = False
        self.self.__message += 'Test file does not exist: '+test_filename
      elif not os.path.exists(gold_filename):
        self.__same = False
        self.self.__message += 'Gold file does not exist: '+gold_filename
      else:
        files_read = True
    #read in files
    if files_read:
      if not correctImport:
        self.self.__message += 'ImageDiff cannot run with scipy version less '+\
          'than 0.15.0, and requires the PIL installed; scipy version is '+\
          str(scipy.__version__)
        self.__same = False
        return(self.__same, self.self.__message)
      try:
        # RAK - The original line...
        # test_image = imread(open(test_filename,'r'))
        # ...didn't work on Windows Python because it couldn't sense the file type
        test_image = imread(test_filename)
      except IOError:
        self.self.__message += 'Unrecognized file type for test image in scipy.imread: '+test_filename
        files_read = False
        return (False, self.self.__message)
      try:
        # RAK - The original line...
        # gold_image = imread(open(gold_filename,'r'))
        # ...didn't work on Windows Python because it couldn't sense the file type
        gold_image = imread(gold_filename)
      except IOError:
        files_read = False
        self.self.__message += 'Unrecognized file type for test image in scipy.imread: '+gold_filename
        return (False, self.self.__message)
      #first check dimensionality
      if gold_image.shape != test_image.shape:
        self.self.__message += 'Gold and test image are not the same shape: '+\
          str(gold_image.shape)+', '+str(test_image.shape)
        self.__same = False
        return (self.__same, self.self.__message)
      #pixelwise comparison
      #TODO in the future we can add greyscale, normalized coloring, etc.
      # For now just do raw comparison of right/wrong pixels
      diff = gold_image - test_image
      only_diffs = diff[abs(diff) > self.__zero_threshold]
      pct_num_diff = only_diffs.size/float(diff.size)
      if pct_num_diff > self.__rel_err:
        self.self.__message += 'Difference between images is too large:'+\
          ' %2.2f pct (allowable: %2.2f)' %(100*pct_num_diff,
                                            100*self.__rel_err)
        self.__same = False
    return (self.__same, self.__message)


class ImageDiffer(Differ):
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
    params.add_param('rel_err', '', 'Relative Error for image files')
    params.add_param('zero_threshold', sys.float_info.min*4.0,
                     'it represents the value below which a float is '+
                     'considered zero in the pixel comparison')
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
    if len(self.specs['rel_err']) > 0:
      self.__rel_err = float(self.specs['rel_err'])
    else:
      self.__rel_err = 1e-10

  def check_output(self):
    """
      Checks that the output matches the gold.
      returns (same, message) where same is true if the
      test passes, or false if the test failes.  message should
      gives a human readable explaination of the differences.
      @ In, None
      @ Out, (same, message), same is true if the tests passes.
    """
    image_files = self._get_test_files()
    gold_files = self._get_gold_files()
    image_diff = ImageDiff(image_files,
                                  gold_files,
                                  relative_error=self.__rel_err,
                                  zero_threshold=self.__zero_threshold)
    return image_diff.diff()

