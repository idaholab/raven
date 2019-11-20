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
import sys

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

  def __init__(self, outFiles, goldFiles, relativeError=1e-10, zeroThreshold=None):
    """
      Create an ImageDiff class
      @ In, outFiles, the files to be compared.
      @ In, goldFiles, the files to be compared to the outFiles.
      @ In, relativeError, float, optional, relative error
      @ In, zeroThreshold, float, optional, if a number is less equal then
                                             abs(zeroThreshold), it will be considered 0
      @ Out, None.
    """
    #assert len(outFiles) == len(goldFiles)
    self.__out_files = outFiles
    self.__gold_files = goldFiles
    self.__message = ""
    self.__same = True
    self.__rel_err = relativeError
    self.__zero_threshold = float(zeroThreshold) if zeroThreshold is not None else 0.0

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
    filesRead = False
    for testFilename, goldFilename in zip(self.__out_files, self.__gold_files):
      if not os.path.exists(testFilename):
        self.__same = False
        self.__message += 'Test file does not exist: '+testFilename
      elif not os.path.exists(goldFilename):
        self.__same = False
        self.__message += 'Gold file does not exist: '+goldFilename
      else:
        filesRead = True
      #read in files
      if filesRead:
        if not correctImport:
          self.__message += 'ImageDiff cannot run with scipy version less '+\
            'than 0.15.0, and requires the PIL installed; scipy version is '+\
            str(scipy.__version__)
          self.__same = False
          return(self.__same, self.__message)
        try:
          # RAK - The original line...
          # testImage = imread(open(testFilename,'r'))
          # ...didn't work on Windows Python because it couldn't sense the file type
          testImage = imread(testFilename)
        except IOError:
          self.__message += 'Unrecognized file type for test image in scipy.imread: '+testFilename
          filesRead = False
          return (False, self.__message)
        try:
          # RAK - The original line...
          # goldImage = imread(open(goldFilename,'r'))
          # ...didn't work on Windows Python because it couldn't sense the file type
          goldImage = imread(goldFilename)
        except IOError:
          filesRead = False
          self.__message += 'Unrecognized file type for test image in scipy.imread: '+goldFilename
          return (False, self.__message)
        #first check dimensionality
        if goldImage.shape != testImage.shape:
          self.__message += 'Gold and test image are not the same shape: '+\
            str(goldImage.shape)+', '+str(testImage.shape)
          self.__same = False
          return (self.__same, self.__message)
        #pixelwise comparison
        #TODO in the future we can add greyscale, normalized coloring, etc.
        # For now just do raw comparison of right/wrong pixels
        diff = goldImage - testImage
        onlyDiffs = diff[abs(diff) > self.__zero_threshold]
        pctNumDiff = onlyDiffs.size/float(diff.size)
        if pctNumDiff > self.__rel_err:
          self.__message += 'Difference between images is too large:'+\
            ' %2.2f pct (allowable: %2.2f)' %(100*pctNumDiff,\
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

  def __init__(self, name, params, testDir):
    """
      Initializer for the class. Takes a String name and a dictionary params
      @ In, name, string, name of the test.
      @ In, params, dictionary, parameters for the class
      @ In, testDir, string, path to the test.
      @ Out, None.
    """
    Differ.__init__(self, name, params, testDir)
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
    imageFiles = self._get_test_files()
    goldFiles = self._get_gold_files()
    imageDiff = ImageDiff(imageFiles,
                           goldFiles,
                           relativeError=self.__rel_err,
                           zeroThreshold=self.__zero_threshold)
    return imageDiff.diff()
