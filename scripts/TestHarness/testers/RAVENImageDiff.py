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
import sys,os,re

try:
  from scipy.misc import imread
  correctImport = True
except ImportError:
  import scipy
  correctImport = False
import numpy as np

import diffUtils as DU

class ImageDiff:
  """
    ImageDiff is used for comparing two image files.
  """

  def __init__(self, testDir, outFile,**kwargs):
    """
      Create an XMLDiff class
      testDir: the directory where the test takes place
      outFile: the files to be compared.  They will be in testDir + outFile
               and testDir + gold + outFile
      args: other arguments that may be included:
    """
    self.__outFile = outFile
    self.__messages = ""
    self.__same = True
    self.__testDir = testDir
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
    filesRead = False
    for outfile in self.__outFile:
      testFilename = os.path.join(self.__testDir,outfile)
      goldFilename = os.path.join(self.__testDir, 'gold', outfile)
      if not os.path.exists(testFilename):
        self.__same = False
        self.__messages += 'Test file does not exist: '+testFilename
      elif not os.path.exists(goldFilename):
        self.__same = False
        self.__messages += 'Gold file does not exist: '+goldFilename
      else:
        filesRead = True
    #read in files
    if filesRead:
      if not correctImport:
        self.__messages+='ImageDiff cannot run with scipy version less than 0.15.0, and requires the PIL installed; scipy version is '+str(scipy.__version__)
        self.__same = False
        return(self.__same,self.__messages)
      try:
        # RAK - The original line...
        # testImage = imread(open(testFilename,'r'))
        # ...didn't work on Windows Python because it couldn't sense the file type
        testImage = imread(testFilename)
      except IOError as e:
        self.__messages += 'Unrecognized file type for test image in scipy.imread: '+testFilename
        filesRead = False
        return (False, self.__messages)
      try:
        # RAK - The original line...
        # goldImage = imread(open(goldFilename,'r'))
        # ...didn't work on Windows Python because it couldn't sense the file type
        goldImage = imread(goldFilename)
      except IOError as e:
        filesRead = False
        self.__messages += 'Unrecognized file type for test image in scipy.imread: '+goldFilename
        return (False, self.__messages)
      #first check dimensionality
      if goldImage.shape != testImage.shape:
        self.__messages += 'Gold and test image are not the same shape: '+str(goldImage.shape)+', '+str(testImage.shape)
        self.__same = False
        return (self.__same, self.__messages)
      #set default options
      DU.setDefaultOptions(self.__options)
      #pixelwise comparison
      #TODO in the future we can add greyscale, normalized coloring, etc.  For now just do raw comparison of right/wrong pixels
      diff = goldImage - testImage
      onlyDiffs = diff[abs(diff)>self.__options['zero_threshold']]
      pctNumDiff = onlyDiffs.size/float(diff.size)
      if pctNumDiff > self.__options['rel_err']:
        self.__messages += 'Difference between images is too large: %2.2f pct (allowable: %2.2f)' %(100*pctNumDiff,100*self.__options['rel_err'])
        self.__same = False
    return (self.__same,self.__messages)
