from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os,re

try:
  from scipy.misc import imread
except ImportError:
  print ('ImageDiff cannot run with scipy version less than 0.15.0')
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
    print('outfiles:',self.__outFile,file=sys.stderr)
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
      try:
        testImage = imread(open(testFilename,'r'))
      except IOError as e:
        self.__messages += 'Unrecognized file type for test image in scipy.imread: '+testFilename
        filesRead = False
      try:
        goldImage = imread(open(goldFilename,'r'))
      except IOError as e:
        filesRead = False
        self.__messages += 'Unrecognized file type for test image in scipy.imread: '+goldFilename
      #first check dimensionality
      if goldImage.shape != testImage.shape:
        self.__messages += 'Gold and test image are not the same shape: '+str(godImage.shape)+', '+testImage.shape
        self.__same = False
      #set default options
      DU.setDefaultOptions(self.__options)
      #pixelwise comparison
      #TODO in the future we can add greyscale, normalized coloring, etc.  For now just do raw comparison of right/wrong pixels
      diff = goldImage - testImage
      onlyDiffs = diff[abs(diff)>self.__options['zero_threshold']]
      pctNumDiff = onlyDiffs.size/float(diff.size)
      if pctNumDiff > self.__options['rel_err']:
        self.__messages += 'Difference between images is too large: %2.2f %' %(100*pctNumDiff)
        self.__same = False
    print('Same?',self.__same,filesRead,self.__messages,file=sys.stderr)
    return (self.__same,self.__messages)
