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
import difflib

from Tester import Differ

class TextDiff:
  """ TextDiff is used for comparing a bunch of xml files.
  """
  def __init__(self, testDir, outFile,**kwargs):
    """ Create a TextDiff class
    testDir: the directory where the test takes place
    outFile: the files to be compared.  They will be in testDir + outFile
               and testDir + gold + outFile
    args: other arguments that may be included:
          - 'comment': indicates the character or string that should be used to denote a comment line
    """
    self.__outFile = outFile
    self.__messages = ""
    self.__same = True
    self.__testDir = testDir
    self.__options = kwargs

  def diff(self):
    """ Run the comparison.
    returns (same,messages) where same is true if all the txt files are the
    same, and messages is a string with all the differences.
    """
    # read in files
    commentSymbol = self.__options['comment']
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
        try:
          testFile = open(testFilename)
          testLines = [line.split(commentSymbol,1)[0].strip() if len(commentSymbol) > 0 else line for line in testFile]
          testLines = [line for line in testLines if len(line) > 0]
          testFile.close()
        except Exception as e:
          self.__same = False
          self.__messages += "Error reading " + testFilename + ":" + str(e) + " "
          filesRead = False
        try:
          goldFile = open(goldFilename)
          goldLines = [line.split(commentSymbol,1)[0].strip() if len(commentSymbol) > 0 else line for line in goldFile]
          goldLines = [line for line in goldLines if len(line) > 0]
          goldFile.close()
        except Exception as e:
          self.__same = False
          self.__messages += "Error reading " + goldFilename + ":" + str(e) + " "
          filesRead = False

        if filesRead:
          diff = list(difflib.unified_diff(testLines,goldLines))
          # deletions = [ line for line in diff if line.startswith('-')]
          # additions = [ line for line in diff if line.startswith('+')]
          if len(diff):
            self.__same = False
            separator = "\n"+" "*4
            self.__messages += "Mismatch between "+testFilename+" and "+goldFilename+separator
            self.__messages += separator.join(diff[2:8]) + separator+'...' + "\n" #truncation prevents too much output
    if '[' in self.__messages or ']' in self.__messages:
      self.__messages = self.__messages.replace('[','(')
      self.__messages = self.__messages.replace(']',')')
    return (self.__same,self.__messages)

class Text(Differ):
  """
  This is the class to use for handling the Text block.
  """
  @staticmethod
  def validParams():
    params = Differ.validParams()
    params.addParam('comment','-20021986',"Character or string denoting comments, all text to the right of the symbol will be ignored in the diff of text files")
    return params

  def __init__(self, name, params):
    """
    Initializer for the class. Takes a String name and a dictionary params
    """
    Differ.__init__(self, name, params)
    self.__text_opts = {'comment': self.specs['comment']}
    self.__text_files = self.specs['output'].split()

  def check_output(self, test_dir):
    """
    Checks that the output matches the gold.
    test_dir: the directory where the test is located.
    returns (same, message) where same is true if the
    test passes, or false if the test failes.  message should
    gives a human readable explaination of the differences.
    """
    text_diff =  TextDiff(test_dir, self.__text_files, **self.__text_opts)
    return text_diff.diff()
