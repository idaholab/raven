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
created on Jul 15, 2015

@author: tompjame
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
import os
import re
import collections

class CUBITparser():
  """
    Import Cubit journal file input, provide methods to add/change entries and print input back
  """
  def __init__(self,inputFile):
    """
      Open and read file content into an ordered dictionary
      @ In, inputFile, File object, object with information about the template input file
      @ Out, None
    """
    self.printTag = 'CUBIT_PARSER'
    if not os.path.exists(inputFile.getAbsFile()):
      raise IOError('Input file not found: '+inputFile.getAbsFile())
    # Initialize file dictionary, storage order, and internal variables
    self.keywordDictionary = collections.OrderedDict()
    self.fileOrderStorage = []
    between_str = ''
    dict_stored = False
    # Open file
    self.inputfile = inputFile.getAbsFile()
    # Generate Global Input Dictionary
    for line in inputFile:
      clear_ws = line.replace(" ", "")
      if clear_ws.startswith('#{'):
        # Catch Aprepro logic
        if 'else' in line or 'ifdef' in line or 'ifndef' in line or 'endif' in line or 'Loop' in line or 'EndLoop' in line:
          between_str += line
        elif'=' in line:
          splitline_clear_ws = re.split('{|<|>|=|}|!', clear_ws)
          splitline = re.split('{|<|>|=|}|!', line)
          # Catch Aprepro if logic
          if splitline_clear_ws[1] != splitline[1].strip():
            between_str += line
          elif splitline_clear_ws[1] == splitline[1].strip():
            if len(between_str) > 0:
              self.fileOrderStorage.append(between_str)
              between_str = ''
            if dict_stored == False:
              self.fileOrderStorage.append(['dict_location'])
              dict_stored = True
            _, keywordAndValue, _ = re.split('#{|}',clear_ws)
            varname, varvalue = keywordAndValue.split('=')
            self.keywordDictionary[varname] = varvalue
      else:
        between_str += line
    if len(between_str) > 0:
      self.fileOrderStorage.append(between_str)

  def modifyInternalDictionary(self,**inDictionary):
    """
      Parse the input dictionary and replace matching keywords in internal dictionary.
      @ In, inDictionary, dict, dictionary containing full longform name and raven sampled var value
      @ Out, None
    """
    for keyword, newvalue in inDictionary.items():
      _, keyword = keyword.split('@')
      self.keywordDictionary[keyword] = newvalue

  def writeNewInput(self,outFile=None):
    """
      Using the fileOrderStorage list, reconstruct the template input with modified keywordDictionary.
      @ In, outFile, string, optional, outFile name
      @ Out, None
    """
    if outFile == None:
      outFile = self.inputfile
    with open(outFile,'w') as IOfile:
      for entry in self.fileOrderStorage:
        if type(entry) == unicode:
          IOfile.writelines(entry)
        elif type(entry) == list:
          for key, value in self.keywordDictionary.items():
            IOfile.writelines('#{ '+key+' = '+str(value)+'}'+'\n')
