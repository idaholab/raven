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
Created on Jul 09, 2015

@author: tompjame
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import re
import collections

class BISONMESHSCRIPTparser():
  """
    Import Bison Mesh Script input, provide methods to add/change entries and print input back
  """
  def __init__(self,inputFile):
    """
      Open and read file content into an ordered dictionary
      @ In, inputFile, File object, object with information about the template input file
      @ Out, None
    """
    self.printTag = 'BISONMESHSCRIPT_PARSER'
    if not os.path.exists(inputFile.getAbsFile()):
      raise IOError('Input file not found: '+inputFile.getAbsFile())
    # Initialize file dictionary, storage order, and internal variables
    self.AllVarDict = collections.OrderedDict()
    self.fileOrderStorage = []
    quote_comment = False
    quote_comment_line = False
    apostrophe_comment = False
    apostrophe_comment_line = False
    between_str = ''
    # Open file
    self.inputfile = inputFile.getAbsFile()
    # self.keywordDictionary dictionary
    for line in inputFile:
      if '"""' in line or "'''" in line:
        if '"""' in line and quote_comment == True:
          quote_comment_line = True
          splitline = line.split('"""')
          between_str += splitline[0] + '"""'
          line = splitline[1]
          quote_comment = not quote_comment
        elif '"""' in line and quote_comment == False:
          quote_comment_line = True
          splitline = line.split('"""')
          line = splitline[0]
          quote_comment = not quote_comment
        elif "'''" in line and apostrophe_comment == True:
          apostrophe_comment_line = True
          splitline = line.split("'''")
          between_str += splitline[0] + "'''"
          line = splitline[1]
          apostrophe_comment = not apostrophe_comment
        elif "'''" in line and apostrophe_comment == False:
          apostrophe_comment_line = True
          splitline = line.split("'''")
          line = splitline[0]
          apostrophe_comment = not apostrophe_comment
        # parse stuff that is left over on the line
        if len(line) == 0:
          between_str += line
        elif line.isspace():
          between_str += line
        elif line.startswith('#'):
          between_str += line
        elif '{}' in line:
          between_str += line
        elif line.startswith('pellets'):
          between_str += line
        else:
          # Append string of non-varying parts of input file to file storage and reset the collection string
          if len(between_str) > 0:
            self.fileOrderStorage.append(between_str)
            between_str = ''
          dictname, varname, varvalue = re.split(r"\['|'] = |'] =|']= ", line)
          if dictname in self.AllVarDict.keys():
            self.AllVarDict[dictname][varname] = varvalue.strip()
          else:
            self.fileOrderStorage.append([dictname])
            self.AllVarDict[dictname] = collections.OrderedDict()
            self.AllVarDict[dictname][varname] = varvalue.strip()
        # Add comment contents to storage for ''' or """ that starts comment block after code on same line
        if quote_comment_line == True and quote_comment == True:
          between_str += '"""' + splitline[1]
        elif apostrophe_comment_line == True and apostrophe_comment == True:
          between_str += "'''" + splitline[1]
        quote_comment_line = False
        apostrophe_comment_line = False
      else:
        # Didn't find a comment block flag
        if quote_comment == True or apostrophe_comment == True:
          between_str += line
          continue
        else:
          # Outside of comment block (in code)
          if len(line) == 0:
            between_str += line
          elif line.isspace():
            between_str += line
          elif line.startswith('#'):
            between_str += line
          elif '{}' in line:
            between_str += line
          elif line.startswith('pellets'):
            between_str += line
          else:
            # Append string of non-varying parts of input file to file storage and reset the collection string
            if len(between_str) > 0:
              self.fileOrderStorage.append(between_str)
              between_str = ''
            dictname, varname, varvalue = re.split(r"\['|'] = |'] =|']= ", line)
            if dictname in self.AllVarDict.keys():
              self.AllVarDict[dictname][varname] = varvalue.strip()
            else:
              self.fileOrderStorage.append([dictname])
              self.AllVarDict[dictname] = collections.OrderedDict()
              self.AllVarDict[dictname][varname] = varvalue.strip()
    if len(between_str) > 0:
      self.fileOrderStorage.append(between_str)

  def modifyInternalDictionary(self,**inDictionary):
    """
      Parse the input dictionary and replace matching keywords in internal dictionary.
      @ In, inDictionary, dict, dictionary containing full longform name and raven sampled var value
      @ Out, None
    """
    for keyword, newvalue in inDictionary.items():
      keyword1, keyword2 = keyword.split('@')[-1].split('|')
      self.AllVarDict[keyword1][keyword2] = newvalue

  def writeNewInput(self,outfile=None):
    """
      Using the fileOrderStorage list, reconstruct the template input with modified keywordDictionary
      @ In, outfile, string, optional, output file name
      @ Out, None
    """
    if outfile==None:
      outfile = self.inputfile
    with open(outfile,'w') as IOfile:
      for e, entry in enumerate(self.fileOrderStorage):
        if type(entry) == unicode:
          IOfile.writelines(entry)
        elif type(entry) == list:
          DictBlockName = self.fileOrderStorage[e][0]
          DictBlock = self.AllVarDict[DictBlockName]
          for key, value in DictBlock.items():
            IOfile.writelines(DictBlockName + "['" + key + "'] = " + str(value) + '\n')
