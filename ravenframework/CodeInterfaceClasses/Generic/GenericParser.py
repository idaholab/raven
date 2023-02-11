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
Created on Mar 10, 2015

@author: talbpaul
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import os
import sys
import numpy as np
from ravenframework.utils import mathUtils
# numpy with version 1.14.0 and upper will change the floating point type and print
# https://docs.scipy.org/doc/numpy-1.14.0/release.html
if int(np.__version__.split('.')[1]) > 13:
  np.set_printoptions(**{'legacy':'1.13'})

def _reprIfFloat(value):
  """
    Uses repr if the value is a float
    @ In, value, any, the value to convert to a string
    @ Out, _reprIfFloat, string, a string conversion of this
  """
  if mathUtils.isAFloat(value):
    return repr(value)
  else:
    return str(value)

class GenericParser():
  """
    import the user-edited input file, build list of strings with replaceable parts
  """
  def __init__(self,inputFiles,prefix='$RAVEN-',postfix='$',defaultDelim=':', formatDelim='|'):
    """
      Accept the input file and parse it by the prefix-postfix breaks. Someday might be able to change prefix,postfix,defaultDelim from input file, but not yet.
      @ In, inputFiles, list, string list of input filenames that might need parsing.
      @ In, prefix, string, optional, the string prefix to find input variables within the input files
      @ In, postfix, string, optional, the string postfix signifying the end of an input variable within an input file
      @ In, defaultDelim, string, optional, the string used between prefix and postfix to set default values
      @ In, formatDelim, string, optional, the string used between prefix and postfix to set the format of the value
      @ Out, None
    """
    self.inputFiles = inputFiles
    self.prefixKey=prefix
    self.postfixKey=postfix
    self.varPlaces = {} # varPlaces[var][inputFile]
    self.defaults  = {} # defaults[var][inputFile]
    self.formats   = {} # formats[var][inputFile]
    self.acceptFormats = {"d":int,"e":float,"E":float,"f":float,"F":float,"g":float,"G":float}
    self.segments  = {} # segments[inputFile]
    self.printTag = 'GENERIC_PARSER'
    for inputFile in self.inputFiles:
      infileName = inputFile.getFilename()#os.path.basename(inputFile)
      self.segments[infileName] = []
      if not os.path.exists(inputFile.getAbsFile()):
        ## Make sure to cast the inputFile to a string as it may be File object.
        raise IOError('Input file not found: ' + str(inputFile))
      seg = ''
      lines = inputFile.readlines()
      inputFile.close()
      for lineNo, line in enumerate(lines):
        while self.prefixKey in line and self.postfixKey in line:
          self.segments[infileName].append(seg)
          start = line.find(self.prefixKey)
          end = line.find(self.postfixKey,start+1)
          # not found in line, since the default self.postfixKey is in self.prefixKey, it can happen.
          if end == -1:
            msg = f'Special wildcard prefix "{self.prefixKey}" is found, but postfix "{self.postfixKey}" is not found!'
            msg += f' Please check your input file "{infileName}"."'
            msg += f' This error happened in line "{lineNo+1}": i.e., "{line}"'
            raise IOError(msg)
          else:
            var = line[start+len(self.prefixKey):end]
            if defaultDelim in var or formatDelim in var:
              optionalPos = [None]*2
              optionalPos[0], optionalPos[1] = var.find(defaultDelim), var.find(formatDelim)
              if optionalPos[0] == -1:
                optionalPos[0]  = sys.maxsize
              if optionalPos[1] == -1:
                optionalPos[1] = sys.maxsize
              defval    = var[optionalPos[0]+1:min(optionalPos[1],len(var))] if optionalPos[0] < optionalPos[1] else var[min(optionalPos[0]+1,len(var)):len(var)]
              varformat = var[min(optionalPos[1]+1,len(var)):len(var)] if optionalPos[0] < optionalPos[1] else var[optionalPos[1]+1:min(optionalPos[0],len(var))]
              var = var[0:min(optionalPos)]
              if var in self.defaults.keys() and optionalPos[0] != sys.maxsize:
                print('multiple default values given for variable',var)
              if var in self.formats.keys() and optionalPos[1] != sys.maxsize:
                print('multiple format values given for variable',var)
              #TODO allow the user to specify take-last or take-first?
              if var not in self.defaults.keys() and optionalPos[0] != sys.maxsize:
                self.defaults[var] = {}
              if var not in self.formats.keys()  and optionalPos[1] != sys.maxsize:
                self.formats[var ] = {}
              if optionalPos[0] != sys.maxsize:
                self.defaults[var][infileName]=defval
              if optionalPos[1] != sys.maxsize:
                # check if the format is valid
                if not any(formVal in varformat for formVal in self.acceptFormats.keys()):
                  try:
                    int(varformat)
                  except ValueError:
                    raise ValueError("the format specified for wildcard "+ line[start+len(self.prefixKey):end] +
                                                       " is unknown. Available are either a plain integer or the following "+" ".join(self.acceptFormats.keys()))
                  self.formats[var][infileName ]=varformat,int
                else:
                  for formVal in self.acceptFormats.keys():
                    if formVal in varformat:
                      self.formats[var][infileName ]=varformat,self.acceptFormats[formVal]
                      break
            self.segments[infileName].append(line[:start])
            self.segments[infileName].append(var)
            if var not in self.varPlaces.keys():
              self.varPlaces[var] = {infileName:[len(self.segments[infileName])-1]}
            elif infileName not in self.varPlaces[var].keys():
              self.varPlaces[var][infileName]=[len(self.segments[infileName])-1]
            else:
              self.varPlaces[var][infileName].append(len(self.segments[infileName])-1)
            #self.segments.append(line[end+1:])
            line=line[end+1:]
            seg = ''
        else:
          seg+=line
      self.segments[infileName].append(seg)

  def modifyInternalDictionary(self,**Kwargs):
    """
      Edits the parsed file stored in self.segments to enter new variable values preparatory to a new run.
      @ In, **Kwargs, dict, dict including moddit (the dictionary of variable:value to replace) and additionalEdits.
      @ Out, None
    """
    modDict = Kwargs['SampledVars']
    self.adlDict = Kwargs.get('additionalEdits',{})
    ioVars = []
    for value in self.adlDict.values():
      if type(value)==dict:
        for k in value.keys():
          ioVars.append(k)
      elif type(value)==list:
        for v in value:
          ioVars.append(v)
      else:
        ioVars.append(value)
    for var in self.varPlaces.keys():
      for inputFile in self.segments.keys():
        for place in self.varPlaces[var][inputFile] if inputFile in self.varPlaces[var].keys() else []:
          if var in modDict.keys():
            if var in self.formats.keys():
              if inputFile in self.formats[var].keys():
                if any(formVal in self.formats[var][inputFile][0] for formVal in self.acceptFormats.keys()):
                  formatstringc = "{:"+self.formats[var][inputFile][0].strip()+"}"
                  self.segments[inputFile][place] = formatstringc.format(self.formats[var][inputFile][1](modDict[var]))
                else:
                  self.segments[inputFile][place] = _reprIfFloat(modDict[var]).strip().rjust(self.formats[var][inputFile][1](self.formats[var][inputFile][0]))
            else:
              self.segments[inputFile][place] = _reprIfFloat(modDict[var])
          elif var in self.defaults.keys():
            if var in self.formats.keys():
              if inputFile in self.formats[var].keys():
                if any(formVal in self.formats[var][inputFile][0] for formVal in self.acceptFormats.keys()):
                  formatstringc = "{:"+self.formats[var][inputFile][0].strip()+"}"
                  self.segments[inputFile][place] = formatstringc.format(self.formats[var][inputFile][1](self.defaults[var][inputFile]))
                else:
                  self.segments[inputFile][place] = _reprIfFloat(self.defaults[var][inputFile]).strip().rjust(self.formats[var][inputFile][1](self.formats[var][inputFile][0]))
            else:
              self.segments[inputFile][place] = self.defaults[var][inputFile]
          elif var in ioVars:
            continue #this gets handled in writeNewInput
          else:
            raise IOError('Generic Parser: Variable '+var+' was not sampled and no default given!')

  def writeNewInput(self,inFiles,origFiles):
    """
      Generates a new input file with the existing parsed dictionary.
      @ In, inFiles, list, Files list of new input files to return
      @ In, origFiles, list, the original list of Files, used for key names
      @ Out, None
    """
    #get the right IO names put in
    case = 'out~'+inFiles[0].getBase() #FIXME the first entry? This is bad! Forces order somewhere in input file
    # however, I can't seem to generate an error with this, so maybe it's okay
    def getFileWithExtension(fileList,ext):
      """
        Just a script to get the file with extension ext from the fileList.
        @ In, fileList, list, the Files list of files to pick from.
        @ In, ext, string, the string extension that the desired filename ends with.
        @ Out, None
      """
      found=False
      for index,inputFile in enumerate(fileList):
        if inputFile.getExt() == ext:
          found=True
          break
      if not found:
        raise IOError('No InputFile with extension '+ext+' found!')
      return index,inputFile

    for var in self.varPlaces.keys():
      for inputFile in self.segments.keys():
        for place in self.varPlaces[var][inputFile] if inputFile in self.varPlaces[var].keys() else []:
          for iotype,adlvar in self.adlDict.items():
            if iotype=='output':
              if var==self.adlDict[iotype]:
                self.segments[inputFile][place] = case
                break
            elif iotype=='input':
              if var in self.adlDict[iotype].keys():
                self.segments[inputFile][place] = getFileWithExtension(inFiles,self.adlDict[iotype][var][0].strip('.'))[1].getAbsFile()
                break
    #now just write the files.
    for f,inFile in enumerate(origFiles):
      outfile = inFiles[f]
      #if os.path.isfile(outfile.getAbsFile()): os.remove(outfile.getAbsFile())
      outfile.open('w')
      outfile.writelines(''.join(self.segments[inFile.getFilename()]))
      outfile.close()
