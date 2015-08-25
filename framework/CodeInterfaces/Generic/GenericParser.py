'''
Created on Mar 10, 2015

@author: talbpaul
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
import os
import sys
import copy
from utils import toBytes, toStrish, compare

class GenericParser():
  '''import the user-edited input file, build list of strings with replacable parts'''
  def __init__(self,inputFiles,prefix='$RAVEN-',postfix='$',defaultDelim=':', formatDelim='|'):
    '''
    Accept the input file and parse it by the prefix-postfix breaks. Someday might be able to change prefix,postfix,defaultDelim from input file, but not yet.
    @ In, inputFiles, string list of input filenames that might need parsing.
    @ In, prefix, the string prefix to find input variables within the input files
    @ In, postfix, the string postfix signifying hte end of an input variable within an input file
    @ In, defaultDelim, the string used between prefix and postfix to set default values
    @ In, formatDelim, the string used between prefix and postfix to set the format of the value
    @Out, None.
    '''
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
      if not os.path.exists(inputFile.getAbsFile()): raise IOError('Input file not found: '+inputFile)
      foundSome = False
      seg = ''
      lines = inputFile.readlines()
      inputFile.close()
      for line in lines:
        while self.prefixKey in line and self.postfixKey in line:
          self.segments[infileName].append(toBytes(seg))
          start = line.find(self.prefixKey)
          end = line.find(self.postfixKey,start+1)
          var = line[start+len(self.prefixKey):end]
          if defaultDelim in var or formatDelim in var:
            optionalPos = [None]*2
            optionalPos[0], optionalPos[1] = var.find(defaultDelim), var.find(formatDelim)
            if optionalPos[0] == -1 : optionalPos[0]  = sys.maxint
            if optionalPos[1] == -1 : optionalPos[1] = sys.maxint
            defval    = var[optionalPos[0]+1:min(optionalPos[1],len(var))] if optionalPos[0] < optionalPos[1] else var[min(optionalPos[0]+1,len(var)):len(var)]
            varformat = var[min(optionalPos[1]+1,len(var)):len(var)] if optionalPos[0] < optionalPos[1] else var[optionalPos[1]+1:min(optionalPos[0],len(var))]
            var = var[0:min(optionalPos)]
            if var in self.defaults.keys() and optionalPos[0] != sys.maxint: print('multiple default values given for variable',var)
            if var in self.formats.keys() and optionalPos[1] != sys.maxint: print('multiple format values given for variable',var)
            #TODO allow the user to specify take-last or take-first?
            if var not in self.defaults.keys() and optionalPos[0] != sys.maxint : self.defaults[var] = {}
            if var not in self.formats.keys()  and optionalPos[1] != sys.maxint : self.formats[var ] = {}
            if optionalPos[0] != sys.maxint: self.defaults[var][infileName]=defval
            if optionalPos[1] != sys.maxint:
              # check if the format is valid
              if not any(formVal in varformat for formVal in self.acceptFormats.keys()):
                try              : int(varformat)
                except ValueError: raise ValueError("the format specified for wildcard "+ line[start+len(self.prefixKey):end] +
                                                     " is unknown. Available are either a plain integer or the following "+" ".join(self.acceptFormats.keys()))
                self.formats[var][infileName ]=varformat,int
              else:
                for formVal in self.acceptFormats.keys():
                  if formVal in varformat: self.formats[var][infileName ]=varformat,self.acceptFormats[formVal]; break
          self.segments[infileName].append(toBytes(line[:start]))
          self.segments[infileName].append(toBytes(var))
          if var not in self.varPlaces.keys(): self.varPlaces[var] = {infileName:[len(self.segments[infileName])-1]}
          elif inputFile not in self.varPlaces[var].keys(): self.varPlaces[var][infileName]=[len(self.segments[infileName])-1]
          else: self.varPlaces[var][infileName].append(len(self.segments[infileName])-1)
          #self.segments.append(toBytes(line[end+1:]))
          line=line[end+1:]
          seg = ''
        else:
          seg+=line
      self.segments[infileName].append(toBytes(seg))

  def modifyInternalDictionary(self,**Kwargs):
    '''
    Edits the parsed file stored in self.segments to enter new variable values preperatory to a new run.
    @ In, **Kwargs, dict including moddit (the dictionary of variable:value to replace) and additionalEdits.
    @Out, None.
    '''
    moddict = Kwargs['SampledVars']
    self.adldict = Kwargs['additionalEdits']
    iovars = []
    for key,value in self.adldict.items():
      if type(value)==dict:
        for k in value.keys():
          iovars.append(k)
      elif type(value)==list:
        for v in value:
          iovars.append(v)
      else:
        iovars.append(value)
    newFileStrings={}
    for var in self.varPlaces.keys():
      for inputFile in self.segments.keys():
        for place in self.varPlaces[var][inputFile] if inputFile in self.varPlaces[var].keys() else []:
          if var in moddict.keys():
            if var in self.formats.keys():
              if inputFile in self.formats[var].keys():
                if any(formVal in self.formats[var][inputFile][0] for formVal in self.acceptFormats.keys()):
                  formatstringc = "{:"+self.formats[var][inputFile][0].strip()+"}"
                  self.segments[inputFile][place] = formatstringc.format(self.formats[var][inputFile][1](moddict[var]))
                else: self.segments[inputFile][place] = str(moddict[var]).strip().rjust(self.formats[var][inputFile][1](self.formats[var][inputFile][0]))
            else: self.segments[inputFile][place] = str(moddict[var])
          elif var in self.defaults.keys():
            if var in self.formats.keys():
              if inputFile in self.formats[var].keys():
                if any(formVal in self.formats[var][inputFile][0] for formVal in self.acceptFormats.keys()):
                  formatstringc = "{:"+self.formats[var][inputFile][0].strip()+"}"
                  self.segments[inputFile][place] = formatstringc.format(self.formats[var][inputFile][1](self.defaults[var][inputFile]))
                else: self.segments[inputFile][place] = str(self.defaults[var][inputFile]).strip().rjust(self.formats[var][inputFile][1](self.formats[var][inputFile][0]))
            else: self.segments[inputFile][place] = self.defaults[var][inputFile]
          elif var in iovars: continue #this gets handled in writeNewInput
          else: raise IOError('Variable '+var+' was not sampled and no default given!')

  def writeNewInput(self,inFiles,origFiles):
    '''
    Generates a new input file with the existing parsed dictionary.
    @ In, inFiles, Files list of new input files to return
    @ In, origFiles, the original list of Files, used for key names
    @Out, None.
    '''
    #get the right IO names put in
    case = 'out~'+inFiles[0].getBase()
    def getFileWithExtension(fileList,ext):
      '''
      Just a script to get the file with extension ext from the fileList.
      @ In, fileList, the Files list of files to pick from.
      @Out, ext, the string extension that the desired filename ends with.
      '''
      found=False
      for index,inputFile in enumerate(fileList):
        if inputFile.getExt() == ext:
          found=True
          break
      if not found: raise IOError('No InputFile with extension '+ext+' found!')
      return index,inputFile

    for var in self.varPlaces.keys():
      for inputFile in self.segments.keys():
        for place in self.varPlaces[var][inputFile] if inputFile in self.varPlaces[var].keys() else []:
          for iotype,adlvar in self.adldict.items():
            if iotype=='output':
              if var==self.adldict[iotype]:
                self.segments[inputFile][place] = case
                break
            elif iotype=='input':
              if var in self.adldict[iotype].keys():
                self.segments[inputFile][place] = getFileWithExtension(inFiles,self.adldict[iotype][var][0].strip('.'))[1].getAbsFile()
                break
    #now just write the files.
    for f,inFile in enumerate(origFiles):
      outfile = inFiles[f]
      outfile.writelines(toBytes(''.join(self.segments[inFile.getFilename()])))
      outfile.close()
