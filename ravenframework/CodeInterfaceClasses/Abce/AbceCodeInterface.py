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
Created on 2023-Jul-12

This is a CodeInterface for the ABCE code.
"""

import os
import re
import warnings
import sqlite3
import pandas as pd

from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..Generic.GenericCodeInterface import GenericParser

class Abce(CodeInterfaceBase):
  """
    This class is used to run the Abce code.
  """

  def generateCommand(self,inputFiles,executable,clargs=None, fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path of ABCE/run.py
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, ignored
      @ In, preExec, ignored
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    if clargs==None:
      raise IOError('No input file was specified in clargs!')
    #check for duplicate extension use
    usedExts = list(ext[0][0] for ext in clargs['input'].values() if len(ext) != 0)
    if len(usedExts) != len(set(usedExts)):
      raise IOError('ABCEInterface cannot handle multiple input files with the same extension.  You may need to write your own interface.')
    for inf in inputFiles:
      ext = '.' + inf.getExt() if inf.getExt() is not None else ''
      try:
        usedExts.remove(ext)
      except ValueError:
        pass
    if len(usedExts) != 0:
      raise IOError('Input extension',','.join(usedExts),'listed in XML node Code, but not found in the list of Input of <Files>')

    def findSettingIndex(inputFiles,ext):
      """
      Find the settings file and return its index in the inputFiles list.
      @ In, inputFiles, list, list of InputFile objects
      @ In, ext, string, extension of the settings.yml file
      @ Out, (index, inputFile), (int, str), index of the settings file in the inputFiles list and the file
      """
      for index,inputFile in enumerate(inputFiles):
        if inputFile.getBase() == 'settings' and inputFile.getExt() == ext:
          return index,inputFile
      raise IOError('No settings file with extension '+ext+' found!')

    def setOutputDir(settingsFile):
      """
      Set the output directory in the settings file.
      @ In, settingsFile, InputFile object, settings.yml file
      @ Out, None
      """
      # the settings file is the settings.yml file the scenario name is in node
      # simulation -> scenario_name
      # the output directory is the directory of the settings file in subdirectory "outputs/$scenario_name"
      # get the scenario name by reading the settings.yml file
      scenarioName = None
      with open(settingsFile.getAbsFile(),'r') as settings:
        for line in settings:
          if 'scenario_name' in line:
            scenarioName = line.split(':')[1].strip()
            # remove the double quotes
            scenarioName = scenarioName.replace('"','')
            break
      self._outputDirectory = os.path.join(os.path.dirname(settingsFile.getAbsFile()),'outputs',scenarioName)
      return None

    #prepend
    fullCommand = ''
    fullCommand += clargs['pre']+' '
    fullCommand += executable
    index=None
    #setup input files and output directory
    self._outputDirectory = None
    #inputs
    for flag,elems in clargs['input'].items():
      if flag == 'noarg':
        continue
      fullCommand += ' '+flag
      for elem in elems:
        ext, delimiter = elem[0], elem[1]
        idx,fname = findSettingIndex(inputFiles,ext.strip('.'))
        setOutputDir(fname)
        fullCommand += delimiter + fname.getFilename()
        if index == None:
          index = idx
    self.caseName = inputFiles[index].getBase()
    outFile = 'out~'+self.caseName
    if 'output' in clargs:
      fullCommand+=' '+clargs['output']+' '+outFile
    fullCommand+=' '+clargs['text']
    fullCommand+=' '+clargs['post']
    returnCommand = [('parallel',fullCommand)],outFile
    print('Execution Command: '+str(returnCommand[0]))
    return returnCommand

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    """
      This method is used to generate an input based on the information passed in.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    infiles=[]
    origfiles=[]
    for inputFile in currentInputFiles:
      if inputFile.getExt() in self.getInputExtension():
        infiles.append(inputFile)
    for inputFile in origInputFiles:
      if inputFile.getExt() in self.getInputExtension():
        origfiles.append(inputFile)
    parser = GenericParser.GenericParser(infiles)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(infiles,origfiles)
    return currentInputFiles

  def finalizeCodeOutput(self, command, codeLogFile, subDirectory):
    """
      Convert SQLite information to RAVEN's prefered formats [Pandas DataFrame]
      Joins together two different SQLite files and also reorders it a bit.
      @ In, command, ignored
      @ In, codeLogFile, ignored
      @ In, subDirectory, string, the subdirectory where the information is.
      @ Out, outDict, dictionary of DataFrames, the assets results
    """
    outDict = {}
    outputFile = os.path.join(self._outputDirectory,'abce_db.db')
    db_conn = sqlite3.connect(outputFile)
    assetsDataFrame = pd.read_sql_query("SELECT asset_id, agent_id, unit_type, start_pd, completion_pd, retirement_pd, cancellation_pd, total_capex, cap_pmt, C2N_reserved from assets", db_conn)
    db_conn.close()
    # print("assetsData", assetsData)
    # read each column and store it in the dictionary
    # column_names are: asset_id agent_id unit_type start_pd completion_pd
    # cancellation_pd retirement_pd total_capex cap_pmt C2N_reserved
    for col in assetsDataFrame.columns:
      outDict[col] = assetsDataFrame[col].values
    # TODO should change it in the future for reading the output file
    # from the code or the database file
    # OutputPlaceHolder should be a list of float("NaN")
    # if the len(assetsData)>0 or just a float("NaN")
    outDict['OutputPlaceHolder'] = float("NaN")*len(assetsDataFrame)
    return outDict
