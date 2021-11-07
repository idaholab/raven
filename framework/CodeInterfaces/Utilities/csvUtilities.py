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
Created on Jun 5, 2015

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, absolute_import
# WARNING if you import unicode_literals here, we fail tests (e.g. framework.testFactorials).  This may be a future-proofing problem. 2015-04.
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
from glob import glob
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
#Internal Modules End--------------------------------------------------------------------------------

class csvUtilityClass(object):
  """
    This utility class is aimed to provide utilities for CSV handling.
  """
  def __init__(self, listOfFiles, linesToSkipAfterHeader=0, delimeter=",", mergeSameVariables=False):
    """
      Constructor
      @ In, listOfFiles, list, list of CSV files that need to be merged. If in one or more "filenames" the special symbol $*$ is present, the class will use the filename as root name and look for all the files with that root. For example:
                                             if  listOfFiles[1] == "aPath/outputChannel$*$":
                                               the code will inquire the directory "aPath" to look for all the files starting with the name "outputChannel" => at end we will have a list of files like "outputChannel_1.csv,outputChannel_ab.csv, etc"
      @ In, linesToSkipAfterHeader, int, optional, the number of lines that need to be skipped after the header
      @ In, delimeter, string, optional, the delimiter of the csv
      @ In, mergeSameVariables, bool, optional, do variables with the same name need to be merged together ? (aka, take only the values of the first occurence)?
      @ Out, None
    """
    if len(listOfFiles) == 0:
      raise IOError("MergeCSV class ERROR: the number of CSV files provided is equal to 0!! it can not merge anything!")
    self.listOfFiles    = []   # list of files
    self.dataContainer  = {}   # dictionary that is going to contain all the data from the multiple CSVs
    self.allHeaders     = []   # it containes all the headers
    filePathToExpand    = []
    self.mergeSameVariables = mergeSameVariables
    for filename in listOfFiles:
      if "$*$" in filename:
        filePathToExpand.append(filename)
      else:
        self.listOfFiles.append(filename)
    if len(filePathToExpand) > 0:
      # we need to look for this files
      for fileToExpand in filePathToExpand:
        self.listOfFiles.extend(glob(os.path.join(os.path.split(fileToExpand)[0],os.path.split(fileToExpand)[1].replace("$*$","*") + ".csv")))

    for filename in self.listOfFiles:
      # open file
      myFile = open (filename,'rb')
      # read the field names
      head = myFile.readline().decode()
      for _ in range(linesToSkipAfterHeader):
        myFile.readline()
      allFieldNames = head.split(delimeter)
      for index in range(len(allFieldNames)):
        allFieldNames[index] = allFieldNames[index].strip()
      if allFieldNames[-1] == "":
        allFieldNames.pop(-1) # it means there is a trailing "'" at the end of the file
      isAlreadyIn = False

      # load the table data (from the csv file) into a numpy nd array
      data = np.atleast_2d(np.loadtxt(myFile,
                                      delimiter=delimeter,
                                      usecols=tuple([i for i in range(len(allFieldNames))])))
      # close file
      myFile.close()
      self.allHeaders.extend(allFieldNames)
      # store the data
      self.dataContainer[filename] = {"headers":allFieldNames,"data":data}

  def mergeCSV(self,outputFileName, options = {}):
    """
      Method that is going to merge multiple csvs in a single one.
      @ In, outputFileName, string, full path of the resulting merged CSV (output file name, eg. /users/userName/output.csv)
      @ In, options, dict, optional, dictionary of options: { "variablesToExpandFrom":"aVariable" (a variable through which the "shorter" CSVs need to be expanded)
                                                              "sameKeySuffix":"integerCounter or filename (default)" (if in the CSVs that need to be merged there are
                                                              multiple occurrences of the same key, the code will append either a letter (A,B,C,D,etc) or an integer counter (1,2,3,etc)

                                                            }
      @ Out, None
    """
    if len(outputFileName.strip()) == 0:
      raise IOError("MergeCSV class ERROR: the outputFileName string is empty!")
    options['returnAsDict'] = False
    self.allHeaders, dataFinal = self.mergeCsvAndReturnOutput(options)
    np.savetxt(outputFileName,dataFinal,delimiter=",",header=",".join(self.allHeaders))

  def mergeCsvAndReturnOutput(self, options = {}):
    """
      Method that is going to read multiple csvs and return the merged results
      @ In, options, dict, optional, dictionary of options: { "variablesToExpandFrom":"aVariable" (a variable through which the "shorter" CSVs need to be expanded)
                                                              "sameKeySuffix":"integerCounter or filename (default)" (if in the CSVs that need to be merged there are
                                                              multiple occurrences of the same key, the code will append either a letter (A,B,C,D,etc) or an integer counter (1,2,3,etc)
                                                              "returnAsDict":True/False, True if the merged values need to be returned as a dictionary, otherwise it returns a tuple
                                                            }
      @ Out, mergedReturn, dict or tuple, merged csvs values (see "returnAsDict" option above to understand what you get)
    """
    # set some default
    sameKeySuffix        = "filename"
    variablesToExpandFrom = []
    returnAsDict = False
    variablesToExpandFrom.append('time')
    if options:
      if "sameKeySuffix" in options.keys():
        sameKeySuffix = options["sameKeySuffix"]
      if "variablesToExpandFrom" in options.keys():
        variablesToExpandFrom = options["variablesToExpandFrom"]
      if "returnAsDict" in options.keys():
        returnAsDict = bool(options["returnAsDict"])
    setHeaders = list(set(self.allHeaders))
    headerCounts = {}
    headerAppender = {}
    for variable in setHeaders:
      headerCounts[variable]   = self.allHeaders.count(variable)
      headerAppender[variable] = 0
    self.allHeaders = []
    variablesToExpandFromValues = {}
    variablesToExpandFromValuesSet = []
    for filename, data in self.dataContainer.items():
      for varToExpandFrom in variablesToExpandFrom:
        if varToExpandFrom in data["headers"]:
          variablesToExpandFromValues[filename] =  data["data"][:,data["headers"].index(varToExpandFrom)]
          variablesToExpandFromValuesSet.extend(variablesToExpandFromValues[filename].tolist())
        else:
          print("in file " + filename + "the variable "+ varToExpandFrom + " has not been found")
      for cnt, head in enumerate(data["headers"]):
        if headerCounts[head] > 1 and head not in variablesToExpandFrom:
          if not self.mergeSameVariables:
            #append a suffix
            if sameKeySuffix == "filename":
              self.dataContainer[filename]["headers"][cnt] = head + "_" + os.path.split(filename)[-1].split(".")[0]
            else:
              headerAppender[variable] += 1
              self.dataContainer[filename]["headers"][cnt] = head + "_" + str(headerAppender[variable])
      self.allHeaders.extend(data["headers"])
    # at this point all the headers are unique
    variablesToExpandFromValuesSet = list(set(variablesToExpandFromValuesSet))
    variablesToExpandFromValuesSet = sorted(variablesToExpandFromValuesSet, key=float)
    variablesToExpandFromValuesSet = np.array(variablesToExpandFromValuesSet)
    variablesToExpandFromValuesSet.shape = (len(variablesToExpandFromValuesSet),1)
    if len(variablesToExpandFromValues.keys()) != len(self.dataContainer.keys()):
      raise Exception ("the variables "+str(variablesToExpandFrom) + " have not been found in all files!!!!")
    dataFinal = np.zeros((len(variablesToExpandFromValuesSet),len(self.allHeaders)))
    # we use a neighbors.KNeighborsRegressor to merge the csvs
    from sklearn import neighbors
    nearest = neighbors.KNeighborsRegressor(n_neighbors=1)
    for filename, data in self.dataContainer.items():
      for _, varToExpandFrom in enumerate(variablesToExpandFrom):
        if varToExpandFrom in data["headers"]:
          index = data["headers"].index(varToExpandFrom)
          dataFinal[:,index] = variablesToExpandFromValuesSet[:,index]
          break
      for headIndex, head in enumerate(data["headers"]):
        if head not in variablesToExpandFrom:
          nearest.fit(np.atleast_2d(data["data"][:,index]).T,data["data"][:,headIndex])   #[nsamples,nfeatures]
          dataFinal[:,self.allHeaders.index(head)] = nearest.predict(variablesToExpandFromValuesSet)[:]
    if returnAsDict:
      mergedReturn = {}
      for variableToAdd in self.allHeaders:
        if self.mergeSameVariables:
          if variableToAdd not in mergedReturn.keys():
            mergedReturn[variableToAdd] = dataFinal[:,self.allHeaders.index(variableToAdd)]
        else:
          mergedReturn[variableToAdd] = dataFinal[:,self.allHeaders.index(variableToAdd)] # dataFinal[:,cnt]
    else:
      mergedReturn = (self.allHeaders,dataFinal)
    return mergedReturn
