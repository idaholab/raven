'''
Created on Jun 5, 2015

@author: alfoa
'''
import os
from glob import glob
import numpy as np

class csvUtilityClass(object):
  """
  This utility class is aimed to provide utilities for CSV handling.
  """
  def __init__(self, listOfFiles):
    """
    Constructor
    @ In, list, required param, listOfFiles, list of CSV files that need to be merged. If in one or more "filenames" the special symbol $*$ is present, the class will use the filename as root name and look for all the files with that root. For example:
                                             if  listOfFiles[1] == "aPath/outputChannel$*$": the code will inquire the directory "aPath" to look for all the files starting with the name "outputChannel" => at end we will have a list of files like "outputChannel_1.csv,outputChannel_ab.csv, etc"

    """
    if len(listOfFiles) == 0: raise IOError("MergeCSV class ERROR: the number of CSV files provided is equal to 0!! it can not merge anything!")
    self.listOfFiles    = []   # list of files
    self.dataContainer  = {}   # dictionary that is going to contain all the data from the multiple CSVs
    self.allHeaders     = []   # it containes all the headers
    filePathToExpand    = []
    for filename in listOfFiles:
      if "$*$" in filename: filePathToExpand.append(filename)
      else                : self.listOfFiles.append(filename)
    if len(filePathToExpand) > 0:
      # we need to look for this files
      for fileToExpand in filePathToExpand: self.listOfFiles.extend(glob(os.path.join(os.path.split(fileToExpand)[0],os.path.split(fileToExpand)[1].replace("$*$","*") + ".csv")))

    for filename in self.listOfFiles:
      # open file
      myFile = open (filename,'rb')
      # read the field names
      head = myFile.readline().decode()
      all_field_names = head.split(',')
      for index in range(len(all_field_names)): all_field_names[index] = all_field_names[index].strip()
      if all_field_names[-1] == "": all_field_names.pop(-1) # it means there is a trailing "'" at the end of the file
      self.allHeaders.extend(all_field_names)
      # load the table data (from the csv file) into a numpy nd array
      data = np.loadtxt(myFile, delimiter=",", usecols=tuple([i for i in range(len(all_field_names))]))
      # close file
      myFile.close()
      # store the data
      self.dataContainer[filename] = {"headers":all_field_names,"data":data}
    print("file read")

  def mergeCSV(self,outputFileName,options = {}):
    from sklearn import neighbors

    """
    Method that is going to merge multiple csvs in a single one.
    @ In, string, required param, outputFileName, full path of the resulting merged CSV (output file name, eg. /users/userName/output.csv)
    @ In, dict, optional param. options, dictionary of options: {
                                                                 "variablesToExpandFrom":"aVariable" (a variable through which the "shorter" CSVs need to be expanded)
                                                                 "sameKeySuffix":"integerCounter or filename (default)" (if in the CSVs that need to be merged there are
                                                                 multiple occurrences of the same key, the code will append either a letter (A,B,C,D,etc) or an integer counter (1,2,3,etc)
                                                                 }
    """
    if len(outputFileName.strip()) == 0: raise IOError("MergeCSV class ERROR: the outputFileName string is empty!")
    # set some default
    sameKeySuffix        = "filename"
    variablesToExpandFrom = ["time"]
    if options:
      if "sameKeySuffix" in options.keys()        : sameKeySuffix         = options["sameKeySuffix"]
      if "variablesToExpandFrom" in options.keys(): variablesToExpandFrom = options["variablesToExpandFrom"]
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
      for cnt, head in enumerate(data["headers"]):
        if headerCounts[head] > 1 and head not in variablesToExpandFrom:
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
    if len(variablesToExpandFromValues.keys()) != len(self.dataContainer.keys()): raise Exception ("the variables "+str(variablesToExpandFrom) + "have not been found in all files!!!!")
    datafinal = np.zeros((len(variablesToExpandFromValuesSet),len(self.allHeaders)))
    # we use a neighbors.KNeighborsRegressor to merge the csvs
    nearest = neighbors.KNeighborsRegressor(n_neighbors=1)
    for filename, data in self.dataContainer.items():
      for _, varToExpandFrom in enumerate(variablesToExpandFrom):
        if varToExpandFrom in data["headers"]:
          index = data["headers"].index(varToExpandFrom)
          break
      for headindex, head in enumerate(data["headers"]):
        nearest.fit(np.atleast_2d(data["data"][:,index]).T,data["data"][:,headindex])   #[nsamples,nfeatures]
        datafinal[:,self.allHeaders.index(head)] = nearest.predict(variablesToExpandFromValuesSet)[:]
    np.savetxt(outputFileName,datafinal,delimiter=",",header=",".join(self.allHeaders))




