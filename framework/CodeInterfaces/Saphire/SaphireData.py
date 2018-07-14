
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
import numpy as np
import os
import copy
from sklearn import neighbors
import re
"""
Created on July 12, 2018

@author: wangc
"""

class SaphireData:
  """
    Class that parses output of SAPHIRE outputs and write a RAVEN compatible CSV
  """

  def __init__(self, outFiles):
    """
      Initialize the class
      @ In, outFiles, list, list of output files of SAPHIRE
      @ Out, None
    """
    self.headerNames = []
    self.outData = []
    for outFile in outFiles:
      outFileName, outFileType = outFile[0], outFile[1]
      if outFileType == 'uncertainty':
        headers, data = self.getUncertainty(outFileName)
        self.headerNames.append(headers)
        self.outData.append(data)
      elif outFileType == 'importance':
        headers, data = self.getImportance(outFileName)
        self.headerNames.append(headers)
        self.outData.append(data)
      elif outFileType == 'quantiles':
        print("File:",outFileName, "with type", outFileType, "is not implemented yet! Skipping" )
        pass
      else:
        raise IOError('The output file', outFileName, 'with type', outFileType, 'is not supported yet!')

  def getUncertainty(self, outName):
    """
      Method to extract the uncertainty information of Event Tree or Fault Tree from SAPHIRE output files
      @ In, outName, string, the name of output file
      @ Out, headerNames, list, list of output variable names
      @ Out, outData, list, list of output variable values
    """
    headerNames = []
    outData = []
    outFile = os.path.abspath(os.path.expanduser(outName))
    data = np.loadtxt(outFile, dtype=object, delimiter=',', skiprows=2)
    headers = data[0]
    for i in range(1, len(data)):
      for j in range(1, len(headers)):
        name = data[i,0].strip().replace(" ", "~")
        header = headers[j].strip().replace(" ", "~")
        headerNames.append(name + '_' + header)
        outData.append(float(data[i,j]))

    return headerNames, outData

  def getImportance(self, outName):
    """
      Method to extract the importance information of Fault Tree from SAPHIRE output files
      @ In, outName, string, the name of output file
      @ Out, headerNames, list, list of output variable names
      @ Out, outData, list, list of output variable values
    """
    headerNames = []
    outData = []
    outFile = os.path.abspath(os.path.expanduser(outName))
    data = np.loadtxt(outFile, dtype=object, delimiter=',', skiprows=2)
    headers = data[0]
    for i in range(1, len(data)):
      for j in range(1, len(headers)):
        name = data[i,0].strip().replace(" ", "~")
        header = headers[j].strip().replace(" ", "~")
        headerNames.append(name + '_' + header)
        outData.append(float(data[i,j]))

    return headerNames, outData

  def writeCSV(self, output):
    """
      Print data into CSV format
      @ In, output, str, the name of output file
      @ Out, None
    """
    outObj = open(output.strip()+".csv", mode='w+') if not output.endswith('csv') else open(output.strip(), mode='w+')
    # create string for header names
    headerString = ",".join(self.headerNames)
    # write & save array as csv file
    np.savetxt(outObj, self.outData, delimiter=',', header=headerString, comments='')
    outObj.close()


