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
Created on July 12, 2018

@author: wangc
"""
import numpy as np
import os
import copy
import re

def _deweird(s):
  """
    Sometimes numpy loadtxt returns strings like "b'stuff'"
    This converts them to "stuff"
    @ In, s, str, possibly weird string
    @ Out, _deweird, str, possibly less weird string
  """
  if type(s) == str and s.startswith("b'") and s.endswith("'"):
    return s[2:-1]
  else:
    return s

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
    self.headerNames = [] # list of variable names in SAPHIRE output files
    self.outData = []     # list of variable values in SAPHIRE output files
    for outFile in outFiles:
      outFileName, outFileType = outFile[0], outFile[1]
      if outFileType == 'uncertainty':
        headers, data = self.getUncertainty(outFileName)
        self.headerNames.extend(headers)
        self.outData.extend(data)
      elif outFileType == 'importance':
        headers, data = self.getImportance(outFileName)
        self.headerNames.extend(headers)
        self.outData.extend(data)
      elif outFileType == 'quantiles':
        print("File:",outFileName, "with type", outFileType, "is not implemented yet! Skipping" )
        pass
      else:
        raise IOError('The output file', outFileName, 'with type', outFileType, 'is not supported yet!')

  def getUncertainty(self, outName):
    """
      Method to extract the uncertainty information of Event Tree or Fault Tree from SAPHIRE output files
      @ In, outName, string, the name of output file
      @ Out, (headerNames,outData), tuple, where headerNames is a list of output variable names and
        outData is a list of output variable values
    """
    headerNames = []
    outData = []
    outFile = os.path.abspath(os.path.expanduser(outName))
    data = np.loadtxt(outFile, dtype=object, delimiter=',', skiprows=2)
    headers = data[0]
    for i in range(1, len(data)):
      for j in range(1, len(headers)):
        name = _deweird(data[i,0]).strip().replace(" ", "~")
        header = _deweird(headers[j]).strip().replace(" ", "~")
        headerNames.append(name + '_' + header)
        outData.append(float(_deweird(data[i,j])))

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
        name = _deweird(data[i,0]).strip().replace(" ", "~")
        header = _deweird(headers[j]).strip().replace(" ", "~")
        headerNames.append(name + '_' + header)
        outData.append(float(_deweird(data[i,j])))

    return headerNames, outData

  def writeCSV(self, output):
    """
      Print data into CSV format
      @ In, output, str, the name of output file
      @ Out, None
    """
    outObj = open(output.strip()+".csv", mode='w+b') if not output.endswith('csv') else open(output.strip(), mode='w+b')
    # create string for header names
    headerString = ",".join(self.headerNames)
    # write & save array as csv file
    # FIXME: There is a problem with the numpy.savetxt, if provided data is 1D array_like, the demiliter will be
    # ignored, and out file format is not correct
    np.savetxt(outObj, [self.outData], delimiter=',', header=headerString, comments='')
    outObj.close()


