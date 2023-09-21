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
Created on Oct 25, 2022
@author: khnguy22 (NCSU)
comments: Interface for PARCS loading pattern optimzation
          Originally, this was based on SIMULATE3 output structure
"""

import numpy
import os

#from ravenframework.utils import utils
class PARCSData:
  """
  Class that parses output of PARCS for a multiple run
  Partially copied from SIMULATE3 interface
  """
  def __init__(self,depletionFile,pinpowerFile):
    """
    Constructor
    @ In, depletionFile, string, depletion file name to be parsed
    @ In, pinpowerFile, string, pinpower file name to be parsed
    @ Out, None
    """
    self.data = {}
    with open(os.path.abspath(os.path.expanduser(depletionFile)),"r") as df:
      self.lines = df.readlines()
    # retrieve data, only needed data for optimization problem
    extractedData = self.getParam()
    self.data['keff'] = {'info_ids':extractedData['info_ids'][0], 'values':extractedData['values'][0]}
    self.data['FDeltaH'] = {'info_ids':extractedData['info_ids'][3], 'values':extractedData['values'][3]}
    self.data["boron"] = {'info_ids':extractedData['info_ids'][4], 'values':extractedData['values'][4]}
    self.data["cycle_length"] = {'info_ids':extractedData['info_ids'][2], 'values':extractedData['values'][2]}
    self.data["PinPowerPeaking"] = {'info_ids':extractedData['info_ids'][1], 'values':extractedData['values'][1]}
    self.data["exposure"] = {'info_ids':extractedData['info_ids'][5], 'values':extractedData['values'][5]}
    # check if something has been found
    if all(v is None for v in self.data.values()):
      raise IOError("No readable outputs have been found!")
    ## for pin power
    self.pinPower = self.getPinPower(pinpowerFile)
#------------------------------------------------------------------------------------------
  #function to retrivedata
  def getSummary(self):
    """
    Get the starting line and endding line of the summary in output file
    @ In, None
    @ Out, (lineBeg, lineEnd), tuple, the line indices for starting line and endding line of the summary in output file
    """
    lineBeg = 0
    lineEnd = 0
    numLines =  len(self.lines)
    for i in range (numLines):
      if self.lines[i].find('summary:')>=0:
        lineBeg = i+4
        for j in range (lineBeg, numLines):
          if self.lines[j].find('========================')>=0:
            lineEnd = j
            break
        break
    return lineBeg, lineEnd

  def getParam(self):
    """
    Extract all the parameters value from output file lines
    @ In, None
    @ Out, outDict, dict, the dictionary containing the read data (None if none found)
                           {'info_ids':list(of ids of data),
                            'values': list}
    """
    lineBeg, lineEnd = self.getSummary()
    cycLength = []
    keff = []
    FQ = []
    FdelH = []
    BU = []
    boronCon = []
    TFuel = []
    TMod = []
    for i in range (lineBeg, lineEnd):
      elem=self.lines[i].split()
      cycLength.append(float(elem[2]))
      keff.append(float(elem[3][:6]))
      fq = float(elem[4].split('(')[0])
      if elem[4].find(')')>0:
        indx=4
      else:
        indx=5
      FQ.append(fq)
      if elem[indx+1].find(')')>0:
        indx =indx+1
      FdelH.append(float(elem[indx+1]))
      BU.append(float(elem[indx+3]))
      boronCon.append(float(elem[indx+6]))
      TMod.append(float(elem[indx+8]))
      TFuel.append(float(elem[indx+7]))
    ## create a check point
    check = [cycLength, keff, FQ, FdelH, BU, boronCon, TFuel, TMod]
    c = [ii for ii in check if not ii]
    if c:
      return ValueError("No values returned. Check output File executed correctly")
    ### get cycle length at 10ppm interpolated
    for i in range (len(boronCon)):
      if (boronCon[i] - 10.0)<1e-3:
        idx_ = i
        break
    EOCboron = 10
    cycLengthEOC = cycLength[idx_-1] +(EOCboron-boronCon[idx_-1])*(cycLength[idx_]-cycLength[idx_-1])\
                  /(boronCon[idx_]-boronCon[idx_-1])
    outDict = {'info_ids':[['eoc_keff'], ['PinPowerPeaking'], ['MaxEFPD'],['MaxFDH'],
                              ['max_boron'], ['exposure']],
               'values': [[keff[-1]], [max(FQ)], [cycLengthEOC], [max(FdelH)],
                              [max(boronCon)], [BU[-1]] ]}
    return outDict

  def writeCSV(self, fileOut):
    """
      Print Data into CSV format
      @ In, fileOut, str, the output file name
      @ Out, None
    """
    fileObject = open(fileOut.strip()+".csv", mode='wb+') if not fileOut.endswith('csv') else open(fileOut.strip(), mode='wb+')
    headers=[]
    nParams = numpy.sum([len(data['info_ids']) for data in self.data.values() if data is not None and type(data) is dict])
    outputMatrix = numpy.zeros((nParams,1))
    index=0
    for data in self.data.values():
      if data is not None and type(data) is dict:
        headers.extend(data['info_ids'])
        for i in range(len(data['info_ids'])):
          outputMatrix[index]= data['values'][i]
          index=index+1
    numpy.savetxt(fileObject, outputMatrix.T, delimiter=',', header=','.join(headers), comments='')
    fileObject.close()

  def getPinPower(self, pinpowerFile):
    """
      Get the pin power from the depletion output file
      @ In, pinpowerFile, string, pinpower file name to be parsed
      @ Out, outputDict, dict, the dictionary containing the read data
                           {'info_ids':list(of ids of data),
                            'values': list}
    """
    with open(os.path.abspath(os.path.expanduser(pinpowerFile)),"r") as df:
      lines = df.readlines()
    numLines = len(lines)
    buStep = []
    faInfo = []
    nodeInfo = []
    pinPower = []
    step = 0
    buStep.append(0)
    for i in range(numLines):
      if lines[i].find("At Time:") >=0:
        step = step+1
        if lines[i].find("Assembly Coordinate (i,j):")>=0:
          temp = lines[i].split()
          nodeInfo.append([temp[10],temp[11],temp[15]])
          n = lines[i+1].split()
          n = int(n[-1])
          pinarray = []
          for jj in range (i+2, i+n+2):
            pinarray.append([float(val) for val in lines[jj].split()[1:]])
          pinPower.append(pinarray)
          faInfo.append([temp[3],temp[4], float(lines[i+n+2].split()[-1])])
          buStep.append(step)

    outputDict = {'info_ids':[['BUStep'], ['FAInfor'], ['nodeInfor'], ['pinPowerMap']],
                  'values': [[buStep],[faInfo], [nodeInfo], [pinPower]]}
    return outputDict
