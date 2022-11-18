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
Last modified on Oct 26, 2022
@author: khnguy22 NCSU
comments: Interface for PARCS loading pattern optimzation
          Originally, this was based on SIMULATE3 output structure
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os, gc, sys, copy, h5py, math
import time
import numpy
import pickle
import random
import os
import copy
import shutil
#from ravenframework.utils import utils
class PARCSData:
  """
  Class that parses output of PARCS for a multiple run
  Partially copied from SIMULATE3 interface
  """
  def __init__(self,depletionfile,pinpowerfile):
    """
    Constructor
    @ In, depletionfile, string or dict, depletion file name to be parsed, read one file at a time
    @ In, pinpowerfile, string or dict, pinpower file name to be parsed, read one file at a time
    @ Out, None
    """
    self.data = {}
    self.lines = open(os.path.abspath(os.path.expanduser(depletionfile)),"r").readlines()
    # retrieve data, only needed data for optimization problem
    extractedData = self.getParam()
    self.data['keff'] = {'info_ids':extractedData['info_ids'][0], 'values':extractedData['values'][0]}
    self.data['FDeltaH'] = {'info_ids':extractedData['info_ids'][3], 'values':extractedData['values'][3]}
    self.data["boron"] = {'info_ids':extractedData['info_ids'][4], 'values':extractedData['values'][4]}
    self.data["cycle_length"] = {'info_ids':extractedData['info_ids'][2], 'values':extractedData['values'][2]}
    self.data["PinPowerPeaking"] = {'info_ids':extractedData['info_ids'][1], 'values':extractedData['values'][1]}
    self.data["exposure"] = {'info_ids':extractedData['info_ids'][5], 'values':extractedData['values'][5]}
    # this is a dummy variable for demonstration
    # Multi-objective --> single objective
    self.data["target"] = self.getTarget()
    # check if something has been found
    if all(v is None for v in self.data.values()):
      raise IOError("No readable outputs have been found!")
    ## for pin power
    self.pinPower = self.getPinpower(pinpowerfile)
#------------------------------------------------------------------------------------------
  #function to retrivedata
  def getSummary(self):
    """
    Get the starting line and endding line of the summary in output file
    @ In, None
    # Out, lineBeg, lineEnd
    """
    lineBeg = 0
    lineEnd = 0
    nlines =  len(self.lines)
    for i in range (nlines):
      if self.lines[i].find('summary:')>=0:
        lineBeg = i+4
        for j in range (lineBeg, nlines):
          if self.lines[j].find('========================')>=0:
            lineEnd = j
            break
        break
    return lineBeg, lineEnd
  def getParam(self):
    """
    Extract all the parameters value from output file lines
    @ In, None
    @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                           {'info_ids':list(of ids of data),
                            'values': list}
    """
    lineBeg, lineEnd = self.getSummary()
    cycLength = []
    k_eff = []
    FQ = []
    FdelH = []
    BU = []
    BoronCon = []
    Tfuel = []
    Tmod = []
    for i in range (lineBeg, lineEnd):
      elem=self.lines[i].split()
      cycLength.append(float(elem[2]))
      k_eff.append(float(elem[3]))
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
      BoronCon.append(float(elem[indx+6]))
      Tmod.append(float(elem[indx+8]))
      Tfuel.append(float(elem[indx+7]))
    ## create a check point
    check = [cycLength, k_eff, FQ, FdelH, BU, BoronCon, Tfuel, Tmod]
    c = [ii for ii in check if not ii]
    if c:
      return ValueError("No values returned. Check output File executed correctly")
    ### get cycle length at 10ppm interpolated
    for i in range (len(BoronCon)):
      if (BoronCon[i] - 0.1)<1e-3:
        idx_ = i
        break
    EOCboron = 10
    cycLengthEOC = cycLength[idx_-1] +(EOCboron-BoronCon[idx_-1])*(cycLength[idx_]-cycLength[idx_-1])\
                  /(BoronCon[idx_]-BoronCon[idx_-1])
    outDict = {'info_ids':[['eoc_keff'], ['PinPowerPeaking'], ['MaxEFPD'],['MaxFDH'],
                              ['max_boron'], ['exposure']],
               'values': [[k_eff[-1]], [max(FQ)], [cycLengthEOC], [max(FdelH)],
                              [max(BoronCon)], [BU[-1]] ]}
    return outDict
  def getTarget(self):
    """
    This is a function to convert the fitness function to be output variable and make the
    problem to be single-objective rather than multi-objective optimzation
    @ In, None
    @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                       {'info_ids':list(of ids of data),
                        'values': list}
    """
    tmp = -1.0*max(0,self.data["boron"]['values'][0] - 1300)\
          -400*max(0,self.data["PinPowerPeaking"]["values"][0]-2.1) \
          -400*max(0,self.data['FDeltaH']["values"][0]-1.48)\
          +self.data["cycle_length"]["values"][0]
    outputDict = {'info_ids':['target'], 'values': [tmp]}
    return outputDict
  def writeCSV(self, fileout):
    """
      Print Data into CSV format
      @ In, fileout, str, the output file name
      @ Out, None
    """
    fileObject = open(fileout.strip()+".csv", mode='wb+') if not fileout.endswith('csv') else open(fileout.strip(), mode='wb+')
    headers=[]
    timeGrid = None
    nParams = numpy.sum([len(data['info_ids']) for data in self.data.values() if data is not None and type(data) is dict])
    ndata = 1
    outputMatrix = numpy.zeros((nParams,1))
    tmp = [data['info_ids'] for data in self.data.values() if data is not None and type(data) is dict]
    index=0
    for data in self.data.values():
      if data is not None and type(data) is dict:
        headers.extend(data['info_ids'])
        for i in range(len(data['info_ids'])):
          outputMatrix[index]= data['values'][i]
          index=index+1
    numpy.savetxt(fileObject, outputMatrix.T, delimiter=',', header=','.join(headers), comments='')
    fileObject.close()
  def getPinpower(self, depletionfile):
    lines = open(os.path.abspath(os.path.expanduser(depletionfile)),"r").readlines()
    n_line = len(lines)
    bu_step = []
    FAinfo = []
    Nodeinfo = []
    Pinpower = []
    step = 0
    bu_step.append(0)
    for i in range(n_line):
      if lines[i].find("At Time:") >=0:
        step = step+1
        if lines[i].find("Assembly Coordinate (i,j):")>=0:
          temp = lines[i].split()
          Nodeinfo.append([temp[10],temp[11],temp[15]])
          N = lines[i+1].split()
          N = int(N[-1])
          pinarray = []
          for jj in range (i+2, i+N+2):
            pinarray.append([float(val) for val in lines[jj].split()[1:]])
          Pinpower.append(pinarray)
          FAinfo.append([temp[3],temp[4], float(lines[i+N+2].split()[-1])])
          bu_step.append(step)

    outputDict = {'info_ids':[['BUStep'], ['FAInfor'], ['nodeInfor'], ['pinPowerMap']],
                  'values': [[bu_step],[FAinfo], [Nodeinfo], [Pinpower]]}
    return outputDict
