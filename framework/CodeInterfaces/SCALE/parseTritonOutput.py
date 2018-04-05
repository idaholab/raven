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
import re
import xml.etree.ElementTree as ET
import numpy as np
import sys
import os
import copy
from sklearn import neighbors
"""
Created on March 25, 2018

@author: alfoa
"""

def _scalingFactorBetweenTimeUOM(targetUOM, inputUOM):
  """
    Method to get the scaling factor based between UOM for time
    @ In, targetUOM, str, the target UOM (e.g. s, m,h, d, y)
    @ In, inputUOM, str, the UOM (e.g. s, m,h, d, y) for which the scaling factor needs to be computed
    @ Out, multiplier, float, the multiplier to apply
  """
  scaling = {'s':1.0,'m':60.0,'h':60.0,'d':24.0,'y':365.25}
  uoms    = ['s','m','h','d','y']
  try:
    targetUOMIndex = uoms.index(targetUOM.strip()[0].lower())
  except ValueError:
    raise ValueError("unricognized UOM <"+targetUOM.strip()+">!")
  try:
    inputUOMIndex = uoms.index(inputUOM.strip()[0].lower())
  except ValueError:
    raise ValueError("unricognized UOM <"+inputUOM.strip()+">!")

  if inputUOMIndex == targetUOMIndex:
    return 1.0
  else:
    exponent =1.0 if targetUOMIndex < inputUOMIndex else -1.0
    targetMultiplier, inputMultiplier = 1.0,1.0
    for uom in uoms[0:targetUOMIndex+1]:
      targetMultiplier*= scaling[uom]
    for uom in uoms[0:inputUOMIndex+1]:
      inputMultiplier*= scaling[uom]
    multiplier = (inputMultiplier/targetMultiplier)**exponent
  return multiplier



class scaleData:
  """
    Class that parses output of scale output (now Triton and/or Orgin only) and write a RAVEN compatible CSV
  """
  def __init__(self,filen):
    """
      Constructor
      @ In, filen, string, file name to be parsed
      @ Out, None
    """
    self.lines            = open(os.path.abspath(os.path.expanduser(filen)),"r").readlines()
    # retrieve keff and kinf
    self.data = {}
    self.data['transportInfo']    = self.retrieveKeff()
    # retrieve nuclide densities
    self.data['nuclideDensities'] = self.retrieveNuclideConcentrations()
    # retrieve mixture powers
    self.data['mixPowers']        = self.retrieveMixtureInfo()
    # origen history overview and concentration tables
    self.data['origenData']  = self.retrieveOrigenData()
    # check if something has been found
    if all(v is None for v in self.data.values()):
      raise IOError("No readable outputs have been found!")
    # check time grid consistency, if not, interpolate by nearest neighbor
    self.dataConsistency()

  def dataConsistency(self):
    """
      Check the info present in self.data
      If the time grids do not correspond, nearest neighbor interpolation
      @ In, None
      @ Out, None
    """
    timeGrid = []
    for data in self.data.values():
      if data is not None:
        if not set(data['time']) <= set(timeGrid):
          timeGrid.extend(data['time'])
          timeGrid = list(set(timeGrid))
    timeGrid.sort()
    for data in self.data.values():
      if data is not None and data['time'] != timeGrid:
        knn = neighbors.KNeighborsRegressor(n_neighbors=1)
        knn.fit(np.atleast_2d([data['time']]).T , data['values'].T)
        data['values'] = knn.predict(np.atleast_2d([timeGrid]).T).T
        data['time']   = timeGrid

  def retrieveOrigenData(self):
    """
      Retrieve Origen Data
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),'info_ids':list(of ids of data),'values':np.ndarray( ntime,nids )}
    """
    histOverview = self.retrieveHistoryOverview()
    overviewTimeUom = histOverview['timeUOM']
    concTables   = self.retrieveConcentrationTables()
    concTimeUom = concTables['timeUOM']
    if overviewTimeUom.strip() != concTimeUom.strip():
      # convert time
      multiplier = _scalingFactorBetweenTimeUOM(overviewTimeUom, concTimeUom)
      test = _scalingFactorBetweenTimeUOM("h", "y")
      test = _scalingFactorBetweenTimeUOM("y", "s")






  def retrieveHistoryOverview(self):
    """
      Retrieve History Overview (Origen)
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),'info_ids':list(of ids of data),'values':np.ndarray( ntime,nids )}
    """
    # history overview
    outputDict = None
    try:
      indexHistOverview = [i+3 for i, x in enumerate(self.lines) if x.strip().startswith('=   History overview for')]
    except StopIteration:
      return outputDict
    if len(indexHistOverview) > 1:
      raise IOError("Multiple cases have been found in ORIGEN output. Currently Only one is handaled!")
    indexHistOverview = indexHistOverview[0]
    headers = self.lines[indexHistOverview-1].split()
    ids = copy.deepcopy(headers)
    ids.pop( ids.index("t"))
    timeUom = self.lines[indexHistOverview].split()[headers.index("t")].replace("(","").replace(")","")
    values = []
    outputDict = {'time':[],'info_ids':ids, 'values':None, 'timeUOM':timeUom}
    startIndex = indexHistOverview
    stepName = "-"
    while stepName.strip() != "":
      startIndex+=1
      components = self.lines[startIndex].split()
      stepName   = ""
      if len(components) > 0:
        stepName = components[0].strip()
        time = float(components[4].strip())
        outputDict['time'].append(time)
        components.pop(headers.index("t"))
        values.append( [float(elm) for elm in  components[:] ])

    outputDict['values'] = np.atleast_2d(values).T
    return outputDict

  def retrieveConcentrationTables(self):
    """
      Retrieve History Nuclide Evolutions (Origen) - Concentration tables
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),'info_ids':list(of ids of data),'values':np.ndarray( ntime,nids )}
    """
    # Concentration tables
    outputDict = None
    try:
      indexHistOverview = [i for i, x in enumerate(self.lines) if x.strip().startswith('=   Nuclide concentrations in')]
    except StopIteration:
      return outputDict
    headers = []
    values = []
    totals = []
    for cnt, indexConcTable in enumerate(indexHistOverview):
      uom = self.lines[indexConcTable].split(",")[0].split()[-1].strip()
      isotopeType = self.lines[indexConcTable].split(",")[-1].split("for case")[0].strip()
      indexConcTable+=4
      timeUom = self.lines[indexConcTable].split()[0][self.lines[indexConcTable].split()[0].index("E+")+4:].strip()
      timeGrid = [float(elm.replace(timeUom.strip(),"")) for elm in  self.lines[indexConcTable].split()]

      if outputDict is None:
        outputDict = {'time':timeGrid,'timeUOM':timeUom, 'info_ids':[], 'values':None}
      startIndex = indexConcTable
      nuclideName = ""
      while not nuclideName.strip().startswith('subtotals'):
        startIndex+=1
        components = self.lines[startIndex].split()
        nuclideName   = ""
        if len(components) > 0:
          components = self.lines[startIndex].split()
          nuclideName = components[0].strip()
          if nuclideName == 'totals':
            nuclideName = "subtotals_" + isotopeType.replace(" ","_")
          if "---" not in nuclideName:
            outputDict['info_ids'].append(nuclideName+"_"+uom.strip())
            values.append( [float(elm) if 'E' in elm else float(elm[:elm.index("-" if "-" in elm else "+")] + "E"+elm[elm.index("-" if "-" in elm else "+"):]) for elm in  components[2:] ])
            if nuclideName.startswith("subtotals"):
              if len(totals) > 0:
                totals = [subtot + values[-1][cnt] for cnt, subtot in enumerate(totals)]
              else:
                totals = values[-1]
    if len(totals)>0:
      values.append(totals)
      outputDict['info_ids'].append('totals'+"_"+uom.strip())

    outputDict['values'] = np.atleast_2d(values).T
    return outputDict


  def retrieveKeff(self):
    """
      Retrieve Summary Keff info
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),'info_ids':list(of ids of data),'values':np.ndarray( ntime,nids )}
    """
    indicesKeff = np.asarray([i for i, x in enumerate(self.lines) if x.strip().startswith("k-eff =")])
    indicesKinf = np.asarray([i for i, x in enumerate(self.lines) if x.strip().startswith("Infinite neutron multiplication")])
    if len(indicesKeff) == 0 and len(indicesKinf) == 0:
      return None
    outputDict = {'time':[],'info_ids':[], 'values':None}
    values = []
    if len(indicesKeff) > 0:
      outputDict['info_ids'] = ['keff','iter_number','keff_delta','max_flux_delta',"kinf","kinf_epsilon","kinf_p","kinf_f","kinf_eta"]
      for cnt, index in enumerate(indicesKeff):
        values.append([])
        #keff
        components = self.lines[index].split()
        time = float(components[4][:-1])
        outputDict['time'].append(time)
        values[cnt].append(float(components[2]))
        comps = self.lines[index-1].split()
        values[cnt].extend([int(comps[0]),float(comps[2]),float(comps[3])])
        # kinf
        kinf_index = indicesKinf[cnt]
        values[cnt].append(float(self.lines[kinf_index].split()[-1]))
        values[cnt].append(float(self.lines[kinf_index-2].split()[-1]))
        values[cnt].append(float(self.lines[kinf_index-3].split()[-1]))
        values[cnt].append(float(self.lines[kinf_index-4].split()[-1]))
        values[cnt].append(float(self.lines[kinf_index-5].split()[-1]))
      outputDict['values'] = np.atleast_2d(values).T
    return outputDict

  def retrieveMixtureInfo(self):
    """
      Retrieve Summary of Triton Mixture Info
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),'info_ids':list(of ids of data),'values':np.ndarray( ntime,nids )}
    """
    indicesMatPowers = np.asarray([i+1 for i, x in enumerate(self.lines) if x.strip().startswith("--- Material powers for depletion pass no")])
    if len(indicesMatPowers) == 0:
      return None
    outputDict = {'time':[],'info_ids':[], 'values':None}
    values = []
    outputDict['info_ids'] = ['bu']
    for cnt, index in enumerate(indicesMatPowers):
      values.append([])
      #bu and powers
      components = self.lines[index].split(",")
      time = float(components[0].split()[2])
      outputDict['time'].append(time)
      bu   = float(components[1].split()[2])
      values[cnt].append(bu)
      startIndex = index + 4
      mixName = ""
      while mixName.lower() != "total":
        startIndex+=1
        components = self.lines[startIndex].split()
        mixName = components[0].strip()
        if mixName.lower() != "total":
          if "tot_power_mix_"+mixName.strip() not in outputDict['info_ids']:
            outputDict['info_ids'].extend( ["tot_power_mix_"+mixName.strip(),"fract_power_mix_"+mixName.strip(),"th_flux_mix_"+mixName.strip(),"tot_flux_mix_"+mixName.strip()]  )
          values[cnt].extend( [float(components[1]),float(components[2]),float(components[4]),float(components[5])] )
    outputDict['values'] = np.atleast_2d(values).T
    return outputDict

  def retrieveNuclideConcentrations(self):
    """
      Retrieve Summary of Triton Nuclide Concentration
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),'info_ids':list(of ids of data),'values':np.ndarray( ntime,nids )}
    """
    outputDict = None
    try:
      indexConc = next(i for i,x in enumerate(self.lines) if x.strip().startswith("| nuclide concentrations"))
    except StopIteration:
      return outputDict

    # retrieve time grid
    uomTime = self.lines[indexConc+1].split("|")[-1].split()[-1].strip()
    timeGrid = [float(elm.replace(uomTime[0],"")) for elm in self.lines[indexConc+2].split("|")[-1].split()]
    values = []
    outputDict = {'time':timeGrid,'info_ids':[], 'values':None}
    startIndex = indexConc + 3
    nuclideName = ""
    while nuclideName.lower() != "total":
      startIndex+=1
      components = self.lines[startIndex].split()
      nuclideName = components[0].strip()
      outputDict['info_ids'].append(nuclideName+"_conc")
      values.append( [float(elm) for elm in  components[2:] ])
    outputDict['values'] = np.atleast_2d(values)
    return outputDict

  def writeCSV(self, fileout):
    """
      Print Data into CSV format
      @ In, fileout, str, the output file name
      @ Out, None
    """
    fileObject = open(fileout.strip()+".csv", mode='w+') if not fileout.endswith('csv') else open(fileout.strip(), mode='w+')
    headers = ['time']
    timeGrid = None
    nParams = np.sum([len(data['info_ids']) for data in self.data.values() if data is not None])+1

    for dataId, data in self.data.items():
      if data is not None:
        startIndex = len(headers)
        headers.extend(data['info_ids'])
        endIndex = len(headers)
        if timeGrid is None:
          timeGrid = data['time']
          # construct Matrix
          outputMatrix = np.zeros( (nParams, len(timeGrid)) )
          outputMatrix[0,:] = timeGrid[:]
        outputMatrix[startIndex:endIndex,:] = data['values'][:,:]
    # print the csv
    np.savetxt(fileObject, outputMatrix.T, delimiter=',', header=','.join(headers), comments='')


if __name__ == '__main__':

  test = scaleData("~/Downloads/decay.out")
  test.writeCSV("figa.csv")
