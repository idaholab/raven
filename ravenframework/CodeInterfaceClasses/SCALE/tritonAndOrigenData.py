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
Created on March 25, 2018

@author: alfoa
"""
import numpy as np
import os
import copy
import re

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
    targetMultiplier, inputMultiplier = 1.0,1.0
    for uom in uoms[0:targetUOMIndex+1]:
      targetMultiplier*= scaling[uom]
    for uom in uoms[0:inputUOMIndex+1]:
      inputMultiplier*= scaling[uom]
    multiplier = inputMultiplier/targetMultiplier
  return multiplier

class origenAndTritonData:
  """
    Class that parses output of scale output (now Triton and/or Orgin only) and write a RAVEN compatible CSV
  """
  def __init__(self,filen, timeUOM='s', outputType="triton"):
    """
      Constructor
      @ In, filen, string or dict, file name to be parsed or dictionary {'origen':filein1,'triton':filein2}
      @ In, timeUOM, string, time uom (e.g. s, m, d, h, y)
      @ In, outputType, string, output type (i.e. origen, triton or combined). If combined, the triton output (last step) is used as exported as initial condition of the origen one
      @ Out, None
    """
    if not isinstance(filen,dict):
      filenames = [filen]
      outTypeDict = {filen:outputType}
    else:
      filenames = [filen[outputType]] if outputType != 'combined' else filen.values()
      outTypeDict = dict(zip(filen.values(), filen.keys()))

    # retrieve keff and kinf
    self.data = {}
    for outFile in filenames:
      self.lines = open(os.path.abspath(os.path.expanduser(outFile)),"r").readlines()
      if outTypeDict[outFile] == 'triton':
        self.data['transportInfo'] = self.retrieveKeff()
        if self.data['transportInfo'] is not None:
          if not self.data['transportInfo']['timeUOM'].strip().startswith(timeUOM):
            self.data['transportInfo']['time'] = (_scalingFactorBetweenTimeUOM(timeUOM, self.data['transportInfo']['timeUOM'])*np.asarray(self.data['transportInfo']['time'])).tolist()
        # retrieve nuclide densities
        self.data['nuclideDensities'] = self.retrieveNuclideConcentrations()
        if self.data['nuclideDensities'] is not None:
          if not self.data['nuclideDensities']['timeUOM'].strip().startswith(timeUOM):
            self.data['nuclideDensities']['time'] = (_scalingFactorBetweenTimeUOM(timeUOM, self.data['nuclideDensities']['timeUOM'])*np.asarray(self.data['nuclideDensities']['time'])).tolist()
        # retrieve mixture powers
        self.data['mixPowers'] = self.retrieveMixtureInfo()
        if self.data['mixPowers'] is not None:
          if not self.data['mixPowers']['timeUOM'].strip().startswith(timeUOM):
            self.data['mixPowers']['time'] = (_scalingFactorBetweenTimeUOM(timeUOM, self.data['mixPowers']['timeUOM'])*np.asarray(self.data['mixPowers']['time'])).tolist()
      elif outTypeDict[outFile] == 'origen':
        # origen history overview and concentration tables
        self.data['histOverviewOrigen'] = self.retrieveHistoryOverview()
        if self.data['histOverviewOrigen'] is not None:
          self.data['concTablesOrigen'] = self.retrieveConcentrationTables()
          if not self.data['concTablesOrigen']['timeUOM'].strip().startswith(timeUOM):
            self.data['concTablesOrigen']['time'] = (_scalingFactorBetweenTimeUOM(timeUOM, self.data['concTablesOrigen']['timeUOM'])*np.asarray(self.data['concTablesOrigen']['time'])).tolist()
          if not self.data['histOverviewOrigen']['timeUOM'].strip().startswith(timeUOM):
            self.data['histOverviewOrigen']['time'] = (_scalingFactorBetweenTimeUOM(timeUOM, self.data['histOverviewOrigen']['timeUOM'])*np.asarray(self.data['histOverviewOrigen']['time'])).tolist()
    # check if something has been found
    if all(v is None for v in self.data.values()):
      raise IOError("No readable outputs have been found!")
    if outputType == 'combined':
      # if origen, histOverviewOrigen is always present
      timeGrid = self.data['histOverviewOrigen']['time']
      if self.data['transportInfo'] is not None:
        self.data['transportInfo']['time'] = timeGrid
        self.data['transportInfo']['values'] = np.array([self.data['transportInfo']['values'][:,-1],]*len(timeGrid)).T
      if self.data['nuclideDensities'] is not None:
        self.data['nuclideDensities']['time'] = timeGrid
        self.data['nuclideDensities']['values'] = np.array([self.data['nuclideDensities']['values'][:,-1],]*len(timeGrid)).T
      if self.data['mixPowers'] is not None:
        self.data['mixPowers']['time'] = timeGrid
        self.data['mixPowers']['values'] = np.array([self.data['mixPowers']['values'][:,-1],]*len(timeGrid)).T
    # check time grid consistency, if not, interpolate by nearest neighbor
    self.dataConsistency()

  def dataConsistency(self):
    """
      Check the info present in self.data
      If the time grids do not correspond, nearest neighbor interpolation
      @ In, None
      @ Out, None
    """
    from sklearn import neighbors
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

  def retrieveHistoryOverview(self):
    """
      Retrieve History Overview (Origen)
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),
                                'timeUOM':string - time units,
                                'info_ids':list(of ids of data),
                                'values':np.ndarray( ntime,nids )}
    """
    # history overview
    outputDict = None

    indexHistOverview = [i+3 for i, x in enumerate(self.lines) if x.strip().startswith('=   History overview for')]
    if len(indexHistOverview) == 0:
      return outputDict
    elif len(indexHistOverview) > 1:
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
                               {'time':list(of time values),
                                'timeUOM':string - time units,
                                'info_ids':list(of ids of data),
                                'values':np.ndarray( ntime,nids )}
    """
    # Concentration tables
    outputDict = None
    try:
      indexHistOverview = [i for i, x in enumerate(self.lines) if x.strip().startswith('=   Nuclide concentrations in')]
    except StopIteration:
      return outputDict
    values = []
    totals = {}
    for cnt, indexConcTable in enumerate(indexHistOverview):
      splitted = self.lines[indexConcTable].split(",")
      if len(splitted) > 1:
        uom = splitted[0].split()[-1].strip()
        isotopeType = splitted[-1].split("for case")[0].strip()
      else:
        uom = splitted[0].split("for case")[0].split()[-1].strip()
        isotopeType = None
      if uom not in totals:
        totals[uom] = []
      indexConcTable+=4
      timeUom = re.split(r'(\d+)', self.lines[indexConcTable].split()[0])[-1].strip()
      timeGrid = [float(elm.replace(timeUom.strip(),"")) for elm in  self.lines[indexConcTable].split()]

      if outputDict is None:
        outputDict = {'time':timeGrid,'timeUOM':timeUom, 'info_ids':[], 'values':None}
      startIndex = indexConcTable
      nuclideName = ""
      while not (nuclideName.strip().startswith('subtotals') or nuclideName.strip().startswith('totals')):
        startIndex+=1
        components = self.lines[startIndex].split()
        nuclideName   = ""
        if len(components) > 0:
          components = self.lines[startIndex].split()
          nuclideName = components[0].strip()
          if nuclideName == 'totals' and isotopeType is not None:
            nuclideName = "subtotals_" + isotopeType.replace(" ","_")
          if "---" not in nuclideName:
            outputDict['info_ids'].append(nuclideName+"_"+uom.strip())
            values.append( [float(elm) if 'E' in elm else float(elm[:elm.index("-" if "-" in elm else "+")] + "E"+elm[elm.index("-" if "-" in elm else "+"):]) for elm in  components[1:] ])
            if nuclideName.startswith("subtotals"):
              if len(totals[uom]) > 0:
                totals[uom] = [subtot + values[-1][cnt] for cnt, subtot in enumerate(totals[uom])]
              else:
                totals[uom] = values[-1]
    for uom in totals:
      if len(totals[uom])>0 and isotopeType is not None:
        values.append(totals[uom])
        outputDict['info_ids'].append('totals'+"_"+uom.strip())

    outputDict['values'] = np.atleast_2d(values)
    return outputDict


  def retrieveKeff(self):
    """
      Retrieve Summary Keff info
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),
                                'timeUOM':string - time units,
                                'info_ids':list(of ids of data),
                                'values':np.ndarray( ntime,nids )}
    """
    indicesKeff = np.asarray([i for i, x in enumerate(self.lines) if x.strip().startswith("k-eff =")])
    indicesKinf = np.asarray([i for i, x in enumerate(self.lines) if x.strip().startswith("Infinite neutron multiplication")])
    if len(indicesKeff) == 0 and len(indicesKinf) == 0:
      return None
    outputDict = {'time':[],'info_ids':[],'timeUOM':None, 'values':None}
    values = []
    if len(indicesKeff) > 0:
      outputDict['info_ids'] = ['keff','iter_number','keff_delta','max_flux_delta',"kinf","kinf_epsilon","kinf_p","kinf_f","kinf_eta"]
      for cnt, index in enumerate(indicesKeff):
        values.append([])
        #keff
        components = self.lines[index].split()
        time = float(components[4][:-1])
        if outputDict['timeUOM'] is None:
          outputDict['timeUOM'] = components[4][-1]
        outputDict['time'].append(time)
        values[cnt].append(float(components[2]))
        comps = self.lines[index-1].split()
        values[cnt].extend([int(comps[0]),float(comps[2]),float(comps[3])])
        # kinf
        kinfIndex = indicesKinf[cnt]
        values[cnt].append(float(self.lines[kinfIndex].split()[-1]))
        values[cnt].append(float(self.lines[kinfIndex-2].split()[-1]))
        values[cnt].append(float(self.lines[kinfIndex-3].split()[-1]))
        values[cnt].append(float(self.lines[kinfIndex-4].split()[-1]))
        values[cnt].append(float(self.lines[kinfIndex-5].split()[-1]))
      outputDict['values'] = np.atleast_2d(values).T
    return outputDict

  def retrieveMixtureInfo(self):
    """
      Retrieve Summary of Triton Mixture Info
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                               {'time':list(of time values),
                                'timeUOM':string - time units,
                                'info_ids':list(of ids of data),
                                'values':np.ndarray( ntime,nids )}
    """
    indicesMatPowers = np.asarray([i+1 for i, x in enumerate(self.lines) if x.strip().startswith("--- Material powers for depletion pass no")])
    if len(indicesMatPowers) == 0:
      return None
    outputDict = {'time':[],'info_ids':[],'timeUOM':None, 'values':None}
    values = []
    outputDict['info_ids'] = ['bu']
    for cnt, index in enumerate(indicesMatPowers):
      values.append([])
      #bu and powers
      components = self.lines[index].split(",")
      time = float(components[0].split()[2])
      outputDict['time'].append(time)
      if outputDict['timeUOM'] is None:
        outputDict['timeUOM'] = components[0].split()[3]
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
                               {'time':list(of time values),
                                'timeUOM':string - time units,
                                'info_ids':list(of ids of data),
                                'values':np.ndarray( ntime,nids )}
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
    outputDict = {'time':timeGrid,'info_ids':[], 'timeUOM':uomTime, 'values':None}
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
    fileObject = open(fileout.strip()+".csv", mode='wb+') if not fileout.endswith('csv') else open(fileout.strip(), mode='wb+')
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
    fileObject.close()

