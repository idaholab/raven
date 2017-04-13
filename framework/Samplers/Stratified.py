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
  This module contains the Stratified sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
from operator import mul
from functools import reduce
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Grid import Grid
from .Sampler import Sampler
from utils import utils
import Distributions
#Internal Modules End--------------------------------------------------------------------------------


class Stratified(Grid):
  """
    Stratified sampler, also known as Latin Hypercube Sampling (LHS). Currently no
    special filling methods are implemented
  """
  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    Grid.__init__(self)
    self.sampledCoordinate    = [] # a list of list for i=0,..,limit a list of the coordinate to be used this is needed for the LHS
    self.printTag = 'SAMPLER Stratified'
    self.globalGrid          = {}    # Dictionary for the globalGrid. These grids are used only for Stratified for ND distributions.

  def localInputAndChecks(self,xmlNode):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ Out, None
    """
    Sampler.readSamplerInit(self,xmlNode)
    Grid.localInputAndChecks(self,xmlNode)
    pointByVar  = [len(self.gridEntity.returnParameter("gridInfo")[variable][2]) for variable in self.gridInfo.keys()]
    if len(set(pointByVar))!=1: self.raiseAnError(IOError,'the latin Hyper Cube requires the same number of point in each dimension')
    self.pointByVar         = pointByVar[0]
    self.inputInfo['upper'] = {}
    self.inputInfo['lower'] = {}

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    Grid.localInitialize(self)
    self.limit = (self.pointByVar-1)
    # For the multivariate normal distribtuion, if the user generates the grids on the transformed space, the user needs to provide the grid for each variables, no globalGrid is needed
    if self.variablesTransformationDict:
      tempFillingCheck = [[None]*(self.pointByVar-1)]*len(self.gridEntity.returnParameter("dimensionNames")) #for all variables
      self.sampledCoordinate = [[None]*len(self.axisName)]*(self.pointByVar-1)
      for i in range(len(tempFillingCheck)): tempFillingCheck[i]  = Distributions.randomPermutation(list(range(self.pointByVar-1)),self) #pick a random interval sequence
      mappingIdVarName = {}
      for cnt, varName in enumerate(self.axisName):
        mappingIdVarName[varName] = cnt
    # For the multivariate normal, if the user wants to generate the grids based on joint distribution, the user needs to provide the globalGrid for all corresponding variables
    else:
      globGridsCount = {}
      dimInfo = self.gridEntity.returnParameter("dimInfo")
      for val in dimInfo.values():
        if val[-1] is not None and val[-1] not in globGridsCount.keys(): globGridsCount[val[-1]] = 0
        globGridsCount[val[-1]] += 1
      diff = -sum(globGridsCount.values())+len(globGridsCount.keys())
      tempFillingCheck = [[None]*(self.pointByVar-1)]*(len(self.gridEntity.returnParameter("dimensionNames"))+diff) #for all variables
      self.sampledCoordinate = [[None]*len(self.axisName)]*(self.pointByVar-1)
      for i in range(len(tempFillingCheck)): tempFillingCheck[i]  = Distributions.randomPermutation(list(range(self.pointByVar-1)),self) #pick a random interval sequence
      cnt = 0
      mappingIdVarName = {}
      for varName in self.axisName:
        if varName not in dimInfo.keys(): mappingIdVarName[varName] = cnt
        else:
          for addKey,value in dimInfo.items():
            if value[1] == dimInfo[varName][1] and addKey not in mappingIdVarName.keys(): mappingIdVarName[addKey] = cnt
        if len(mappingIdVarName.keys()) == len(self.axisName): break
        cnt +=1

    for nPoint in range(self.pointByVar-1): self.sampledCoordinate[nPoint]= [tempFillingCheck[mappingIdVarName[varName]][nPoint] for varName in self.axisName]

  def localGenerateInput(self,model,myInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    varCount = 0
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    weight = 1.0
    for varName in self.axisName:
      # new implementation for ND LHS
      if not "<distribution>" in varName:
        if self.variables2distributionsMapping[varName]['totDim']>1 and self.variables2distributionsMapping[varName]['reducedDim'] == 1:    # to avoid double count of weight for ND distribution; I need to count only one variable instaed of N
          if self.variablesTransformationDict:
            distName = self.variables2distributionsMapping[varName]['name']
            for distVarName in self.distributions2variablesMapping[distName]:
              for kkey in utils.first(distVarName.keys()).strip().split(','):
                self.inputInfo['distributionName'][kkey] = self.toBeSampled[varName]
                self.inputInfo['distributionType'][kkey] = self.distDict[varName].type
            ndCoordinate = np.zeros(len(self.distributions2variablesMapping[distName]))
            dxs = np.zeros(len(self.distributions2variablesMapping[distName]))
            centerCoordinate = np.zeros(len(self.distributions2variablesMapping[distName]))
            positionList = self.distributions2variablesIndexList[distName]
            for var in self.distributions2variablesMapping[distName]:
              # if the varName is a comma separated list of strings the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
              variable = utils.first(var.keys()).strip()
              position = utils.first(var.values())
              upper = self.gridEntity.returnShiftedCoordinate(self.gridEntity.returnIteratorIndexes(),{variable:self.sampledCoordinate[self.counter-1][varCount]+1})[variable]
              lower = self.gridEntity.returnShiftedCoordinate(self.gridEntity.returnIteratorIndexes(),{variable:self.sampledCoordinate[self.counter-1][varCount]})[variable]
              varCount += 1
              if self.gridInfo[variable] == 'CDF':
                coordinate = lower + (upper-lower)*Distributions.random()
                ndCoordinate[positionList.index(position)] = self.distDict[variable].inverseMarginalDistribution(coordinate,variable)
                dxs[positionList.index(position)] = self.distDict[variable].inverseMarginalDistribution(max(upper,lower),variable)-self.distDict[variable].inverseMarginalDistribution(min(upper,lower),variable)
                centerCoordinate[positionList.index(position)] = (self.distDict[variable].inverseMarginalDistribution(upper,variable)+self.distDict[variable].inverseMarginalDistribution(lower,variable))/2.0
                for kkey in variable.strip().split(','):
                  self.values[kkey] = ndCoordinate[positionList.index(position)]
                  self.inputInfo['upper'][kkey] = self.distDict[variable].inverseMarginalDistribution(max(upper,lower),variable)
                  self.inputInfo['lower'][kkey] = self.distDict[variable].inverseMarginalDistribution(min(upper,lower),variable)
              elif self.gridInfo[variable] == 'value':
                dxs[positionList.index(position)] = max(upper,lower) - min(upper,lower)
                centerCoordinate[positionList.index(position)] = (upper + lower)/2.0
                coordinateCdf = self.distDict[variable].marginalCdf(lower) + (self.distDict[variable].marginalCdf(upper) - self.distDict[variable].marginalCdf(lower))*Distributions.random()
                coordinate = self.distDict[variable].inverseMarginalDistribution(coordinateCdf,variable)
                ndCoordinate[positionList.index(position)] = coordinate
                for kkey in variable.strip().split(','):
                  self.values[kkey] = coordinate
                  self.inputInfo['upper'][kkey] = max(upper,lower)
                  self.inputInfo['lower'][kkey] = min(upper,lower)
            self.inputInfo['ProbabilityWeight-'+varName.replace(",","!")] = self.distDict[varName].cellIntegral(centerCoordinate,dxs)
            weight *= self.inputInfo['ProbabilityWeight-'+varName.replace(",","!")]
            self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(ndCoordinate)
          else:
            if self.gridInfo[varName] == 'CDF':
              upper = self.gridEntity.returnShiftedCoordinate(self.gridEntity.returnIteratorIndexes(),{varName:self.sampledCoordinate[self.counter-1][varCount]+1})[varName]
              lower = self.gridEntity.returnShiftedCoordinate(self.gridEntity.returnIteratorIndexes(),{varName:self.sampledCoordinate[self.counter-1][varCount]})[varName]
              varCount += 1
              coordinate = lower + (upper-lower)*Distributions.random()
              gridCoordinate, distName =  self.distDict[varName].ppf(coordinate), self.variables2distributionsMapping[varName]['name']
              for distVarName in self.distributions2variablesMapping[distName]:
                for kkey in utils.first(distVarName.keys()).strip().split(','):
                  self.inputInfo['distributionName'][kkey], self.inputInfo['distributionType'][kkey], self.values[kkey] = self.toBeSampled[varName], self.distDict[varName].type, np.atleast_1d(gridCoordinate)[distVarName.values()[0]-1]
              # coordinate stores the cdf values, we need to compute the pdf for SampledVarsPb
              self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(np.atleast_1d(gridCoordinate).tolist())
              weight *= max(upper,lower) - min(upper,lower)
              self.inputInfo['ProbabilityWeight-'+varName.replace(",","!")] = max(upper,lower) - min(upper,lower)
            else: self.raiseAnError(IOError,"Since the globalGrid is defined, the Stratified Sampler is only working when the sampling is performed on a grid on a CDF. However, the user specifies the grid on " + self.gridInfo[varName])
      if ("<distribution>" in varName) or self.variables2distributionsMapping[varName]['totDim']==1:   # 1D variable
        # if the varName is a comma separated list of strings the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
        upper = self.gridEntity.returnShiftedCoordinate(self.gridEntity.returnIteratorIndexes(),{varName:self.sampledCoordinate[self.counter-1][varCount]+1})[varName]
        lower = self.gridEntity.returnShiftedCoordinate(self.gridEntity.returnIteratorIndexes(),{varName:self.sampledCoordinate[self.counter-1][varCount]})[varName]
        varCount += 1
        if self.gridInfo[varName] =='CDF':
          coordinate = lower + (upper-lower)*Distributions.random()
          ppfValue = self.distDict[varName].ppf(coordinate)
          ppfLower = self.distDict[varName].ppf(min(upper,lower))
          ppfUpper = self.distDict[varName].ppf(max(upper,lower))
          weight *= self.distDict[varName].cdf(ppfUpper) - self.distDict[varName].cdf(ppfLower)
          self.inputInfo['ProbabilityWeight-'+varName.replace(",","-")] = self.distDict[varName].cdf(ppfUpper) - self.distDict[varName].cdf(ppfLower)
          self.inputInfo['SampledVarsPb'][varName]  = self.distDict[varName].pdf(ppfValue)
        elif self.gridInfo[varName] == 'value':
          coordinateCdf = self.distDict[varName].cdf(min(upper,lower)) + (self.distDict[varName].cdf(max(upper,lower))-self.distDict[varName].cdf(min(upper,lower)))*Distributions.random()
          if coordinateCdf == 0.0: self.raiseAWarning(IOError,"The grid lower bound and upper bound in value will generate ZERO cdf value!!!")
          coordinate = self.distDict[varName].ppf(coordinateCdf)
          weight *= self.distDict[varName].cdf(max(upper,lower)) - self.distDict[varName].cdf(min(upper,lower))
          self.inputInfo['ProbabilityWeight-'+varName.replace(",","-")] = self.distDict[varName].cdf(max(upper,lower)) - self.distDict[varName].cdf(min(upper,lower))
          self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(coordinate)
        for kkey in varName.strip().split(','):
          self.inputInfo['distributionName'][kkey] = self.toBeSampled[varName]
          self.inputInfo['distributionType'][kkey] = self.distDict[varName].type
          if self.gridInfo[varName] =='CDF':
            self.values[kkey] = ppfValue
            self.inputInfo['upper'][kkey] = ppfUpper
            self.inputInfo['lower'][kkey] = ppfLower
          elif self.gridInfo[varName] =='value':
            self.values[kkey] = coordinate
            self.inputInfo['upper'][kkey] = max(upper,lower)
            self.inputInfo['lower'][kkey] = min(upper,lower)

    self.inputInfo['PointProbability'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight' ] = weight
    self.inputInfo['SamplerType'] = 'Stratified'
