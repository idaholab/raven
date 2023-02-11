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
  This module contains the Grid sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
# for future compatibility with Python 3------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
# End compatibility block for Python 3--------------------------------------------------------------

# External Modules----------------------------------------------------------------------------------
import sys
import copy
from operator import mul
from functools import reduce
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .Sampler import Sampler
from ..utils import utils
from ..utils import InputData, InputTypes
from .. import GridEntities
# Internal Modules End------------------------------------------------------------------------------

class Grid(Sampler):
  """
    Samples the model on a given (by input) set of points
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Grid, cls).getInputSpecification()
    # grid input
    gridInput = InputData.parameterInputFactory("grid", contentType=InputTypes.StringType)
    gridInput.addParam("type", InputTypes.StringType)
    gridInput.addParam("construction", InputTypes.StringType)
    gridInput.addParam("steps", InputTypes.IntegerType)
    # old outer distribution input
    oldSubOutDist =  inputSpecification.popSub("Distribution")
    newOuterDistributionInput = InputData.parameterInputFactory("Distribution", baseNode=oldSubOutDist)
    # old variable input
    oldSub = inputSpecification.popSub("variable")
    newVariableInput = InputData.parameterInputFactory("variable", baseNode=oldSub)
    # update variable input with new grid input
    newVariableInput.addSub(gridInput)
    inputSpecification.addSub(newVariableInput)
    # update outer distribution input with new grid input
    newOuterDistributionInput.addSub(gridInput)
    inputSpecification.addSub(newOuterDistributionInput)

    return inputSpecification

  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    super().__init__()
    self.printTag = 'SAMPLER GRID'
    self.axisName = []                 # the name of each axis (variable)
    self.gridInfo = {}                 # {'name of the variable':Type}  --> Type: CDF/Value
    self.externalgGridCoord = False    # boolean attribute. True if the coordinate list has been filled by external source (see factorial sampler)
    self.gridCoordinate = []           # current grid coordinates
    self.gridEntity = None

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    self.gridEntity = GridEntities.factory.returnInstance('GridEntity')
    #TODO remove using xmlNode
    if 'limit' in paramInput.parameterValues:
      self.raiseAnError(IOError,'limit is not used in Grid sampler')
    self.limit = 1
    # FIXME: THIS READ MORE XML MUST BE CONVERTED IN THE INPUTPARAMETER COLLECTOR!!!!!!!
    self.gridEntity._readMoreXml(xmlNode,dimensionTags=["variable", "Distribution"], dimTagsPrefix={"Distribution": "<distribution>"})
    grdInfo = self.gridEntity.returnParameter("gridInfo")
    for axis, value in grdInfo.items():
      self.gridInfo[axis] = value[0]
    if len(self.toBeSampled.keys()) != len(grdInfo.keys()):
      self.raiseAnError(IOError, 'inconsistency between number of variables and grid specification')
    self.axisName = list(grdInfo.keys())
    self.axisName.sort()
    # check that grid in CDF contains values in the [0,1] interval
    for key in grdInfo:
      if grdInfo[key][0] == 'CDF':
        valueArrays = grdInfo[key][2]
        if min(valueArrays)<0.0 or max(valueArrays)>1.0:
          self.raiseAnError(IOError, ("Grid sampler " + str(self.name) + ": Grid associated with variable " + str(key) + " is outside the [0,1] interval"))

  def localGetInitParams(self):
    """
      Appends a given dictionary with class specific member variables and their
      associated initialized values.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for variable,value in self.gridInfo.items():
      paramDict[f'{variable} is sampled using a grid in '] = value

    return paramDict

  def localGetCurrentSetting(self):
    """
      Appends a given dictionary with class specific information regarding the
      current status of the object.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for var, value in self.values.items():
      paramDict[f'coordinate {var} has value'] = value

    return paramDict

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    self.gridEntity.initialize()
    self.limit = self.gridEntity.len()

  def localGenerateInput(self, model, oldInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    self.inputInfo['distributionName'] = {} # Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} # Used to determine which distribution type is used
    weight = 1.0
    recastDict = {}
    for i in range(len(self.axisName)):
      varName = self.axisName[i]
      if self.gridInfo[varName] == 'CDF':
        if self.distDict[varName].getDimensionality() == 1:
          recastDict[varName] = [self.distDict[varName].ppf]
        else:
          recastDict[varName] = [self.distDict[varName].inverseMarginalDistribution,[self.variables2distributionsMapping[varName]['dim']-1]]
      elif self.gridInfo[varName] == 'value':
        gridLB = self.gridEntity.gridInitDict['lowerBounds'][varName]
        gridUB = self.gridEntity.gridInitDict['upperBounds'][varName]
        if self.variables2distributionsMapping[varName]['totDim'] == 1:
          distLB = self.distDict[varName].lowerBound
          distUB = self.distDict[varName].upperBound
        else:
          dim = self.variables2distributionsMapping[varName]['dim'] - 1
          distLB = self.distDict[varName].lowerBound[dim]
          distUB = self.distDict[varName].upperBound[dim]
        if gridLB < distLB or gridUB > distUB:
          self.raiseAnError(IOError, (f'Grids defined for "{varName}" in range ({gridLB}, {gridUB}) are outside the range' +\
                                      f'of the given distribution "{self.distDict[varName].type}" ({distLB}, {distUB})'))
      else:
        self.raiseAnError(IOError, f'{self.gridInfo[varName]} is not known as value keyword for type. Sampler: {self.name}')

    if self.externalgGridCoord:
      currentIndexes = self.gridEntity.returnIteratorIndexesFromIndex(self.gridCoordinate)
      coordinates = self.gridEntity.returnCoordinateFromIndex(self.gridCoordinate, True, recastDict)
    else:
      currentIndexes = self.gridEntity.returnIteratorIndexes()
      coordinates = self.gridEntity.returnPointAndAdvanceIterator(True,recastDict)
    if coordinates is None:
      self.raiseADebug('Grid finished with restart points!  Moving on...')
      raise utils.NoMoreSamplesNeeded
    coordinatesPlusOne  = self.gridEntity.returnShiftedCoordinate(currentIndexes,dict.fromkeys(self.axisName,1))
    coordinatesMinusOne = self.gridEntity.returnShiftedCoordinate(currentIndexes,dict.fromkeys(self.axisName,-1))
    for i in range(len(self.axisName)):
      varName = self.axisName[i]
      # compute the SampledVarsPb for 1-D distribution
      if ("<distribution>" in varName) or (self.variables2distributionsMapping[varName]['totDim'] == 1):
        for key in varName.strip().split(','):
          self.inputInfo['distributionName'][key] = self.toBeSampled[varName]
          self.inputInfo['distributionType'][key] = self.distDict[varName].type
          self.values[key] = coordinates[varName]
          self.inputInfo['SampledVarsPb'][key] = self.distDict[varName].pdf(self.values[key])
      # compute the SampledVarsPb for N-D distribution
      else:
        if self.variables2distributionsMapping[varName]['reducedDim'] == 1:
          # to avoid double count;
          distName = self.variables2distributionsMapping[varName]['name']
          ndCoordinate=[0]*len(self.distributions2variablesMapping[distName])
          positionList = self.distributions2variablesIndexList[distName]
          for var in self.distributions2variablesMapping[distName]:
            variable = utils.first(var.keys())
            position = utils.first(var.values())
            ndCoordinate[positionList.index(position)] = float(coordinates[variable.strip()])
            for key in variable.strip().split(','):
              self.inputInfo['distributionName'][key] = self.toBeSampled[variable]
              self.inputInfo['distributionType'][key] = self.distDict[variable].type
              self.values[key] = coordinates[variable]
          # Based on the discussion with Diego, we will use the following to compute SampledVarsPb.
          self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(ndCoordinate)
      # Compute the ProbabilityWeight
      if ("<distribution>" in varName) or (self.variables2distributionsMapping[varName]['totDim']==1):
        if self.distDict[varName].getDistType() == 'Discrete':
          gridWeight = self.distDict[varName].pdf(coordinates[varName])
        else:
          if self.gridInfo[varName]=='CDF':
            if coordinatesPlusOne[varName] != sys.maxsize and coordinatesMinusOne[varName] != -sys.maxsize:
              midPlusCDF   = (coordinatesPlusOne[varName]+self.distDict[varName].cdf(self.values[key]))/2.0
              midMinusCDF  = (coordinatesMinusOne[varName]+self.distDict[varName].cdf(self.values[key]))/2.0
            if coordinatesMinusOne[varName] == -sys.maxsize:
              midPlusCDF   = (coordinatesPlusOne[varName]+self.distDict[varName].cdf(self.values[key]))/2.0
              midMinusCDF  = 0.0
            if coordinatesPlusOne[varName] == sys.maxsize:
              midPlusCDF   = 1.0
              midMinusCDF  = (coordinatesMinusOne[varName]+self.distDict[varName].cdf(self.values[key]))/2.0
            gridWeight = midPlusCDF - midMinusCDF
          else:
            # Value
            if coordinatesPlusOne[varName] != sys.maxsize and coordinatesMinusOne[varName] != -sys.maxsize:
              midPlusValue   = (self.values[key]+coordinatesPlusOne[varName])/2.0
              midMinusValue  = (self.values[key]+coordinatesMinusOne[varName])/2.0
              gridWeight = self.distDict[varName].cdf(midPlusValue) - self.distDict[varName].cdf(midMinusValue)
            if coordinatesMinusOne[varName] == -sys.maxsize:
              midPlusValue = (self.values[key]+coordinatesPlusOne[varName])/2.0
              gridWeight = self.distDict[varName].cdf(midPlusValue) - 0.0
            if coordinatesPlusOne[varName] == sys.maxsize:
              midMinusValue  = (self.values[key]+coordinatesMinusOne[varName])/2.0
              gridWeight = 1.0 - self.distDict[varName].cdf(midMinusValue)
        self.inputInfo['ProbabilityWeight-'+varName] = gridWeight
        weight *= gridWeight
      # ND variable
      else:
        if self.variables2distributionsMapping[varName]['reducedDim'] == 1:
          # to avoid double count of weight for ND distribution; I need to count only one variable instaed of N
          distName = self.variables2distributionsMapping[varName]['name']
          ndCoordinate=np.zeros(len(self.distributions2variablesMapping[distName]))
          dxs=np.zeros(len(self.distributions2variablesMapping[distName]))
          positionList = self.distributions2variablesIndexList[distName]
          for var in self.distributions2variablesMapping[distName]:
            variable = utils.first(var.keys()).strip()
            position = utils.first(var.values())
            if self.gridInfo[variable]=='CDF':
              if coordinatesPlusOne[variable] != sys.maxsize and coordinatesMinusOne[variable] != -sys.maxsize:
                up   = self.distDict[variable].inverseMarginalDistribution(coordinatesPlusOne[variable] ,self.variables2distributionsMapping[variable]['dim']-1)
                down = self.distDict[variable].inverseMarginalDistribution(coordinatesMinusOne[variable],self.variables2distributionsMapping[variable]['dim']-1)
                dxs[positionList.index(position)] = (up - down)/2.0
                ndCoordinate[positionList.index(position)] = coordinates[variable] - (coordinates[variable] - down)/2.0 + dxs[positionList.index(position)]/2.0
              if coordinatesMinusOne[variable] == -sys.maxsize:
                up = self.distDict[variable].inverseMarginalDistribution(coordinatesPlusOne[variable] ,self.variables2distributionsMapping[variable]['dim']-1)
                dxs[positionList.index(position)] = (coordinates[variable.strip()]+up)/2.0 - self.distDict[varName].returnLowerBound(positionList.index(position))
                ndCoordinate[positionList.index(position)] = ((coordinates[variable.strip()]+up)/2.0 + self.distDict[varName].returnLowerBound(positionList.index(position)))/2.0
              if coordinatesPlusOne[variable] == sys.maxsize:
                down = self.distDict[variable].inverseMarginalDistribution(coordinatesMinusOne[variable],self.variables2distributionsMapping[variable]['dim']-1)
                dxs[positionList.index(position)] = self.distDict[varName].returnUpperBound(positionList.index(position)) - (coordinates[variable.strip()]+down)/2.0
                ndCoordinate[positionList.index(position)] = (self.distDict[varName].returnUpperBound(positionList.index(position)) + (coordinates[variable.strip()]+down)/2.0) /2.0
            else:
              if coordinatesPlusOne[variable] != sys.maxsize and coordinatesMinusOne[variable] != -sys.maxsize:
                dxs[positionList.index(position)] = (coordinatesPlusOne[variable] - coordinatesMinusOne[variable])/2.0
                ndCoordinate[positionList.index(position)] = coordinates[variable.strip()] - (coordinates[variable.strip()]-coordinatesMinusOne[variable])/2.0 + dxs[positionList.index(position)]/2.0
              if coordinatesMinusOne[variable] == -sys.maxsize:
                dxs[positionList.index(position)]          =  (coordinates[variable.strip()]+coordinatesPlusOne[variable])/2.0 - self.distDict[varName].returnLowerBound(positionList.index(position))
                ndCoordinate[positionList.index(position)] = ((coordinates[variable.strip()]+coordinatesPlusOne[variable])/2.0 + self.distDict[varName].returnLowerBound(positionList.index(position)))/2.0
              if coordinatesPlusOne[variable] == sys.maxsize:
                dxs[positionList.index(position)]          =  self.distDict[varName].returnUpperBound(positionList.index(position)) - (coordinates[variable.strip()]+coordinatesMinusOne[variable])/2.0
                ndCoordinate[positionList.index(position)] = (self.distDict[varName].returnUpperBound(positionList.index(position)) + (coordinates[variable.strip()]+coordinatesMinusOne[variable])/2.0) /2.0
          self.inputInfo['ProbabilityWeight-'+distName] = self.distDict[varName].cellIntegral(ndCoordinate,dxs)
          weight *= self.distDict[varName].cellIntegral(ndCoordinate,dxs)
    self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = copy.deepcopy(weight)
    self.inputInfo['SamplerType'] = 'Grid'

  def flush(self):
    """
      Reset Sampler attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    if self.gridEntity is not None:
      self.gridEntity.flush()
