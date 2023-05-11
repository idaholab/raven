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
Created on Mar 30, 2015

@author: alfoa
"""
import abc
import sys

# External Modules----------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .EntityFactoryBase import EntityFactory
from .utils.utils import UreturnPrintTag, partialEval, floatConversion, compare, metaclass_insert
from .BaseClasses import BaseEntity
from .utils import TreeStructure as ETS
from .utils.RAVENiterators import ravenArrayIterator
# Internal Modules End------------------------------------------------------------------------------

class GridBase(metaclass_insert(abc.ABCMeta, BaseEntity)):
  """
    Base Class that needs to be used when a new Grid class is generated
    It provides all the methods to create, modify, and handle a grid in the phase space.
  """
  @classmethod
  def __len__(cls):
    """
    Overload __len__ method.
    @ In, None
    @ Out, __len__, int, total number of nodes
    """
    return 0

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = UreturnPrintTag("GRID ENTITY")
    self.gridContainer = {} # dictionary that contains all the key feature of the grid
    self.gridInitDict = {}

  @classmethod
  def _readMoreXml(cls, xmlNode, dimensionTags=None, dimTagsPrefix=None):
    """
      XML reader for the grid statement.
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this class
      @ In, dimensionTag, list, optional, names of the tag that represents the grid dimensions
      @ In, dimTagsPrefix, dict, optional, eventual prefix to use for defining the dimName
      @ Out, None
    """

  @classmethod
  def initialize(cls, initDictionary=None):
    """
      Initialization method. The full grid is created in this method.
      @ In, initDictionary, dict, optional, dictionary of input arguments needed to create a Grid:
        {dimensionNames:[]}, required,list of axis names (dimensions' IDs)
        {lowerBounds:{}}, required, dictionary of lower bounds for each dimension
        {upperBounds:{}}, required, dictionary of upper bounds for each dimension
        {volumetriRatio:float or stepLength:dict}, required, p.u. volumetric ratio of the grid or dictionary of stepLengths ({'varName:list,etc'}
        {computeCells:bool},optional, boolean to ask to compute the cells ids and verteces coordinates, default = False
        {transformationMethods:{}}, optional, dictionary of methods to transform p.u. step size into a transformed system of coordinate
        !!!!!!
        if the self.gridInitDict is != None (info read from XML node), this method looks for the information in that dictionary first and after it checks the initDict object
        !!!!!!
      @ Out, None
    """

  @classmethod
  def retrieveCellIds(cls, listOfPoints):
    """
      This method is aimed to retrieve the cell IDs that are contained in certain boundaries provided as list of points
      @ In, listOfPoints, list, list of points that represent the boundaries ([listOfFirstBound, listOfSecondBound])
      @ Out, setOfCells, list, list of cells' ids
    """

  @classmethod
  def returnGridAsArrayOfCoordinates(cls):
    """
      Return the grid as an array of coordinates
      @ In, None
      @ Out, fullReshapedCoordinates, np.array, reshaped array
    """

  def returnParameter(self, parameterName):
    """
      Method to return one of the initialization parameters
      @ In, parameterName, string, name of the parameter to be returned
      @ Out, pointer, object, pointer to the requested parameter
    """
    if parameterName not in self.gridContainer:
      self.raiseAnError(Exception, f'parameter {parameterName} unknown among ones in GridEntity class.')
    pointer = self.gridContainer[parameterName]

    return pointer

  def updateParameter(self, parameterName, newValue, upContainer=True):
    """
      Method to update one of the initialization parameters
      @ In, parameterName, string, name of the parameter to be updated
      @ In, newValue, float, new value
      @ In, upContainer, bool, optional, True if gridContainer needs to be updated, else gridInit
      @ Out, None
    """
    if upContainer:
      self.gridContainer[parameterName] = newValue
    else:
      self.gridInitDict[parameterName ] = newValue

  def addCustomParameter(self, parameterName, value):
    """
      Method to add a new parameter in the Grid Entity
      @ In, parameterName, string, name of the parameter to be added
      @ In, value, float, new value
      @ Out, None
    """
    if parameterName in self.gridContainer:
      self.raiseAnError(Exception, f'parameter {parameterName} already present in GridEntity!')
    self.updateParameter(parameterName, value)

  @classmethod
  def resetIterator(cls):
    """
      Reset internal iterator
      @ In, None
      @ Out, None
    """

  @classmethod
  def returnIteratorIndexes(cls, returnDict=True):
    """
      Return the iterator indexes
      @ In, returnDict, bool, optional, returnDict if true, the Indexes are returned in dictionary format
      @ Out, returnDict, tuple or dict, the tuple or dictionary of the indexes
    """

  @classmethod
  def returnIteratorIndexesFromIndex(cls, listOfIndexes):
    """
      Return internal iterator indexes from list of coordinates in the list
      @ In, listOfIndexes, list, list of grid coordinates
      @ Out, returnDict, tuple or dict, the tuple or dictionary of the indexes
    """

  @classmethod
  def returnShiftedCoordinate(cls, coordinates, shiftingSteps):
    """
      Method to return the coordinate that is a # shiftingStep away from the input coordinate
      For example, if 1D grid= {'dimName':[1,2,3,4]}, coordinate is 3 and  shiftingStep is -2,
      the returned coordinate will be 1
      @ In,  coordinates, dict, dictionary of coordinates. {'dimName1':startingCoordinate1,dimName2:startingCoordinate2,...}
      @ In,  shiftingSteps, dict, dict of shifiting steps. {'dimName1':shiftingStep1,dimName2:shiftingStep2,...}
      @ Out, outputCoordinates, dict, dictionary of shifted coordinates' values {dimName:value1,...}
    """

  @classmethod
  def returnPointAndAdvanceIterator(cls, returnDict=False, recastMethods={}):
    """
      Method to return a point in the grid. This method will return the coordinates of the point to which the iterator is pointing
      In addition, it advances the iterator in order to point to the following coordinate
      @ In, returnDict, bool, optional, flag to request the output in dictionary format or not.
                               if True a dict ( {dimName1:
                                 coordinate1,dimName2:coordinate2,etc} is returned
                               if False a tuple is returned (coordinate1,coordinate2,etc
      @ In, recastMethods, dict, optional, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
      @ Out, coordinate, tuple, tuple containing the coordinates
    """

  @classmethod
  def returnCoordinateFromIndex(cls, multiDimIndex, returnDict=False, recastMethods={}):
    """
      Method to return a point in the grid. This method will return the coordinates of the point is requested by multiDimIndex
      In addition, it advances the iterator in order to point to the following coordinate
      @ In, multiDimIndex, tuple, tuple containing the Id of the point needs to be returned (e.g. 3 dim grid,  (xID,yID,zID))
      @ In, returnDict, bool, optional, flag to request the output in dictionary format or not.
                                         if True a dict ( {dimName1:
                                           coordinate1,dimName2:coordinate2,etc} is returned
                                         if False a tuple is riturned (coordinate1,coordinate2,etc)
      @ In, recastMethods, dict, optional, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
      @ Out, coordinate, tuple or dict, tuple containing the coordinates
    """

  def flush(self):
    """
      Reset GridBase attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """

class GridEntity(GridBase):
  """
    Class that defines a Grid in the phase space. This class should be used by all the Classes that need a Grid entity.
    It provides all the methods to create, modify, and handle a grid in the phase space.
  """
  @staticmethod
  def transformationMethodFromCustom(x):
    """
      Static method to create a transformationFunction from a set of points. Those points are going to be "transformed" in 0-1 space
      @ In, x, array-like, set of points
      @ Out, transformFunction, instance, instance of the transformation method (callable like f(newPoint))
    """
    return interp1d(x, np.linspace(0.0, 1.0, len(x)), kind='nearest')

  def len(self):
    """
      Size of the grid
      @ In, None
      @ Out, len, int, total number of nodes
    """
    return self.gridContainer['gridLength'] if 'gridLength' in self.gridContainer else 0

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.gridContainer['dimensionNames']        = []                 # this is the ordered list of the variable names (ordering match self.gridStepSize anfd the ordering in the test matrixes)
    self.gridContainer['gridVectors']           = {}                 # {'name of the variable':numpy.ndarray['the coordinate']}
    self.gridContainer['bounds']                = {'upperBounds':{},'lowerBounds':{}} # dictionary of lower and upper bounds
    self.gridContainer['gridLength']            = 0                  # this the total number of nodes in the grid
    self.gridContainer['gridShape']             = None               # shape of the grid (tuple)
    self.gridContainer['gridMatrix']            = None               # matrix containing the cell ids (unique integer identifier that map a set on nodes with respect an hypervolume)
    self.gridContainer['gridCoorShape']         = None               # shape of the matrix containing all coordinate of all points in the grid
    self.gridContainer['gridCoord']             = None               # the matrix containing all coordinate of all points in the grid
    self.gridContainer['nVar']                  = 0                  # this is the number of grid dimensions
    self.gridContainer['transformationMethods'] = None               # Dictionary of methods to transform the coordinate from 0-1 values to something else. These methods are pointed and passed into the initialize method. {varName:method}
    self.gridContainer['cellIDs']               = {}                 # Cell IDs and verteces coordinates
    self.gridContainer['vertexToCellIds']       = {}                 # mapping between vertices and cell ids
    self.gridContainer['initDictionary']        = None               # dictionary of initialization parameters passed in the initialize method
    self.constructTensor                        = False              # True if we need to construct the tensor product of the the ND grid (full grid) or just the iterator (False)
    self.uniqueCellNumber                       = 0                  # number of unique cells
    self.gridIterator                           = None               # the grid iterator
    self.gridInitDict                           = {}                 # dictionary with initialization grid info from _readMoreXML. If None, the "initialize" method will look for all the information in the in Dictionary
    self.volumetricRatio                        = None               # volumetric ratio (optional if steplenght is read or passed in initDict)
    self.nVar = None
    self.dimName = None
    self.gridInitDict = {}

  def _readMoreXml(self, xmlNode, dimensionTags=None, dimTagsPrefix=None):
    """
      XML reader for the grid statement.
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this class
      @ In, dimensionTag, list, optional, names of the tag that represents the grid dimensions
      @ In, dimTagsPrefix, dict, optional, eventual prefix to use for defining the dimName
      @ Out, None
    """
    self.gridInitDict = {'dimensionNames':[],'lowerBounds':{},'upperBounds':{},'stepLength':{}}
    gridInfo = {}
    dimInfo = {}
    for child in xmlNode:
      self.dimName = None
      if dimensionTags is not None:
        if child.tag in dimensionTags:
          self.dimName = child.attrib['name']
          if dimTagsPrefix is not None:
            self.dimName = dimTagsPrefix[child.tag] + self.dimName if child.tag in dimTagsPrefix else self.dimName
      if child.tag == "grid":
        gridInfo[self.dimName] = self._readGridStructure(child, xmlNode)
      for childChild in child:
        if childChild.tag == "grid":
          gridInfo[self.dimName] = self._readGridStructure(childChild,child)
        if 'dim' in childChild.attrib:
          dimID = str(len(self.gridInitDict['dimensionNames']) + 1) if self.dimName is None else self.dimName
          try:
            dimInfo[dimID] = [int(childChild.attrib['dim']),None]
          except ValueError:
            self.raiseAnError(ValueError, "cannot convert 'dim' attribute in integer!")
    # check for globalGrid type of structure
    globalGrids = {}
    for key in list(gridInfo.keys()): # list() to create copy so pop can be used
      splitted = key.split(":")
      if splitted[0].strip() == 'globalGrid':
        globalGrids[splitted[1]] = gridInfo.pop(key)
    for key in gridInfo:
      if gridInfo[key][0].strip() == 'globalGrid':
        if gridInfo[key][-1].strip() not in globalGrids:
          self.raiseAnError(IOError, f'global grid for dimension named {key} has not been found!')
        if key in dimInfo:
          dimInfo[key][-1] = gridInfo[key][-1].strip()
        gridInfo[key] = globalGrids[gridInfo[key][-1].strip()]
      self.gridInitDict['lowerBounds'][key] = min(gridInfo[key][-1])
      self.gridInitDict['upperBounds'][key] = max(gridInfo[key][-1])
      self.gridInitDict['stepLength' ][key] = [round(gridInfo[key][-1][k+1] - gridInfo[key][-1][k],14) for k in range(len(gridInfo[key][-1])-1)] if gridInfo[key][1] == 'custom' else [round(gridInfo[key][-1][1] - gridInfo[key][-1][0],14)]
    self.gridContainer['gridInfo'    ]      = gridInfo
    self.gridContainer['dimInfo'] = dimInfo

  def _readGridStructure(self, child, parent):
    """
      This method is aimed to read the grid structure in the xml node
      @ In, child, xml.etree.ElementTree.Element, the xml node containing the grid info
      @ In, parent, xml.etree.ElementTree.Element, the xml node that contains the node in which the grid info are defined
      @ Out, gridStruct, tuple, the grid structure read ((type, construction type, upper and lower bounds), gridName)
    """
    if child.tag =='grid':
      gridStruct, gridName = self._fillGrid(child)
      if self.dimName is None:
        self.dimName = str(len(self.gridInitDict['dimensionNames'])+1)
      if parent.tag != 'globalGrid':
        self.gridInitDict['dimensionNames'].append(self.dimName)
      else:
        if gridName is None:
          self.raiseAnError(IOError, 'grid defined in globalGrid block must have the attribute "name"!')
        self.dimName = parent.tag + ':' + gridName

      return gridStruct

  def _fillGrid(self, child):
    """
      This method is aimed to fill the grid structure from an XML node
      @ In, child, xml.etree.ElementTree.Element, the xml node containing the grid info
      @ Out, gridStruct, tuple, the grid structure read ((type, construction type, upper and lower bounds), gridName)
    """
    constrType = None
    if 'construction' in child.attrib.keys():
      constrType = child.attrib['construction']
    if 'type' not in child.attrib.keys():
      self.raiseAnError(IOError, "Each <grid> XML node needs to have the attribute type!!!!")
    nameGrid = None
    if constrType in ['custom','equal']:
      bounds = [floatConversion(element) for element in child.text.split()]
      bounds.sort()
      lower, upper = min(bounds), max(bounds)
      if 'name' in child.attrib.keys():
        nameGrid = child.attrib['name']
    if constrType == 'custom':
      gridStruct = (child.attrib['type'],constrType,bounds),nameGrid
    elif constrType == 'equal':
      if len(bounds) != 2:
        self.raiseAnError(IOError, f'body of grid XML node needs to contain 2 values (lower and upper bounds).Tag = {child.tag}')
      if 'steps' not in child.attrib.keys():
        self.raiseAnError(IOError, 'the attribute step needs to be inputted when "construction" attribute == equal!')
      gridStruct = (child.attrib['type'],constrType,np.linspace(lower,upper,partialEval(child.attrib['steps'])+1)),nameGrid
    elif child.attrib['type'] == 'globalGrid':
      gridStruct = (child.attrib['type'],constrType,child.text),nameGrid
    else:
      self.raiseAnError(IOError, f'construction type unknown! Got: {constrType}')

    return gridStruct

  def initialize(self, initDictionary=None):
    """
      Initialization method. The full grid is created in this method.
      @ In, initDictionary, dict, optional, dictionary of input arguments needed to create a Grid:
      {dimensionNames:[]}, required,list of axis names (dimensions' IDs)
      {lowerBounds:{}}, required, dictionary of lower bounds for each dimension
      {upperBounds:{}}, required, dictionary of upper bounds for each dimension
      {volumetriRatio:float or stepLength:dict}, required, p.u. volumetric ratio of the grid or dictionary of stepLengths ({'varName:list,etc'}
      {excludeBounds:{'lowerBounds':bool,'upperBounds':bool}}, optional, dictionary of dictionaries that determines if the lower or upper bounds should be excluded or not
      {computeCells:bool},optional, boolean to ask to compute the cells ids and verteces coordinates, default = False
      {constructTensor:bool},optional, boolean to ask to compute the full grid (True) or just the ND iterator
      {transformationMethods:{}}, optional, dictionary of methods to transform p.u. step size into a transformed system of coordinate. the transformationMethods dictionary needs to be provided as follow:
                                  {"dimensionName1":[instanceOfMethod,optional *args (in case the method takes as input other parameters in addition to a coordinate],
                                   or
                                   "dimensionName2":[instanceOfMethod]
                                  }
      !!!!!!
      if the self.gridInitDict is != None (info read from XML node), this method looks for the information in that dictionary first and after it checks the initDict object
      !!!!!!
      @ Out, None
    """
    self.raiseAMessage("Starting initialization of grid ")
    if len(self.gridInitDict.keys()) == 0 and initDictionary is None:
      self.raiseAnError(Exception, 'No initialization parameters have been provided!!')
    # grep the keys that have been read
    readKeys        = []
    initDict        = initDictionary if initDictionary is not None else {}
    computeCells    = bool(initDict.get('computeCells', False))
    self.constructTensor = bool(initDict['constructTensor']) if 'constructTensor' in initDict else False
    if  len(self.gridInitDict.keys()) != 0:
      readKeys = list(self.gridInitDict.keys())
    if initDict is not None:
      if type(initDict).__name__ != "dict":
        self.raiseAnError(Exception, 'The in argument is not a dictionary!')
    if "dimensionNames" not in list(initDict.keys())+readKeys:
      self.raiseAnError(Exception, '"dimensionNames" key is not present in the initialization dictionary!')
    if "lowerBounds" not in list(initDict.keys())+readKeys:
      self.raiseAnError(Exception, '"lowerBounds" key is not present in the initialization dictionary')
    if "lowerBounds" not in readKeys:
      if type(initDict["lowerBounds"]).__name__ != "dict":
        self.raiseAnError(Exception, 'The lowerBounds entry is not a dictionary')
    if "upperBounds" not in list(initDict.keys())+readKeys:
      self.raiseAnError(Exception, '"upperBounds" key is not present in the initialization dictionary')
    if "upperBounds" not in readKeys:
      if type(initDict["upperBounds"]).__name__ != "dict":
        self.raiseAnError(Exception, 'The upperBounds entry is not a dictionary')
    if "transformationMethods" in initDict:
      self.gridContainer['transformationMethods'] = initDict["transformationMethods"]
    self.nVar                            = len(self.gridInitDict["dimensionNames"]) if "dimensionNames" in self.gridInitDict else len(initDict["dimensionNames"])
    self.gridContainer['dimensionNames'] = self.gridInitDict["dimensionNames"] if "dimensionNames" in self.gridInitDict else initDict["dimensionNames"]
    #expand iterator with list()
    self.gridContainer['dimensionNames'] = list(self.gridContainer['dimensionNames'])
    upperkeys                            = list(self.gridInitDict["upperBounds"].keys() if "upperBounds" in self.gridInitDict else initDict["upperBounds"  ].keys())
    lowerkeys                            = list(self.gridInitDict["lowerBounds"].keys() if "lowerBounds" in self.gridInitDict else initDict["lowerBounds"  ].keys())
    self.gridContainer['dimensionNames'].sort()
    upperkeys.sort()
    lowerkeys.sort()
    if upperkeys != lowerkeys != self.gridContainer['dimensionNames']:
      self.raiseAnError(Exception, 'dimensionNames and keys in upperBounds and lowerBounds dictionaries do not correspond')
    self.gridContainer['bounds']["upperBounds" ] = self.gridInitDict["upperBounds"] if "upperBounds" in self.gridInitDict else initDict["upperBounds"]
    self.gridContainer['bounds']["lowerBounds"]  = self.gridInitDict["lowerBounds"] if "lowerBounds" in self.gridInitDict else initDict["lowerBounds"]
    if "volumetricRatio" not in initDict and "stepLength" not in list(initDict.keys())+readKeys:
      self.raiseAnError(Exception, '"volumetricRatio" or "stepLength" key is not present in the initialization dictionary')
    if "volumetricRatio" in initDict and "stepLength" in list(initDict.keys())+readKeys:
      self.raiseAWarning('"volumetricRatio" and "stepLength" keys are both present! the "volumetricRatio" has priority!')
    if "volumetricRatio" in initDict:
      self.volumetricRatio = initDict["volumetricRatio"]
      # build the step size in 0-1 range such as the differential volume is equal to the tolerance
      stepLength, ratioRelative = [], self.volumetricRatio**(1./float(self.nVar))
      for varId in range(len(self.gridContainer['dimensionNames'])):
        stepLength.append([ratioRelative*(self.gridContainer['bounds']["upperBounds" ][self.gridContainer['dimensionNames'][varId]] - self.gridContainer['bounds']["lowerBounds" ][self.gridContainer['dimensionNames'][varId]])])
    else:
      if "stepLength" not in readKeys:
        if type(initDict["stepLength"]).__name__ != "dict":
          self.raiseAnError(Exception, 'The stepLength entry is not a dictionary')
      stepLength = []
      for dimName in self.gridContainer['dimensionNames']:
        stepLength.append(initDict["stepLength"][dimName] if  "stepLength" not in readKeys else self.gridInitDict["stepLength"][dimName])

    # check if the lower or upper bounds need to be excluded
    excludeBounds   = initDict.get('excludeBounds', {'lowerBounds': False,'upperBounds': False})
    if 'lowerBounds' not in excludeBounds:
      excludeBounds['lowerBounds'] = False
    if 'upperBounds' not in excludeBounds:
      excludeBounds['upperBounds'] = False

    #moving forward building all the information set
    pointByVar                           = [None]*self.nVar  #list storing the number of point by cooridnate
    self.gridContainer['initDictionary'] = initDict
    #building the grid point coordinates
    for varId, varName in enumerate(self.gridContainer['dimensionNames']):
      checkBounds = True
      if len(stepLength[varId]) == 1:
        # equally spaced or volumetriRatio. (the use of np.finfo(float).eps is only to avoid round-off error, the upperBound is included in the mesh)
        # Any number greater than zero and less than one should suffice
        if self.volumetricRatio is not None:
          lowerBound = self.gridContainer['bounds']["lowerBounds"][varName] if not excludeBounds['lowerBounds'] else self.gridContainer['bounds']["lowerBounds"][varName] + stepLength[varId][-1]
          upperBound = self.gridContainer['bounds']["upperBounds"][varName]
          self.gridContainer['gridVectors'][varName] = np.arange(lowerBound, upperBound, stepLength[varId][-1])
        else:
          # maljdan: Enhancing readability of this conditional by using local
          # variables. This portion of the conditional is for evenly-spaced
          # grid cells. In the call to np.arange, if the one was to use ub as
          # the upper bound argument, then you are allowing room for round-off
          # error as sometimes this will include the point ub and sometimes it
          # will not. Instead, we will explicitly remove it from the np.arange
          # call by moving the upper bound slightly inward (ub-myEps), and then
          # explicitly concatenating ub to the end of the generated list.
          # Note, that any number greater than zero and less than one should
          # suffice for the multiplier used in myEps. My first thought was to
          # use machine precision (np.finfo(float).eps), but this could also
          # numerical instabilities. The former value was using 1e-3, but it was
          # multiplied by ub not the step size. I selected 0.5 because it seems
          # like a goldilocks number. Any larger value will bias values toward
          # ub in terms of numerical stability, and any value lower will bias
          # towards the next lower grid cell. 0.5 puts us as far from making a
          # mistake in either direction as possible.
          stepSize   = stepLength[varId][-1]
          lowerBound = self.gridContainer['bounds']["lowerBounds"][varName] if not excludeBounds['lowerBounds'] else self.gridContainer['bounds']["lowerBounds"][varName] + stepSize
          upperBound = self.gridContainer['bounds']["upperBounds"][varName] if not excludeBounds['upperBounds'] else self.gridContainer['bounds']["upperBounds"][varName] - stepSize
          myEps = stepSize * 0.5 # stepSize * np.finfo(float).eps
          self.gridContainer['gridVectors'][varName] = np.concatenate((np.arange(lowerBound, upperBound-myEps, stepSize), np.atleast_1d(upperBound)))
      else:
        # custom grid
        # it is not very efficient, but this approach is only for custom grids => limited number of discretizations
        gridMesh = [self.gridContainer['bounds']["lowerBounds"][varName]]
        for stepLengthi in stepLength[varId]:
          gridMesh.append(round(gridMesh[-1], 14) + round(stepLengthi, 14))
        if len(gridMesh) == 1:
          checkBounds = False
        self.gridContainer['gridVectors'][varName] = np.asarray(gridMesh)
      if checkBounds and compare(self.gridContainer['bounds']["lowerBounds" ][varName], self.gridContainer['bounds']["upperBounds" ][varName]):
        self.raiseAnError(IOError, f"the lowerBound and upperBound for dimension named {varName} are the same!. lowerBound = {self.gridContainer['bounds']['lowerBounds'][varName]}" +
                                   f" and upperBound = {self.gridContainer['bounds']['upperBounds'][varName]}")
      lowerBound = self.gridContainer['bounds']["lowerBounds"][varName] if not excludeBounds['lowerBounds'] else self.gridContainer['bounds']["lowerBounds"][varName] + stepLength[varId][-1]
      upperBound = self.gridContainer['bounds']["upperBounds"][varName] if not excludeBounds['upperBounds'] else self.gridContainer['bounds']["upperBounds"][varName] - stepLength[varId][-1]
      if not compare(max(self.gridContainer['gridVectors'][varName]), upperBound) and self.volumetricRatio is None:
        self.raiseAnError(IOError, f"the maximum value in the grid is different than the upperBound! upperBound: {upperBound}" +
                                   f" != maxValue in grid: {max(self.gridContainer['gridVectors'][varName])}")
      if not compare(min(self.gridContainer['gridVectors'][varName]),lowerBound):
        self.raiseAnError(IOError, f"the minimum value in the grid is different than the lowerBound! lowerBound: {lowerBound}" +
                                   f" != minValue in grid: {min(self.gridContainer['gridVectors'][varName])}")
      if self.gridContainer['transformationMethods'] is not None:
        if varName in self.gridContainer['transformationMethods']:
          self.gridContainer['gridVectors'][varName] = np.asarray([self.gridContainer['transformationMethods'][varName][0](coor) for coor in self.gridContainer['gridVectors'][varName]])
      pointByVar[varId]                              = int(np.shape(self.gridContainer['gridVectors'][varName])[0])
    self.gridContainer['gridShape']                  = tuple   (pointByVar)                             # tuple of the grid shape
    self.gridContainer['gridLength']                 = int(np.prod (np.asarray(pointByVar, dtype=np.float64))) # total number of point on the grid
    self.gridContainer['gridCoorShape']              = tuple   (pointByVar+[self.nVar])                # shape of the matrix containing all coordinate of all points in the grid
    if self.constructTensor:
      self.gridContainer['gridCoord'] = np.zeros(self.gridContainer['gridCoorShape'])   # the matrix containing all coordinate of all points in the grid
    self.uniqueCellNumber                               = np.prod ([element-1 for element in pointByVar]) # number of unique cells
    # filling the coordinate on the grid
    self.gridIterator = ravenArrayIterator(arrayIn=self.gridContainer['gridCoord']) if self.constructTensor else ravenArrayIterator(shape=self.gridContainer['gridShape'])
    if computeCells:
      gridIterCells =  ravenArrayIterator(arrayIn=np.zeros(shape=(2,)*self.nVar,dtype=int))
      origin = [-1]*self.nVar
      pp     = [element - 1 for element in pointByVar]
      cellID = int(initDict['startingCellId']) if 'startingCellId' in  initDict else 1
    if self.constructTensor or computeCells:
      while not self.gridIterator.finished:
        if self.constructTensor:
          coordinateID  = self.gridIterator.multiIndex[-1]
          dimName       = self.gridContainer['dimensionNames'][coordinateID]
          valuePosition = self.gridIterator.multiIndex[coordinateID]
          self.gridContainer['gridCoord'][self.gridIterator.multiIndex] = self.gridContainer['gridVectors'][dimName][valuePosition]
        if computeCells:
          if all(np.greater(pp,list(self.gridIterator.multiIndex[:
            -1]))) and list(self.gridIterator.multiIndex[:-1]) != origin:
            self.gridContainer['cellIDs'][cellID] = []
            origin = list(self.gridIterator.multiIndex[:-1])
            while not gridIterCells.finished:
              vertex = tuple(np.array(origin)+gridIterCells.multiIndex)
              self.gridContainer['cellIDs'][cellID].append(vertex)
              if vertex in self.gridContainer['vertexToCellIds'].keys():
                self.gridContainer['vertexToCellIds'][vertex].append(cellID)
              else:
                self.gridContainer['vertexToCellIds'][vertex] = [cellID]
              gridIterCells.iternext()
            gridIterCells.reset()
            cellID += 1
        self.gridIterator.iternext()
      if len(self.gridContainer['cellIDs'].keys()) != self.uniqueCellNumber and computeCells:
        self.raiseAnError(IOError, "number of cells detected != than the number of actual cells!")
      self.resetIterator()

    self.raiseAMessage("Grid "+"initialized...")

  def retrieveCellIds(self, listOfPoints, containedOnly=False):
    """
      This method is aimed to retrieve the cell IDs that are contained in certain boundaries provided as list of points
      @ In, listOfPoints, list, list of points that represent the boundaries ([listOfFirstBound, listOfSecondBound])
      @ In, containedOnly, bool, optional, flag to ask for cells contained in the listOfPoints or just cells that touch the listOfPoints, default False
      @ Out, previousSet, list, list of cell ids
    """
    cellIds = []
    for cntb, bound in enumerate(listOfPoints):
      cellIds.append([])
      for point in bound:
        cellIds[cntb].extend(self.gridContainer['vertexToCellIds'][tuple(point)])
      if cntb == 0:
        previousSet = set(cellIds[cntb])
      if containedOnly:
        previousSet = set(previousSet).intersection(cellIds[cntb])
      else:
        previousSet.update(cellIds[cntb])

    return list(set(previousSet))

  def returnGridAsArrayOfCoordinates(self):
    """
      Return the grid as an array of coordinates
      @ In, None
      @ Out, returnCoordinates, np.array, array of coordinates
    """
    returnCoordinates =  self.__returnCoordinatesReshaped((self.gridContainer['gridLength'], self.nVar))

    return returnCoordinates

  def __returnCoordinatesReshaped(self, newShape):
    """
      Method to return the grid Coordinates reshaped with respect an in Shape
      @ In, newShape, tuple, newer shape
      @ Out, returnCoordinates, np.array, array of coordinates
    """
    returnCoordinates = self.gridContainer['gridCoord']
    returnCoordinates.shape = newShape

    return returnCoordinates

  def resetIterator(self):
    """
      Reset internal iterator
      @ In, None
      @ Out, None
    """
    self.gridIterator.reset()

  def returnIteratorIndexes(self, returnDict=True):
    """
      Return the iterator indexes
      @ In, returnDict, bool, optional, returnDict if true, the Indexes are returned in dictionary format
      @ Out, coordinates, tuple or dictionary, coordinates
    """
    currentIndexes = self.gridIterator.multiIndex
    if not returnDict:
      return currentIndexes
    coordinates = {}
    for cnt, key in enumerate(self.gridContainer['dimensionNames']):
      coordinates[key] = currentIndexes[cnt]

    return coordinates

  def returnIteratorIndexesFromIndex(self, listOfIndexes):
    """
      Return internal iterator indexes from list of coordinates in the list
      @ In, listOfIndexes, list, list of grid coordinates
      @ Out, coordinates, tuple or dictionary, coordinates
    """
    coordinates = {}
    for cnt, key in enumerate(self.gridContainer['dimensionNames']):
      coordinates[key] = listOfIndexes[cnt]

    return coordinates

  def returnShiftedCoordinate(self, coordinates, shiftingSteps):
    """
      Method to return the coordinate that is a # shiftingStep away from the input coordinate
      For example, if 1D grid= {'dimName':[1,2,3,4]}, coordinate is 3 and  shiftingStep is -2,
      the returned coordinate will be 1
      @ In,  coordinates, dict, dictionary of coordinates. {'dimName1':startingCoordinate1,dimName2:startingCoordinate2,...}
      @ In,  shiftingSteps, dict, dict of shifiting steps. {'dimName1':shiftingStep1,dimName2:shiftingStep2,...}
      @ Out, outputCoordinates, dict, dictionary of shifted coordinates' values {dimName:value1,...}
    """
    outputCoordinates = {}
    # create multiindex
    multiindex = []
    for varName in self.gridContainer['dimensionNames']:
      if varName in coordinates and varName in shiftingSteps:
        multiindex.append(coordinates[varName] + shiftingSteps[varName])
      elif varName in coordinates and not varName in shiftingSteps:
        multiindex.append(coordinates[varName])
      else:
        multiindex.append(0)
    outputCoors = self.returnCoordinateFromIndex(multiindex,returnDict=True)
    for varName in shiftingSteps:
      outputCoordinates[varName] = outputCoors[varName]

    return outputCoordinates

  def returnPointAndAdvanceIterator(self, returnDict=False, recastMethods={}):
    """
      Method to return a point in the grid. This method will return the coordinates of the point to which the iterator is pointing
      In addition, it advances the iterator in order to point to the following coordinate
      @ In, returnDict, bool, optional, flag to request the output in dictionary format or not.
                               if True a dict ( {dimName1:
                                 coordinate1,dimName2:coordinate2,etc} is returned
                               if False a tuple is riturned (coordinate1,coordinate2,etc
      @ In, recastMethods, dict, optional, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
      @ Out, coordinate, tuple, tuple containing the coordinates
    """
    if not self.gridIterator.finished:
      coordinates = self.returnCoordinateFromIndex(self.gridIterator.multiIndex,returnDict,recastMethods)
      for _ in range(self.nVar if self.constructTensor else 1):
        self.gridIterator.iternext()
    else:
      coordinates = None

    return coordinates

  def returnCoordinateFromIndex(self, multiDimIndex, returnDict=False, recastMethods={}):
    """
      Method to return a point in the grid. This method will return the coordinates of the point is requested by multiDimIndex
      In addition, it advances the iterator in order to point to the following coordinate
      @ In, multiDimIndex, tuple, tuple containing the Id of the point needs to be returned (e.g. 3 dim grid,  (xID,yID,zID))
      @ In, returnDict, bool, optional, flag to request the output in dictionary format or not.
                                         if True a dict ( {dimName1:
                                           coordinate1,dimName2:coordinate2,etc} is returned
                                         if False a tuple is riturned (coordinate1,coordinate2,etc)
      @ In, recastMethods, dict, optional, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
      @ Out, coordinate, tuple or dict, tuple containing the coordinates
    """
    coordinates = [None]*self.nVar if returnDict is False else {}
    for cnt, key in enumerate(self.gridContainer['dimensionNames']):
      vvkey = cnt if not returnDict else key
      # if out of bound, we set the coordinate to maxsize
      if multiDimIndex[cnt] < 0:
        coordinates[vvkey] = -sys.maxsize
      elif multiDimIndex[cnt] > len(self.gridContainer['gridVectors'][key])-1:
        coordinates[vvkey] = sys.maxsize
      else:
        if key in recastMethods:
          coordinates[vvkey] = recastMethods[key][0](self.gridContainer['gridVectors'][key][multiDimIndex[cnt]],*recastMethods[key][1] if len(recastMethods[key]) > 1 else [])
        else:
          coordinates[vvkey] = self.gridContainer['gridVectors'][key][multiDimIndex[cnt]]
    if not returnDict:
      coordinates = tuple(coordinates)

    return coordinates

  def flush(self):
    """
      Reset GridEntity attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self.gridContainer['dimensionNames'] = []
    self.gridContainer['gridVectors'] = {}
    self.gridContainer['bounds'] = {'upperBounds':{},'lowerBounds':{}}
    self.gridContainer['gridLength'] = 0
    self.gridContainer['gridShape'] = None
    self.gridContainer['gridMatrix'] = None
    self.gridContainer['gridCoorShape'] = None
    self.gridContainer['gridCoord'] = None
    self.gridContainer['nVar'] = 0
    self.gridContainer['transformationMethods'] = None
    self.gridContainer['cellIDs'] = {}
    self.gridContainer['vertexToCellIds'] = {}
    self.gridContainer['initDictionary'] = None
    self.uniqueCellNumber = 0
    self.gridIterator = None
    self.nVar = None

class MultiGridEntity(GridBase):
  """
    This class is dedicated to the creation and handling of N-Dimensional Grid.
    In addition, it handles an hierarchical multi-grid approach (creating a mapping from coarse and finer grids in
    an adaptive meshing approach). The strategy for mesh (grid) refining is driven from outside.
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.multiGridActivated = False         # boolean flag to check if the multigrid approach has been activated
    self.subGridVolumetricRatio = None      # initial subgrid volumetric ratio
    self.grid = ETS.HierarchicalTree(self.__createNewNode("InitialGrid",
                                        {"grid":factory.returnInstance("GridEntity"),
                                        "level":"1"})) # grid hierarchical Container
    self.multiGridIterator = ["1", None]    # multi grid iterator [first position is the level ID, the second it the multi-index]
    self.mappingLevelName = {'1':None}      # mapping between grid level and node name
    self.nVar = None

  def __len__(self):
    """
    Overload __len__ method. It iterates over all the hierarchical structure of grids
    @ In, None
    @ Out, totalLength, integer, total number of nodes
    """
    totalLength = 0
    for node in self.grid.iter():
      totalLength += len(node.get('grid'))

    return totalLength

  def _readMoreXml(self, xmlNode, dimensionTags=None, dimTagsPrefix=None):
    """
      XML reader for the Multi-grid statement.
      @ In, xmlNode, xml.etree.ElementTree, XML element node that represents the portion of the input that belongs to this class
      @ In, dimensionTag, list, optional, names of the tag that represents the grid dimensions
      @ In, dimTagsPrefix, dict, optional, eventual prefix to use for defining the dimName
      @ Out, None
    """
    self.grid.getrootnode().get("grid")._readMoreXml(xmlNode, dimensionTags, dimTagsPrefix)

  def initialize(self, initDictionary=None):
    """
      Initialization method. The full grid is created in this method.
      @ In, initDictionary, dict, optional, dictionary of input arguments needed to create a Grid:
       {dimensionNames:[]}, required,list of axis names (dimensions' IDs)
       {lowerBounds:{}}, required, dictionary of lower bounds for each dimension
       {upperBounds:{}}, required, dictionary of upper bounds for each dimension
       { volumetriRatio:float or stepLength:dict}, required, p.u. volumetric ratio of the grid or dictionary of stepLengths ({'varName:list,etc'}
       {subGridVolumetricRatio:float}, optional, p.u. volumetric ratio of the subGrid, default  1.e5
       {computeCells:bool},optional, boolean to ask to compute the cells ids and verteces coordinates, default = False
       {transformationMethods:{}}, optional, dictionary of methods to transform p.u. step size into a transformed system of coordinate
       !!!!!!
       if the self.gridInitDict is != None (info read from XML node), this method looks for the information in that dictionary first and after it checks the initDict object
       !!!!!!
      @ Out, None
    """
    if "rootName" in initDictionary.keys():
      self.grid.updateNodeName("root",initDictionary["rootName"])
      self.mappingLevelName['1'] = initDictionary["rootName"]
    self.grid.getrootnode().get("grid").initialize(initDictionary)
    self.nVar = self.grid.getrootnode().get("grid").nVar
    self.multiGridIterator[1] = self.grid.getrootnode().get("grid").returnIteratorIndexes(False)

  def retrieveCellIds(self,listOfPoints,nodeName=None, containedOnly = True):
    """
      This method is aimed to retrieve the cell IDs that are contained in certain bounaried provided as list of points
      @ In, listOfPoints, list, list of points that represent the boundaries ([listOfFirstBound, listOfSecondBound])
      @ In, nodeName, string, optional, node from which the cell IDs needs to be retrieved. If not present, all the cells are going to be retrieved
      @ In, containedOnly, bool, optional, flag to ask for cells contained in the listOfPoints or just cells that touch the listOfPoints, default True
      @ Out, setOfCells, list, list of cells ids
    """
    setOfCells = []
    if nodeName is None:
      for node in self.grid.iter():
        setOfCells.extend(node.get('grid').retrieveCellIds(listOfPoints, containedOnly))
    else:
      node = self.grid.find(nodeName)
      setOfCells.extend(node.get('grid').retrieveCellIds(listOfPoints, containedOnly))

    return setOfCells

  def getAllNodesNames(self, startingNode=None):
    """
      Get all the nodes' names
      @ In, startingNode, string, optional, node name
      @ Out, returnNames, list, list of names
    """
    if startingNode is not None:
      snode = self.grid.find(startingNode)
      returnNames = [node.name for node in snode.iter()]
    else:
      returnNames = self.mappingLevelName.values()

    return returnNames

  def __createNewNode(self, nodeName, attributes={}):
    """
      Create a new node in the grid
      @ In, nodeName, string, new node name
      @ In, attributes, dict, initial attributes
      @ Out, node, Node, new node
    """
    node = ETS.HierarchicalNode(nodeName)
    node.add("grid", factory.returnInstance("GridEntity"))
    for key, attribute in attributes.items():
      node.add(key,attribute)

    return node

  def _getMaxCellIds(self):
    """
      This method is aimed to retrieve the maximum cell Ids among all the nodes
      @ In, None
      @ Out, maxCellId, int, the maximum cell id
    """
    maxCellId = 0
    for node in self.grid.iter():
      maxLocalCellId = max(node.get('grid').returnParameter('cellIDs').keys())
      maxCellId = maxLocalCellId if maxLocalCellId > maxCellId else maxCellId

    return maxCellId

  def updateSubGrid(self, parentNode, refineDict):
    """
      Method aimed to update all the sub-grids of a parent Node.
      This method is going to delete the sub-grids of the parent Node and reconstruct them
      @ In, parentNode, string, name of the parent node whose sub-grids need to be updated
      @ In, refineDict, dict, dictionary with information to refine the parentNode grid
      @ Out, None
    """
    parentNode = self.grid.find(parentNode)
    if parentNode == -1:
      self.raiseAnError(Exception, f"parent Node named {parentNode} has not been found!")
    parentNode.clearBranch()
    self.refineGrid(refineDict)

  def refineGrid(self, refineDict):
    """
      Method aimed to refine all the grids that are related to the cellIds specified in the refineDict
      @ In, refineDict, dict, dictionary with information to refine the parentNode grid:
           {cellIDs:listOfCellIdsToBeRefined}
           {refiningNumSteps:numberOfStepsToUseForTheRefinement}
      @ Out, None
    """
    if "refiningNumSteps" not in refineDict and "volumetricRatio" not in refineDict:
      self.raiseAnError(IOError, "the refining Number of steps or the volumetricRatio has not been provided!!!")
    cellIdsToRefine, didWeFoundCells = refineDict['cellIDs'], dict.fromkeys(refineDict['cellIDs'], False)
    maxCellId = self._getMaxCellIds()
    for node in self.grid.iter():
      parentNodeCellIds  = node.get("grid").returnParameter('cellIDs')
      level, nodeCellIds = node.get("level"), parentNodeCellIds.keys()
      foundCells = set(nodeCellIds).intersection(cellIdsToRefine)
      if len(foundCells) > 0:
        parentGrid = node.get("grid")
        initDict   = parentGrid.returnParameter("initDictionary")
        if "transformationMethods" in initDict:
          initDict.pop("transformationMethods")
        for idcnt, fcellId in enumerate(foundCells):
          didWeFoundCells[fcellId] = True
          newGrid = factory.returnInstance("GridEntity")
          verteces = parentNodeCellIds[fcellId]
          lowerBounds,upperBounds = dict.fromkeys(parentGrid.returnParameter('dimensionNames'), sys.float_info.max), dict.fromkeys(parentGrid.returnParameter('dimensionNames'), -sys.float_info.max)
          for vertex in verteces:
            coordinates = parentGrid.returnCoordinateFromIndex(vertex, True, recastMethods=initDict["transformationMethods"] if "transformationMethods" in initDict.keys() else {})
            for key in lowerBounds:
              lowerBounds[key], upperBounds[key] = min(lowerBounds[key],coordinates[key]), max(upperBounds[key],coordinates[key])
          initDict["lowerBounds"], initDict["upperBounds"] = lowerBounds, upperBounds
          if "volumetricRatio" in refineDict.keys():
            initDict["volumetricRatio"] = refineDict["volumetricRatio"]
          else:
            if "volumetricRatio" in initDict:
              initDict.pop("volumetricRatio")
            initDict["stepLength"] = {}
            for key in lowerBounds:
              initDict["stepLength"][key] = [(upperBounds[key] - lowerBounds[key])/float(refineDict["refiningNumSteps"])]
          initDict["startingCellId"] = maxCellId+1
          newGrid.initialize(initDict)
          maxCellId   = max(newGrid.returnParameter('cellIDs').keys())
          refinedNode = self.__createNewNode(node.name+"_cell:"+str(fcellId),{"grid":newGrid,"level":level+"."+str(idcnt)})
          self.mappingLevelName[level+"."+str(idcnt)] = node.name+"_cell:"+str(fcellId)
          node.appendBranch(refinedNode)
      foundAll = all(item is True for item in set(didWeFoundCells.values()))
      if foundAll:
        break
    if not foundAll:
      self.raiseAnError(Exception, f"the following cell IDs have not been found: {' '.join([cellId for cellId, value in didWeFoundCells.items() if value is True])}")

  def returnGridAsArrayOfCoordinates(self, nodeName=None, returnDict=False):
    """
      Return the grid as an array of coordinates
      @ In, nodeName, string, optional, node name
      @ In, returnDict, bool, return a dictionary with the coordinates for all sub-grid, default = False
      @ Out, fullReshapedCoordinates, ndarray or dict (dependeing on returnDict flag), numpy array or dictionary of numpy arrays containing all the coordinates shaped as (fullgridLength,self.nVar) or (sub-gridLength, self.nVar)
    """
    if not returnDict:
      fullReshapedCoordinates = np.zeros((0,self.nVar))
      if nodeName is None:
        for node in self.grid.iter():
          fullReshapedCoordinates = np.concatenate((fullReshapedCoordinates, node.get('grid').returnGridAsArrayOfCoordinates()))
      else:
        fullReshapedCoordinates = self.grid.find(nodeName).get('grid').returnGridAsArrayOfCoordinates()
    else:
      fullReshapedCoordinates = {}
      if nodeName is None:
        for node in self.grid.iter():
          fullReshapedCoordinates[node.name] = node.get('grid').returnGridAsArrayOfCoordinates()
      else:
        fullReshapedCoordinates[nodeName] = self.grid.find(nodeName).get('grid').returnGridAsArrayOfCoordinates()

    return fullReshapedCoordinates

  def resetIterator(self):
    """
      Reset internal iterator
      @ In, None
      @ Out, None
    """
    for node in self.grid.iter():
      node.get('grid').resetIterator()
    self.multiGridIterator = [self.grid.getrootnode().get("level"),self.grid.getrootnode().get("grid").returnIteratorIndexes(False)]

  def returnIteratorIndexes(self, returnDict=True):
    """
      Return the iterator current indexes
      @ In, returnDict, bool, returnDict if true, the Indexes are returned in dictionary format
      @ Out, indexes, tuple or dictionary, current indexes
    """
    node = self.grid.find(self.mappingLevelName[self.multiGridIterator[0]])
    indexes = node.get('grid').returnIteratorIndexes(returnDict)

    return indexes

  def returnIteratorIndexesFromIndex(self, indexes):
    """
      Return internal iterator indexes from list of coordinates in the list
      @ In, indexes, list or tuple, if tuple -> tuple[0] multi-grid level, tuple[1] list of grid coordinates
                                  if list  -> list of grid coordinates. The multi-grid level is gonna be taken from self.multiGridIterator
      @ Out, returnIndexes, tuple or dictionary, current indexes
    """
    if isinstance(indexes, tuple):
      level, listOfIndexes = indexes[0], indexes[1]
    elif isinstance(indexes, list):
      level, listOfIndexes = self.multiGridIterator[0], indexes
    else:
      self.raiseAnError(Exception, "returnIteratorIndexesFromIndex method accepts a list or tuple only!")
    node = self.grid.find(self.mappingLevelName[level])
    returnIndexes = node.get('grid').returnIteratorIndexesFromIndex(listOfIndexes)

    return returnIndexes

  def returnShiftedCoordinate(self, coords, shiftingSteps):
    """
      Method to return the coordinate that is a # shiftingStep away from the input coordinate
      For example, if 1D grid= {'dimName':[1,2,3,4]}, coordinate is 3 and  shiftingStep is -2,
      the returned coordinate will be 1
      @ In, coords, dict or tuple, if  dict  -> dictionary of coordinates. {'dimName1':startingCoordinate1,dimName2:startingCoordinate2,...}.The multi-grid level is gonna be taken from self.multiGridIterator
                                  if  tuple -> tuple[0] multi-grid level, tuple[1] dictionary of coordinates. {'dimName1':
                                    startingCoordinate1,dimName2:startingCoordinate2,...}
      @ In, shiftingSteps, dict, dict of shifiting steps. {'dimName1':shiftingStep1,dimName2:shiftingStep2,...}
      @ Out, outputCoordinates, dict, dictionary of shifted coordinates' values {dimName:value1,...}
    """
    if isinstance(coords, tuple):
      level, coordinates = coords[0], coords[1]
    elif isinstance(coords, dict):
      level, coordinates = self.multiGridIterator[0], coords
    else:
      self.raiseAnError(Exception, "returnShiftedCoordinate method accepts a coords or tuple only!")
    node = self.grid.find(self.mappingLevelName[level])
    outputCoordinates = node.get('grid').returnShiftedCoordinate(coordinates,shiftingSteps)

    return outputCoordinates

  def returnPointAndAdvanceIterator(self, returnDict=False, recastMethods={}):
    """
      Method to return a point in the grid. This method will return the coordinates of the point to which the iterator is pointing
      In addition, it advances the iterator in order to point to the following coordinate
      @ In, returnDict, bool, optional, flag to request the output in dictionary format or not.
                               if True a dict ( {dimName1:
                                 coordinate1,dimName2:coordinate2,etc} is returned
                               if False a tuple is riturned (coordinate1,coordinate2,etc
      @ In, recastMethods, dict, optional, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
      @ Out, coordinate, tuple or dict, tuple (if returnDict=False) or dict (if returnDict=True) containing the coordinates
    """
    startingNode = self.grid.find(self.mappingLevelName[self.multiGridIterator[0]])
    coordinates = None
    for node in startingNode.iter():
      subGrid =  node.get('grid')
      if not subGrid.gridIterator.finished:
        coordinates = subGrid.returnCoordinateFromIndex(subGrid.gridIterator.multiIndex,returnDict,recastMethods)
        for _ in range(self.nVar):
          subGrid.gridIterator.iternext()
        break
      self.multiGridIterator[0], self.multiGridIterator[1] = node.get("level"), node.get("grid").returnIteratorIndexes(False)

    return coordinates

  def returnCoordinateFromIndex(self, multiDimNDIndex, returnDict=False, recastMethods={}):
    """
      Method to return a point in the grid. This method will return the coordinates of the point is requested by multiDimIndex
      In addition, it advances the iterator in order to point to the following coordinate
      @ In, multiDimNDIndex, tuple, tuple containing the Id of the point needs to be returned (e.g. 3 dim grid,  (xID,yID,zID))
      @ In, returnDict, bool, optional, flag to request the output in dictionary format or not.
                                         if True a dict ( {dimName1:
                                           coordinate1,dimName2:coordinate2,etc} is returned
                                         if False a tuple is riturned (coordinate1,coordinate2,etc)
      @ In, recastMethods, dict, optional, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
      @ Out, coordinate, tuple or dict, tuple (if returnDict=False) or dict (if returnDict=True) containing the coordinates
    """
    if isinstance(multiDimNDIndex[0], Number):
      level, multiDimIndex = self.multiGridIterator[0], multiDimNDIndex
    else:
      level, multiDimIndex = multiDimNDIndex[0], multiDimNDIndex[1]
    node = self.grid.find(self.mappingLevelName[level])

    return node.get('grid').returnCoordinateFromIndex(multiDimIndex, returnDict, recastMethods)

  def returnParameter(self, parameterName, nodeName=None):
    """
      Method to return one of the initialization parameters
      @ In, parameterName, string, name of the parameter to be returned
      @ In, nodeName, string, optional, name of the node from which we need to retrieve the parameter. If not present, the parameter is going to be retrieved from all nodes (and subnodes). If nodeName == *, the parameter is gonna retrieved from the self
      @ Out, paramDict, dict, dictionary of pointers to the requested parameter
    """
    if nodeName is not None and nodeName != "*":
      node = self.grid.find(nodeName)
      if node is None:
        self.raiseAnError(Exception, f'node {nodeName} has not been found in the MultiGrid hierarchal tree!')
      return node.get("grid").returnParameter(parameterName)
    elif nodeName == "*":
      if parameterName not in self.gridContainer:
        self.raiseAnError(Exception, f'parameter {parameterName} unknown among ones in MultiGridEntity class.')
      return self.gridContainer[parameterName]
    else:
      paramDict = {}
      for node in self.grid.iter():
        paramDict[node.name] = node.get("grid").returnParameter(parameterName)

    return paramDict

  def updateParameter(self, parameterName, newValue, upContainer=True, nodeName=None):
    """
      Method to update one of the initialization parameters
      @ In, parameterName, string, name of the parameter to be updated
      @ In, newValue, object, newer value
      @ In, upContainer, bool, optional, True if gridContainer needs to be updated, else gridInit
      @ In, nodeName, string, optional, name of the node in which we need to update the parameter. If not present, the parameter is going to be updated in all nodes (and subnodes).If nodeName == *, the parameter is gonna updated in self
      @ Out, None
    """
    if nodeName is not None:
      node = self.grid.find(nodeName)
      if node is None:
        self.raiseAnError(Exception, f'node {nodeName} has not been found in the MultiGrid hierarchal tree!')
      node.get("grid").updateParameter(parameterName, newValue, upContainer)
    elif nodeName == "*":
      if upContainer:
        self.gridContainer[parameterName] = newValue
      else:
        self.gridInitDict[parameterName ] = newValue
    else:
      for node in self.grid.iter():
        node.get("grid").updateParameter(parameterName, newValue, upContainer)

  def addCustomParameter(self, parameterName, value, nodeName=None):
    """
      Method to add a new parameter in the MultiGrid Entity
      @ In, parameterName, string, name of the parameter to be added
      @ In, value, float, new value
      @ In, nodeName, string, optional, name of the node in which we need to update the parameter. If not present, the parameter is going to be updated in all nodes (and subnodes).If nodeName == *, the parameter is gonna updated in self
      @ Out, None
    """
    if nodeName is not None:
      node = self.grid.find(nodeName)
      if node is None:
        self.raiseAnError(Exception, f'node {nodeName} has not been found in the MultiGrid hierarchal tree!')
    elif nodeName == "*":
      if parameterName in self.gridContainer:
        self.raiseAnError(Exception, f'parameter {parameterName} already present in MultiGridEntity!')
    else:
      for node in self.grid.iter():
        if parameterName in node.get("grid").gridContainer:
          self.raiseAnError(Exception, f'parameter {parameterName} already present in MultiGridEntity subnode {node.name}!')
    self.updateParameter(parameterName, value, True, nodeName)

  def flush(self):
    """
      Reset GridBase attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()

factory = EntityFactory('GridEntities')
factory.registerType('GridEntity', GridEntity)
factory.registerType('MultiGridEntity', MultiGridEntity)
