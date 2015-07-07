"""
Created on Mar 30, 2015

@author: alfoa
"""

#External Modules------------------------------------------------------------------------------------
#import itertools
import numpy as np
from scipy.interpolate import interp1d
import sys
import abc
import itertools
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import UreturnPrintTag,partialEval,compare, metaclass_insert
from BaseClasses import BaseType
from MessageHandler import MessageHandler
import TreeStructure as ETS
#import TreeStructure as TS
#Internal Modules End--------------------------------------------------------------------------------


class GridBase(metaclass_insert(abc.ABCMeta,BaseType)):
  """
  Base Class that needs to be used when a new Grid class is generated
  It provides all the methods to create, modify, and handle a grid in the phase space.
  """
  @classmethod
  def __len__(self):
    """
    Overload __len__ method.
    @ In, None
    @ Out, integer, total number of nodes
    """
    return 0

  def __init__(self,messageHandler=None):
    """
      Constructor
    """
    if messageHandler != None: self.setMessageHandler(messageHandler)
    self.printTag                               = UreturnPrintTag("GRID ENTITY")
    self.gridContainer                          = {}                             # dictionary that contains all the key feature of the grid

  @classmethod
  def _readMoreXml(self,xmlNode,dimensionTags=None,messageHandler=None,dimTagsPrefix=None):
    """
     XML reader for the grid statement.
     @ In, ETree object, xml node from where the info need to be retrieved
     @ In, dimensionTag, optional, list, names of the tag that represents the grid dimensions
     @ In, dimTagsPrefix, optional, dict, eventual prefix to use for defining the dimName
     @ Out, None
    """
    pass
  
  @classmethod
  def initialize(self,initDictionary=None):
    """
    Initialization method. The full grid is created in this method.
    @ In, initDictionary, dictionary, optional, dictionary of input arguments needed to create a Grid:
      {dimensionNames:[]}, required,list of axis names (dimensions' IDs)
      {lowerBounds:{}}, required, dictionary of lower bounds for each dimension
      {upperBounds:{}}, required, dictionary of upper bounds for each dimension
      {volumetriRatio:float or stepLenght:dict}, required, p.u. volumetric ratio of the grid or dictionary of stepLenghts ({'varName:list,etc'}
      {computeCells:bool},optional, boolean to ask to compute the cells ids and verteces coordinates, default = False
      {transformationMethods:{}}, optional, dictionary of methods to transform p.u. step size into a transformed system of coordinate
      !!!!!!
      if the self.gridInitDict is != None (info read from XML node), this method looks for the information in that dictionary first and after it checks the initDict object
      !!!!!!
    """
    pass

  @classmethod
  def retrieveCellIds(self,listOfPoints):
    """
     This method is aimed to retrieve the cell IDs that are contained in certain bounaried provided as list of points
     @ In, listOfPoints, list, list of points that represent the boundaries ([listOfFirstBound, listOfSecondBound])
    """
    pass

  @classmethod
  def returnGridAsArrayOfCoordinates(self):
    """
    Return the grid as an array of coordinates
    """
    pass

  def returnParameter(self,parameterName):
    """
    Method to return one of the initialization parameters
    @ In, string, parameterName, name of the parameter to be returned
    @ Out, object, pointer to the requested parameter
    """
    if parameterName not in self.gridContainer.keys(): self.raiseAnError(Exception,'parameter '+parameterName+'unknown among ones in GridEntity class.')
    return self.gridContainer[parameterName]

  def updateParameter(self,parameterName, newValue, upContainer=True):
    """
    Method to update one of the initialization parameters
    @ In, string, parameterName, name of the parameter to be updated
    @ In, boolean, optional, upContainer, True if gridContainer needs to be updated, else gridInit
    @ Out, None
    """
    if upContainer: self.gridContainer[parameterName] = newValue
    else          : self.gridInitDict[parameterName ] = newValue

  def addCustomParameter(self,parameterName, Value):
    """
    Method to add a new parameter in the Grid Entity
    @ In, string, parameterName, name of the parameter to be added
    @ Out, None
    """
    if parameterName in self.gridContainer.keys(): self.raiseAnError(Exception,'parameter '+parameterName+'already present in GridEntity!')
    self.updateParameter(parameterName, Value)
  
  @classmethod
  def resetIterator(self):
    """
    Reset internal iterator
    @ In, None
    @ Out, None
    """
    pass
  
  @classmethod
  def returnIteratorIndexes(self,returnDict = True):
    """
    Return the iterator indexes
    @ In, boolean,returnDict if true, the Indexes are returned in dictionary format
    @ Out, tuple or dictionary
    """
    pass
  
  @classmethod
  def returnIteratorIndexesFromIndex(self, listOfIndexes):
    """
    Return internal iterator indexes from list of coordinates in the list
    @ In, list,listOfIndexes, list of grid coordinates
    @ Out, dictionary
    """
    pass
  
  @classmethod
  def returnShiftedCoordinate(self,coordinates,shiftingSteps):
    """
    Method to return the coordinate that is a # shiftingStep away from the input coordinate
    For example, if 1D grid= {'dimName':[1,2,3,4]}, coordinate is 3 and  shiftingStep is -2,
    the returned coordinate will be 1
    @ In,  dict, coordinates, dictionary of coordinates. {'dimName1':startingCoordinate1,dimName2:startingCoordinate2,...}
    @ In,  dict, shiftingSteps, dict of shifiting steps. {'dimName1':shiftingStep1,dimName2:shiftingStep2,...}
    @ Out, dict, outputCoordinates, dictionary of shifted coordinates' values {dimName:value1,...}
    """
    pass
  
  @classmethod
  def returnPointAndAdvanceIterator(self, returnDict=False, recastMethods={}):
    """
    Method to return a point in the grid. This method will return the coordinates of the point to which the iterator is pointing
    In addition, it advances the iterator in order to point to the following coordinate
    @ In, boolean, optional, returnDict, flag to request the output in dictionary format or not.
                               if True a dict ( {dimName1:coordinate1,dimName2:coordinate2,etc} is returned
                               if False a tuple is riturned (coordinate1,coordinate2,etc
    @ In, dict, optional, recastMethods, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
    @ Out, tuple, coordinate, tuple containing the coordinates
    """
    pass
  
  @classmethod
  def returnCoordinateFromIndex(self, multiDimIndex, returnDict=False, recastMethods={}):
    """
    Method to return a point in the grid. This method will return the coordinates of the point is requested by multiDimIndex
    In addition, it advances the iterator in order to point to the following coordinate
    @ In, tuple, multiDimIndex, tuple containing the Id of the point needs to be returned (e.g. 3 dim grid,  (xID,yID,zID))
    @ In, boolean, optional, returnDict, flag to request the output in dictionary format or not.
                                         if True a dict ( {dimName1:coordinate1,dimName2:coordinate2,etc} is returned
                                         if False a tuple is riturned (coordinate1,coordinate2,etc)
    @ In, dict, optional, recastMethods, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
    @ Out, tuple or dict, coordinate, tuple containing the coordinates
    """
    pass




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
    @ Out, transformFunction, instance of the transformation method (callable like f(newPoint))
    """
    return interp1d(x, np.linspace(0.0, 1.0, len(x)), kind='nearest')

  def __len__(self):
    """
    Overload __len__ method.
    @ In, None
    @ Out, integer, total number of nodes
    """
    return self.gridContainer['gridLenght'] if 'gridLenght' in self.gridContainer.keys() else 0

  def __init__(self,messageHandler):
    GridBase.__init__(self,messageHandler)
    self.gridContainer['dimensionNames']        = []                 # this is the ordered list of the variable names (ordering match self.gridStepSize anfd the ordering in the test matrixes)
    self.gridContainer['gridVectors']           = {}                 # {'name of the variable':numpy.ndarray['the coordinate']}
    self.gridContainer['bounds']                = {'upperBounds':{},'lowerBounds':{}} # dictionary of lower and upper bounds
    self.gridContainer['gridLenght']            = 0                  # this the total number of nodes in the grid
    self.gridContainer['gridShape']             = None               # shape of the grid (tuple)
    self.gridContainer['gridMatrix']            = None               # matrix containing the cell ids (unique integer identifier that map a set on nodes with respect an hypervolume)
    self.gridContainer['gridCoorShape']         = None               # shape of the matrix containing all coordinate of all points in the grid
    self.gridContainer['gridCoord']             = None               # the matrix containing all coordinate of all points in the grid
    self.gridContainer['nVar']                  = 0                  # this is the number of grid dimensions
    self.gridContainer['transformationMethods'] = None               # Dictionary of methods to transform the coordinate from 0-1 values to something else. These methods are pointed and passed into the initialize method. {varName:method}
    self.gridContainer['cellIDs']               = {}                 # Cell IDs and verteces coordinates
    self.gridContainer['vertexToCellIds']       = {}
    self.uniqueCellNumber                       = 0                  # number of unique cells
    self.gridIterator                           = None               # the grid iterator
    self.gridInitDict                           = {}                 # dictionary with initialization grid info from _readMoreXML. If None, the "initialize" method will look for all the information in the in Dictionary
    self.volumetricRatio                        = None               # volumetric ratio (optional if steplenght is read or passed in initDict)

  def _readMoreXml(self,xmlNode,dimensionTags=None,messageHandler=None,dimTagsPrefix=None):
    """
     XML reader for the grid statement.
     @ In, ETree object, xml node from where the info need to be retrieved
     @ In, dimensionTag, optional, list, names of the tag that represents the grid dimensions
     @ In, dimTagsPrefix, optional, dict, eventual prefix to use for defining the dimName
     @ Out, None
    """
    if messageHandler != None: self.setMessageHandler(messageHandler)
    self.gridInitDict = {'dimensionNames':[],'lowerBounds':{},'upperBounds':{},'stepLenght':{}}
    gridInfo = {}
    dimInfo = {}
    for child in xmlNode:
      self.dimName = None
      if dimensionTags != None:
        if child.tag in dimensionTags:
          self.dimName = child.attrib['name']
          if dimTagsPrefix != None: self.dimName = dimTagsPrefix[child.tag] + self.dimName if child.tag in dimTagsPrefix.keys() else self.dimName
      if child.tag == "grid":
        gridInfo[self.dimName] = self._readGridStructure(child,xmlNode)
      for childChild in child:
        if childChild.tag == "grid": gridInfo[self.dimName] = self._readGridStructure(childChild,child)
        if 'dim' in childChild.attrib.keys():
          dimID = str(len(self.gridInitDict['dimensionNames'])+1) if self.dimName == None else self.dimName
          try              : dimInfo[dimID] = [int(childChild.attrib['dim']),None]
          except ValueError: self.raiseAnError(ValueError, "can not convert 'dim' attribute in integer!")
    #check for global_grid type of structure
    globalGrids = {}
    for key in gridInfo.keys():
      splitted = key.split(":")
      if splitted[0].strip() == 'global_grid': globalGrids[splitted[1]] = gridInfo.pop(key)
    for key in gridInfo.keys():
      if gridInfo[key][0].strip() == 'global_grid':
        if gridInfo[key][-1].strip() not in globalGrids.keys(): self.raiseAnError(IOError,'global grid for dimension named '+key+'has not been found!')
        if key in dimInfo.keys(): dimInfo[key][-1] = gridInfo[key][-1].strip()
        gridInfo[key] = globalGrids[gridInfo[key][-1].strip()]
      self.gridInitDict['lowerBounds'           ][key] = min(gridInfo[key][-1])
      self.gridInitDict['upperBounds'           ][key] = max(gridInfo[key][-1])
      self.gridInitDict['stepLenght'            ][key] = [round(gridInfo[key][-1][k+1] - gridInfo[key][-1][k],14) for k in range(len(gridInfo[key][-1])-1)] if gridInfo[key][1] == 'custom' else [round(gridInfo[key][-1][1] - gridInfo[key][-1][0],14)]
    self.gridContainer['gridInfo'               ]      = gridInfo
    self.gridContainer['dimInfo'] = dimInfo

  def _readGridStructure(self,child,parent):
    if child.tag =='grid':
      gridStruct, gridName = self._fillGrid(child)
      if self.dimName == None: self.dimName = str(len(self.gridInitDict['dimensionNames'])+1)
      if parent.tag != 'global_grid': self.gridInitDict['dimensionNames'].append(self.dimName)
      else:
        if gridName == None: self.raiseAnError(IOError,'grid defined in global_grid block must have the attribute "name"!')
        self.dimName = parent.tag + ':' + gridName
      return gridStruct

  def _fillGrid(self,child):
    constrType = None
    if 'construction' in child.attrib.keys(): constrType = child.attrib['construction']
    if 'type' not in child.attrib.keys()    : self.raiseAnError(IOError,"Each <grid> XML node needs to have the attribute type!!!!")
    nameGrid = None
    if constrType in ['custom','equal']:
      bounds = [partialEval(element) for element in child.text.split()]
      bounds.sort()
      lower, upper = min(bounds), max(bounds)
      if 'name' in child.attrib.keys(): nameGrid = child.attrib['name']
    if constrType == 'custom': return (child.attrib['type'],constrType,bounds),nameGrid
    elif constrType == 'equal':
      if len(bounds) != 2: self.raiseAnError(IOError,'body of grid XML node needs to contain 2 values (lower and upper bounds).Tag = '+child.tag)
      if 'steps' not in child.attrib.keys(): self.raiseAnError(IOError,'the attribute step needs to be inputted when "construction" attribute == equal!')
      return (child.attrib['type'],constrType,np.linspace(lower,upper,partialEval(child.attrib['steps'])+1)),nameGrid
    elif child.attrib['type'] == 'global_grid': return (child.attrib['type'],constrType,child.text),nameGrid
    else: self.raiseAnError(IOError,'construction type unknown! Got: ' + str(constrType))

  def initialize(self,initDictionary=None):
    """
    Initialization method. The full grid is created in this method.
    @ In, initDictionary, dictionary, optional, dictionary of input arguments needed to create a Grid:
      {dimensionNames:[]}, required,list of axis names (dimensions' IDs)
      {lowerBounds:{}}, required, dictionary of lower bounds for each dimension
      {upperBounds:{}}, required, dictionary of upper bounds for each dimension
      {volumetriRatio:float or stepLenght:dict}, required, p.u. volumetric ratio of the grid or dictionary of stepLenghts ({'varName:list,etc'}
      {computeCells:bool},optional, boolean to ask to compute the cells ids and verteces coordinates, default = False
      {transformationMethods:{}}, optional, dictionary of methods to transform p.u. step size into a transformed system of coordinate
      !!!!!!
      if the self.gridInitDict is != None (info read from XML node), this method looks for the information in that dictionary first and after it checks the initDict object
      !!!!!!
    """
    self.raiseAMessage("Starting initialization grid...")
    if len(self.gridInitDict.keys()) == 0 and initDictionary == None: self.raiseAnError(Exception,'No initialization parameters have been provided!!')
    # grep the keys that have been read
    readKeys = []
    initDict = initDictionary if initDictionary != None else {}
    computeCells = bool(initDict['computeCells']) if 'computeCells' in initDict.keys() else False
    if  len(self.gridInitDict.keys()) != 0: readKeys = self.gridInitDict.keys()
    if initDict != None:
      if type(initDict).__name__ != "dict": self.raiseAnError(Exception,'The in argument is not a dictionary!')
    if "dimensionNames" not in initDict.keys()+readKeys: self.raiseAnError(Exception,'"dimensionNames" key is not present in the initialization dictionary!')
    if "lowerBounds" not in initDict.keys()+readKeys: self.raiseAnError(Exception,'"lowerBounds" key is not present in the initialization dictionary')
    if "lowerBounds" not in readKeys:
      if type(initDict["lowerBounds"]).__name__ != "dict": self.raiseAnError(Exception,'The lowerBounds entry is not a dictionary')
    if "upperBounds" not in initDict.keys()+readKeys: self.raiseAnError(Exception,'"upperBounds" key is not present in the initialization dictionary')
    if "upperBounds" not in readKeys:
      if type(initDict["upperBounds"]).__name__ != "dict": self.raiseAnError(Exception,'The upperBounds entry is not a dictionary')
    if "transformationMethods" in initDict.keys(): self.gridContainer['transformationMethods'] = initDict["transformationMethods"]
    self.nVar                            = len(self.gridInitDict["dimensionNames"]) if "dimensionNames" in self.gridInitDict.keys() else len(initDict["dimensionNames"])
    self.gridContainer['dimensionNames'] = self.gridInitDict["dimensionNames"] if "dimensionNames" in self.gridInitDict.keys() else initDict["dimensionNames"]
    upperkeys                            = self.gridInitDict["upperBounds"].keys() if "upperBounds" in self.gridInitDict.keys() else initDict["upperBounds"  ].keys()
    lowerkeys                            = self.gridInitDict["lowerBounds"].keys() if "lowerBounds" in self.gridInitDict.keys() else initDict["lowerBounds"  ].keys()
    self.gridContainer['dimensionNames'].sort()
    upperkeys.sort()
    lowerkeys.sort()
    if upperkeys != lowerkeys != self.gridContainer['dimensionNames']: self.raiseAnError(Exception,'dimensionNames and keys in upperBounds and lowerBounds dictionaries do not correspond')
    self.gridContainer['bounds']["upperBounds" ] = self.gridInitDict["upperBounds"] if "upperBounds" in self.gridInitDict.keys() else initDict["upperBounds"]
    self.gridContainer['bounds']["lowerBounds"]  = self.gridInitDict["lowerBounds"] if "lowerBounds" in self.gridInitDict.keys() else initDict["lowerBounds"]
    if "volumetricRatio" not in initDict.keys() and "stepLenght" not in initDict.keys()+readKeys: self.raiseAnError(Exception,'"volumetricRatio" or "stepLenght" key is not present in the initialization dictionary')
    if "volumetricRatio"  in initDict.keys() and "stepLenght" in initDict.keys()+readKeys: self.raiseAWarning('"volumetricRatio" and "stepLenght" keys are both present! the "volumetricRatio" has priority!')
    if "volumetricRatio" in initDict.keys():
      self.volumetricRatio = initDict["volumetricRatio"]
      # build the step size in 0-1 range such as the differential volume is equal to the tolerance
      stepLenght, ratioRelative = [], self.volumetricRatio**(1./float(self.nVar))
      for varId in range(len(self.gridContainer['dimensionNames'])): stepLenght.append([ratioRelative*(self.gridContainer['bounds']["upperBounds" ][self.gridContainer['dimensionNames'][varId]] - self.gridContainer['bounds']["lowerBounds" ][self.gridContainer['dimensionNames'][varId]])])
    else:
      if "stepLenght" not in readKeys:
        if type(initDict["stepLenght"]).__name__ != "dict": self.raiseAnError(Exception,'The stepLenght entry is not a dictionary')
      stepLenght = []
      for dimName in self.gridContainer['dimensionNames']: stepLenght.append(initDict["stepLenght"][dimName] if  "stepLenght" not in readKeys else self.gridInitDict["stepLenght"][dimName])
    #moving forward building all the information set
    pointByVar                                   = [None]*self.nVar  #list storing the number of point by cooridnate
    #building the grid point coordinates
    for varId, varName in enumerate(self.gridContainer['dimensionNames']):
      if len(stepLenght[varId]) == 1:
        # equally spaced or volumetriRatio. (the substruction of stepLenght*10e-3 is only to avoid that for roundoff error, the dummy upperbound is included in the mesh)
        if self.volumetricRatio != None: self.gridContainer['gridVectors'][varName] = np.arange(self.gridContainer['bounds']["lowerBounds"][varName],self.gridContainer['bounds']["upperBounds" ][varName],stepLenght[varId][-1])
        else                           : self.gridContainer['gridVectors'][varName] = np.concatenate((np.arange(self.gridContainer['bounds']["lowerBounds"][varName],self.gridContainer['bounds']["upperBounds" ][varName]-self.gridContainer['bounds']["upperBounds" ][varName]*1.e-3,stepLenght[varId][-1]),np.atleast_1d(self.gridContainer['bounds']["upperBounds" ][varName])))
      else:
        # custom grid
        # it is not very efficient, but this approach is only for custom grids => limited number of discretizations
        gridMesh = [self.gridContainer['bounds']["lowerBounds"][varName]]
        for stepLenghti in stepLenght[varId]: gridMesh.append(round(gridMesh[-1],14)+round(stepLenghti,14))
        self.gridContainer['gridVectors'][varName] = np.asarray(gridMesh)
      if not compare(round(max(self.gridContainer['gridVectors'][varName]),14), round(self.gridContainer['bounds']["upperBounds" ][varName],14)) and self.volumetricRatio == None: self.raiseAnError(IOError,"the maximum value in the grid is bigger that upperBound! upperBound: "+
                                                                                                                                      str(self.gridContainer['bounds']["upperBounds" ][varName]) +
                                                                                                                                      " < maxValue in grid: "+str(max(self.gridContainer['gridVectors'][varName])))
      if not compare(round(min(self.gridContainer['gridVectors'][varName]),14),round(self.gridContainer['bounds']["lowerBounds" ][varName],14)): self.raiseAnError(IOError,"the minimum value in the grid is lower that lowerBound! lowerBound: "+
                                                                                                                                      str(self.gridContainer['bounds']["lowerBounds"][varName]) +
                                                                                                                                      " > minValue in grid: "+str(min(self.gridContainer['gridVectors'][varName])))
      if self.gridContainer['transformationMethods'] != None:
        if varName in self.gridContainer['transformationMethods'].keys():
          self.gridContainer['gridVectors'][varName] = np.asarray([self.gridContainer['transformationMethods'][varName](coor) for coor in self.gridContainer['gridVectors'][varName]])
      pointByVar[varId]                               = np.shape(self.gridContainer['gridVectors'][varName])[0]
    self.gridContainer['gridShape']                 = tuple   (pointByVar)                            # tuple of the grid shape
    self.gridContainer['gridLenght']                = np.prod (pointByVar)                            # total number of point on the grid
    self.gridContainer['gridMatrix']                = np.zeros(self.gridContainer['gridShape'])       # grid where the values of the goalfunction are stored
    self.gridContainer['gridCoorShape']             = tuple   (pointByVar+[self.nVar])                # shape of the matrix containing all coordinate of all points in the grid
    self.gridContainer['gridCoord']                 = np.zeros(self.gridContainer['gridCoorShape'])   # the matrix containing all coordinate of all points in the grid
    self.uniqueCellNumber                           = np.prod ([element-1 for element in pointByVar]) # number of unique cells
    #filling the coordinate on the grid
    self.gridIterator = np.nditer(self.gridContainer['gridCoord'],flags=['multi_index'])
    gridIterCells = np.nditer(np.zeros(shape=(2,)*self.nVar,dtype=int),flags=['multi_index'])
  
    origin, pp, cellID = [-1]*self.nVar, [element - 1 for element in pointByVar], int(initDict['startingCellId']) if 'startingCellId' in  initDict.keys() else 1
    while not self.gridIterator.finished:
      coordinateID                          = self.gridIterator.multi_index[-1]
      dimName                               = self.gridContainer['dimensionNames'][coordinateID]
      valuePosition                         = self.gridIterator.multi_index[coordinateID]
      self.gridContainer['gridCoord'][self.gridIterator.multi_index] = self.gridContainer['gridVectors'][dimName][valuePosition]
      if computeCells:
        if all(np.greater(pp,list(self.gridIterator.multi_index[:-1]))) and list(self.gridIterator.multi_index[:-1]) != origin:
          self.gridContainer['cellIDs'][cellID] = []
          origin = list(self.gridIterator.multi_index[:-1])
          while not gridIterCells.finished:
            vertex = tuple(np.array(origin)+gridIterCells.multi_index)
            self.gridContainer['cellIDs'][cellID].append(vertex)
            try   : self.gridContainer['vertexToCellIds'][vertex].append(cellID)
            except: self.gridContainer['vertexToCellIds'][vertex] = [cellID]
            gridIterCells.iternext()
          gridIterCells.reset() 
          cellID+=1
      self.gridIterator.iternext()  
    if len(self.gridContainer['cellIDs'].keys()) != self.uniqueCellNumber and computeCells: self.raiseAnError(IOError, "number of cells detected != than the number of actual cells!")
    self.resetIterator()
    self.raiseAMessage("Grid initialized...")

  def retrieveCellIds(self,listOfPoints):
    """
     This method is aimed to retrieve the cell IDs that are contained in certain bounaried provided as list of points
     @ In, listOfPoints, list, list of points that represent the boundaries ([listOfFirstBound, listOfSecondBound])
    """
    cellIds = []
    for cntb, bound in enumerate(listOfPoints):
      cellIds.append([])
      for point in bound: cellIds[cntb].extend(self.gridContainer['vertexToCellIds'][tuple(point)])
      if cntb == 0: previousSet = set(cellIds[cntb])
      previousSet = set(previousSet).intersection(cellIds[cntb])
    return list(previousSet)

  def returnGridAsArrayOfCoordinates(self):
    """
    Return the grid as an array of coordinates
    """
    return self.__returnCoordinatesReshaped((self.gridContainer['gridLenght'],self.nVar))

  def __returnCoordinatesReshaped(self,newShape):
    """
     Method to return the grid Coordinates reshaped with respect an in Shape
     @ In, newShape, tuple, newer shape
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

  def returnIteratorIndexes(self,returnDict = True):
    """
    Return the iterator indexes
    @ In, boolean,returnDict if true, the Indexes are returned in dictionary format
    @ Out, tuple or dictionary
    """
    currentIndexes = self.gridIterator.multi_index
    if not returnDict: return currentIndexes
    coordinates = {}
    for cnt, key in enumerate(self.gridContainer['dimensionNames']): coordinates[key] = currentIndexes[cnt]
    return coordinates

  def returnIteratorIndexesFromIndex(self, listOfIndexes):
    """
    Return internal iterator indexes from list of coordinates in the list
    @ In, list,listOfIndexes, list of grid coordinates
    @ Out, dictionary
    """
    coordinates = {}
    for cnt, key in enumerate(self.gridContainer['dimensionNames']): coordinates[key] = listOfIndexes[cnt]
    return coordinates

  def returnShiftedCoordinate(self,coordinates,shiftingSteps):
    """
    Method to return the coordinate that is a # shiftingStep away from the input coordinate
    For example, if 1D grid= {'dimName':[1,2,3,4]}, coordinate is 3 and  shiftingStep is -2,
    the returned coordinate will be 1
    @ In,  dict, coordinates, dictionary of coordinates. {'dimName1':startingCoordinate1,dimName2:startingCoordinate2,...}
    @ In,  dict, shiftingSteps, dict of shifiting steps. {'dimName1':shiftingStep1,dimName2:shiftingStep2,...}
    @ Out, dict, outputCoordinates, dictionary of shifted coordinates' values {dimName:value1,...}
    """
    outputCoordinates = {}
    # create multiindex
    multiindex = []
    for varName in self.gridContainer['dimensionNames']:
      if varName in coordinates.keys() and varName in shiftingSteps.keys()      : multiindex.append(coordinates[varName] + shiftingSteps[varName])
      elif varName in coordinates.keys() and not varName in shiftingSteps.keys(): multiindex.append(coordinates[varName])
      else                                                                      : multiindex.append(0)
    outputCoors = self.returnCoordinateFromIndex(multiindex,returnDict=True)
    for varName in shiftingSteps.keys(): outputCoordinates[varName] = outputCoors[varName]
    return outputCoordinates

  def returnPointAndAdvanceIterator(self, returnDict=False, recastMethods={}):
    """
    Method to return a point in the grid. This method will return the coordinates of the point to which the iterator is pointing
    In addition, it advances the iterator in order to point to the following coordinate
    @ In, boolean, optional, returnDict, flag to request the output in dictionary format or not.
                               if True a dict ( {dimName1:coordinate1,dimName2:coordinate2,etc} is returned
                               if False a tuple is riturned (coordinate1,coordinate2,etc
    @ In, dict, optional, recastMethods, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
    @ Out, tuple, coordinate, tuple containing the coordinates
    """
    if not self.gridIterator.finished:
      coordinates = self.returnCoordinateFromIndex(self.gridIterator.multi_index,returnDict,recastMethods)
      for _ in range(self.nVar): self.gridIterator.iternext()
    else: coordinates = None
    return coordinates

  def returnCoordinateFromIndex(self, multiDimIndex, returnDict=False, recastMethods={}):
    """
    Method to return a point in the grid. This method will return the coordinates of the point is requested by multiDimIndex
    In addition, it advances the iterator in order to point to the following coordinate
    @ In, tuple, multiDimIndex, tuple containing the Id of the point needs to be returned (e.g. 3 dim grid,  (xID,yID,zID))
    @ In, boolean, optional, returnDict, flag to request the output in dictionary format or not.
                                         if True a dict ( {dimName1:coordinate1,dimName2:coordinate2,etc} is returned
                                         if False a tuple is riturned (coordinate1,coordinate2,etc)
    @ In, dict, optional, recastMethods, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
    @ Out, tuple or dict, coordinate, tuple containing the coordinates
    """

    coordinates = [None]*self.nVar if returnDict == False else {}
    for cnt, key in enumerate(self.gridContainer['dimensionNames']):
      vvkey = cnt if not returnDict else key
      # if out of bound, we set the coordinate to maxsize
      if multiDimIndex[cnt] < 0: coordinates[vvkey] = -sys.maxsize
      elif multiDimIndex[cnt] > len(self.gridContainer['gridVectors'][key])-1: coordinates[vvkey] = sys.maxsize
      else:
        if key in recastMethods.keys(): coordinates[vvkey] = recastMethods[key][0](self.gridContainer['gridVectors'][key][multiDimIndex[cnt]],*recastMethods[key][1] if len(recastMethods[key]) > 1 else [])
        else                          : coordinates[vvkey] = self.gridContainer['gridVectors'][key][multiDimIndex[cnt]]

    if not returnDict: coordinates = tuple(coordinates)
    return coordinates

class MultiGridEntity(GridBase):
  """
    This class is dedicated to the creation and handling of N-Dimensional Grid.
    In addition, it handles an hierarchical multi-grid approach (creating a mapping from coarse and finer grids in
    an adaptive meshing approach). The strategy for mesh(grid) refining is driven from outside.
  """
  def __init__(self,messageHandler):
    """
      Constructor
    """
    GridBase.__init__(self, messageHandler)
    self.multiGridActivated     = False                   # boolean flag to check if the multigrid approach has been activated
    self.subGridVolumetricRatio = None                    # initial subgrid volumetric ratio
    node = ETS.Node("InitialGrid")
    node.add("grid",returnInstance("GridEntity",self.messageHandler))
    node.add("level","1")
    self.grid               = ETS.NodeTree(node)         # grid hierarchical Container
    self.multiGridIterator  = [node.get("level"), None]  # multi grid iterator [first position is the level ID, the second it the multi-index]   
    self.mappingLevelName   = {'1':'InitialGrid'}        # mapping between grid level and node name  
   
  def __len__(self):
    """
    Overload __len__ method. It iterates over all the hierarchical structure of grids
    @ In, None
    @ Out, totalLenght, integer, total number of nodes
    """
    totalLenght = 0
    for node in self.grid.iter():
      totalLenght += len(node.get('grid'))
    return totalLenght

  def _readMoreXml(self,xmlNode,dimensionTags=None,messageHandler=None,dimTagsPrefix=None):
    """
     XML reader for the Multi-grid statement.
     @ In, ETree object, xml node from where the info need to be retrieved
     @ In, dimensionTag, optional, list, names of the tag that represents the grid dimensions
     @ In, dimTagsPrefix, optional, dict, eventual prefix to use for defining the dimName
     @ Out, None
    """
    self.grid.getrootnode().get("grid")._readMoreXml(xmlNode,dimensionTags,messageHandler,dimTagsPrefix)

  def initialize(self,initDictionary=None):
    """
    Initialization method. The full grid is created in this method.
    @ In, initDictionary, dictionary, optional, dictionary of input arguments needed to create a Grid:
      {dimensionNames:[]}, required,list of axis names (dimensions' IDs)
      {lowerBounds:{}}, required, dictionary of lower bounds for each dimension
      {upperBounds:{}}, required, dictionary of upper bounds for each dimension
      {volumetriRatio:float or stepLenght:dict}, required, p.u. volumetric ratio of the grid or dictionary of stepLenghts ({'varName:list,etc'}
      {subGridVolumetricRatio:float}, optional, p.u. volumetric ratio of the subGrid, default  1.e5
      {computeCells:bool},optional, boolean to ask to compute the cells ids and verteces coordinates, default = False
      {transformationMethods:{}}, optional, dictionary of methods to transform p.u. step size into a transformed system of coordinate
      !!!!!!
      if the self.gridInitDict is != None (info read from XML node), this method looks for the information in that dictionary first and after it checks the initDict object
      !!!!!!
    """
    self.grid.getrootnode().get("grid").initialize(initDictionary)
    if initDictionary != None: self.subGridVolumetricRatio = float(initDictionary['subGridVolumetricRatio']) if 'subGridVolumetricRatio' in initDictionary.keys() else 1.e-5
    self.nVar = self.grid.getrootnode().get("grid").nVar
    self.multiGridIterator[1] = self.grid.getrootnode().get("grid").returnIteratorIndexes(False)

  def retrieveCellIds(self,listOfPoints):
    """
     This method is aimed to retrieve the cell IDs that are contained in certain bounaried provided as list of points
     @ In, listOfPoints, list, list of points that represent the boundaries ([listOfFirstBound, listOfSecondBound])
    """
    setOfCells = []
    for node in self.grid.iter():
      setOfCells.extend(node.get('grid').retrieveCellIds(listOfPoints))
    return setOfCells
  
  def refineGrid(self,refineDict):
    """
    aaaaaa
    """
    cellIdsToRefine, didWeFoundCells = refineDict['cellIDs'], dict.fromkeys(refineDict['cellIDs'], False)
    for node in self.grid.iter():
      level, nodeCellIds = node.get("level"), node.get("grid").returnParameter('cellIDs').keys()
      foundCells = set(nodeCellIds).intersection(cellIdsToRefine)
      if len(foundCells) > 0:
        for idcnt, fcellId in enumerate(foundCells):
          didWeFoundCells[fcellId] = True
      
      
      self.grid.getrootnode().get("grid").initialize(initDictionary)
  
  def returnGridAsArrayOfCoordinates(self):
    """
    Return the grid as an array of coordinates
    @ In, None
    @ Out, fullReshapedCoordinates, ndarray, numpy array containing all the coordinates shaped as (fullGridLenght,self.nVar)
    """
    fullReshapedCoordinates = np.zeros((0,self.nVar))
    for node in self.grid.iter():
      fullReshapedCoordinates = np.concatenate((fullReshapedCoordinates,node.get('grid').returnGridAsArrayOfCoordinates()))                  
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

  def returnIteratorIndexes(self,returnDict = True):
    """
    Return the iterator current indexes
    @ In, boolean,returnDict if true, the Indexes are returned in dictionary format
    @ Out, tuple or dictionary
    """
    node = self.grid.find(self.mappingLevelName[self.multiGridIterator[0]])
    return node.get('grid').returnIteratorIndexes(returnDict)

  def returnIteratorIndexesFromIndex(self, Indexes):
    """
    Return internal iterator indexes from list of coordinates in the list
    @ In, Indexes, list or tuple, if tuple -> tuple[0] multi-grid level, tuple[1] list of grid coordinates
                                  if list  -> list of grid coordinates. The multi-grid level is gonna be taken from self.multiGridIterator
    @ Out, dictionary
    """
    if   type(Indexes) == tuple : level, listOfIndexes = Indexes[0], Indexes[1]
    elif type(Indexes) == list  : level, listOfIndexes = self.multiGridIterator[0], Indexes
    else                        : self.raiseAnError(Exception,"returnIteratorIndexesFromIndex method accepts a list or tuple only!")
    node = self.grid.find(self.mappingLevelName[level])
    return node.get('grid').returnIteratorIndexesFromIndex(listOfIndexes)

  def returnShiftedCoordinate(self,coords,shiftingSteps):
    """
    Method to return the coordinate that is a # shiftingStep away from the input coordinate
    For example, if 1D grid= {'dimName':[1,2,3,4]}, coordinate is 3 and  shiftingStep is -2,
    the returned coordinate will be 1
    @ In,  dict or tuple, coords, if  dict  -> dictionary of coordinates. {'dimName1':startingCoordinate1,dimName2:startingCoordinate2,...}.The multi-grid level is gonna be taken from self.multiGridIterator
                                  if  tuple -> tuple[0] multi-grid level, tuple[1] dictionary of coordinates. {'dimName1':startingCoordinate1,dimName2:startingCoordinate2,...}
    @ In,  dict, shiftingSteps, dict of shifiting steps. {'dimName1':shiftingStep1,dimName2:shiftingStep2,...}
    @ Out, dict, outputCoordinates, dictionary of shifted coordinates' values {dimName:value1,...}
    """
    if   type(coords) == tuple  : level, coordinates = coords[0], coords[1]
    elif type(coords) == dict   : level, coordinates = self.multiGridIterator[0], coords
    else                        : self.raiseAnError(Exception,"returnShiftedCoordinate method accepts a coords or tuple only!")    
    node = self.grid.find(self.mappingLevelName[level])
    return node.get('grid').returnShiftedCoordinate(coordinates,shiftingSteps)

  def returnPointAndAdvanceIterator(self, returnDict=False, recastMethods={}):
    """
    Method to return a point in the grid. This method will return the coordinates of the point to which the iterator is pointing
    In addition, it advances the iterator in order to point to the following coordinate
    @ In, boolean, optional, returnDict, flag to request the output in dictionary format or not.
                               if True a dict ( {dimName1:coordinate1,dimName2:coordinate2,etc} is returned
                               if False a tuple is riturned (coordinate1,coordinate2,etc
    @ In, dict, optional, recastMethods, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
    @ Out, tuple, coordinate, tuple containing the coordinates
    """
    startingNode = self.grid.find(self.mappingLevelName[self.multiGridIterator[0]])
    coordinates = None
    for node in startingNode.iter():
      subGrid =  node.get('grid')
      if not subGrid.gridIterator.finished:
        coordinates = subGrid.returnCoordinateFromIndex(subGrid.gridIterator.multi_index,returnDict,recastMethods)
        for _ in range(self.nVar): subGrid.gridIterator.iternext()
        break
      self.multiGridIterator[0], self.multiGridIterator[1] = node.get("level"), node.get("grid").returnIteratorIndexes(False)
    return coordinates

  def returnCoordinateFromIndex(self, multiDimNDIndex, returnDict=False, recastMethods={}):
    """
    Method to return a point in the grid. This method will return the coordinates of the point is requested by multiDimIndex
    In addition, it advances the iterator in order to point to the following coordinate
    @ In, tuple, multiDimNDIndex, tuple containing the Id of the point needs to be returned (e.g. 3 dim grid,  (xID,yID,zID))
    @ In, boolean, optional, returnDict, flag to request the output in dictionary format or not.
                                         if True a dict ( {dimName1:coordinate1,dimName2:coordinate2,etc} is returned
                                         if False a tuple is riturned (coordinate1,coordinate2,etc)
    @ In, dict, optional, recastMethods, dictionary containing the methods that need to be used for trasforming the coordinates
                                         ex. {'dimName1':[methodToTransformCoordinate,*args]}
    @ Out, tuple or dict, coordinate, tuple containing the coordinates
    """
    try: 
      multiDimNDIndex[0]/1
      level, multiDimIndex = self.multiGridIterator[0], multiDimNDIndex
    except TypeError: 
      level, multiDimIndex = multiDimNDIndex[0], multiDimNDIndex[1]
    node = self.grid.find(self.mappingLevelName[level])
    return node.get('grid').returnCoordinateFromIndex(multiDimIndex, returnDict, recastMethods)

"""
 Internal Factory of Classes
"""
__base                             = 'GridEntities'
__interFaceDict                    = {}
__interFaceDict['GridEntity'     ] = GridEntity
__interFaceDict['MultiGridEntity'] = MultiGridEntity
__knownTypes                       = __interFaceDict.keys()

def knownTypes():
  """
   Method to return the types known by this module
   @ In, None
   @ Out, __knownTypes, dict, dictionary of known types (e.g. [GridEntity, MultiGridEntity, etc.])
  """
  return __knownTypes

def returnInstance(Type,caller,messageHandler=None):
  """
   Method to return an instance of a class defined in this module
   @ In, Type, string, Class name (e.g. GridEntity)
   @ In, caller, instance, instance of the caller object
   @ In, messageHandler, optional instance, instance of the messageHandler system 
   @ Out, __interFaceDict[Type], instance, instance of the requested class
  """
  try: return __interFaceDict[Type](messageHandler)
  except KeyError: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
