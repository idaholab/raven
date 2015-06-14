'''
Created on Mar 30, 2015

@author: alfoa
'''

#External Modules------------------------------------------------------------------------------------
#import itertools
import numpy as np
from scipy.interpolate import interp1d
import sys
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import UreturnPrintTag,partialEval,compare
from BaseClasses import BaseType
from MessageHandler import MessageHandler
#import TreeStructure as TS
#Internal Modules End--------------------------------------------------------------------------------


class GridEntity(BaseType):
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
    if messageHandler != None: self.setMessageHandler(messageHandler)
    self.printTag                               = UreturnPrintTag("GRID ENTITY")
    self.gridContainer                          = {}                 # dictionary that contains all the key feature of the grid
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
    self.uniqueCellNumber                       = 0                  # number of unique cells
    self.gridIterator                           = None               # the grid iterator
    self.gridInitDict                           = {}                 # dictionary with initialization grid info from _readMoreXML. If None, the "initialize" method will look for all the information in the in Dictionary

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
#         if dimName == None: dimName = str(len(self.gridInitDict['dimensionNames'])+1)
#         gridStruct, gridName = self._fillGrid(child)
#         if child.tag != 'global_grid': self.gridInitDict['dimensionNames'].append(dimName)
#         else:
#           if gridName == None: self.raiseAnError(IOError,'grid defined in global_grid block must have the attribute "name"!')
#           dimName = child.tag + ':' + gridName
#         gridInfo[dimName] = gridStruct
      for childChild in child:
        if childChild.tag == "grid": gridInfo[self.dimName] = self._readGridStructure(childChild,child)
        if 'dim' in childChild.attrib.keys():
          dimID = str(len(self.gridInitDict['dimensionNames'])+1) if self.dimName == None else self.dimName
          try              : dimInfo[dimID] = [int(childChild.attrib['dim']),None]
          except ValueError: self.raiseAnError(ValueError, "can not convert 'dim' attribute in integer!")
#         if childChild.tag =='grid':
#           gridStruct, gridName = self._fillGrid(childChild)
#           if dimName == None: dimName = str(len(self.gridInitDict['dimensionNames'])+1)
#           if child.tag != 'global_grid': self.gridInitDict['dimensionNames'].append(dimName)
#           else:
#             if gridName == None: self.raiseAnError(IOError,'grid defined in global_grid block must have the attribute "name"!')
#             dimName = child.tag + ':' + gridName
#           gridInfo[dimName] = gridStruct
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
        #  if dimInfo[key] != 1:
        #    self.gridInitDict['dimensionNames'].pop(self.gridInitDict['dimensionNames'].index(key))
        #    #gridInfo.pop(key)
        #    gridInfo[key] = globalGrids[gridInfo[key][-1].strip()]
        #    continue
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
      {transformationMethods:{}}, optional, dictionary of methods to transform p.u. step size into a transformed system of coordinate
      !!!!!!
      if the self.gridInitDict is != None (info read from XML node), this method looks for the information in that dictionary first and after it checks the initDict object
      !!!!!!
    """
    if len(self.gridInitDict.keys()) == 0 and initDictionary == None: self.raiseAnError(Exception,'No initialization parameters have been provided!!')
    # grep the keys that have been read
    readKeys = []
    initDict = initDictionary if initDictionary != None else {}
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
        self.gridContainer['gridVectors'][varName] = np.concatenate((np.arange(self.gridContainer['bounds']["lowerBounds"][varName],self.gridContainer['bounds']["upperBounds" ][varName]-self.gridContainer['bounds']["upperBounds" ][varName]*1.e-3,stepLenght[varId][-1]),np.atleast_1d(self.gridContainer['bounds']["upperBounds" ][varName])))
      else:
        # custom grid
        # it is not very efficient, but this approach is only for custom grids => limited number of discretizations
        gridMesh = [self.gridContainer['bounds']["lowerBounds"][varName]]
        for stepLenghti in stepLenght[varId]: gridMesh.append(round(gridMesh[-1],14)+round(stepLenghti,14))
        self.gridContainer['gridVectors'][varName] = np.asarray(gridMesh)
      if not compare(round(max(self.gridContainer['gridVectors'][varName]),14), round(self.gridContainer['bounds']["upperBounds" ][varName],14)): self.raiseAnError(IOError,"the maximum value in the grid is bigger that upperBound! upperBound: "+
                                                                                                                                      str(self.gridContainer['bounds']["upperBounds" ][varName]) +
                                                                                                                                      " < maxValue in grid: "+str(max(self.gridContainer['gridVectors'][varName])))
      if not compare(round(min(self.gridContainer['gridVectors'][varName]),14),round(self.gridContainer['bounds']["lowerBounds" ][varName],14)): self.raiseAnError(IOError,"the minimum value in the grid is lower that lowerBound! lowerBound: "+
                                                                                                                                      str(self.gridContainer['bounds']["lowerBounds"][varName]) +
                                                                                                                                      " > minValue in grid: "+str(min(self.gridContainer['gridVectors'][varName])))
      if self.gridContainer['transformationMethods'] != None:
        if varName in self.gridContainer['transformationMethods'].keys():
          self.gridContainer['gridVectors'][varName] = np.asarray([self.gridContainer['transformationMethods'][varName](coor) for coor in self.self.gridContainer['gridVectors'][varName]])
      pointByVar[varId]                               = np.shape(self.gridContainer['gridVectors'][varName])[0]
    self.gridContainer['gridShape']                 = tuple   (pointByVar)          # tuple of the grid shape
    self.gridContainer['gridLenght']                = np.prod (pointByVar)          # total number of point on the grid
    self.gridContainer['gridMatrix']                = np.zeros(self.gridContainer['gridShape'])      # grid where the values of the goalfunction are stored
    self.gridContainer['gridCoorShape']             = tuple(pointByVar+[self.nVar])                  # shape of the matrix containing all coordinate of all points in the grid
    self.gridContainer['gridCoord']                 = np.zeros(self.gridContainer['gridCoorShape'])  # the matrix containing all coordinate of all points in the grid
    self.uniqueCellNumber                           = self.gridContainer['gridLenght']/2**self.nVar
    #filling the coordinate on the grid
    self.gridIterator = np.nditer(self.gridContainer['gridCoord'],flags=['multi_index'])
    while not self.gridIterator.finished:
      coordinateID                          = self.gridIterator.multi_index[-1]
      dimName                               = self.gridContainer['dimensionNames'][coordinateID]
      valuePosition                         = self.gridIterator.multi_index[coordinateID]
      self.gridContainer['gridCoord'][self.gridIterator.multi_index] = self.gridContainer['gridVectors'][dimName][valuePosition]
      #print(self.gridIterator.multi_index)
      self.gridIterator.iternext()
    self.resetIterator()

  def returnGridAsArrayOfCoordinates(self):
    """
    Return the grid as an array of coordinates
    """
    return self.returnCoordinatesReshaped((self.gridContainer['gridLenght'],self.nVar))

  def returnCoordinatesReshaped(self,newShape):
    """
     Method to return the grid Coordinates reshaped with respect an in Shape
     @ In, newShape, tuple, newer shape
    """
    returnCoordinates = self.gridContainer['gridCoord']
    returnCoordinates.shape = newShape
    return returnCoordinates

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
    #coordinate = self.gridContainer['gridCoord'][multiDimIndex]
    return coordinates

# class MultiGridEntity(object):
#   '''
#     This class is dedicated to the creation and handling of N-Dimensional Grid.
#     In addition, it handles an hirarchical multi-grid approach (creating a mapping from coarse and finer grids in
#     an adaptive meshing approach
#   '''
#   def __init__(self):
#     '''
#       Constructor
#     '''
#     self.grid = TS.NodeTree(TS.Node("Level-0-grid"))


__base                             = 'GridEntities'
__interFaceDict                    = {}
__interFaceDict['GridEntity'     ] = GridEntity
__interFaceDict['MultiGridEntity'] = GridEntity
__knownTypes                       = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller,messageHandler=None):
  try: return __interFaceDict[Type](messageHandler)
  except KeyError: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
