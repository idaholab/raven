'''
Created on Mar 30, 2015

@author: alfoa
'''

#External Modules------------------------------------------------------------------------------------
#import itertools
import numpy as np
from scipy.interpolate import interp1d
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import UreturnPrintTag,UreturnPrintPostTag,partialEval
from BaseClasses import BaseType
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
    
  def __init__(self):
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

  def _readMoreXml(self,xmlNode,dimensionTags=None):
    """
     XML reader for the grid statement. 
     @ In, ETree object, xml node from where the info need to be retrieved 
     @ In, dimensionTag, optional, list, names of the tag that represents the grid dimensions
     @ Out, None
    """
    self.gridInitDict = {'dimensionNames':[],'lowerBounds':{},'upperBounds':{},'stepLenght':{}}
    gridInfo = {}
    for child in xmlNode:
      dimName = None
      if dimensionTags != None:
        if child.tag in dimensionTags: dimName = child.attrib['name']    
      if child.tag == "grid":
        if dimName == None: dimName = str(len(self.gridInitDict['dimensionNames'])+1)
        gridStruct, gridName = self._fillGrid(child)
        if child.tag != 'global_grid': self.gridInitDict['dimensionNames'].append(dimName)
        else: 
          if gridName == None: self.raiseAnError(IOError,'grid defined in global_grid block must have the attribute "name"!')
          dimName = child.tag + ':' + gridName
        gridInfo[dimName] = gridStruct
      # to be removed when better strategy for NDimensional is found
      readGrid = True
      for childChild in child:
        if 'dim' in childChild.attrib.keys():
          readGrid = False
          if partialEval(childChild.attrib['dim']) == 1: readGrid = True
          break
      # end to be removed
      for childChild in child:
        if childChild.tag =='grid' and readGrid:
          gridStruct, gridName = self._fillGrid(childChild)
          if dimName == None: dimName = str(len(self.gridInitDict['dimensionNames'])+1)
          if child.tag != 'global_grid': self.gridInitDict['dimensionNames'].append(dimName)
          else: 
            if gridName == None: self.raiseAnError(IOError,'grid defined in global_grid block must have the attribute "name"!')
            dimName = child.tag + ':' + gridName
          gridInfo[dimName] = gridStruct
    #check for global_grid type of structure
    globalGrids = {}
    for key in gridInfo.keys():
      splitted = key.split(":")
      if splitted[0].strip() == 'global_grid': globalGrids[splitted[1]] = gridInfo.pop(key)
    for key in gridInfo.keys():
      if gridInfo[key][0].strip() == 'global_grid':
        if gridInfo[key][-1].strip() not in globalGrids.keys(): self.raiseAnError(IOError,'global grid for dimension named '+key+'has not been found!')
        gridInfo[key] = globalGrids[gridInfo[key][-1].strip()]
      self.gridInitDict['lowerBounds'           ][key] = min(gridInfo[key][-1])
      self.gridInitDict['upperBounds'           ][key] = max(gridInfo[key][-1])
      self.gridInitDict['stepLenght'            ][key] = [gridInfo[key][-1][k+1] - gridInfo[key][-1][k] for k in range(len(gridInfo[key][-1])-1)] if gridInfo[key][1] == 'custom' else [gridInfo[key][-1][1] - gridInfo[key][-1][0]]
    self.gridContainer['gridInfo'] = gridInfo 
              




  def _fillGrid(self,child):
    constrType = None  
    if 'construction' in child.attrib.keys(): constrType = child.attrib['construction']
    nameGrid = None
    if constrType in ['custom','equal']:
      bounds = [partialEval(element) for element in child.text.split()]
      bounds.sort()
      lower, upper = min(bounds), max(bounds)
      if 'name' in child.attrib.keys(): nameGrid = child.attrib['name']
    if constrType == 'custom': return (child.attrib['type'],constrType,bounds),nameGrid
    elif constrType == 'equal':
      if len(bounds) != 2: self.raiseAnError(IOError,'body of grid XML node needs to contain 2 values (lower and upper bounds)')
      if 'steps' not in child.attrib.keys(): self.raiseAnError(IOError,'the attribute step needs to be inputted when "construction" attribute == equal!')
      return (child.attrib['type'],constrType,np.linspace(lower,upper,partialEval(child.attrib['steps'])+1)),nameGrid
    elif child.attrib['type'] == 'global_grid': return (child.attrib['type'],constrType,child.text),nameGrid
    else: self.raiseAnError(IOError,'construction type unknown! Got: ' + str(constrType))

  def initialize(self,initDict):
    """
    Initialization method. The full grid is created in this method.
    @ In, dictionary, dictionary of input arguments needed to create a Grid:
      {dimensionNames:[]}, required,list of axis names (dimensions' IDs)
      {lowerBounds:{}}, required, dictionary of lower bounds for each dimension 
      {upperBounds:{}}, required, dictionary of upper bounds for each dimension
      {volumetriRatio:float}, required, p.u. volumetric ratio of the grid 
      {transformationMethods:{}}, optional, dictionary of methods to transform p.u. step size into a transformed system of coordinate
    """
    if type(initDict).__name__ != "dict": self.raiseAnError(Exception,'The in argument is not a dictionary!')
    if "dimensionNames" not in initDict.keys(): self.raiseAnError(Exception,'"dimensionNames" key is not present in the initialization dictionary!')
    if "lowerBounds" not in initDict.keys(): self.raiseAnError(Exception,'"lowerBounds" key is not present in the initialization dictionary')
    if type(initDict["lowerBounds"]).__name__ != "dict": self.raiseAnError(Exception,'The lowerBounds entry is not a dictionary')
    if "upperBounds" not in initDict.keys(): self.raiseAnError(Exception,'"upperBounds" key is not present in the initialization dictionary')
    if type(initDict["upperBounds"]).__name__ != "dict": self.raiseAnError(Exception,'The upperBounds entry is not a dictionary')
    if "transformationMethods" in initDict.keys(): self.gridContainer['transformationMethods'] = initDict["transformationMethods"]
    self.nVar                            = len(initDict["dimensionNames"])
    self.gridContainer['dimensionNames'] = initDict["dimensionNames"]
    upperkeys                            = initDict["upperBounds"   ].keys()
    lowerkeys                            = initDict["lowerBounds"   ].keys()
    self.gridContainer['dimensionNames'].sort()
    upperkeys.sort()
    lowerkeys.sort()
    if upperkeys != lowerkeys != self.gridContainer['dimensionNames']: self.raiseAnError(Exception,'dimensionNames and keys in upperBounds and lowerBounds dictionaries do not correspond')
    self.gridContainer['bounds']["upperBounds" ] = initDict["upperBounds"]
    self.gridContainer['bounds']["lowerBounds"]  = initDict["lowerBounds"]
    if "volumetricRatio" not in initDict.keys() and "stepLenght" not in initDict.keys(): self.raiseAnError(Exception,'"volumetricRatio" or "stepLenght" key is not present in the initialization dictionary')
    if "volumetricRatio" in initDict.keys():
      self.volumetricRatio                         = initDict["volumetricRatio"]
      stepLenght                                   = [self.volumetricRatio**(1./float(self.nVar))]*self.nVar # build the step size in 0-1 range such as the differential volume is equal to the tolerance
    else:
      if type(initDict["stepLenght"]).__name__ != "dict": self.raiseAnError(Exception,'The stepLenght entry is not a dictionary')
      stepLenght = []
      for dimName in self.gridContainer['dimensionNames']: stepLenght.append(initDict["stepLenght"][dimName])
      self.volumetricRatio = np.sum(stepLenght)**(1/self.nVar) # in this case it is an average => it "represents" the average volumentric ratio...not too much sense. Andrea
    #here we build lambda function to return the coordinate of the grid point
    stepParam                                    = lambda x: [stepLenght[self.gridContainer['dimensionNames'].index(x)]*(self.gridContainer['bounds']["upperBounds" ][x]-self.gridContainer['bounds']["lowerBounds"][x]), 
                                                                          self.gridContainer['bounds']["lowerBounds"][x], self.gridContainer['bounds']["upperBounds" ][x]]
    #moving forward building all the information set
    pointByVar                                   = [None]*self.nVar  #list storing the number of point by cooridnate
    #building the grid point coordinates
    for varId, varName in enumerate(self.gridContainer['dimensionNames']):
      [stpLenght, start, end]     = stepParam(varName)
      start                      += 0.5*stpLenght
      if self.gridContainer['transformationMethods'] != None: 
        self.self.gridContainer['gridVectors'][varName] = np.asarray([self.gridContainer['transformationMethods'][varName](coor) for coor in  np.arange(start,end,stpLenght)])
      else:
        self.self.gridContainer['gridVectors'][varName] = np.arange(start,end,stpLenght)
      pointByVar[varId]                               = np.shape(self.self.gridContainer['gridVectors'][varName])[0]
    self.gridContainer['gridShape']                 = tuple   (pointByVar)          # tuple of the grid shape
    self.gridContainer['gridLenght']                = np.prod (pointByVar)          # total number of point on the grid
    self.gridContainer['gridMatrix']                = np.zeros(self.gridContainer['gridShape'])      # grid where the values of the goalfunction are stored
    self.gridContainer['gridCoorShape']             = tuple(pointByVar+[self.nVar]) # shape of the matrix containing all coordinate of all points in the grid
    self.gridContainer['gridCoord']                 = np.zeros(self.gridContainer['gridCoorShape'])  # the matrix containing all coordinate of all points in the grid
    self.uniqueCellNumber                           = self.gridContainer['gridLenght']/2**self.nVar
    #filling the coordinate on the grid
    self.gridIterator = np.nditer(self.gridContainer['gridCoord'],flags=['multi_index'])
    while not self.gridIterator.finished:
      coordinateID                          = self.gridIterator.multi_index[-1]
      dimName                               = self.gridContainer['dimensionNames'][coordinateID]
      valuePosition                         = self.gridIterator.multi_index[coordinateID]
      self.gridContainer['gridCoord'][self.gridIterator.multi_index] = self.self.gridContainer['gridVectors'][dimName][valuePosition]
      self.gridIterator.iternext()
    self.resetIterator()

  def returnParameter(self,parameterName):
    """
    Method to return one of the initialization parameters
    @ In, string, parameterName, name of the parameter to be returned
    @ Out, object, pointer to the requested parameter
    """
    if parameterName not in self.gridContainer.keys(): self.raiseAnError(Exception,'parameter '+parameterName+'unknown among ones in GridEntity class.')
    return self.gridContainer[parameterName]

  def updateParameter(self,parameterName, newValue):
    """
    Method to update one of the initialization parameters
    @ In, string, parameterName, name of the parameter to be updated 
    @ Out, None
    """
    self.gridContainer[parameterName] = newValue

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
  
  def returnPointAndAdvanceIterator(self):
    """
    Method to return a point in the grid. This method will return the coordinates of the point to which the iterator is pointing
    In addition, it advances the iterator in order to point to the following coordinate
    @ In, None
    @ Out, tuple, coordinate, tuple containing the coordinates
    """
    if not self.gridIterator.finished:
      coordinate = self.returnCoordinateFromIndex(self.gridIterator.multi_index)
      self.gridIterator.iternext()
    else: coordinate = None
    return coordinate

  def returnCoordinateFromIndex(self,multiDimIndex):
    """
    Method to return a point in the grid. This method will return the coordinates of the point is requested by multiDimIndex
    In addition, it advances the iterator in order to point to the following coordinate
    @ In, tuple, multiDimIndex, tuple containing the Id of the point needs to be returned (e.g. 3 dim grid,  (xID,yID,zID))
    @ Out, tuple, coordinate, tuple containing the coordinates
    """
    coordinate = self.gridContainer['gridCoord'][multiDimIndex]
    return coordinate

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

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
