'''
Created on Mar 30, 2015

@author: alfoa
'''
import TreeStructure as TS
import itertools
import numpy as np
from utils import returnPrintTag,returnPrintPostTag

class GridEntity(object):
  """
  Class that defines a Grid in the phase space. This class should be used by all the Classes that need a Grid entity.
  It provides all the methods to create, modify, and handle a grid in the phase space.
  """
  def __init__(self):
    self.printTag                               = returnPrintTag("GRID ENTITY")
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
    if type(initDict).__name__ != "dict": raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> The in argument is not a dictionary')
    if "dimensionNames" not in initDict.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> "dimensionNames" key is not present in the initialization dictionary')
    if "lowerBounds" not in initDict.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> "lowerBounds" key is not present in the initialization dictionary')
    if type(initDict["lowerBounds"]).__name__ != "dict": raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> The lowerBounds entry is not a dictionary')
    if "upperBounds" not in initDict.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> "upperBounds" key is not present in the initialization dictionary')
    if type(initDict["upperBounds"]).__name__ != "dict": raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> The upperBounds entry is not a dictionary')
    if "transformationMethods" in initDict.keys(): self.gridContainer['transformationMethods'] = initDict["transformationMethods"]
    self.nVar                            = len(initDict["dimensionNames"])
    self.gridContainer['dimensionNames'] = initDict["dimensionNames"]
    upperkeys                            = initDict["upperBounds"   ].keys()
    lowerkeys                            = initDict["lowerBounds"   ].keys()
    self.gridContainer['dimensionNames'].sort()
    upperkeys.sort()
    lowerkeys.sort()
    if upperkeys != lowerkeys != self.gridContainer['dimensionNames']: raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> dimensionNames and keys in upperBounds and lowerBounds dictionaries do not correspond')
    self.gridContainer['bounds']["upperBounds" ] = initDict["upperBounds"]
    self.gridContainer['bounds']["lowerBounds"]  = initDict["lowerBounds"]
    if "volumetricRatio" not in initDict.keys() and "stepLenght" not in initDict.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> "volumetricRatio" or "stepLenght" key is not present in the initialization dictionary')
    if "volumetricRatio" in initDict.keys():
      self.volumetricRatio                         = initDict["volumetricRatio"]
      stepLenght                                   = [self.volumetricRatio**(1./float(self.nVar))]*self.nVar # build the step size in 0-1 range such as the differential volume is equal to the tolerance
    else:
      if type(initDict["stepLenght"]).__name__ != "dict": raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> The stepLenght entry is not a dictionary')
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
    if parameterName not in self.gridContainer.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> parameter '+parameterName+'unknown among ones in GridEntity class.')
    return self.gridContainer[parameterName]
  
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


