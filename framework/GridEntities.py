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
    #combinations = list(itertools.product(*precondlistoflist))
  def initialize(self,initDict):
    if type(initDict).__name__ != "dict": raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> The in argument is not a dictionary')
    if "dimensionNames" not in initDict.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> "dimensionNames" key is not present in the initialization dictionary')
    if "lowerBounds" not in initDict.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> "lowerBounds" key is not present in the initialization dictionary')
    if type(initDict["lowerBounds"]).__name__ != "dict": raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> The lowerBounds entry is not a dictionary')
    if "upperBounds" not in initDict.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> "upperBounds" key is not present in the initialization dictionary')
    if type(initDict["upperBounds"]).__name__ != "dict": raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> The upperBounds entry is not a dictionary')
    if "gridTol" not in initDict.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> "gridTol" key is not present in the initialization dictionary')
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
    self.gridTol                                 = initDict["gridTol"    ]
    stepLenght                                   = self.gridTol**(1./float(self.nVar)) # build the step size in 0-1 range such as the differential volume is equal to the tolerance
    #here we build lambda function to return the coordinate of the grid point
    stepParam                                    = lambda x: [stepLenght*(self.gridContainer['bounds']["upperBounds" ][x]-self.gridContainer['bounds']["lowerBounds"][x]), 
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
    localIter = np.nditer(self.gridContainer['gridCoord'],flags=['multi_index'])
    while not localIter.finished:
      coordinateID                          = localIter.multi_index[-1]
      axisName                              = self.gridContainer['dimensionNames'][coordinateID]
      valuePosition                         = localIter.multi_index[coordinateID]
      self.gridContainer['gridCoord'][localIter.multi_index] = self.self.gridContainer['gridVectors'][axisName][valuePosition]
      localIter.iternext()
  
  def returnParameter(self,parameterName):
    if parameterName not in self.gridContainer.keys(): raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> parameter '+parameterName+'unknown among ones in GridEntity class.')
    return self.gridContainer[parameterName]
      
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


