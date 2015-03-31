'''
Created on July 10, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import sys
import numpy as np
from sklearn import tree
from scipy import spatial
#from scipy import interpolate
from scipy import integrate
import os
from glob import glob
import copy
import Datas
import math
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import mathUtils
#from utils import utils.toString, utils.toBytes, utils.first, utils.returnPrintTag, utils.returnPrintPostTag
from Assembler import Assembler
import SupervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

#def error(*objs):
#  print("ERROR: ", *objs, file=sys.stderr)

'''
  ***************************************
  *  SPECIALIZED PostProcessor CLASSES  *
  ***************************************
'''

class BasePostProcessor(Assembler):
  """"This is the base class for postprocessors"""
  def __init__(self):
    self.type              = self.__class__.__name__  # pp type
    self.name              = self.__class__.__name__  # pp name
    self.assemblerObjects  = {}                       # {MainClassName(e.g.Distributions):[class(e.g.Models),type(e.g.ROM),objectName]}
    self.requiredAssObject = (False,([],[]))          # tuple. utils.first entry boolean flag. True if the XML parser must look for assembler objects;
                                                      # second entry tuple.utils.first entry list of object can be retrieved, second entry multiplicity (-1,-2,-n means optional (max 1 object,2 object, no number limit))
    self.debug             = False
    self.assemblerDict     = {}  # {'class':[['subtype','name',instance]]}

  def initialize(self, runInfo, inputs, initDict) :
    #if 'externalFunction' in initDict.keys(): self.externalFunction = initDict['externalFunction']
    self.inputs           = inputs

  def inputToInternal(self,currentInput): return [(copy.deepcopy(currentInput))]

  def run(self, Input): pass

class SafestPoint(BasePostProcessor):
  '''
  It searches for the probability-weighted safest point inside the space of the system controllable variables
  '''
  def __init__(self):
    BasePostProcessor.__init__(self)
    self.controllableDist = {}                                    #dictionary created upon the .xml input file reading. It stores the distributions for each controllale variable.
    self.nonControllableDist = {}                                 #dictionary created upon the .xml input file reading. It stores the distributions for each non-controllale variable.
    self.controllableGrid = {}                                    #dictionary created upon the .xml input file reading. It stores the grid type ('value' or 'CDF'), the number of steps and the step length for each controllale variable.
    self.nonControllableGrid = {}                                 #dictionary created upon the .xml input file reading. It stores the grid type ('value' or 'CDF'), the number of steps and the step length for each non-controllale variable.
    self.gridInfo = {}                                            #dictionary contaning the grid type ('value' or 'CDF'), the grid construction type ('equal', set by default) and the list of sampled points for each variable.
    self.controllableOrd = []                                     #list contaning the controllable variables' names in the same order as they appear inside the controllable space (self.controllableSpace)
    self.nonControllableOrd = []                                  #list contaning the controllable variables' names in the same order as they appear inside the non-controllable space (self.nonControllableSpace)
    self.surfPointsMatrix = None                                  #2D-matrix containing the coordinates of the points belonging to the failure boundary (coordinates are derived from both the controllable and non-controllable space)
    self.stat = returnInstance('BasicStatistics')                 #instantiation of the 'BasicStatistics' processor, which is used to compute the expected value of the safest point through the coordinates and probability values collected in the 'run' function
    self.stat.what = ['expectedValue']
    self.requiredAssObject = (True,(['Distribution'],['n']))
    self.printTag = utils.returnPrintTag('POSTPROCESSOR SAFESTPOINT')

  def _localGenerateAssembler(self,initDict):
    ''' see generateAssembler method '''
    for varName, distName in self.controllableDist.items():
      if distName not in initDict['Distributions'].keys():
        raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> distribution ' +distName+ ' not found.')
      self.controllableDist[varName] = initDict['Distributions'][distName]
    for varName, distName in self.nonControllableDist.items():
      if distName not in initDict['Distributions'].keys():
        raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> distribution ' +distName+ ' not found.')
      self.nonControllableDist[varName] = initDict['Distributions'][distName]

  def _localReadMoreXML(self,xmlNode):
    for child in xmlNode:
      if child.tag == 'controllable':
        for childChild in child:
          if childChild.tag == 'variable':
            varName = childChild.attrib['name']
            for childChildChild in childChild:
              if childChildChild.tag == 'distribution':
                self.controllableDist[varName] = childChildChild.text
              elif childChildChild.tag == 'grid':
                if 'type' in childChildChild.attrib.keys():
                  if 'steps' in childChildChild.attrib.keys():
                    self.controllableGrid[varName] = (childChildChild.attrib['type'], int(childChildChild.attrib['steps']), float(childChildChild.text))
                  else:
                    raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> number of steps missing after the grid call.')
                else:
                  raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> grid type missing after the grid call.')
              else:
                raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> invalid labels after the variable call. Only "distribution" and "grid" are accepted.')
          else:
            raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> invalid or missing labels after the controllable variables call. Only "variable" is accepted.')
      elif child.tag == 'non-controllable':
        for childChild in child:
          if childChild.tag == 'variable':
            varName = childChild.attrib['name']
            for childChildChild in childChild:
              if childChildChild.tag == 'distribution':
                self.nonControllableDist[varName] = childChildChild.text
              elif childChildChild.tag == 'grid':
                if 'type' in childChildChild.attrib.keys():
                  if 'steps' in childChildChild.attrib.keys():
                    self.nonControllableGrid[varName] = (childChildChild.attrib['type'], int(childChildChild.attrib['steps']), float(childChildChild.text))
                  else:
                    raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> number of steps missing after the grid call.')
                else:
                  raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> grid type missing after the grid call.')
              else:
                raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> invalid labels after the variable call. Only "distribution" and "grid" are accepted.')
          else:
            raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> invalid or missing labels after the controllable variables call. Only "variable" is accepted.')
      #else:
      #  if child.tag != 'Assembler': raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> invalid or missing labels after the post-processor call. Only "controllable", "non-controllable" and "Assembler" are accepted.')
    if self.debug:
      print('CONTROLLABLE DISTRIBUTIONS:')
      print(self.controllableDist)
      print('CONTROLLABLE GRID:')
      print(self.controllableGrid)
      print('NON-CONTROLLABLE DISTRIBUTIONS:')
      print(self.nonControllableDist)
      print('NON-CONTROLLABLE GRID:')
      print(self.nonControllableGrid)

  def initialize(self,runInfo,inputs,initDict):
    self.__gridSetting__()
    self.__gridGeneration__()
    self.inputToInternal(inputs)
    self.stat.parameters['targets'] = self.controllableOrd
    self.stat.initialize(runInfo,inputs,initDict)
    if self.debug:
      print('GRID INFO:')
      print(self.gridInfo)
      print('N-DIMENSIONAL CONTROLLABLE SPACE:')
      print(self.controllableSpace)
      print('N-DIMENSIONAL NON-CONTROLLABLE SPACE:')
      print(self.nonControllableSpace)
      print('CONTROLLABLE VARIABLES ORDER:')
      print(self.controllableOrd)
      print('NON-CONTROLLABLE VARIABLES ORDER:')
      print(self.nonControllableOrd)
      print('SURFACE POINTS MATRIX:')
      print(self.surfPointsMatrix)

  def __gridSetting__(self,constrType='equal'):
    for varName in self.controllableGrid.keys():
      if self.controllableGrid[varName][0] == 'value':
        self.__stepError__(float(self.controllableDist[varName].lowerBound),float(self.controllableDist[varName].upperBound),self.controllableGrid[varName][1],self.controllableGrid[varName][2],varName)
        self.gridInfo[varName] = (self.controllableGrid[varName][0], constrType, [float(self.controllableDist[varName].lowerBound)+self.controllableGrid[varName][2]*i for i in range(self.controllableGrid[varName][1]+1)])
      elif self.controllableGrid[varName][0] == 'CDF':
        self.__stepError__(0,1,self.controllableGrid[varName][1],self.controllableGrid[varName][2],varName)
        self.gridInfo[varName] = (self.controllableGrid[varName][0], constrType, [self.controllableGrid[varName][2]*i for i in range(self.controllableGrid[varName][1]+1)])
      else:
        raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> inserted invalid grid type. Only "value" and "CDF" are accepted.')
    for varName in self.nonControllableGrid.keys():
      if self.nonControllableGrid[varName][0] == 'value':
        self.__stepError__(float(self.nonControllableDist[varName].lowerBound),float(self.nonControllableDist[varName].upperBound),self.nonControllableGrid[varName][1],self.nonControllableGrid[varName][2],varName)
        self.gridInfo[varName] = (self.nonControllableGrid[varName][0], constrType, [float(self.nonControllableDist[varName].lowerBound)+self.nonControllableGrid[varName][2]*i for i in range(self.nonControllableGrid[varName][1]+1)])
      elif self.nonControllableGrid[varName][0] == 'CDF':
        self.__stepError__(0,1,self.nonControllableGrid[varName][1],self.nonControllableGrid[varName][2],varName)
        self.gridInfo[varName] = (self.nonControllableGrid[varName][0], constrType, [self.nonControllableGrid[varName][2]*i for i in range(self.nonControllableGrid[varName][1]+1)])
      else:
        raise NameError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> inserted invalid grid type. Only "value" and "CDF" are accepted.')

  def __stepError__(self,lowerBound,upperBound,steps,tol,varName):
    if upperBound-lowerBound<steps*tol:
      raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> inserted number of steps or tolerance for variable ' +varName+ ' exceeds its limit.')

  def __gridGeneration__(self):
    NotchesByVar = [None]*len(self.controllableGrid.keys())
    controllableSpaceSize = None
    for varId, varName in enumerate(self.controllableGrid.keys()):
      NotchesByVar[varId] = self.controllableGrid[varName][1]+1
      self.controllableOrd.append(varName)
    controllableSpaceSize = tuple(NotchesByVar+[len(self.controllableGrid.keys())])
    self.controllableSpace = np.zeros(controllableSpaceSize)
    iterIndex = np.nditer(self.controllableSpace,flags=['multi_index'])
    while not iterIndex.finished:
      coordIndex = iterIndex.multi_index[-1]
      varName = self.controllableGrid.keys()[coordIndex]
      notchPos = iterIndex.multi_index[coordIndex]
      if self.gridInfo[varName][0] == 'CDF':
        valList = []
        for probVal in self.gridInfo[varName][2]:
          valList.append(self.controllableDist[varName].cdf(probVal))
        self.controllableSpace[iterIndex.multi_index] = valList[notchPos]
      else:
        self.controllableSpace[iterIndex.multi_index] = self.gridInfo[varName][2][notchPos]
      iterIndex.iternext()
    NotchesByVar = [None]*len(self.nonControllableGrid.keys())
    nonControllableSpaceSize = None
    for varId, varName in enumerate(self.nonControllableGrid.keys()):
      NotchesByVar[varId] = self.nonControllableGrid[varName][1]+1
      self.nonControllableOrd.append(varName)
    nonControllableSpaceSize = tuple(NotchesByVar+[len(self.nonControllableGrid.keys())])
    self.nonControllableSpace = np.zeros(nonControllableSpaceSize)
    iterIndex = np.nditer(self.nonControllableSpace,flags=['multi_index'])
    while not iterIndex.finished:
      coordIndex = iterIndex.multi_index[-1]
      varName = self.nonControllableGrid.keys()[coordIndex]
      notchPos = iterIndex.multi_index[coordIndex]
      if self.gridInfo[varName][0] == 'CDF':
        valList = []
        for probVal in self.gridInfo[varName][2]:
          valList.append(self.nonControllableDist[varName].cdf(probVal))
        self.nonControllableSpace[iterIndex.multi_index] = valList[notchPos]
      else:
        self.nonControllableSpace[iterIndex.multi_index] = self.gridInfo[varName][2][notchPos]
      iterIndex.iternext()

  def inputToInternal(self,currentInput):
    for item in currentInput:
      if item.type == 'TimePointSet':
        self.surfPointsMatrix = np.zeros((len(item.getParam('output',item.getParaKeys('outputs')[-1])),len(self.gridInfo.keys())+1))
        k=0
        for varName in self.controllableOrd:
          self.surfPointsMatrix[:,k] = item.getParam('input',varName)
          k+=1
        for varName in self.nonControllableOrd:
          self.surfPointsMatrix[:,k] = item.getParam('input',varName)
          k+=1
        self.surfPointsMatrix[:,k] = item.getParam('output',item.getParaKeys('outputs')[-1])

  def run(self,Input):
    nearestPointsInd = []
    dataCollector = Datas.returnInstance('TimePointSet')
    dataCollector.type = 'TimePointSet'
    surfTree = spatial.KDTree(copy.copy(self.surfPointsMatrix[:,0:self.surfPointsMatrix.shape[-1]-1]))
    self.controllableSpace.shape = (np.prod(self.controllableSpace.shape[0:len(self.controllableSpace.shape)-1]),self.controllableSpace.shape[-1])
    self.nonControllableSpace.shape = (np.prod(self.nonControllableSpace.shape[0:len(self.nonControllableSpace.shape)-1]),self.nonControllableSpace.shape[-1])
    if self.debug:
      print('RESHAPED CONTROLLABLE SPACE:')
      print(self.controllableSpace)
      print('RESHAPED NON-CONTROLLABLE SPACE:')
      print(self.nonControllableSpace)
    for ncLine in range(self.nonControllableSpace.shape[0]):
      queryPointsMatrix = np.append(self.controllableSpace,np.tile(self.nonControllableSpace[ncLine,:],(self.controllableSpace.shape[0],1)),axis=1)
      print('QUERIED POINTS MATRIX:')
      print(queryPointsMatrix)
      nearestPointsInd = surfTree.query(queryPointsMatrix)[-1]
      distList = []
      indexList = []
      probList = []
      for index in range(len(nearestPointsInd)):
        if self.surfPointsMatrix[np.where(np.prod(surfTree.data[nearestPointsInd[index],0:self.surfPointsMatrix.shape[-1]-1] == self.surfPointsMatrix[:,0:self.surfPointsMatrix.shape[-1]-1],axis=1))[0][0],-1] == 1:
          distList.append(np.sqrt(np.sum(np.power(queryPointsMatrix[index,0:self.controllableSpace.shape[-1]]-surfTree.data[nearestPointsInd[index],0:self.controllableSpace.shape[-1]],2))))
          indexList.append(index)
      if distList == []:
        raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> no safest point found for the current set of non-controllable variables: ' +str(self.nonControllableSpace[ncLine,:])+ '.')
      else:
        for cVarIndex in range(len(self.controllableOrd)):
          dataCollector.updateInputValue(self.controllableOrd[cVarIndex],copy.copy(queryPointsMatrix[indexList[distList.index(max(distList))],cVarIndex]))
        for ncVarIndex in range(len(self.nonControllableOrd)):
          dataCollector.updateInputValue(self.nonControllableOrd[ncVarIndex],copy.copy(queryPointsMatrix[indexList[distList.index(max(distList))],len(self.controllableOrd)+ncVarIndex]))
          if queryPointsMatrix[indexList[distList.index(max(distList))],len(self.controllableOrd)+ncVarIndex] == self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].lowerBound:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]/float(2)
            else:
              prob = self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].lowerBound+self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]/float(2))
          elif queryPointsMatrix[indexList[distList.index(max(distList))],len(self.controllableOrd)+ncVarIndex] == self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].upperBound:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]/float(2)
            else:
              prob = 1-self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].upperBound-self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]/float(2))
          else:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]
            else:
              prob = self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(queryPointsMatrix[indexList[distList.index(max(distList))],len(self.controllableOrd)+ncVarIndex]+self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]/float(2))-self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(queryPointsMatrix[indexList[distList.index(max(distList))],len(self.controllableOrd)+ncVarIndex]-self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]/float(2))
          probList.append(prob)
      dataCollector.updateOutputValue('Probability',np.prod(probList))
      dataCollector.updateMetadata('ProbabilityWeight',np.prod(probList))
    dataCollector.updateMetadata('ExpectedSafestPointCoordinates',self.stat.run(dataCollector)['expectedValue'])
    if self.debug:
      print(dataCollector.getParametersValues('input'))
      print(dataCollector.getParametersValues('output'))
      print(dataCollector.getMetadata('ExpectedSafestPointCoordinates'))
    return dataCollector

  def collectOutput(self,finishedjob,output):
    if finishedjob.returnEvaluation() == -1:
      raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> no available output to collect (the run is likely not over yet).')
    else:
      dataCollector = finishedjob.returnEvaluation()[1]
      if output.type != 'TimePointSet':
        raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> output item type must be "TimePointSet".')
      else:
        if not output.isItEmpty():
          raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> output item must be empty.')
        else:
          for key,value in dataCollector.getParametersValues('input').items():
            for val in value: output.updateInputValue(key, val)
          for key,value in dataCollector.getParametersValues('output').items():
            for val in value: output.updateOutputValue(key,val)
          for key,value in dataCollector.getAllMetadata().items(): output.updateMetadata(key,value)

class ComparisonStatistics(BasePostProcessor):
  """
  ComparisonStatistics is to calculate statistics that compare
  two different codes or code to experimental data.
  """

  class CompareGroup:
    def __init__(self):
      self.dataPulls = []
      self.referenceData = {}

  def __init__(self):
    BasePostProcessor.__init__(self)
    self.dataDict = {} #Dictionary of all the input data, keyed by the name
    self.compareGroups = [] #List of each of the groups that will be compared
    #self.dataPulls = [] #List of data references that will be used
    #self.referenceData = [] #List of reference (experimental) data
    self.methodInfo = {} #Information on what stuff to do.

  def inputToInternal(self,currentInput):
    return [(currentInput)]

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    #print("runInfo",runInfo,"inputs",inputs,"initDict",initDict)

  def _localReadMoreXML(self,xmlNode):
    for outer in xmlNode:
      if outer.tag == 'compare':
        compareGroup = ComparisonStatistics.CompareGroup()
        for child in outer:
          if child.tag == 'data':
            dataName = child.text
            splitName = dataName.split("|")
            name, kind = splitName[:2]
            rest = splitName[2:]
            compareGroup.dataPulls.append([name, kind, rest])
            #print("xml dataName",dataName,self.dataPulls[-1])
          elif child.tag == 'reference':
            compareGroup.referenceData = dict(child.attrib)
        self.compareGroups.append(compareGroup)
      if outer.tag == 'kind':
        self.methodInfo['kind'] = outer.text
        if 'num_bins' in outer.attrib:
          self.methodInfo['num_bins'] = int(outer.attrib['num_bins'])
        if 'bin_method' in outer.attrib:
          self.methodInfo['bin_method'] = outer.attrib['bin_method'].lower()


  def run(self, Input): # inObj,workingDir=None):
    """
     Function to finalize the filter => execute the filtering
     @ Out, None      : To add description
    """
    dataDict = {}
    for aInput in Input: dataDict[aInput.name] = aInput
    return dataDict
    #print("input",Input,"input name",Input.name,"input input",Input.getParametersValues('inputs'),
    #      "input output",Input.getParametersValues('outputs'))

  def collectOutput(self,finishedjob,output):
    if self.debug: print("finishedjob",finishedjob,"output",output)
    if finishedjob.returnEvaluation() == -1: raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> no available output to collect.')
    else: self.dataDict.update(finishedjob.returnEvaluation()[1])

    dataToProcess = []
    for compareGroup in self.compareGroups:
      dataPulls = compareGroup.dataPulls
      reference = compareGroup.referenceData
      foundDatas = []
      for name, kind, rest in dataPulls:
        data = self.dataDict[name].getParametersValues(kind)
        #print("dataPull",dataPull) #("result",self.dataDict[name].getParametersValues(kind))
        if len(rest) == 1:
          #print("dataPart",data[rest[0]])
          #print(data.keys())
          foundDatas.append(data[rest[0]])
      dataToProcess.append((dataPulls,foundDatas,reference))
    #print("dataToProcess",dataToProcess)
    csv = open(output,"w")
#    def printCsv(*args):
#      print(*args,file=csv,sep=',')
    for dataPulls, datas, reference in dataToProcess:
      graphData = []
      if "mean" in reference:
          refDataStats = {"mean":float(reference["mean"]),
                            "stdev":float(reference["sigma"]),
                            "min_bin_size":float(reference["sigma"])/2.0}
          refPdf = lambda x:mathUtils.normal(x,refDataStats["mean"],refDataStats["stdev"])
          refCdf = lambda x:mathUtils.normalCdf(x,refDataStats["mean"],refDataStats["stdev"])
          graphData.append((refDataStats,refCdf,refPdf,"ref"))
      for dataPull, data in zip(dataPulls,datas):
        dataStats = self.processData(dataPull, data, self.methodInfo)
        dataKeys = set(dataStats.keys())
        utils.printCsv(csv,'"'+str(dataPull)+'"')
        utils.printCsv(csv,'"num_bins"',dataStats['num_bins'])
        counts = dataStats['counts']
        bins = dataStats['bins']
        countSum = sum(counts)
        binBoundaries = [dataStats['low']]+bins+[dataStats['high']]
        utils.printCsv(csv,'"bin_boundary"','"bin_midpoint"','"bin_count"','"normalized_bin_count"','"f_prime"','"cdf"')
        cdf = [0.0]*len(counts)
        midpoints = [0.0]*len(counts)
        cdfSum = 0.0
        for i in range(len(counts)):
          f_0 = counts[i]/countSum
          cdfSum += f_0
          cdf[i] = cdfSum
          midpoints[i] = (binBoundaries[i]+binBoundaries[i+1])/2.0
        cdfFunc = mathUtils.createInterp(midpoints,cdf,0.0,1.0,'quadratic')
        fPrimeData = [0.0]*len(counts)
        for i in range(len(counts)):
          h = binBoundaries[i+1] - binBoundaries[i]
          nCount = counts[i]/countSum #normalized count
          f_0 = cdf[i]
          if i + 1 < len(counts):
            f_1 = cdf[i+1]
          else:
            f_1 = 1.0
          if i + 2 < len(counts):
            f_2 = cdf[i+2]
          else:
            f_2 = 1.0
          #f_prime = (f_1 - f_0)/h
          #print(f_0,f_1,f_2,h,f_prime)
          fPrime = (-1.5*f_0 + 2.0*f_1 + -0.5*f_2)/h
          fPrimeData[i] = fPrime
          utils.printCsv(csv,binBoundaries[i+1],midpoints[i],counts[i],nCount,fPrime,cdf[i])
        pdfFunc = mathUtils.createInterp(midpoints,fPrimeData,0.0,0.0,'linear')
        dataKeys -= set({'num_bins','counts','bins'})
        for key in dataKeys:
          utils.printCsv(csv,'"'+key+'"',dataStats[key])
        print("data_stats",dataStats)
        graphData.append((dataStats, cdfFunc, pdfFunc,str(dataPull)))
      mathUtils.printGraphs(csv, graphData)
      for i in range(len(graphData)):
        dataStat = graphData[i][0]
        def delist(l):
          if type(l).__name__ == 'list':
            return '_'.join([delist(x) for x in l])
          else:
            return str(l)
        newFileName = output[:-4]+"_"+delist(dataPulls)+"_"+str(i)+".csv"
        #print("data_stat",type(data_stat),data_stat.__sizeof__,data_stat)
        if type(dataStat).__name__ != 'dict':
          assert(False)
          continue
        dataPairs = []
        for key in sorted(dataStat.keys()):
          value = dataStat[key]
          if type(value).__name__ in ["int","float"]:
            dataPairs.append((key,value))
        extraCsv = open(newFileName,"w")
        extraCsv.write(",".join(['"'+str(x[0])+'"' for x in dataPairs]))
        extraCsv.write("\n")
        extraCsv.write(",".join([str(x[1]) for x in dataPairs]))
        extraCsv.write("\n")
        extraCsv.close()
        #print(new_filename,"data_pairs",data_pairs)
      utils.printCsv(csv)

  def processData(self,dataPull, data, methodInfo):
      ret = {}
      try:
        sortedData = data.tolist()
      except:
        sortedData = list(data)
      sortedData.sort()
      low = sortedData[0]
      high = sortedData[-1]
      dataRange = high - low
      #print("data",dataPull,"average",sum(data)/len(data))
      ret['low'] = low
      ret['high'] = high
      #print("low",low,"high",high,end=' ')
      if not 'bin_method' in methodInfo:
        numBins = methodInfo.get("num_bins",10)
      else:
        binMethod = methodInfo['bin_method']
        dataN = len(sortedData)
        if binMethod == 'square-root':
          numBins = int(math.ceil(math.sqrt(dataN)))
        elif binMethod == 'sturges':
          numBins = int(math.ceil(mathUtils.log2(dataN)+1))
        else:
          print(returnPrintPostTag('ERROR')+"Unknown bin_method "+binMethod)
          numBins = 5
      ret['num_bins'] = numBins
      #print("num_bins",num_bins)
      kind = methodInfo.get("kind","uniform_bins")
      if kind == "uniform_bins":
        bins = [low+x*dataRange/numBins for x in range(1,numBins)]
        ret['min_bin_size'] = dataRange/numBins
      elif kind == "equal_probability":
        stride = len(sortedData)//numBins
        bins = [sortedData[x] for x in range(stride-1,len(sortedData)-stride+1,stride)]
        if len(bins) > 1:
          ret['min_bin_size'] = min(map(lambda x,y: x - y,bins[1:],bins[:-1]))
        else:
          ret['min_bin_size'] = dataRange
      counts = mathUtils.countBins(sortedData,bins)
      ret['bins'] = bins
      ret['counts'] = counts
      ret.update(mathUtils.calculateStats(sortedData))
      skewness = ret["skewness"]
      delta = math.sqrt((math.pi/2.0)*(abs(skewness)**(2.0/3.0))/
                        (abs(skewness)**(2.0/3.0)+((4.0-math.pi)/2.0)**(2.0/3.0)))
      delta = math.copysign(delta,skewness)
      alpha = delta/math.sqrt(1.0-delta**2)
      variance = ret["sample_variance"]
      omega = variance/(1.0-2*delta**2/math.pi)
      mean = ret['mean']
      xi = mean - omega*delta*math.sqrt(2.0/math.pi)
      ret['alpha'] = alpha
      ret['omega'] = omega
      ret['xi'] = xi
      #print("bins",bins,"counts",counts)
      return ret

class PrintCSV(BasePostProcessor):
  """
  PrintCSV PostProcessor class. It prints a CSV file loading data from a hdf5 database or other sources
  """
  def __init__(self):
    BasePostProcessor.__init__(self)
    self.paramters  = ['all']
    self.inObj      = None
    self.workingDir = None
    self.printTag = utils.returnPrintTag('POSTPROCESSOR PRINTCSV')
  def inputToInternal(self,currentInput): return [(currentInput)]

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.workingDir               = os.path.join(runInfo['WorkingDir'],runInfo['stepName']) #generate current working dir
    runInfo['TempWorkingDir']     = self.workingDir
    try:                            os.mkdir(self.workingDir)
    except:                         print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '->current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    #if type(inputs[-1]).__name__ == "HDF5" : self.inObj = inputs[-1]      # this should go in run return but if HDF5, it is not pickable

  def _localReadMoreXML(self,xmlNode):
    """
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'parameters':
        param = child.text
        if(param.lower() != 'all'): self.paramters = param.strip().split(',')
        else: self.paramters[param]

  def collectOutput(self,finishedjob,output):
    # Check the input type
    if finishedjob.returnEvaluation() == -1: raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '->  No available Output to collect (Run probabably is not finished yet)')
    self.inObj = finishedjob.returnEvaluation()[1]
    if(self.inObj.type == "HDF5"):
      #  Input source is a database (HDF5)
      #  Retrieve the ending groups' names
      endGroupNames = self.inObj.getEndingGroupNames()
      histories = {}

      #  Construct a dictionary of all the histories
      for index in range(len(endGroupNames)): histories[endGroupNames[index]] = self.inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
      #  If file, split the strings and add the working directory if present
      for key in histories:
        #  Loop over histories
        #  Retrieve the metadata (posion 1 of the history tuple)
        attributes = histories[key][1]
        #  Construct the header in csv format (first row of the file)
        headers = b",".join([histories[key][1]['output_space_headers'][i] for i in
                             range(len(attributes['output_space_headers']))])
        #  Construct history name
        hist = key
        #  If file, split the strings and add the working directory if present
        if self.workingDir:
          if os.path.split(output)[1] == '': output = output[:-1]
          splitted_1 = os.path.split(output)
          output = splitted_1[1]
        splitted = output.split('.')
        #  Create csv files' names
        addfile = splitted[0] + '_additional_info_' + hist + '.'+splitted[1]
        csvfilen = splitted[0] + '_' + hist + '.'+splitted[1]
        #  Check if workingDir is present and in case join the two paths
        if self.workingDir:
          addfile  = os.path.join(self.workingDir,addfile)
          csvfilen = os.path.join(self.workingDir,csvfilen)

        #  Open the files and save the data
        with open(csvfilen, 'wb') as csvfile, open(addfile, 'wb') as addcsvfile:
          #  Add history to the csv file
          np.savetxt(csvfile, histories[key][0], delimiter=",",header=utils.toString(headers))
          csvfile.write(b' \n')
          #  process the attributes in a different csv file (different kind of informations)
          #  Add metadata to additional info csv file
          addcsvfile.write(b'# History Metadata, \n')
          addcsvfile.write(b'# ______________________________,' + b'_'*len(key)+b','+b'\n')
          addcsvfile.write(b'#number of parameters,\n')
          addcsvfile.write(utils.toBytes(str(attributes['n_params']))+b',\n')
          addcsvfile.write(b'#parameters,\n')
          addcsvfile.write(headers+b'\n')
          addcsvfile.write(b'#parent_id,\n')
          addcsvfile.write(utils.toBytes(attributes['parent_id'])+b'\n')
          addcsvfile.write(b'#start time,\n')
          addcsvfile.write(utils.toBytes(str(attributes['start_time']))+b'\n')
          addcsvfile.write(b'#end time,\n')
          addcsvfile.write(utils.toBytes(str(attributes['end_time']))+b'\n')
          addcsvfile.write(b'#number of time-steps,\n')
          addcsvfile.write(utils.toBytes(str(attributes['n_ts']))+b'\n')
          # remove because not needed!!!!!!
#             for cnt,item in enumerate(attributes['metadata']):
#               if 'initiator_distribution' in item.keys():
#                 init_dist = attributes['initiator_distribution']
#                 addcsvfile.write(b'#number of branches in this history,\n')
#                 addcsvfile.write(utils.toBytes(str(len(init_dist)))+b'\n')
#                 string_work = ''
#                 for i in range(len(init_dist)):
#                   string_work_2 = ''
#                   for j in init_dist[i]: string_work_2 = string_work_2 + str(j) + ' '
#                   string_work = string_work + string_work_2 + ','
#                 addcsvfile.write(b'#initiator distributions,\n')
#                 addcsvfile.write(utils.toBytes(string_work)+b'\n')
#               if 'end_timestep' in item.keys():
#                 string_work = ''
#                 end_ts = attributes['end_timestep']
#                 for i in xrange(len(end_ts)): string_work = string_work + str(end_ts[i]) + ','
#                 addcsvfile.write('#end time step,\n')
#                 addcsvfile.write(str(string_work)+'\n')
#               if 'branch_changed_param' in attributes['metadata'][-1].keys():
#                 string_work = ''
#                 branch_changed_param = attributes['branch_changed_param']
#                 for i in range(len(branch_changed_param)):
#                   string_work_2 = ''
#                   for j in branch_changed_param[i]:
#                     if not j: string_work_2 = string_work_2 + 'None' + ' '
#                     else: string_work_2 = string_work_2 + str(j) + ' '
#                   string_work = string_work + string_work_2 + ','
#                 addcsvfile.write(b'#changed parameters,\n')
#                 addcsvfile.write(utils.toBytes(str(string_work))+b'\n')
#               if 'branch_changed_param_value' in attributes['metadata'][-1].keys():
#                 string_work = ''
#                 branch_changed_param_value = attributes['branch_changed_param_value']
#                 for i in range(len(branch_changed_param_value)):
#                   string_work_2 = ''
#                   for j in branch_changed_param_value[i]:
#                     if not j: string_work_2 = string_work_2 + 'None' + ' '
#                     else: string_work_2 = string_work_2 + str(j) + ' '
#                   string_work = string_work + string_work_2 + ','
#                 addcsvfile.write(b'#changed parameters values,\n')
#                 addcsvfile.write(utils.toBytes(str(string_work))+b'\n')
#               if 'conditional_prb' in attributes['metadata'][-1].keys():
#                 string_work = ''
#                 cond_pbs = attributes['conditional_prb']
#                 for i in range(len(cond_pbs)):
#                   string_work_2 = ''
#                   for j in cond_pbs[i]:
#                     if not j: string_work_2 = string_work_2 + 'None' + ' '
#                     else: string_work_2 = string_work_2 + str(j) + ' '
#                   string_work = string_work + string_work_2 + ','
#                 addcsvfile.write(b'#conditional probability,\n')
#                 addcsvfile.write(utils.toBytes(str(string_work))+b'\n')
#               if 'PbThreshold' in attributes['metadata'][-1].keys():
#                 string_work = ''
#                 pb_thresholds = attributes['PbThreshold']
#                 for i in range(len(pb_thresholds)):
#                   string_work_2 = ''
#                   for j in pb_thresholds[i]:
#                     if not j: string_work_2 = string_work_2 + 'None' + ' '
#                     else: string_work_2 = string_work_2 + str(j) + ' '
#                   string_work = string_work + string_work_2 + ','
#                 addcsvfile.write(b'#Probability threshold,\n')
#                 addcsvfile.write(utils.toBytes(str(string_work))+b'\n')
          addcsvfile.write(b' \n')
    else: raise NameError (self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> for input type ' + self.inObj.type + ' not yet implemented.')

  def run(self, Input): # inObj,workingDir=None):
    """
     Function to finalize the filter => execute the filtering
     @ Out, None      : Print of the CSV file
    """
    return Input[-1]

class BasicStatistics(BasePostProcessor):
  """
    BasicStatistics filter class. It computes all the most popular statistics
  """
  def __init__(self):
    BasePostProcessor.__init__(self)
    self.parameters        = {}                                                                                                      #parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.acceptedCalcParam = ['covariance','NormalizedSensitivity','sensitivity','pearson','expectedValue','sigma','variationCoefficient','variance','skewness','kurtosis','median','percentile']  # accepted calculation parameters
    self.what              = self.acceptedCalcParam                                                                                  # what needs to be computed... default...all
    self.methodsToRun      = []                                                                                                      # if a function is present, its outcome name is here stored... if it matches one of the known outcomes, the pp is going to use the function to compute it
    self.externalFunction  = []
    self.printTag          = utils.returnPrintTag('POSTPROCESSOR BASIC STATISTIC')
    self.requiredAssObject = (True,(['Function'],[-1]))
    self.biased            = False

  def inputToInternal(self,currentInp):
    # each post processor knows how to handle the coming inputs. The BasicStatistics postprocessor accept all the input type (files (csv only), hdf5 and datas
    if type(currentInp) == list  : currentInput = currentInp [-1]
    else                         : currentInput = currentInp
    if type(currentInput) == dict:
      if 'targets' in currentInput.keys(): return
    inputDict = {'targets':{},'metadata':{}}
    try: inType = currentInput.type
    except:
      if type(currentInput) in [str,bytes,unicode]: inType = "file"
      elif type(currentInput) in [list]: inType = "list"
      else: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> BasicStatistics postprocessor accepts files,HDF5,Data(s) only! Got '+ str(type(currentInput)))
    if inType not in ['file','HDF5','TimePointSet','list']: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> BasicStatistics postprocessor accepts files,HDF5,Data(s) only! Got '+ str(inType) + '!!!!')
    if inType == 'file':
      if currentInput.endswith('csv'): pass
    if inType == 'HDF5': pass # to be implemented
    if inType in ['TimePointSet']:
      for targetP in self.parameters['targets']:
        if   targetP in currentInput.getParaKeys('input' ): inputDict['targets'][targetP] = currentInput.getParam('input' ,targetP)
        elif targetP in currentInput.getParaKeys('output'): inputDict['targets'][targetP] = currentInput.getParam('output',targetP)
      inputDict['metadata'] = currentInput.getAllMetadata()
#     # now we check if the sampler that genereted the samples are from adaptive... in case... create the grid
      if inputDict['metadata'].keys().count('SamplerType') > 0: pass

    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']

  def _localReadMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag =="what":
        self.what = child.text
        if self.what == 'all': self.what = self.acceptedCalcParam
        else:
          for whatc in self.what.split(','):
            if whatc not in self.acceptedCalcParam: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> BasicStatistics postprocessor asked unknown operation ' + whatc + '. Available '+str(self.acceptedCalcParam))
          self.what = self.what.split(',')
      if child.tag =="parameters"   : self.parameters['targets'] = child.text.split(',')
      if child.tag =="methodsToRun" : self.methodsToRun          = child.text.split(',')
      if child.tag =="biased"       :
          if child.text.lower() in utils.stringsThatMeanTrue(): self.biased = True

  def collectOutput(self,finishedjob,output):
    #output
    parameterSet = list(set(list(self.parameters['targets'])))
    if finishedjob.returnEvaluation() == -1: raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '->  No available Output to collect (Run probabably is not finished yet)')
    outputDict = finishedjob.returnEvaluation()[1]
    methodToTest = []
    for key in self.methodsToRun:
      if key not in self.acceptedCalcParam: methodToTest.append(key)
    if type(output) in [str,unicode,bytes]:
      availextens = ['csv','txt']
      outputextension = output.split('.')[-1].lower()
      if outputextension not in availextens:
        print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '->BasicStatistics postprocessor output extension you input is '+outputextension)
        print('                     Available are '+str(availextens)+ '. Convertint extension to '+str(availextens[0])+'!')
        outputextension = availextens[0]
      if outputextension != 'csv': separator = ' '
      else                       : separator = ','
      basicStatFilename = os.path.join(self.__workingDir,output[:output.rfind('.')]+'.'+outputextension)
      if self.debug:
        print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '->' + "workingDir",self.__workingDir,"output",output.split('.'))
        print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: dumping output in file named ' + basicStatFilename)
      with open(basicStatFilename, 'wb') as basicStatdump:
        basicStatdump.write('BasicStatistics '+separator+str(self.name)+'\n')
        basicStatdump.write('----------------'+separator+'-'*len(str(self.name))+'\n')
        for targetP in parameterSet:
          if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: writing variable '+ targetP)
          basicStatdump.write('Variable'+ separator + targetP +'\n')
          basicStatdump.write('--------'+ separator +'-'*len(targetP)+'\n')
          for what in outputDict.keys():
            if what not in ['covariance','pearson','NormalizedSensitivity','sensitivity'] + methodToTest:
              if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: writing variable '+ targetP + '. Parameter: '+ what)
              basicStatdump.write(what+ separator + '%.8E' % outputDict[what][targetP]+'\n')
        maxLenght = max(len(max(parameterSet, key=len))+5,16)
        for what in outputDict.keys():
          if what in ['covariance','pearson','NormalizedSensitivity','sensitivity']:
            if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: writing parameter matrix '+ what )
            basicStatdump.write(what+' \n')
            if outputextension != 'csv': basicStatdump.write(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in parameterSet])+'\n')
            else                       : basicStatdump.write('matrix' + separator+''.join([str(item) + separator for item in parameterSet])+'\n')
            for index in range(len(parameterSet)):
              if outputextension != 'csv': basicStatdump.write(parameterSet[index] + ' '*(maxLenght-len(parameterSet[index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict[what][index]])+'\n')
              else                       : basicStatdump.write(parameterSet[index] + ''.join([separator +'%.8E' % item for item in outputDict[what][index]])+'\n')
        if self.externalFunction:
          if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: writing External Function results')
          basicStatdump.write('\n' +'EXT FUNCTION \n')
          basicStatdump.write('------------ \n')
          for what in self.methodsToRun:
            if what not in self.acceptedCalcParam:
              if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: writing External Function parameter '+ what )
              basicStatdump.write(what+ separator + '%.8E' % outputDict[what]+'\n')
    elif output.type == 'Datas':
      if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: dumping output in data object named ' + output.name)
      for what in outputDict.keys():
        if what not in ['covariance','pearson','NormalizedSensitivity','sensitivity'] + methodToTest:
          for targetP in parameterSet:
            if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: dumping variable '+ targetP + '. Parameter: '+ what + '. Metadata name = '+ targetP+'|'+what)
            output.updateMetadata(targetP+'|'+what,outputDict[what][targetP])
        else:
          if what not in methodToTest:
            if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: dumping matrix '+ what + '. Metadata name = ' + what + '. Targets stored in ' + 'targets|'+what)
            output.updateMetadata('targets|'+what,parameterSet)
            output.updateMetadata(what,outputDict[what])
      if self.externalFunction:
        if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: dumping External Function results')
        for what in self.methodsToRun:
          if what not in self.acceptedCalcParam:
            output.updateMetadata(what,outputDict[what])
            if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics postprocessor: dumping External Function parameter '+ what)
    elif output.type == 'HDF5' : print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '->BasicStatistics postprocessor: Output type '+ str(output.type) + ' not yet implemented. Skip it !!!!!')
    else: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> BasicStatistics postprocessor: Output type '+ str(output.type) + ' unknown!!')

  def run(self, InputIn):
    """
     Function to finalize the filter => execute the filtering
     @ In , dictionary       : dictionary of data to process
     @ Out, dictionary       : Dictionary with results
    """
    Input  = self.inputToInternal(InputIn)
    outputDict = {}

    if self.externalFunction:
      # there is an external function
      for what in self.methodsToRun:
        outputDict[what] = self.externalFunction.evaluate(what,Input['targets'])
        # check if "what" corresponds to an internal method
        if what in self.acceptedCalcParam:
          if what not in ['pearson','covariance','NormalizedSensitivity','sensitivity']:
            if type(outputDict[what]) != dict: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> BasicStatistics postprocessor: You have overwritten the "'+what+'" method through an external function, it must be a dictionary!!')
          else:
            if type(outputDict[what]) != np.ndarray: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> BasicStatistics postprocessor: You have overwritten the "'+what+'" method through an external function, it must be a numpy.ndarray!!')
            if len(outputDict[what].shape) != 2:     raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> BasicStatistics postprocessor: You have overwritten the "'+what+'" method through an external function, it must be a 2D numpy.ndarray!!')

    #setting some convenience values
    parameterSet = list(set(list(self.parameters['targets'])))  #@Andrea I am using set to avoid the test: if targetP not in outputDict[what].keys()
    N            = [np.asarray(Input['targets'][targetP]).size for targetP in parameterSet]
    pbPresent    = Input['metadata'].keys().count('ProbabilityWeight')>0

    if 'ProbabilityWeight' not in Input['metadata'].keys():
      if Input['metadata'].keys().count('SamplerType') > 0:
        if Input['metadata']['SamplerType'][0] != 'MC' : print('POSTPROC: Warning -> BasicStatistics postprocessor can not compute expectedValue without ProbabilityWeights. Use unit weight')
      else: print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '->BasicStatistics postprocessor can not compute expectedValue without ProbabilityWeights. Use unit weight')
      pbweights    = np.zeros(len(Input['targets'][self.parameters['targets'][0]]),dtype=np.float)
      pbweights[:] = 1.0/pbweights.size # it was an Integer Division (1/integer) => 0!!!!!!!! Andrea
    else: pbweights       = Input['metadata']['ProbabilityWeight']
    sumSquarePbWeights  = np.sum(np.square(pbweights))
    sumPbWeights        = np.sum(pbweights)
    # if here because the user could have overwritten the method through the external function
    if 'expectedValue' not in outputDict.keys(): outputDict['expectedValue'] = {}
    expValues = np.zeros(len(parameterSet))
    for myIndex, targetP in enumerate(parameterSet):
      outputDict['expectedValue'][targetP]= np.average(Input['targets'][targetP],weights=pbweights)
      expValues[myIndex] = outputDict['expectedValue'][targetP]

    for what in self.what:
      if what not in outputDict.keys(): outputDict[what] = {}
      #sigma
      if what == 'sigma':
        for myIndex, targetP in enumerate(parameterSet):
          outputDict[what][targetP] = np.sqrt(np.average((Input['targets'][targetP]-expValues[myIndex])**2,weights=pbweights)/(sumPbWeights-sumSquarePbWeights/sumPbWeights))
      #variance
      if what == 'variance':
        for myIndex, targetP in enumerate(parameterSet):
          outputDict[what][targetP] = np.average((Input['targets'][targetP]-expValues[myIndex])**2,weights=pbweights)/(sumPbWeights-sumSquarePbWeights/sumPbWeights)
      #coefficient of variation (sigma/mu)
      if what == 'variationCoefficient':
        for myIndex, targetP in enumerate(parameterSet):
          sigma = np.sqrt(np.average((Input['targets'][targetP]-expValues[myIndex])**2,weights=pbweights)/(sumPbWeights-sumSquarePbWeights/sumPbWeights))
          outputDict[what][targetP] = sigma/outputDict['expectedValue'][targetP]
      #kurtosis
      if what == 'kurtosis':
        for myIndex, targetP in enumerate(parameterSet):
          if pbPresent:
              sigma = np.sqrt(np.average((Input['targets'][targetP]-expValues[myIndex])**2, weights=pbweights))
              outputDict[what][targetP] = np.average(((Input['targets'][targetP]-expValues[myIndex])**4), weights=pbweights)/sigma**4
          else:
            outputDict[what][targetP] = -3.0 + (np.sum((np.asarray(Input['targets'][targetP]) - expValues[myIndex])**4)/(N[myIndex]-1))/(np.sum((np.asarray(Input['targets'][targetP]) - expValues[myIndex])**2)/float(N[myIndex]-1))**2
      #skewness
      if what == 'skewness':
        for myIndex, targetP in enumerate(parameterSet):
          if pbPresent:
            sigma = np.sqrt(np.average((Input['targets'][targetP]-expValues[myIndex])**2, weights=pbweights))
            outputDict[what][targetP] = np.average((((Input['targets'][targetP]-expValues[myIndex])/sigma)**3), weights=pbweights)
          else:
            outputDict[what][targetP] = (np.sum((np.asarray(Input['targets'][targetP]) - expValues[myIndex])**3)*(N[myIndex]-1)**-1)/(np.sum((np.asarray(Input['targets'][targetP]) - expValues[myIndex])**2)/float(N[myIndex]-1))**1.5
      #median
      if what == 'median':
        for targetP in parameterSet: outputDict[what][targetP]  = np.median(Input['targets'][targetP])
      #percentile
      if what == 'percentile':
        outputDict.pop(what)
        if what+'_5%'  not in outputDict.keys(): outputDict[what+'_5%']  ={}
        if what+'_95%' not in outputDict.keys(): outputDict[what+'_95%'] ={}
        for targetP in self.parameters['targets'  ]:
          if targetP not in outputDict[what+'_5%'].keys():
            outputDict[what+'_5%'][targetP]  = np.percentile(Input['targets'][targetP],5)
          if targetP not in outputDict[what+'_95%'].keys():
            outputDict[what+'_95%'][targetP]  = np.percentile(Input['targets'][targetP],95)
      #cov matrix
      if what == 'covariance':
        feat = np.zeros((len(Input['targets'].keys()),utils.first(Input['targets'].values()).size))
        for myIndex, targetP in enumerate(parameterSet): feat[myIndex,:] = Input['targets'][targetP][:]
        outputDict[what] = self.covariance(feat, weights=pbweights)
      #pearson matrix
      if what == 'pearson':
        feat = np.zeros((len(Input['targets'].keys()),utils.first(Input['targets'].values()).size))
        for myIndex, targetP in enumerate(parameterSet): feat[myIndex,:] = Input['targets'][targetP][:]
        outputDict[what] = self.corrCoeff(feat, weights=pbweights) #np.corrcoef(feat)
      #sensitivity matrix
      if what == 'sensitivity':
        feat = np.zeros((len(Input['targets'].keys()),utils.first(Input['targets'].values()).size))
        for myIndex, targetP in enumerate(parameterSet): feat[myIndex,:] = Input['targets'][targetP][:]
        covMatrix = self.covariance(feat, weights=pbweights)
        variance  = np.zeros(len(list(parameterSet)))
        for myIndex, targetP in enumerate(parameterSet):
          variance[myIndex] = np.average((Input['targets'][targetP]-expValues[myIndex])**2,weights=pbweights)/(sumPbWeights-sumSquarePbWeights/sumPbWeights)
        for myIndex in range(len(parameterSet)):
          outputDict[what][myIndex] = covMatrix[myIndex,:]/variance
      #Normalizzate sensitivity matrix: linear regression slopes normalizited by the mean (% change)/(% change)
      if what == 'NormalizedSensitivity':
        feat = np.zeros((len(Input['targets'].keys()),utils.first(Input['targets'].values()).size))
        for myIndex, targetP in enumerate(parameterSet): feat[myIndex,:] = Input['targets'][targetP][:]
        covMatrix = self.covariance(feat, weights=pbweights)
        variance  = np.zeros(len(list(parameterSet)))
        for myIndex, targetP in enumerate(parameterSet):
          variance[myIndex] = np.average((Input['targets'][targetP]-expValues[myIndex])**2,weights=pbweights)/(sumPbWeights-sumSquarePbWeights/sumPbWeights)
        for myIndex in range(len(parameterSet)):
          outputDict[what][myIndex] = ((covMatrix[myIndex,:]/variance)*expValues)/expValues[myIndex]

    # print on screen
    print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> BasicStatistics '+str(self.name)+'pp outputs')
    methodToTest = []
    for key in self.methodsToRun:
      if key not in self.acceptedCalcParam: methodToTest.append(key)
    for targetP in parameterSet:
      print('        *************'+'*'*len(targetP)+'***')
      print('        * Variable * '+ targetP +'  *')
      print('        *************'+'*'*len(targetP)+'***')
      for what in outputDict.keys():
        if what not in ['covariance','pearson','NormalizedSensitivity','sensitivity'] + methodToTest:
          print('              ','**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***')
          print('              ','* '+what+' * ' + '%.8E' % outputDict[what][targetP]+'  *')
          print('              ','**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***')
    maxLenght = max(len(max(parameterSet, key=len))+5,16)
    if 'covariance' in outputDict.keys():
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght,'*         Covariance        *')
      print(' '*maxLenght,'*****************************')

      print(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in parameterSet]))
      for index in range(len(parameterSet)):
        print(parameterSet[index] + ' '*(maxLenght-len(parameterSet[index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['covariance'][index]]))
    if 'pearson' in outputDict.keys():
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght,'*    Pearson/Correlation    *')
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in parameterSet]))
      for index in range(len(parameterSet)):
        print(parameterSet[index] + ' '*(maxLenght-len(parameterSet[index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['pearson'][index]]))
    if 'sensitivity' in outputDict.keys():
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght,'*        Sensitivity        *')
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in parameterSet]))
      for index in range(len(parameterSet)):
        print(parameterSet[index] + ' '*(maxLenght-len(parameterSet[index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['sensitivity'][index]]))
    if 'NormalizedSensitivity' in outputDict.keys():
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght,'*   Normalized Sensitivity  *')
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in parameterSet]))
      for index in range(len(parameterSet)):
        print(parameterSet[index] + ' '*(maxLenght-len(parameterSet[index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['NormalizedSensitivity'][index]]))

    if self.externalFunction:
      print(' '*maxLenght,'+++++++++++++++++++++++++++++')
      print(' '*maxLenght,'+ OUTCOME FROM EXT FUNCTION +')
      print(' '*maxLenght,'+++++++++++++++++++++++++++++')
      for what in self.methodsToRun:
        if what not in self.acceptedCalcParam:
          print('              ','**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***')
          print('              ','* '+what+' * ' + '%.8E' % outputDict[what]+'  *')
          print('              ','**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***')
    return outputDict

  def covariance(self, feature, weights=None, rowvar=1):
      """
      This method calculates the covariance Matrix for the given data.
      Unbiased unweighted covariance matrix, weights is None, bias is 0 (default)
      Biased unweighted covariance matrix,   weights is None, bias is 1
      Unbiased weighted covariance matrix,   weights is not None, bias is 0
      Biased weighted covariance matrix,     weights is not None, bias is 1
      can be calcuated depending on the selection of the inputs.
      @Inputs  -> feature, weights, bias, rowvar
      @Outputs -> covMatrix
      """
      X    = np.array(feature, ndmin=2, dtype=np.result_type(feature, np.float64))
      diff = np.zeros(feature.shape, dtype=np.result_type(feature, np.float64))
      if X.shape[0] == 1: rowvar = 1
      if rowvar:
          N = X.shape[1]
          axis = 0
      else:
          N = X.shape[0]
          axis = 1
      if weights != None:
          sumWeights       = np.sum(weights)
          sumSquareWeights = np.sum(np.square(weights))
          diff = X - np.atleast_2d(np.average(X, axis=1-axis, weights=weights)).T
      else:
          diff = X - np.mean(X, axis=1-axis, keepdims=True)
      if weights != None:
          if not self.biased: fact = sumWeights/(sumWeights*sumWeights - sumSquareWeights)
          else:               fact = 1/sumWeights
      else:
          if not self.biased: fact = float(1.0/(N-1))
          else:               fact = float(1.0/N)
      if fact <= 0:
          warnings.warn("Degrees of freedom <= 0", RuntimeWarning)
          fact = 0.0
      if not rowvar:
          covMatrix = (np.dot(diff.T, diff.conj())*fact).squeeze()
      else:
          covMatrix = (np.dot(diff, diff.T.conj())*fact).squeeze()
      return covMatrix

  def corrCoeff(self, feature, weights=None, rowvar=1):
      covM = self.covariance(feature, weights, rowvar)
      try:
        d = np.diag(covM)
      except ValueError:  # scalar covariance
      # nan if incorrect value (nan, inf, 0), 1 otherwise
        return covM / covM
      return covM / np.sqrt(np.multiply.outer(d, d))
#
#
#

class LoadCsvIntoInternalObject(BasePostProcessor):
  """
    LoadCsvIntoInternalObject pp class. It is in charge of loading CSV files into one of the internal object (Data(s) or HDF5)
  """
  def __init__(self):
    BasePostProcessor.__init__(self)
    self.sourceDirectory = None
    self.listOfCsvFiles = []
    self.printTag = utils.returnPrintTag('POSTPROCESSOR LoadCsv')

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']
    if '~' in self.sourceDirectory               : self.sourceDirectory = os.path.expanduser(self.sourceDirectory)
    if not os.path.isabs(self.sourceDirectory)   : self.sourceDirectory = os.path.normpath(os.path.join(self.__workingDir,self.sourceDirectory))
    if not os.path.exists(self.sourceDirectory)  : raise IOError(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + "-> The directory indicated for PostProcessor "+ self.name + "does not exist. Path: "+self.sourceDirectory)
    for _dir,_,_ in os.walk(self.sourceDirectory): self.listOfCsvFiles.extend(glob(os.path.join(_dir,"*.csv")))
    if len(self.listOfCsvFiles) == 0             : raise IOError(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + "-> The directory indicated for PostProcessor "+ self.name + "does not contain any csv file. Path: "+self.sourceDirectory)
    self.listOfCsvFiles.sort()

  def inputToInternal(self,currentInput): return self.listOfCsvFiles

  def _localReadMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag =="directory": self.sourceDirectory = child.text
    if not self.sourceDirectory: raise IOError(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + "-> The PostProcessor "+ self.name + "needs a directory for loading the csv files!")

  def collectOutput(self,finishedjob,output):
    #output
    '''collect the output file in the output object'''
    for index,csvFile in enumerate(self.listOfCsvFiles):

      attributes={"prefix":str(index),"input_file":self.name,"type":"csv","name":os.path.join(self.sourceDirectory,csvFile)}
      metadata = finishedjob.returnMetadata()
      if metadata:
        for key in metadata: attributes[key] = metadata[key]
      try:                   output.addGroup(attributes,attributes)
      except AttributeError:
        output.addOutput(os.path.join(self.sourceDirectory,csvFile),attributes)
        if metadata:
          for key,value in metadata.items(): output.updateMetadata(key,value,attributes)

  def run(self, InputIn):  return self.listOfCsvFiles

class LimitSurface(BasePostProcessor):
  """
    LimitSurface filter class. It computes the limit surface associated to a dataset
  """

  def __init__(self):
    BasePostProcessor.__init__(self)
    self.parameters        = {}               #parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.surfPoint         = None             #coordinate of the points considered on the limit surface
    self.testMatrix        = None             #This is the n-dimensional matrix representing the testing grid
    self.oldTestMatrix     = None             #This is the test matrix to use to store the old evaluation of the function
    self.functionValue     = {}               #This a dictionary that contains np vectors with the value for each variable and for the goal function
    self.ROM               = None
    self.externalFunction  = None
    self.subGridTol        = 1.0e-4
    self.requiredAssObject = (True,(['ROM','Function'],[-1,1]))
    self.printTag = utils.returnPrintTag('POSTPROCESSOR LIMITSURFACE')

  def inputToInternal(self,currentInp):
    # each post processor knows how to handle the coming inputs. The BasicStatistics postprocessor accept all the input type (files (csv only), hdf5 and datas
    if type(currentInp) == list: currentInput = currentInp[-1]
    else                         : currentInput = currentInp
    if type(currentInp) == dict:
      if 'targets' in currentInput.keys(): return
    inputDict = {'targets':{},'metadata':{}}
    try: inType = currentInput.type
    except:
      if type(currentInput) in [str,bytes,unicode]: inType = "file"
      else: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> LimitSurface postprocessor accepts files,HDF5,Data(s) only! Got '+ str(type(currentInput)))
    if inType == 'file':
      if currentInput.endswith('csv'): pass
    if inType == 'HDF5': pass # to be implemented
    if inType in ['TimePointSet']:
      for targetP in self.parameters['targets']:
        if   targetP in currentInput.getParaKeys('input' ): inputDict['targets'][targetP] = currentInput.getParam('input' ,targetP)
        elif targetP in currentInput.getParaKeys('output'): inputDict['targets'][targetP] = currentInput.getParam('output',targetP)
      inputDict['metadata'] = currentInput.getAllMetadata()
    # to be added
    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.externalFunction = self.assemblerDict['Function'][0][3]
    if 'ROM' not in self.assemblerDict.keys():
      mySrting= ','.join(list(self.parameters['targets']))
      self.ROM = SupervisedLearning.returnInstance('SciKitLearn',**{'SKLtype':'neighbors|KNeighborsClassifier','Features':mySrting,'Target':self.externalFunction.name})
    else: self.ROM = self.assemblerDict['ROM'][0][3]
    self.ROM.reset()
    self.__workingDir = runInfo['WorkingDir']
    indexes = [-1,-1]
    for index,inp in enumerate(self.inputs):
      if type(inp) in [str,bytes,unicode]: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> LimitSurface PostProcessor only accepts Data(s) as inputs!')
      if inp.type in ['TimePointSet','TimePoint']: indexes[0] = index
    if indexes[0] == -1: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> LimitSurface PostProcessor needs a TimePoint or TimePointSet as INPUT!!!!!!')
    else:
      # check if parameters are contained in the data
      inpKeys = self.inputs[indexes[0]].getParaKeys("inputs")
      outKeys = self.inputs[indexes[0]].getParaKeys("outputs")
      self.paramType ={}
      for param in self.parameters['targets']:
        if param not in inpKeys+outKeys: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> LimitSurface PostProcessor: The param '+ param+' not contained in Data '+self.inputs[indexes[0]].name +' !')
        if param in inpKeys: self.paramType[param] = 'inputs'
        else:                self.paramType[param] = 'outputs'
    self.nVar        = len(self.parameters['targets'])         #Total number of variables
    stepLenght        = self.subGridTol**(1./float(self.nVar)) #build the step size in 0-1 range such as the differential volume is equal to the tolerance
    self.axisName     = []                                     #this list is the implicit mapping of the name of the variable with the grid axis ordering self.axisName[i] = name i-th coordinate
    self.gridVectors  = {}
    #here we build lambda function to return the coordinate of the grid point depending if the tolerance is on probability or on volume
    stepParam = lambda x: [stepLenght*(max(self.inputs[indexes[0]].getParam(self.paramType[x],x))-min(self.inputs[indexes[0]].getParam(self.paramType[x],x))),
                                       min(self.inputs[indexes[0]].getParam(self.paramType[x],x)),
                                       max(self.inputs[indexes[0]].getParam(self.paramType[x],x))]

    #moving forward building all the information set
    pointByVar = [None]*self.nVar                              #list storing the number of point by cooridnate
    #building the grid point coordinates
    for varId, varName in enumerate(self.parameters['targets']):
      self.axisName.append(varName)
      [myStepLenght, start, end]  = stepParam(varName)
      if start == end:
        start = start - 0.001*start
        end   = end   + 0.001*end
        myStepLenght = stepLenght*(end - start)
      stepLenght
      start                      += 0.5*myStepLenght
      self.gridVectors[varName]   = np.arange(start,end,myStepLenght)
      pointByVar[varId]           = np.shape(self.gridVectors[varName])[0]
    self.gridShape                = tuple   (pointByVar)          #tuple of the grid shape
    self.testGridLenght           = np.prod (pointByVar)          #total number of point on the grid
    self.testMatrix               = np.zeros(self.gridShape)      #grid where the values of the goalfunction are stored
    self.oldTestMatrix            = np.zeros(self.gridShape)      #swap matrix fro convergence test
    self.gridCoorShape            = tuple(pointByVar+[self.nVar]) #shape of the matrix containing all coordinate of all points in the grid
    self.gridCoord                = np.zeros(self.gridCoorShape)  #the matrix containing all coordinate of all points in the grid
    #filling the coordinate on the grid
    myIterator = np.nditer(self.gridCoord,flags=['multi_index'])
    while not myIterator.finished:
      coordinateID  = myIterator.multi_index[-1]
      axisName      = self.axisName[coordinateID]
      valuePosition = myIterator.multi_index[coordinateID]
      self.gridCoord[myIterator.multi_index] = self.gridVectors[axisName][valuePosition]
      myIterator.iternext()
    self.axisStepSize = {}
    for varName in self.parameters['targets']:
      self.axisStepSize[varName] = np.asarray([self.gridVectors[varName][myIndex+1]-self.gridVectors[varName][myIndex] for myIndex in range(len(self.gridVectors[varName])-1)])

    print('Initiate training')
    self.functionValue.update(self.inputs[indexes[0]].getParametersValues('input'))
    self.functionValue.update(self.inputs[indexes[0]].getParametersValues('output'))
    #recovery the index of the last function evaluation performed
    if self.externalFunction.name in self.functionValue.keys(): indexLast = len(self.functionValue[self.externalFunction.name])-1
    else                                                      : indexLast = -1

    #index of last set of point tested and ready to perform the function evaluation
#
    indexEnd  = len(self.functionValue[self.axisName[0]])-1
    tempDict  = {}

    if self.externalFunction.name in self.functionValue.keys():
      self.functionValue[self.externalFunction.name] = np.append( self.functionValue[self.externalFunction.name], np.zeros(indexEnd-indexLast))
    else: self.functionValue[self.externalFunction.name] = np.zeros(indexEnd+1)

    for myIndex in range(indexLast+1,indexEnd+1):
      for key, value in self.functionValue.items(): tempDict[key] = value[myIndex]
      #self.hangingPoints= self.hangingPoints[    ~(self.hangingPoints==np.array([tempDict[varName] for varName in self.axisName])).all(axis=1)     ][:]
      self.functionValue[self.externalFunction.name][myIndex] =  self.externalFunction.evaluate('residuumSign',tempDict)
      if abs(self.functionValue[self.externalFunction.name][myIndex]) != 1.0: raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> LimitSurface: the function evaluation of the residuumSign method needs to return a 1 or -1!')
      if self.externalFunction.name in self.inputs[indexes[0]].getParaKeys('inputs'): self.inputs[indexes[0]].self.updateInputValue (self.externalFunction.name,self.functionValue[self.externalFunction.name][myIndex])
      if self.externalFunction.name in self.inputs[indexes[0]].getParaKeys('output'): self.inputs[indexes[0]].self.updateOutputValue(self.externalFunction.name,self.functionValue[self.externalFunction.name][myIndex])
    if np.sum(self.functionValue[self.externalFunction.name]) == float(len(self.functionValue[self.externalFunction.name])) or np.sum(self.functionValue[self.externalFunction.name]) == -float(len(self.functionValue[self.externalFunction.name])):
      raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> LimitSurface: all the Function evaluations brought to the same result (No Limit Surface has been crossed...). Increase or change the data set!')

#
    #printing----------------------
    if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Mapping of the goal function evaluation performed')
    if self.debug:
      print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Already evaluated points and function values:')
      keyList = list(self.functionValue.keys())
      print(','.join(keyList))
      for index in range(indexEnd+1):
        print(','.join([str(self.functionValue[key][index]) for key in keyList]))
    #printing----------------------
    tempDict = {}
    for name in self.axisName: tempDict[name] = np.asarray(self.functionValue[name])
    tempDict[self.externalFunction.name] = self.functionValue[self.externalFunction.name]
    self.ROM.train(tempDict)
    print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Training performed')
    if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Training finished')
  def _localReadMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    """
    child = xmlNode.find("parameters")
    if child == None: raise IOError(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> No Parameters specified in XML input!!!!')
    self.parameters['targets'] = child.text.split(',')
    child = xmlNode.find("tolerance")
    if child != None: self.subGridTol = float(child.text)

  def collectOutput(self,finishedjob,output):
    #output
    if finishedjob.returnEvaluation() == -1: raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> No available Output to collect (Run probabably is not finished yet)')
    print(finishedjob.returnEvaluation())
    limitSurf = finishedjob.returnEvaluation()[1]
    if limitSurf[0]!=None:
      for varName in output.getParaKeys('inputs'):
        for varIndex in range(len(self.axisName)):
          if varName == self.axisName[varIndex]:
            output.removeInputValue(varName)
            for value in limitSurf[0][:,varIndex]: output.updateInputValue(varName,copy.copy(value))
      output.removeOutputValue('OutputPlaceOrder')
      for value in limitSurf[1]: output.updateOutputValue('OutputPlaceOrder',copy.copy(value))

  def run(self, InputIn): # inObj,workingDir=None):
    """
     Function to finalize the filter => execute the filtering
     @ In , dictionary       : dictionary of data to process
     @ Out, dictionary       : Dictionary with results
    """
    #Input  = self.inputToInternal(InputIn)
#     print('Initiate training')
#     self.functionValue.update(InputIn[-1].getParametersValues('input'))
#     self.functionValue.update(InputIn[-1].getParametersValues('output'))
#     #recovery the index of the last function evaluation performed
#     if self.externalFunction.name in self.functionValue.keys(): indexLast = len(self.functionValue[self.externalFunction.name])-1
#     else                                                      : indexLast = -1
#
#     #index of last set of point tested and ready to perform the function evaluation
# #
#     indexEnd  = len(self.functionValue[self.axisName[0]])-1
#     tempDict  = {}
#
#     if self.externalFunction.name in self.functionValue.keys():
#       self.functionValue[self.externalFunction.name] = np.append( self.functionValue[self.externalFunction.name], np.zeros(indexEnd-indexLast))
#     else: self.functionValue[self.externalFunction.name] = np.zeros(indexEnd+1)
#
#     for myIndex in range(indexLast+1,indexEnd+1):
#       for key, value in self.functionValue.items(): tempDict[key] = value[myIndex]
#       #self.hangingPoints= self.hangingPoints[    ~(self.hangingPoints==np.array([tempDict[varName] for varName in self.axisName])).all(axis=1)     ][:]
#       self.functionValue[self.externalFunction.name][myIndex] =  self.externalFunction.evaluate('residuumSign',tempDict)
#       if abs(self.functionValue[self.externalFunction.name][myIndex]) != 1.0: raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> LimitSurface: the function evaluation of the residuumSign method needs to return a 1 or -1!')
#       if self.externalFunction.name in InputIn[-1].getParaKeys('inputs'): InputIn[-1].self.updateInputValue (self.externalFunction.name,self.functionValue[self.externalFunction.name][myIndex])
#       if self.externalFunction.name in InputIn[-1].getParaKeys('output'): InputIn[-1].self.updateOutputValue(self.externalFunction.name,self.functionValue[self.externalFunction.name][myIndex])
#     if np.sum(self.functionValue[self.externalFunction.name]) == float(len(self.functionValue[self.externalFunction.name])) or np.sum(self.functionValue[self.externalFunction.name]) == -float(len(self.functionValue[self.externalFunction.name])):
#       raise Exception(self.printTag+': ' +utils.returnPrintPostTag("ERROR") + '-> LimitSurface: all the Function evaluations brought to the same result (No Limit Surface has been crossed...). Increase or change the data set!')
#
# #
#     #printing----------------------
#     if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Mapping of the goal function evaluation performed')
#     if self.debug:
#       print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Already evaluated points and function values:')
#       keyList = list(self.functionValue.keys())
#       print(','.join(keyList))
#       for index in range(indexEnd+1):
#         print(','.join([str(self.functionValue[key][index]) for key in keyList]))
#     #printing----------------------
#     tempDict = {}
#     for name in self.axisName: tempDict[name] = np.asarray(self.functionValue[name])
#     tempDict[self.externalFunction.name] = self.functionValue[self.externalFunction.name]
#     print("lupo")
#     print(self.ROM.__dict__)
#     print("lup2")
#     print(self.ROM.SupervisedEngine.values()[0].__dict__)
#     print("lup3")
#     self.ROM.train(tempDict)
#
#     print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Training performed')
#     if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Training finished')
    np.copyto(self.oldTestMatrix,self.testMatrix)                                #copy the old solution for convergence check
    self.testMatrix.shape     = (self.testGridLenght)                            #rearrange the grid matrix such as is an array of values
    self.gridCoord.shape      = (self.testGridLenght,self.nVar)                  #rearrange the grid coordinate matrix such as is an array of coordinate values
    tempDict ={}
    for  varId, varName in enumerate(self.axisName): tempDict[varName] = self.gridCoord[:,varId]
    self.testMatrix[:]        = self.ROM.evaluate(tempDict)                      #get the prediction on the testing grid
    self.testMatrix.shape     = self.gridShape                                   #bring back the grid structure
    self.gridCoord.shape      = self.gridCoorShape                               #bring back the grid structure
    if self.debug: print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Prediction performed')
    #here next the points that are close to any change are detected by a gradient (it is a pre-screener)
    toBeTested = np.squeeze(np.dstack(np.nonzero(np.sum(np.abs(np.gradient(self.testMatrix)),axis=0))))
    #printing----------------------
    if self.debug:
      print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface:  Limit surface candidate points')
      for coordinate in np.rollaxis(toBeTested,0):
        myStr = ''
        for iVar, varnName in enumerate(self.axisName): myStr +=  varnName+': '+str(coordinate[iVar])+'      '
        print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: ' + myStr+'  value: '+str(self.testMatrix[tuple(coordinate)]))
    #printing----------------------
    #check which one of the preselected points is really on the limit surface
    listsurfPoint = []
    myIdList      = np.zeros(self.nVar)
    for coordinate in np.rollaxis(toBeTested,0):
      myIdList[:] = coordinate
      if int(self.testMatrix[tuple(coordinate)])<0: #we seek the frontier sitting on the -1 side
        for iVar in range(self.nVar):
          if coordinate[iVar]+1<self.gridShape[iVar]: #coordinate range from 0 to n-1 while shape is equal to n
            myIdList[iVar]+=1
            if self.testMatrix[tuple(myIdList)]>=0:
              listsurfPoint.append(copy.copy(coordinate))
              break
            myIdList[iVar]-=1
          if coordinate[iVar]>0:
            myIdList[iVar]-=1
            if self.testMatrix[tuple(myIdList)]>=0:
              listsurfPoint.append(copy.copy(coordinate))
              break
            myIdList[iVar]+=1
    #printing----------------------
    if self.debug:
      print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: Limit surface points:')
      for coordinate in listsurfPoint:
        myStr = ''
        for iVar, varnName in enumerate(self.axisName): myStr +=  varnName+': '+str(coordinate[iVar])+'      '
        print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> LimitSurface: ' + myStr+'  value: '+str(self.testMatrix[tuple(coordinate)]))
    #printing----------------------

    #if the number of point on the limit surface is > than zero than save it
    outputPlaceOrder = np.zeros(len(listsurfPoint))
    if len(listsurfPoint)>0:
      self.surfPoint = np.ndarray((len(listsurfPoint),self.nVar))
      for pointID, coordinate in enumerate(listsurfPoint):
        self.surfPoint[pointID,:] = self.gridCoord[tuple(coordinate)]
        outputPlaceOrder[pointID] = pointID

    return self.surfPoint,outputPlaceOrder

#
#
#

class ExternalPostProcessor(BasePostProcessor):
  """
    ExternalPostProcessor class. It will apply an arbitrary python function to
    a dataset and append each specified function's output to the output data
    object, thus the function should produce a scalar value per row of data. I
    have no idea what happens if the function produces multiple outputs.
  """
  def __init__(self):
    '''
      Initialization.
    '''
    BasePostProcessor.__init__(self)
    self.methodsToRun = []              # A list of strings specifying what
                                        # methods the user wants to compute from
                                        # the external interfaces

    self.externalInterfaces = []          # A list of Function objects that
                                        # hopefully contain definitions for all
                                        # of the methods the user wants

    self.printTag = utils.returnPrintTag('POSTPROCESSOR EXTERNAL FUNCTION')
    self.requiredAssObject = (True,(['Function'],['n']))

  def errorString(self,message):
    """
      Function to format an error string for printing.
      @ In, message: A string describing the error
      @ Out, A formatted string with the appropriate tags listed
    """
    # This function can be promoted for printing error functions more easily and
    # consistently.
    return (self.printTag + ': ' + utils.returnPrintPostTag('ERROR') + '-> '
           + self.__class__.__name__ + ': ' + message)

  def warningString(self,message):
    """
      Function to format a warning string for printing.
      @ In, message: A string describing the warning
      @ Out, A formatted string with the appropriate tags listed
    """
    # This function can be promoted for printing error functions more easily and
    # consistently.
    return (self.printTag + ': ' + utils.returnPrintPostTag('Warning') + '-> '
           + self.__class__.__name__ + ': ' + message)

  def messageString(self,message):
    """
      Function to format a message string for printing.
      @ In, message: A string describing the message
      @ Out, A formatted string with the appropriate tags listed
    """
    # This function can be promoted for printing error functions more easily and
    # consistently.
    return (self.printTag + ': ' + utils.returnPrintPostTag('Message') + '-> '
           + self.__class__.__name__ + ': ' + message)

  def inputToInternal(self,currentInp):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, currentInp: Some form of data object or list of data objects handed
                        to the post-processor
      @ Out, An input dictionary this object can process
    """

    if type(currentInp) == dict:
      if 'targets' in currentInp.keys():
        return

    currentInput = currentInp
    if type(currentInput) != list:
      currentInput = [currentInput]

    inputDict = {'targets':{},'metadata':{}}
    metadata = []
    for item in currentInput:
      inType = None
      if hasattr(item,'type'):
        inType = item.type
      elif type(item).__name__ in ["str","unicode","bytes"]:
        inType = "file"
      elif type(item) in [list]:
        inType = "list"

      if inType not in ['file','HDF5','TimePointSet','list']:
        print(self.warningString('Input type ' + type(item).__name__ + ' not'
                               + ' recognized. I am going to skip it.'))
      elif inType == 'file':
        if currentInput.endswith('csv'):
          # TODO
          print(self.warningString('Input type ' + inType + ' not yet '
                                 + 'implemented. I am going to skip it.'))
      elif inType == 'HDF5':
        # TODO
          print(self.warningString('Input type ' + inType + ' not yet '
                                 + 'implemented. I am going to skip it.'))
      elif inType == 'TimePointSet':
        for param in item.getParaKeys('input'):
          inputDict['targets'][param] = item.getParam('input', param)
        for param in item.getParaKeys('output'):
          inputDict['targets'][param] = item.getParam('output', param)
        metadata.append(item.getAllMetadata())

      #Not sure if we need it, but keep a copy of every inputs metadata
      inputDict['metadata'] = metadata

    if len(inputDict['targets'].keys()) == 0: raise IOError(self.errorString("No input variables have been found in the input objects!"))
    for interface in self.externalInterfaces:
      for method in self.methodsToRun:
        # The function should reference self and use the same variable names
        # as the xml file
        for param in interface.parameterNames():
          if param not in inputDict['targets']:
            raise IOError(self.errorString('variable \"' + param + '\" unknown.'
                                          + ' Please verify your external'
                                          + ' script ('
                                          + interface.functionFile
                                          + ') variables match the data'
                                          + ' available in your dataset.'))

    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']
    for key in self.assemblerDict.keys():
      if 'Function' in key:
        indice = 0
        for value in self.assemblerDict[key]:
          self.externalInterfaces.append(self.assemblerDict[key][indice][3])
          indice += 1

  def _localReadMoreXML(self,xmlNode):
    """
      Function to grab the names of the methods this post-processor will be
      using
      @ In, xmlNode    : Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'method':
        methods = child.text.split(',')
        self.methodsToRun.extend(methods)

  def collectOutput(self,finishedJob,output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob: A JobHandler object that is in charge of running this
                         post-processor
      @ In, output: The object where we want to place our computed results
      @ Out, None
    """
    if finishedJob.returnEvaluation() == -1:
      ##TODO This does not feel right
      raise Exception(self.errorString('No available Output to collect (Run '
                                       + 'probably did not finish yet)'))
    inputList = finishedJob.returnEvaluation()[0]
    outputDict = finishedJob.returnEvaluation()[1]

    if type(output).__name__ in ["str","unicode","bytes"]:
      print(self.warningString('Output type ' + type(output).__name__ + ' not'
                               + ' yet implemented. I am going to skip it.'))
    elif output.type == 'Datas':
      print(self.warningString('Output type ' + type(output).__name__ + ' not'
                               + ' yet implemented. I am going to skip it.'))
    elif output.type == 'HDF5':
      print(self.warningString('Output type ' + type(output).__name__ + ' not'
                               + ' yet implemented. I am going to skip it.'))
    elif output.type == 'TimePointSet':
      requestedInput = output.getParaKeys('input')
      requestedOutput = output.getParaKeys('output')
      ## The user can simply ask for a computation that may exist in multiple
      ## interfaces, in that case, we will need to qualify their names for the
      ## output. The names should already be qualified from the outputDict.
      ## However, the user may have already qualified the name, so make sure and
      ## test whether the unqualified name exists in the requestedOutput before
      ## replacing it.
      for key,replacements in outputDict['qualifiedNames'].iteritems():
        if key in requestedOutput:
          requestedOutput.remove(key)
          requestedOutput.extend(replacements)

      ## Grab all data from the outputDict and anything else requested not
      ## present in the outputDict will be copied from the input data.
      ## TODO: User may want to specify which dataset the parameter comes from.
      ##       For now, we assume that if we find more than one an error will
      ##      occur.
      ## FIXME: There is an issue that the data size should be determined before
      ##        entering this loop, otherwise if say a scalar is first added,
      ##        then dataLength will be 1 and everything longer will be placed
      ##        in the Metadata.
      ##        How do we know what size the output data should be?
      dataLength = None
      for key in requestedInput+requestedOutput:
        storeInOutput = True
        value = []
        if key in outputDict:
          value = outputDict[key]
        else:
          foundCount = 0
          if key in requestedInput:
            for inputData in inputList:
              if key in inputData.getParametersValues('input').keys():
                value = inputData.getParametersValues('input')[key]
                foundCount += 1
          else:
            for inputData in inputList:
                if key in inputData.getParametersValues('output').keys():
                  value = inputData.getParametersValues('output')[key]
                  foundCount += 1

          if foundCount == 0:
            raise IOError(self.errorString(key + ' not found in the input '
                                            + 'object or the computed output '
                                            + 'object.'))
          elif foundCount > 1:
            raise IOError(self.errorString(key + ' is ambiguous since it occurs'
                                            + ' in multiple input objects.'))

        ## We need the size to ensure the data size is consistent, but there
        ## is no guarantee the data is not scalar, so this check is necessary
        myLength = 1
        if not hasattr(value, "__iter__"):
          value = [value]
        myLength = len(value)

        if dataLength is None:
          dataLength = myLength
        elif dataLength != myLength:
          print(self.warningString('Requested output for ' + key + ' has a'
                                    + ' non-conformant data size ('
                                    + str(dataLength) + ' vs ' + str(myLength)
                                    + '), it is being placed in the metadata.'))
          storeInOutput = False

        ## Finally, no matter what, place the requested data somewhere
        ## accessible
        if storeInOutput:
          if key in requestedInput:
            for val in value:
              output.updateInputValue(key, val)
          else:
            for val in value:
              output.updateOutputValue(key, val)
        else:
          if not hasattr(value, "__iter__"):
            value = [value]
          for val in value:
            output.updateMetadata(key, val)

    else:
      raise IOError(errorString('Unknown output type: ' + str(output.type)))

  def run(self, InputIn):
    """
     Function to finalize the filter => execute the filtering
     @ In , dictionary       : dictionary of data to process
     @ Out, dictionary       : Dictionary with results
    """
    Input  = self.inputToInternal(InputIn)
    outputDict = {'qualifiedNames' : {}}
    ## This will map the name to its appropriate interface and method
    ## in the case of a function being defined in two separate files, we
    ## qualify the output by appending the name of the interface from which it
    ## originates
    methodMap = {}

    ## First check all the requested methods are available and if there are
    ## duplicates then qualify their names for the user
    for method in self.methodsToRun:
      matchingInterfaces = []
      for interface in self.externalInterfaces:
        if method in interface.availableMethods():
          matchingInterfaces.append(interface)


      if len(matchingInterfaces) == 0:
        print(self.warningString(method + ' not found. I will skip it.'))
      elif len(matchingInterfaces) == 1:
        methodMap[method] = (matchingInterfaces[0],method)
      else:
        outputDict['qualifiedNames'][method] = []
        for interface in matchingInterfaces:
          methodName = interface.name + '.' + method
          methodMap[methodName] = (interface,method)
          outputDict['qualifiedNames'][method].append(methodName)

    ## Evaluate the method and add it to the outputDict, also if the method
    ## adjusts the input data, then you should update it as well.
    for methodName,(interface,method) in methodMap.iteritems():
      outputDict[methodName] = interface.evaluate(method,Input['targets'])
      for target in Input['targets']:
        if hasattr(interface,target):
          outputDict[target] = getattr(interface, target)

    return outputDict

'''
 Interface Dictionary (factory) (private)
'''
__base                                       = 'PostProcessor'
__interFaceDict                              = {}
__interFaceDict['SafestPoint'              ] = SafestPoint
__interFaceDict['PrintCSV'                 ] = PrintCSV
__interFaceDict['BasicStatistics'          ] = BasicStatistics
__interFaceDict['LoadCsvIntoInternalObject'] = LoadCsvIntoInternalObject
__interFaceDict['LimitSurface'             ] = LimitSurface
__interFaceDict['ComparisonStatistics'     ] = ComparisonStatistics
__interFaceDict['External'                 ] = ExternalPostProcessor
__knownTypes                                 = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

def returnInstance(Type):
  """
    function used to generate a Filter class
    @ In, Type : Filter type
    @ Out,Instance of the Specialized Filter class
  """
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
