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
  This module contains the Limit Surface Search sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from collections import OrderedDict
import copy
import numpy as np
from operator import mul
from functools import reduce
from scipy import spatial
from math import ceil
import sys
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
#from PostProcessors import LimitSurface
from PostProcessors import BasicStatistics
from .AdaptiveSampler import AdaptiveSampler
from .MonteCarlo import MonteCarlo
import Distributions
from AMSC_Object import AMSC_Object
from utils import randomUtils
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------


class AdaptiveMonteCarlo(AdaptiveSampler):
  """
    A sampler that will adaptively locate the limit surface of a given problem
  """

  statErVals = ['expectedValue',
                'median',
                'variance',
                'sigma',
                'skewness',
                'kurtosis']
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(AdaptiveMonteCarlo, cls).getInputSpecification()
    convergenceInput = InputData.parameterInputFactory('Convergence')
    convergenceInput.addSub(InputData.parameterInputFactory('limit', contentType=InputTypes.IntegerType, strictMode=True))
    convergenceInput.addSub(InputData.parameterInputFactory('forceIteration', contentType=InputTypes.BoolType, strictMode=True))
    convergenceInput.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType, strictMode=True))
    for statEr in cls.statErVals:
      statErSpecification = InputData.parameterInputFactory(statEr, contentType=InputTypes.FloatType)
      statErSpecification.addParam("prefix", InputTypes.StringType)
      statErSpecification.addParam("var", InputTypes.StringListType)
      convergenceInput.addSub(statErSpecification)

    inputSpecification.addSub(convergenceInput)

    targetEvaluationInput = InputData.parameterInputFactory("TargetEvaluation", contentType=InputTypes.StringType)
    targetEvaluationInput.addParam("type", InputTypes.StringType)
    targetEvaluationInput.addParam("class", InputTypes.StringType)
    inputSpecification.addSub(targetEvaluationInput)
    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    AdaptiveSampler.__init__(self)
    self.persistence         = 5                #this is the number of times the error needs to fell below the tolerance before considering the sim converged
    self.forceIteration      = False            #this flag control if at least a self.limit number of iteration should be done
    self.axisName            = None             #this is the ordered list of the variable names (ordering match self.gridStepSize and the ordering in the test matrixes)
    self.solutionExport      = None             #This is the data used to export the solution (it could also not be present)
    self.nVar                = 0                #this is the number of the variable sampled
    self.basicStatPP      = None                # post-processor to compute the basic statistics
    self.converged      = False                 # flag that is set to True when the sampler converged
    self.threshold      = 0                     # Post-rank function value
    self.printTag            = 'SAMPLER ADAPTIVE MC'
    self.addAssemblerObject('TargetEvaluation','n')


  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == "convergence":
        for grandchild in child.subparts:
          tag = grandchild.getName()
          if tag == "limit":
            self.limit = grandchild.value
            if self.limit is None:
              self.raiseAnError(IOError,self,'Adaptive Monte Carlo sampler '+self.name+' needs the limit block (number of samples) in the Convergence block')
          elif tag == "persistence":
            self.persistence = grandchild.value
            self.raiseADebug('Persistence is set at',self.gainGrowthFactor)
          elif tag == "forceIteration":
            self.forceIteration = grandchild.value

  def localInitialize(self,solutionExport=None):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, solutionExport, DataObjects, optional, a PointSet to hold the solution (a list of limit surface points)
      @ Out, None
    """
    self.converged        = False
    self.basicStatPP   = BasicStatistics(self.messageHandler)
    if 'TargetEvaluation' in self.assemblerDict.keys():
      self.lastOutput = self.assemblerDict['TargetEvaluation'][0][3]
    self.solutionExport    = solutionExport
    # check if solutionExport is actually a "DataObjects" type "PointSet"
    if solutionExport.type != "PointSet":
      self.raiseAnError(IOError,'solutionExport type is not a PointSet. Got '+ solutionExport.type +'!')
    # set number of job request-able after a new evaluation

    self.nVar         = len(self.distDict.keys())              # Total number of variables
    bounds          = {"lowerBounds":{},"upperBounds":{}}
    transformMethod = {}
    for varName in self.distDict.keys():
      if self.toleranceWeight!='cdf':
        bounds["lowerBounds"][varName.replace('<distribution>','')], bounds["upperBounds"][varName.replace('<distribution>','')] = self.distDict[varName].lowerBound, self.distDict[varName].upperBound
      else:
        bounds["lowerBounds"][varName.replace('<distribution>','')], bounds["upperBounds"][varName.replace('<distribution>','')] = 0.0, 1.0
        transformMethod[varName.replace('<distribution>','')] = [self.distDict[varName].ppf]
    #moving forward building all the information set
    self.axisName = list(self.distDict.keys())
    self.axisName.sort()


    # initialize BasicStatistics PP
    self.basicStatPP._initFromDict()
    self.basicStatPP.assemblerDict = self.assemblerDict
    self.basicStatPP._initializeLSpp({'WorkingDir':None},[self.lastOutput],{'computeCells':self.tolerance != self.subGridTol})
    matrixShape = self.basicStatPP.getTestMatrix().shape
    self.persistenceMatrix[self.name+"LSpp"]  = np.zeros(matrixShape) #matrix that for each point of the testing grid tracks the persistence of the limit surface position
    self.oldTestMatrix[self.name+"LSpp"]      = np.zeros(matrixShape) #swap matrix fro convergence test
    self.hangingPoints                        = np.ndarray((0, self.nVar))
    self.raiseADebug('Initialization done')


  ###############
  # Run Methods #
  ###############

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    """
      General function (available to all samplers) that finalize the sampling
      calculation just ended. In this case, The function is aimed to check if
      all the batch calculations have been performed
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
      @ Out, None
    """
    #check if all sampling is done
    if self.jobHandler.isFinished():
      self.batchDone = True
    else:
      self.batchDone = False
    #batchDone is used to check if the sampler should find new points.

  def localGenerateInput(self,model,oldInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    #  Alternatively, though I don't think we do this yet:
    #  compute the direction normal to the surface, compute the derivative
    #  normal to the surface of the probability, check the points where the
    #  derivative probability is the lowest

    # create values dictionary
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    self.raiseADebug('generating input')
    varSet=False

    # DM: This sequence gets used repetitively, so I am promoting it to its own
    #  variable
    axisNames = [key.replace('<distribution>','') for key in self.axisName]

    if self.surfPoint is not None and len(self.surfPoint) > 0:
      if self.batchStrategy == 'none':
        self.__scoreCandidates()
        maxDistance, maxGridId, maxId =  0.0, "", 0
        for key, value in sorted(self.invPointPersistence.items()):
          if key != self.exceptionGrid and self.surfPoint[key] is not None:
            localMax = np.max(self.scores[key])
            if localMax > maxDistance:
              maxDistance, maxGridId, maxId  = localMax, key,  np.argmax(self.scores[key])
        if maxDistance > 0.0:
          for varIndex, _ in enumerate([key.replace('<distribution>','') for key in self.axisName]):
            self.values[self.axisName[varIndex]] = copy.copy(float(self.surfPoint[maxGridId][maxId,varIndex]))
            self.inputInfo['SampledVarsPb'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
            self.inputInfo['ProbabilityWeight-'+self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
          varSet=True
        else:
          self.raiseADebug('Maximum score is 0.0')
      elif self.batchStrategy.startswith('max'):
        ########################################################################
        ## Initialize the queue with as many points as requested or as many as
        ## possible
        if len(self.toProcess) == 0:
          self.__scoreCandidates()
          edges = []

          flattenedSurfPoints = list()
          flattenedBandPoints = list()
          flattenedScores     = list()
          for key in self.bandIndices.keys():
            flattenedSurfPoints = flattenedSurfPoints + list(self.surfPoint[key])
            flattenedScores = flattenedScores + list(self.scores[key])
            flattenedBandPoints = flattenedBandPoints + self.listSurfPoint[key] + self.bandIndices[key]

          flattenedSurfPoints = np.array(flattenedSurfPoints)
          for i,iCoords in enumerate(flattenedBandPoints):
            for j in range(i+1, len(flattenedBandPoints)):
              jCoords = flattenedBandPoints[j]
              ijValidNeighbors = True
              for d in range(len(jCoords)):
                if abs(iCoords[d] - jCoords[d]) > 1:
                  ijValidNeighbors = False
                  break
              if ijValidNeighbors:
                edges.append((i,j))
                edges.append((j,i))

          names = axisNames[:] #make copy
          names.append('score')
          amsc = AMSC_Object(X=flattenedSurfPoints, Y=flattenedScores,
                             w=None, names=names, graph='none',
                             gradient='steepest', normalization='feature',
                             persistence='difference', edges=edges, debug=False)
          plevel = self.simplification*(max(flattenedScores)-min(flattenedScores))
          partitions = amsc.StableManifolds(plevel)
          mergeSequence = amsc.GetMergeSequence()
          maxIdxs = list(set(partitions.keys()))

          thresholdLevel = self.threshold*(max(flattenedScores)-min(flattenedScores))+min(flattenedScores)
          # Sort the maxima based on decreasing function value, thus the top
          # candidate is the first element.
          if self.batchStrategy.endswith('V'):
            sortedMaxima = sorted(maxIdxs, key=lambda idx: flattenedScores[idx], reverse=True)
          else:
          # Sort the maxima based on decreasing persistence value, thus the top
          # candidate is the first element.
            sortedMaxima = sorted(maxIdxs, key=lambda idx: mergeSequence[idx][1], reverse=True)
          B = min(self.maxBatchSize,len(sortedMaxima))
          for idx in sortedMaxima[0:B]:
            if flattenedScores[idx] >= thresholdLevel:
              self.toProcess.append(flattenedSurfPoints[idx,:])
          if len(self.toProcess) == 0:
            self.toProcess.append(flattenedSurfPoints[np.argmax(flattenedScores),:])
        ########################################################################
        ## Select one sample
        selectedPoint = self.toProcess.pop()
        for varIndex, varName in enumerate(axisNames):
          self.values[self.axisName[varIndex]] = float(selectedPoint[varIndex])
          self.inputInfo['SampledVarsPb'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
          self.inputInfo['ProbabilityWeight-'+self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
        varSet=True
      elif self.batchStrategy == 'naive':
        ########################################################################
        ## Initialize the queue with as many points as requested or as many as
        ## possible
        if len(self.toProcess) == 0:
          self.__scoreCandidates()
          sortedIndices = sorted(range(len(self.scores)), key=lambda k: self.scores[k],reverse=True)
          B = min(self.maxBatchSize,len(sortedIndices))
          for idx in sortedIndices[0:B]:
            self.toProcess.append(self.surfPoint[idx,:])
          if len(self.toProcess) == 0:
            self.toProcess.append(self.surfPoint[np.argmax(self.scores),:])
        ########################################################################
        ## Select one sample
        selectedPoint = self.toProcess.pop()
        for varIndex, varName in enumerate(axisNames):
          self.values[self.axisName[varIndex]] = float(selectedPoint[varIndex])
          self.inputInfo['SampledVarsPb'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
          self.inputInfo['ProbabilityWeight-'+self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
        varSet=True

    if not varSet:
      #here we are still generating the batch
      for key in sorted(self.distDict.keys()):
        if self.toleranceWeight=='cdf':
          self.values[key]                       = self.distDict[key].ppf(float(randomUtils.random()))
        else:
          self.values[key]                       = self.distDict[key].lowerBound+(self.distDict[key].upperBound-self.distDict[key].lowerBound)*float(randomUtils.random())
        self.inputInfo['distributionName'][key]  = self.toBeSampled[key]
        self.inputInfo['distributionType'][key]  = self.distDict[key].type
        self.inputInfo['SampledVarsPb'   ][key]  = self.distDict[key].pdf(self.values[key])
        self.inputInfo['ProbabilityWeight-'+key] = self.distDict[key].pdf(self.values[key])
        self.addMetaKeys(['ProbabilityWeight-'+key])
    self.inputInfo['PointProbability'    ]      = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    # the probability weight here is not used, the post processor is going to recreate the grid associated and use a ROM for the probability evaluation
    self.inputInfo['ProbabilityWeight']         = self.inputInfo['PointProbability']
    self.hangingPoints                          = np.vstack((self.hangingPoints,copy.copy(np.array([self.values[axis] for axis in self.axisName]))))
    self.raiseADebug('At counter '+str(self.counter)+' the generated sampled variables are: '+str(self.values))
    self.inputInfo['SamplerType'] = 'AdaptiveMonteCarlo'
    self.inputInfo['subGridTol' ] = self.subGridTol

  def localGetCurrentSetting(self):
    """
      Appends a given dictionary with class specific information regarding the
      current status of the object.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    if self.solutionExport!=None:
      paramDict['The solution is exported in '    ] = 'Name: ' + self.solutionExport.name + 'Type: ' + self.solutionExport.type
    if self.goalFunction!=None  :
      paramDict['The function used is '] = self.goalFunction.name
    return paramDict

  def localStillReady(self,ready): #,lastOutput=None
    """
      first perform some check to understand what it needs to be done possibly perform an early return
      ready is returned
      lastOutput should be present when the next point should be chosen on previous iteration and convergence checked
      lastOutput it is not considered to be present during the test performed for generating an input batch
      ROM if passed in it is used to construct the test matrix otherwise the nearest neighbor value is used
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    self.raiseADebug('From method localStillReady...')
    # if the limit surface search has converged, we return False
    if self.converged:
      return False
    #test on what to do
    if not ready:
      return ready #if we exceeded the limit just return that we are done
    if type(self.lastOutput) == dict:
      if self.lastOutput == None and not self.limitSurfacePP.ROM.amITrained:
        return ready
    else:
      #if the last output is not provided I am still generating an input batch, if the rom was not trained before we need to start clean
      if len(self.lastOutput) == 0 and not self.limitSurfacePP.ROM.amITrained:
        return ready
    #first evaluate the goal function on the newly sampled points and store them in mapping description self.functionValue RecontructEnding
    oldSizeLsFunctionValue = 0 if len(self.limitSurfacePP.getFunctionValue()) == 0 else len(self.limitSurfacePP.getFunctionValue()[self.goalFunction.name])
    if type(self.lastOutput) == dict:
      self.limitSurfacePP._initializeLSppROM(self.lastOutput,False)
    else:
      if len(self.lastOutput) > 0:
        self.limitSurfacePP._initializeLSppROM(self.lastOutput,False)
    self.raiseADebug('Classifier ' +self.name+' has been trained!')
    self.oldTestMatrix = copy.deepcopy(self.limitSurfacePP.getTestMatrix("all",exceptionGrid=self.exceptionGrid))    #copy the old solution (contained in the limit surface PP) for convergence check
    # evaluate the Limit Surface coordinates (return input space coordinates, evaluation vector and grid indexing)
    self.surfPoint, evaluations, self.listSurfPoint = self.limitSurfacePP.run(returnListSurfCoord = True, exceptionGrid=self.exceptionGrid, merge=False)
    self.raiseADebug('Limit Surface has been computed!')
    newSizeLsFunctionValue = len(self.limitSurfacePP.getFunctionValue()[self.goalFunction.name])  if self.goalFunction.name in self.limitSurfacePP.getFunctionValue().keys() else 0
    # check hanging points
    if self.goalFunction.name in self.limitSurfacePP.getFunctionValue().keys():
      indexLast = len(self.limitSurfacePP.getFunctionValue()[self.goalFunction.name])-1
    else:
      indexLast = -1
    #index of last set of point tested and ready to perform the function evaluation
    indexEnd  = len(self.limitSurfacePP.getFunctionValue()[self.axisName[0].replace('<distribution>','')])-1
    tempDict  = {}
    for myIndex in range(indexLast+1,indexEnd+1):
      for key, value in self.limitSurfacePP.getFunctionValue().items():
        tempDict[key] = value[myIndex]
      if len(self.hangingPoints) > 0:
        self.hangingPoints = self.hangingPoints[
          ~(self.hangingPoints==np.array([tempDict[varName]
                                          for varName in [key.replace('<distribution>','')
                                                          for key in self.axisName]])).all(axis=1)][:]
    for key,value in self.limitSurfacePP.getTestMatrix("all",exceptionGrid=self.exceptionGrid).items():
      self.persistenceMatrix[key] += value

    # get the test matrices' dictionaries to test the error
    testMatrixDict = list(self.limitSurfacePP.getTestMatrix("all",exceptionGrid=self.exceptionGrid).values())
    oldTestMatrixDict = list(self.oldTestMatrix.values())
    # the first test matrices in the list are always represented by the coarse grid
    # (if subGridTol activated) or the only grid available
    coarseGridTestMatix, coarseGridOldTestMatix = testMatrixDict.pop(0), oldTestMatrixDict.pop(0)
    # compute the Linf norm with respect the location of the LS
    testError = np.sum(np.abs(np.subtract(coarseGridTestMatix,coarseGridOldTestMatix)))
    if self.sizeGrid is None:
      self.sizeGrid = float(coarseGridTestMatix.size)
    if len(testMatrixDict) > 0:
      # compute the error
      if self.sizeSubGrid is None:
        self.sizeSubGrid = float(np.asarray(testMatrixDict).size)
      testError += np.sum(np.abs(np.subtract(testMatrixDict,oldTestMatrixDict)))/(self.sizeGrid+self.sizeSubGrid)
    else:
      testError/= self.sizeGrid

    if (testError > self.errorTolerance) or newSizeLsFunctionValue == oldSizeLsFunctionValue:
      # we still have error
      ready, self.repetition = True, 0
    else:
      # we are increasing persistence
      self.repetition +=1
    if self.persistence<self.repetition:
      ready =  False
      if self.subGridTol != self.tolerance \
         and evaluations is not None \
         and not self.refinedPerformed and self.limitSurfacePP.crossedLimitSurf:
        # we refine the grid since we converged on the coarse one. we use the "ceil" method in order to be sure
        # that the volumetric cell weight is <= of the subGridTol
        self.raiseAMessage("Grid refinement activated! Refining the evaluation grid!")
        self.limitSurfacePP.refineGrid(int(ceil((self.tolerance/self.subGridTol)**(1.0/self.nVar))))
        self.exceptionGrid, self.refinedPerformed, ready, self.repetition = self.name + "LSpp", True, True, 0
        self.persistenceMatrix.update(copy.deepcopy(self.limitSurfacePP.getTestMatrix("all",exceptionGrid=self.exceptionGrid)))
        self.errorTolerance = self.subGridTol
      else:
        self.converged = True
        if not self.limitSurfacePP.crossedLimitSurf:
          self.raiseAWarning("THE LIMIT SURFACE has NOT been crossed. The search FAILED!!!")
    self.raiseAMessage('counter: '+str(self.counter)+'       Error: {:9.6E} Repetition: {:5d}'.format(testError,self.repetition) )
    #if the number of point on the limit surface is > than compute persistence
    realAxisNames, cnt = [key.replace('<distribution>','') for key in self.axisName], 0
    if self.solutionExport is not None:
      rlz = {}
      # reset solution export
      self.solutionExport.reset()
    for gridID,listsurfPoint in self.listSurfPoint.items():
      if len(listsurfPoint)>0:
        self.invPointPersistence[gridID] = np.ones(len(listsurfPoint))
        if self.firstSurface == False:
          for pointID, coordinate in enumerate(listsurfPoint):
            self.invPointPersistence[gridID][pointID]=abs(self.persistenceMatrix[gridID][tuple(coordinate)])
          maxPers = np.max(self.invPointPersistence[gridID])
          if maxPers != 0:
            self.invPointPersistence[gridID] = (maxPers-self.invPointPersistence[gridID])/maxPers
        else:
          self.firstSurface = False
        if self.solutionExport is not None:
          # construct the realizations dict
          localRlz = {varName: (self.surfPoint[gridID][:,varIndex] if varName not in rlz else np.concatenate(( rlz[varName],self.surfPoint[gridID][:,varIndex] )) ) for varIndex,varName in enumerate(realAxisNames) }
          localRlz[self.goalFunction.name] = evaluations[gridID] if self.goalFunction.name not in rlz else np.concatenate( (rlz[self.goalFunction.name],evaluations[gridID])  )
          rlz.update(localRlz)
    # add the full realizations
    if self.solutionExport is not None:
      if len(rlz):
        self.solutionExport.load(rlz,style='dict')

    # Keep track of some extra points that we will add to thicken the limit
    # surface candidate set
    self.bandIndices = OrderedDict()
    for gridID,points in self.listSurfPoint.items():
      setSurfPoint = set()
      self.bandIndices[gridID] = set()
      for surfPoint in points:
        setSurfPoint.add(tuple(surfPoint))
      newIndices = set(setSurfPoint)
      for step in range(1,self.thickness):
        prevPoints = set(newIndices)
        newIndices = set()
        for i,iCoords in enumerate(prevPoints):
          for d in range(len(iCoords)):
            offset = np.zeros(len(iCoords),dtype=int)
            offset[d] = 1
            if iCoords[d] - offset[d] > 0:
              newIndices.add(tuple(iCoords - offset))
            if iCoords[d] + offset[d] < self.oldTestMatrix[gridID].shape[d]-1:
              newIndices.add(tuple(iCoords + offset))
        self.bandIndices[gridID].update(newIndices)
      self.bandIndices[gridID] = self.bandIndices[gridID].difference(setSurfPoint)
      self.bandIndices[gridID] = list(self.bandIndices[gridID])
      for coordinate in self.bandIndices[gridID]:
        self.surfPoint[gridID] = np.vstack((self.surfPoint[gridID],self.limitSurfacePP.gridCoord[gridID][coordinate]))
    if self.converged:
      self.raiseAMessage(self.name + " converged!")
    return ready
