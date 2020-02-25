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
import numpy as np
import sys

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
#from PostProcessors import LimitSurface
from PostProcessors import BasicStatistics
from .AdaptiveSampler import AdaptiveSampler
from .MonteCarlo import MonteCarlo
import Distributions
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------


class AdaptiveMonteCarlo(AdaptiveSampler,MonteCarlo):
  """
    A sampler that will adaptively locate the limit surface of a given problem
  """
  statScVals = BasicStatistics.scalarVals
  statErVals = BasicStatistics.steVals
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
    for metric in cls.statErVals:
      statEr, ste = metric.split('_')
      if statEr in cls.statScVals:
        statErSpecification = InputData.parameterInputFactory(statEr, contentType=InputTypes.StringListType)
        statErSpecification.addParam("prefix", InputTypes.StringType)
        statErSpecification.addParam("tol", InputTypes.FloatType)
        convergenceInput.addSub(statErSpecification)
    inputSpecification.addSub(convergenceInput)
    targetEvaluationInput = InputData.parameterInputFactory("TargetEvaluation", contentType=InputTypes.StringType)
    targetEvaluationInput.addParam("type", InputTypes.StringType)
    targetEvaluationInput.addParam("class", InputTypes.StringType)
    inputSpecification.addSub(targetEvaluationInput)
    inputSpecification.addSub(InputData.parameterInputFactory("initialSeed", contentType=InputTypes.IntegerType))
    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    AdaptiveSampler.__init__(self)
    MonteCarlo.__init__(self)
    self.persistence         = 5                # this is the number of times the error needs to fell below the tolerance before considering the sim converged
    self.persistenceCounter  = 0                # Counter for the persistence
    self.forceIteration      = False            # flag control if at least a self.limit number of iteration should be done
    self.solutionExport      = None             # data used to export the solution (it could also not be present)
    self.tolerance           = {}               # dictionary stores the tolerance for each variables
    self.converged           = False            # flag convergence
    self.basicStatPP         = None             # post-processor to compute the basic statistics
    self.converged           = False            # flag that is set to True when the sampler converged
    self.printTag            = 'SAMPLER ADAPTIVE MC'
    self.addAssemblerObject('TargetEvaluation','n')

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    self.toDo = {}
    for child in paramInput.subparts:
      if child.getName() == "Convergence":
        for grandchild in child.subparts:
          tag = grandchild.getName()
          if tag == "limit":
            self.limit = grandchild.value
            if self.limit is None:
              self.raiseAnError(IOError,self,'Adaptive Monte Carlo sampler '+self.name+' needs the limit block (number of samples) in the Convergence block')

          elif tag == "persistence":
            self.persistence = grandchild.value
            self.raiseADebug('Persistence is set at',self.persistence)
          elif tag == "forceIteration":
            self.forceIteration = grandchild.value
          elif tag in self.statScVals:
            if 'prefix' not in grandchild.parameterValues:
              self.raiseAnError(IOError, "No prefix is provided for node: ", tag)
            if 'tol' not in grandchild.parameterValues:
              self.raiseAnError(IOError, "No tolerance is provided for metric: ", tag)
            prefix = grandchild.parameterValues['prefix']
            tol = grandchild.parameterValues['tol']
            if tag not in self.toDo.keys():
              self.toDo[tag] = [] # list of {'targets':(), 'prefix':str}
            self.toDo[tag].append({'targets':set(grandchild.value),
                                  'prefix':prefix,
                                  'tol':tol
                                  })
          else:
            self.raiseAWarning('Unrecognized convergence node "',tag,'" has been ignored!')
        assert (len(self.toDo)>0), self.raiseAnError(IOError, ' No target have been assigned to convergence node')
      elif child.getName() == "initialSeed":
        self.initSeed = child.value
    for metric, infos in self.toDo.items():
      steMetric = metric + '_ste'
      if steMetric in self.statErVals:
        for info in infos:
          prefix = info['prefix']
          for target in info['targets']:
            metaVar = prefix + '_ste_' + target
            self.tolerance[metaVar] = info['tol']

  def localInitialize(self,solutionExport=None):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, solutionExport, DataObjects, optional, a PointSet to hold the solution
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

    self.basicStatPP.what = self.toDo.keys()
    self.basicStatPP.toDo = self.toDo
    self.basicStatPP.initialize({'WorkingDir':None},[self.lastOutput],{'Output':[]})
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
    if self.counter>1:
      output = self.basicStatPP.run(self.lastOutput)
      output['solutionUpdate'] = np.asarray([self.counter - 1])
      self.solutionExport.addRealization(output)
      self.checkConvergence(output)


  def checkConvergence(self,output):
    '''
      Determine convergence for Adaptive MonteCarlo
      @ In, output, dictionary containing the results from Basic Statistic
      @ Out, None
    '''
    if self.forceIteration:
      self.converged = False
    else:
      converged = all(abs(tol) > abs(output[metric][0]) for metric,tol in self.tolerance.items())
      if converged:
        self.raiseAMessage('Checking target convergence for standard error and tolerance')
        for metric,tol in self.tolerance.items():
          self.raiseAMessage('Target \"{}\" standard error {:>2.2e} < tolerance {:>2.2e}'.format(''.join(metric.split('_ste')), output[metric][0], tol))
        self.persistenceCounter += 1
        # check if we've met persistence requirement; if not, keep going
        if self.persistenceCounter >= self.persistence:
          self.raiseAMessage(' ... {} converged {} times consecutively!'.format(self.name,self.persistenceCounter))
          self.converged = True
        else:
          self.raiseAMessage(' ... {} converged {} times, required persistence is {}.'.format(self.name,self.persistenceCounter,self.persistence))


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
    if self.converged:
      return False
    else:
      return ready
