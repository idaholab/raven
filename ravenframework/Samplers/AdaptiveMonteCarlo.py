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
  This module contains the Adaptive Monte Carlo sampling strategy

  Created on Feb 20, 2020
  @author: ZHOUJ2

"""
import numpy as np

from ..Models.PostProcessors import factory as ppFactory
from ..utils import InputData, InputTypes
from .AdaptiveSampler import AdaptiveSampler
from .MonteCarlo import MonteCarlo


class AdaptiveMonteCarlo(AdaptiveSampler, MonteCarlo):
  """
    A sampler that will adaptively locate the limit surface of a given problem
  """
  bS = ppFactory.returnClass('BasicStatistics')
  statScVals = bS.scalarVals
  statErVals = bS.steVals
  usableStats = []
  for errMetric in statErVals:
    metric, _ = errMetric.split('_')
    if metric in statScVals:
      usableStats.append((metric, errMetric))

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
    # TODO this class should use MonteCarlo's "limit" definition, probably?
    convergenceInput = InputData.parameterInputFactory('Convergence')
    convergenceInput.addSub(InputData.parameterInputFactory('limit', contentType=InputTypes.IntegerType))
    convergenceInput.addSub(InputData.parameterInputFactory('forceIteration', contentType=InputTypes.BoolType))
    convergenceInput.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType))
    for metric, _ in cls.usableStats:
      statErSpecification = InputData.parameterInputFactory(metric, contentType=InputTypes.StringListType)
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

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, ok, dict, {varName: manual description} for each solution export option
    """
    # cannot be determined before run-time due to variables and prefixes.
    ok = super(AdaptiveMonteCarlo, cls).getSolutionExportVariableNames()
    new = {'solutionUpdate': 'iteration (or step) number for convergence algorithm',
           '{PREFIX}_{VAR}': 'value of metric with given prefix {PREFIX} for variable {VAR} at current step in convergence',
           '{PREFIX}_ste_{VAR}': 'estimate of error in metric with given prefix {PREFIX} for variable {VAR} at current step in convergence',
          }
    ok.update(new)

    return ok

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    MonteCarlo.__init__(self)
    AdaptiveSampler.__init__(self)
    self.persistence = 5          # this is the number of times the error needs to fell below the tolerance before considering the sim converged
    self.persistenceCounter = 0   # Counter for the persistence
    self.forceIteration = False   # flag control if at least a self.limit number of iteration should be done
    self.solutionExport = None    # data used to export the solution (it could also not be present)
    self.tolerance = {}           # dictionary stores the tolerance for each variables
    self.basicStatPP = None       # post-processor to compute the basic statistics
    self.converged = False        # flag that is set to True when the sampler converged
    self.printTag = 'SAMPLER ADAPTIVE MC'
    self.toDo = None              # BasicStatistics metrics to calculate

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
          elif tag == "persistence":
            self.persistence = grandchild.value
            self.raiseADebug(f'Persistence is set at {self.persistence}')
          elif tag == "forceIteration":
            self.forceIteration = grandchild.value
          elif tag in self.statScVals:
            if 'prefix' not in grandchild.parameterValues:
              self.raiseAnError(IOError, f"No prefix is provided for node: {tag}")
            if 'tol' not in grandchild.parameterValues:
              self.raiseAnError(IOError, f"No tolerance is provided for metric: {tag}")
            prefix = grandchild.parameterValues['prefix']
            tol = grandchild.parameterValues['tol']
            if tag not in self.toDo:
              self.toDo[tag] = [] # list of {'targets':(), 'prefix':str}
            self.toDo[tag].append({'targets':set(grandchild.value),
                                   'prefix':prefix,
                                   'tol':tol
                                  })
          else:
            self.raiseAWarning(f'Unrecognized convergence node "{tag}" has been ignored!')
        assert (len(self.toDo) > 0), self.raiseAnError(IOError, ' No target have been assigned to convergence node')
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
    if self.limit is None:
      self.raiseAnError(IOError, f'{self.type} requires a <limit> to be specified!')

  def localInitialize(self, solutionExport=None):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, solutionExport, DataObjects, optional, a PointSet to hold the solution
      @ Out, None
    """
    self.converged = False
    self.basicStatPP = ppFactory.returnInstance('BasicStatistics')
    # check if solutionExport is actually a "DataObjects" type "PointSet"
    if self._solutionExport.type != "PointSet":
      self.raiseAnError(IOError, f'solutionExport type is not a PointSet. Got {self._solutionExport.type}!')

    self.basicStatPP.what = self.toDo.keys()
    self.basicStatPP.toDo = self.toDo
    self.basicStatPP.initialize({'WorkingDir': None}, [self._targetEvaluation], {'Output': []})
    self.raiseADebug('Initialization done')

  ###############
  # Run Methods #
  ###############

  def localFinalizeActualSampling(self, jobObject, model, myInput):
    """
      General function (available to all samplers) that finalize the sampling
      calculation just ended. In this case, The function is aimed to check if
      all the batch calculations have been performed
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
      @ Out, None
    """
    if self.counter > 1:
      output = self.basicStatPP.run(self._targetEvaluation)
      output['solutionUpdate'] = np.asarray([self.counter - 1])
      self._solutionExport.addRealization(output)
      self.checkConvergence(output)

  def checkConvergence(self, output):
    """
      Determine convergence for Adaptive MonteCarlo
      @ In, output, dict, dictionary containing the results from Basic Statistic
      @ Out, None
    """
    if self.forceIteration:
      self.converged = False
    else:
      checker = [output[metric][0] for metric in self.tolerance.keys()]
      isNan = True if 'nan' in checker else np.isnan(checker).any()
      if isNan:
        self.converged = False
        return
      converged = all(abs(tol) > abs(output[metric][0]) for metric, tol in self.tolerance.items())
      if converged:
        self.raiseAMessage('Checking target convergence for standard error and tolerance')
        for metric, tol in self.tolerance.items():
          self.raiseAMessage(f'Target \"{"".join(metric.split("_ste"))}\" standard error {output[metric][0]:>2.2e} < tolerance {tol:>2.2e}')
        self.persistenceCounter += 1
        # check if we've met persistence requirement; if not, keep going
        if self.persistenceCounter >= self.persistence:
          self.raiseAMessage(f' ... {self.name} converged {self.persistenceCounter} times consecutively!')
          self.converged = True
        else:
          self.raiseAMessage(f' ... {self.name} converged {self.persistenceCounter} times, required persistence is {self.persistence}.')

  def localStillReady(self, ready):
    """
      first perform some check to understand what it needs to be done possibly perform an early return
      ready is returned
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    if self.converged:
      return False

    return ready

  def _formatSolutionExportVariableNames(self, acceptable):
    """
      Does magic formatting for variables, based on this class's needs.
      Extend in inheritors as needed.
      @ In, acceptable, set, set of acceptable entries for solution export for this entity
      @ Out, new, set, modified set of acceptable variables with all formatting complete
    """
    # remaking the list is easier than using the existing one
    acceptable = AdaptiveSampler._formatSolutionExportVariableNames(self, acceptable)
    new = []
    while acceptable:
      # populate each template
      template = acceptable.pop()
      # the only "magic" entries have PREFIX and VAR in them
      if '{VAR}' in template and '{PREFIX}' in template:
        for _, info in self.toDo.items():
          # each metric may have several entries (for different prefixes, tolerances, etc)
          for entry in info:
            prefix = entry['prefix']
            targets = entry['targets']
            for target in targets:
              new.append(template.format(PREFIX=prefix, VAR=target))
      # if not a "magic" entry, just carry it along
      else:
        new.append(template)
    return set(new)

  def flush(self):
    """
      Reset AdaptiveMonteCarlo attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self.persistenceCounter = 0
    self.basicStatPP = None
