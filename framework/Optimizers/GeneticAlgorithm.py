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
  Genetic Algorithm class for global optimization.
  This class contains the API and interface for performing
  Genetic Algorithm-based optimization. Multiple strategies for
  mutations, cross-overs, etc. are available.
  Created June,3,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
  References
    ----------
    .. [1] Holland, John H. "Genetic algorithms." Scientific American 267.1 (1992): 66-73.
       [2] Z. Michalewicz, "Genetic Algorithms. + Data Structures. = Evolution Programs," Third, Revised
           and Extended Edition, Springer (1996).
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
from scipy.special import comb
from collections import deque, defaultdict
import xarray as xr
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils, randomUtils, InputData, InputTypes
from utils.gaUtils import dataArrayToDict, datasetToDataArray
from .RavenSampled import RavenSampled
from .parentSelectors.parentSelectors import returnInstance as parentSelectionReturnInstance
from .crossOverOperators.crossovers import returnInstance as crossoversReturnInstance
from .mutators.mutators import returnInstance as mutatorsReturnInstance
from .survivorSelectors.survivorSelectors import returnInstance as survivorSelectionReturnInstance
from .fitness.fitness import returnInstance as fitnessReturnInstance
from .repairOperators.repair import returnInstance as repairReturnInstance
#Internal Modules End--------------------------------------------------------------------------------

class GeneticAlgorithm(RavenSampled):
  """
    This class performs Genetic Algorithm optimization ...
  """
  convergenceOptions = {'objective': r""" provides the desired value for the convergence criterion of the objective function
                        ($\epsilon^{obj}$). In essence this is solving the inverse problem of finding the design variable
                         at a given objective value, i.e., convergence is reached when: $$ Objective = \epsilon^{obj}$$.
                        \default{1e-6}, if no criteria specified""",
                        'AHDp': r""" provides the desired value for the Average Hausdorff Distance between populations""",
                        'AHD': r""" provides the desired value for the Hausdorff Distance between populations"""}
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    RavenSampled.__init__(self)
    self._parentSelection = None                                 # mechanism for parent selection
    self._convergenceCriteria = defaultdict(mathUtils.giveZero)  # names and values for convergence checks
    self._acceptHistory = {}                                     # acceptability
    self._acceptRerun = {}                                       # by traj, if True then override accept for point rerun
    self._convergenceInfo = {}                                   # by traj, the persistence and convergence information for most recent opt
    self._requiredPersistence = 0                                # consecutive persistence required to mark convergence
    self.needDenormalized() # the default in all optimizers is to normalize the data which is not the case here
    self.batchId = 0
    self.population = None # panda Dataset container containing the population at the beginning of each generation iteration
    self.popAge = None     # population age
    self.fitness = None    # population fitness
    self.ahdp = np.NaN     # p-Average Hausdorff Distance between populations
    self.ahd  = np.NaN     # Hausdorff Distance between populations

  ##########################
  # Initialization Methods #
  ##########################

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(GeneticAlgorithm, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{GeneticAlgorithm} optimizer is a metaheuristic approach
                            to perform a global search in large design spaces. The methodology rose
                            from the process of natural selection, and like others in the large class
                            of the evolutionary algorithms, it utilizes genetic operations such as
                            selection, crossover, and mutations to avoid being stuck in local minima
                            and hence facilitates finding the global minima. More information can
                            be found in:
                            Holland, John H. "Genetic algorithms." Scientific American 267.1 (1992): 66-73."""

    # GA Params
    GAparams = InputData.parameterInputFactory('GAparams', strictMode=True,
        printPriority=108,
        descr=r""" Genetic Algorithm Parameters:\begin{itemize}
                                                  \item populationSize.
                                                  \item parentSelectors:
                                                                    \begin{itemize}
                                                                      \item rouletteWheel.
                                                                      \item tournamentSelection.
                                                                      \item rankSelection.
                                                                    \end{itemize}
                                                 \item Reproduction:
                                                                  \begin{itemize}
                                                                    \item crossover:
                                                                      \begin{itemize}
                                                                        \item onePointCrossover.
                                                                        \item twoPointsCrossover.
                                                                        \item uniformCrossover
                                                                      \end{itemize}
                                                                    \item mutators:
                                                                      \begin{itemize}
                                                                        \item swapMutator.
                                                                        \item scrambleMutator.
                                                                        \item inversionMutator.
                                                                        \item bitFlipMutator.
                                                                      \end{itemize}
                                                                    \end{itemize}
                                                \item survivorSelectors:
                                                                      \begin{itemize}
                                                                        \item ageBased.
                                                                        \item fitnessBased.
                                                                      \end{itemize}
                                                \end{itemize}""")
    # Population Size
    populationSize = InputData.parameterInputFactory('populationSize', strictMode=True,
        contentType=InputTypes.IntegerType,
        printPriority=108,
        descr=r"""The number of chromosomes in each population.""")
    GAparams.addSub(populationSize)
    # Parent Selection
    parentSelection = InputData.parameterInputFactory('parentSelection', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""A node containing the criterion based on which the parents are selected. This can be a
                  fitness proportional selection such as:
                  a. \textbf{\textit{rouletteWheel}},
                  b. \textbf{\textit{stochasticUniversalSampling}},
                  c. \textbf{\textit{Tournament}},
                  d. \textbf{\textit{Rank}}, or
                  e. \textbf{\textit{randomSelection}}""")
    GAparams.addSub(parentSelection)

    # Reproduction
    reproduction = InputData.parameterInputFactory('reproduction', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the reproduction methods.
                  This accepts subnodes that specifies the types of crossover and mutation.""")
    reproduction.addParam("nParents", InputTypes.IntegerType, True,
                          descr="number of parents to be considered in the reproduction phase")
    # 1.  Crossover
    crossover = InputData.parameterInputFactory('crossover', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented crossover mechanisms.
                  This includes: a.    One Point Crossover,
                                 b.    MultiPoint Crossover,
                                 c.    Uniform Crossover,
                                 d.    Whole Arithmetic Recombination, or
                                 e.    Davis’ Order Crossover.""")
    crossover.addParam("type", InputTypes.StringType, True,
                       descr="type of crossover operation to be used (e.g., OnePoint, MultiPoint, or Uniform)")
    crossoverPoint = InputData.parameterInputFactory('points', strictMode=True,
        contentType=InputTypes.IntegerListType,
        printPriority=108,
        descr=r""" point/gene(s) at which crossover will occur.""")
    crossover.addSub(crossoverPoint)
    crossoverProbability = InputData.parameterInputFactory('crossoverProb', strictMode=True,
        contentType=InputTypes.FloatType,
        printPriority=108,
        descr=r""" The probability governing the crossover step, i.e., the probability that if exceeded crossover will occur.""")
    crossover.addSub(crossoverProbability)
    reproduction.addSub(crossover)
    # 2.  Mutation
    mutation = InputData.parameterInputFactory('mutation', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented mutation mechanisms.
                  This includes: a.    Bit Flip,
                                 b.    Random Resetting,
                                 c.    Swap,
                                 d.    Scramble, or
                                 e.    Inversion.""")
    mutation.addParam("type", InputTypes.StringType, True,
                      descr="type of mutation operation to be used (e.g., bit, swap, or scramble)")
    mutationLocs = InputData.parameterInputFactory('locs', strictMode=True,
        contentType=InputTypes.IntegerListType,
        printPriority=108,
        descr=r""" locations at which mutation will occur.""")
    mutation.addSub(mutationLocs)
    mutationProbability = InputData.parameterInputFactory('mutationProb', strictMode=True,
        contentType=InputTypes.FloatType,
        printPriority=108,
        descr=r""" The probability governing the mutation step, i.e., the probability that if exceeded mutation will occur.""")
    mutation.addSub(mutationProbability)
    reproduction.addSub(mutation)
    GAparams.addSub(reproduction)

    # Survivor Selection
    survivorSelection = InputData.parameterInputFactory('survivorSelection', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented survivor selection mechanisms.
                  This includes: a.    ageBased, or
                                 b.    fitnessBased.""")
    GAparams.addSub(survivorSelection)

    # Fitness
    fitness = InputData.parameterInputFactory('fitness', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented fitness functions.
                  This includes: a.    invLinear: $fitness = -(a \\times obj + b \\times penalty)$.
                                 b.    logistic: $fitness = \\frac{1}{1+e^{a \\times (obj-b)}}$.
                                 c.    feasibleFirst: $fitness = \[ \\begin{cases}
                                                                      -obj & g_j(x)\\geq 0 \\forall j \\
                                                                      -obj_{worst} - \\Sigma_{j=1}^{J}<g_j(x)> & otherwise \\
                                                                    \\end{cases}
                                                                \]$""")
    fitness.addParam("type", InputTypes.StringType, True,
                     descr=r"""[invLin, logistic, feasibleFirst]""")
    objCoeff = InputData.parameterInputFactory('a', strictMode=True,
        contentType=InputTypes.FloatType,
        printPriority=108,
        descr=r""" a: coefficient of objective function.""")
    fitness.addSub(objCoeff)
    penaltyCoeff = InputData.parameterInputFactory('b', strictMode=True,
        contentType=InputTypes.FloatType,
        printPriority=108,
        descr=r""" b: coefficient of constraint penalty.""")
    fitness.addSub(penaltyCoeff)
    GAparams.addSub(fitness)
    specs.addSub(GAparams)

    # convergence
    conv = InputData.parameterInputFactory('convergence', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the desired convergence criteria for the optimization algorithm.
              Note that convergence is met when any one of the convergence criteria is met. If no convergence
              criteria are given, then the defaults are used.""")
    specs.addSub(conv)
    for name,descr in cls.convergenceOptions.items():
      conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatType,descr=descr,printPriority=108  ))

    # Persistence
    conv.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType,
        printPriority = 109,
        descr=r"""provides the number of consecutive times convergence should be reached before a trajectory
              is considered fully converged. This helps in preventing early false convergence."""))
    specs.addSub(conv)
    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, cls, the class for which we are retrieving the solution export
      @ Out, ok, dict, {varName: description} for valid solution export variable names
    """
    # cannot be determined before run-time due to variables and prefixes.
    ok = super(GeneticAlgorithm, cls).getSolutionExportVariableNames()
    new = {}
    # new = {'': 'the size of step taken in the normalized input space to arrive at each optimal point'}
    new['conv_{CONV}'] = 'status of each given convergence criteria'
    new['fitness'] = 'fitness of the current chromosome'
    new['age'] = 'age of current chromosome'
    new['batchId'] = 'Id of the batch to whom the chromosome belongs'
    new['AHDp'] = 'p-Average Hausdorff Distance between populations'
    new['AHD'] = 'Hausdorff Distance between populations'
    ok.update(new)
    return ok

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    RavenSampled.handleInput(self, paramInput)
    # GAparams
    gaParamsNode = paramInput.findFirst('GAparams')
    # populationSize
    populationSizeNode = gaParamsNode.findFirst('populationSize')
    self._populationSize = populationSizeNode.value
    # parent selection
    parentSelectionNode = gaParamsNode.findFirst('parentSelection')
    self._parentSelectionType = parentSelectionNode.value
    self._parentSelectionInstance = parentSelectionReturnInstance(self,name = parentSelectionNode.value)
    # reproduction node
    reproductionNode = gaParamsNode.findFirst('reproduction')
    self._nParents = reproductionNode.parameterValues['nParents']
    self._nChildren = int(2*comb(self._nParents,2))
    # crossover node
    crossoverNode = reproductionNode.findFirst('crossover')
    self._crossoverType = crossoverNode.parameterValues['type']
    if crossoverNode.findFirst('points') is None:
      self._crossoverPoints = None
    else:
      self._crossoverPoints = crossoverNode.findFirst('points').value
    self._crossoverProb = crossoverNode.findFirst('crossoverProb').value
    self._crossoverInstance = crossoversReturnInstance(self,name = self._crossoverType)
    # mutation node
    mutationNode = reproductionNode.findFirst('mutation')
    self._mutationType = mutationNode.parameterValues['type']
    if mutationNode.findFirst('locs') is None:
      self._mutationLocs = None
    else:
      self._mutationLocs = mutationNode.findFirst('locs').value
    self._mutationProb = mutationNode.findFirst('mutationProb').value
    self._mutationInstance = mutatorsReturnInstance(self,name = self._mutationType)
    # Survivor selection
    survivorSelectionNode = gaParamsNode.findFirst('survivorSelection')
    self._survivorSelectionType = survivorSelectionNode.value
    self._survivorSelectionInstance = survivorSelectionReturnInstance(self,name = self._survivorSelectionType)
    # Fitness
    fitnessNode = gaParamsNode.findFirst('fitness')
    self._fitnessType = fitnessNode.parameterValues['type']
    self._objCoeff = fitnessNode.findFirst('a').value if fitnessNode.findFirst('a') is not None else None
    self._penaltyCoeff = fitnessNode.findFirst('b').value if fitnessNode.findFirst('b') is not None else None
    self._fitnessInstance = fitnessReturnInstance(self,name = self._fitnessType)
    self._repairInstance = repairReturnInstance(self,name='replacementRepair')  # currently only replacement repair is implemented,
                                                                                # if other repair methods are implemented then
                                                                                # ##TODO: make the repair type a user input
    # Convergence Criterion
    convNode = paramInput.findFirst('convergence')
    if convNode is not None:
      for sub in convNode.subparts:
        if sub.getName() == 'persistence':
          self._requiredPersistence = sub.value
        else:
          self._convergenceCriteria[sub.name] = sub.value
    if not self._convergenceCriteria:
      self.raiseAWarning('No convergence criteria given; using defaults.')
      self._convergenceCriteria['objective'] = 1e-6
    if self._requiredPersistence is None:
      self.raiseADebug('No persistence given; setting to 1.')
      self._requiredPersistence = 1


  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)

    meta = ['batchId']
    self.addMetaKeys(meta)
    self.batch = self._populationSize*(self.counter==0)+self._nChildren*(self.counter>0)
    if self._populationSize != len(self._initialValues):
      self.raiseAnError(IOError, 'Number of initial values provided for each variable is {}, while the population size is {}'.format(len(self._initialValues),self._populationSize,self._populationSize))
    for _, init in enumerate(self._initialValues): # TODO: this should be single traj
      self._submitRun(init,0,self.getIteration(0)+1)

  def initializeTrajectory(self, traj=None):
    """
      Handles the generation of a trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, new trajectory number
    """
    traj = RavenSampled.initializeTrajectory(self)
    self._acceptHistory[traj] = deque(maxlen=self._maxHistLen)
    self._acceptRerun[traj] = False
    self._convergenceInfo[traj] = {'persistence': 0}
    for criteria in self._convergenceCriteria:
      self._convergenceInfo[traj][criteria] = False
    return traj

  def needDenormalized(self):
    """
      Determines if the currently used algorithms should be normalizing the input space or not
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    # overload as needed in inheritors
    return True

  ###############
  # Run Methods #
  ###############

  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      This is called by localFinalizeActualSampling, and hence should contain the main skeleton.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, xr.Dataset, new batched realizations
      @ Out, None
    """
    # The whole skeleton should be here, this should be calling all classes and _private methods.
    traj = info['traj']
    for t in self._activeTraj[1:]:
      self._closeTrajectory(t, 'cancel', 'Currently GA is single trajectory',0)#, None
    self.incrementIteration(traj)
    info['step'] = self.counter

    # Developer note: each algorithm step is indicated by a number followed by the generation number
    # e.g., '5 @ n-1' refers to step 5 for generation n-1 (i.e., previous generation)
    # for more details refer to GRP-Raven-development/Disceret_opt channel on MS Teams

    # 5 @ n-1: Survivor Selection from previous iteration (children+parents merging from previous generation)

    # 5.1 @ n-1: fitnessCalculation(rlz)
    # perform fitness calculation for newly obtained children (rlz)
    population = datasetToDataArray(rlz, list(self.toBeSampled)) # TODO: rename
    objectiveVal = list(np.atleast_1d(rlz[self._objectiveVar].data))
    # Compute constraint function g_j(x) for all constraints (j = 1 .. J)
    # and all x's (individuals) in the population
    g0 = np.zeros((np.shape(population)[0],len(self._constraintFunctions)+len(self._impConstraintFunctions)))
    g = xr.DataArray(g0,
                     dims=['chromosome','Constraint'],
                     coords={'chromosome':np.arange(np.shape(population)[0]),
                             'Constraint':[y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]})
    for index,individual in enumerate(population):
      newOpt = individual
      opt = objectiveVal[index]
      for constIndex,constraint in enumerate(self._constraintFunctions + self._impConstraintFunctions):
        if constraint in self._constraintFunctions:
          g.data[index, constIndex] = self._handleExplicitConstraints(newOpt,constraint)
        else:
          g.data[index, constIndex] = self._handleImplicitConstraints(newOpt, opt,constraint)
    fitness = self._fitnessInstance(rlz, objVar=self._objectiveVar, a=self._objCoeff, b=self._penaltyCoeff, penalty=None,constraintFunction=g)
    acceptable = 'first' if self.counter==1 else 'accepted'
    self._collectOptPoint(population,fitness,objectiveVal)
    self._resolveNewGeneration(traj, rlz, objectiveVal, fitness, info)

    if self._activeTraj:
      # 5.2@ n-1: Survivor selection(rlz)
      # update population container given obtained children
      if self.counter > 1:
        population,fitness,age = self._survivorSelectionInstance(age=self.popAge, variables=list(self.toBeSampled), population=self.population, fitness=self.fitness, newRlz=rlz,offSpringsFitness=fitness)
        self.popAge = age
        self.population = population
      else:
        self.population = population
      self.objectiveVal = rlz[self._objectiveVar].data
      self.fitness = fitness

      # 1 @ n: Parent selection from population
      # pair parents together by indexes
      parents = self._parentSelectionInstance(population,variables=list(self.toBeSampled),fitness=fitness,nParents=self._nParents)

      # 2 @ n: Crossover from set of parents
      # create childrenCoordinates (x1,...,xM)
      childrenXover = self._crossoverInstance(parents=parents,variables=list(self.toBeSampled),crossoverProb=self._crossoverProb,points=self._crossoverPoints)

      # 3 @ n: Mutation
      # perform random directly on childrenCoordinates
      childrenMutated = self._mutationInstance(offSprings=childrenXover, distDict = self.distDict,locs = self._mutationLocs, mutationProb=self._mutationProb,variables=list(self.toBeSampled))

      # 4 @ n: repair/replacement
      # repair should only happen if multiple genes in a single chromosome have the same values (),
      # and at the same time the sampling of these genes should be with Out replacement.
      needsRepair = False
      for chrom in range(self._nChildren):
        unique = set(childrenMutated.data[chrom,:])
        if len(childrenMutated.data[chrom,:]) != len(unique):
          for var in self.toBeSampled.keys(): ## TODO: there must be a smarter way to check if a variables strategy is without replacement
            if (hasattr(self.distDict[var],'strategy') and self.distDict[var].strategy == 'withoutReplacement'):
              needsRepair = True
              break
      if needsRepair:
        children = self._repairInstance(childrenMutated,variables=list(self.toBeSampled),distInfo=self.distDict)
      else:
        children = childrenMutated
      # Make sure no children are exactly similar to parents
      flag = True
      counter = 0
      while flag and counter < self._populationSize:
        counter += 1
        repeated =[]
        for i in range(np.shape(population.data)[0]):
          for j in range(i,np.shape(children.data)[0]):
            if all(population.data[i,:]==children.data[j,:]):
              repeated.append(j)
        repeated = list(set(repeated))
        if repeated:
          newChildren = self._mutationInstance(offSprings=children[repeated,:], distDict = self.distDict, locs = self._mutationLocs, mutationProb=self._mutationProb,variables=list(self.toBeSampled))
          children.data[repeated,:] = newChildren.data
        else:
          flag = False

      self.batch = np.shape(children)[0]

      daChildren = xr.DataArray(children,
                              dims=['chromosome','Gene'],
                              coords={'chromosome': np.arange(np.shape(children)[0]),
                                      'Gene':list(self.toBeSampled)})
      # 5 @ n: Submit children batch
      # submit children coordinates (x1,...,xm), i.e., self.childrenCoordinates
      for i in range(np.shape(daChildren)[0]):
        newRlz={}
        for _,var in enumerate(self.toBeSampled.keys()):
          newRlz[var] = float(daChildren.loc[i,var].values)
        self._submitRun(newRlz, traj, self.getIteration(traj))

  def _submitRun(self, point, traj, step, moreInfo=None):
    """
      Submits a single run with associated info to the submission queue
      @ In, point, dict, point to submit
      @ In, traj, int, trajectory identifier
      @ In, step, int, iteration number identifier
      @ In, moreInfo, dict, optional, additional run-identifying information to track
      @ Out, None
    """
    info = {}
    if moreInfo is not None:
      info.update(moreInfo)
    info.update({'traj': traj,
                  'step': step
                })
    # NOTE: Currently, GA treats explicit and implicit constraints similarly
    # while box constraints (Boundary constraints) are automatically handled via limits of the distribution
    #
    self.raiseADebug('Adding run to queue: {} | {}'.format(self.denormalizeData(point), info))
    self._submissionQueue.append((point, info))

  # END queuing Runs
  # * * * * * * * * * * * * * * * *

  def _resolveNewGeneration(self, traj, rlz, objectiveVal, fitness, info):
    """
      Store a new Generation after checking convergence
      @ In, traj, int, trajectory for this new point
      @ In, rlz, dict, realized realization
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, fitness, xr.DataArray, fitness values at each chromosome of the realization
      @ In, info, dict, identifying information about the realization
    """
    self.raiseADebug('*'*80)
    self.raiseADebug('Trajectory {} iteration {} resolving new state ...'.format(traj, info['step']))
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    acceptable = 'accepted' if self.counter >1 else 'first'
    old = self.population
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    if converged:
      self._closeTrajectory(traj, 'converge', 'converged', self.bestObjective)
    # NOTE: the solution export needs to be updated BEFORE we run rejectOptPoint or extend the opt
    #       point history.
    if self._writeSteps == 'every':
      for i in range(rlz.sizes['RAVEN_sample_ID']):
        rlzDict = dict((var,np.atleast_1d(rlz[var].data)[i]) for var in self.toBeSampled.keys())
        rlzDict[self._objectiveVar] = np.atleast_1d(rlz[self._objectiveVar].data)[i]
        rlzDict['fitness'] = np.atleast_1d(fitness.data)[i]
        self._updateSolutionExport(traj, rlzDict, acceptable,None)
    # decide what to do next
    if acceptable in ['accepted', 'first']:
      # record history
      bestRlz = {}
      bestRlz[self._objectiveVar] = self.bestObjective
      bestRlz['fitness'] = self.bestFitness
      bestRlz.update(self.bestPoint)
      self._optPointHistory[traj].append((bestRlz, info))
    elif acceptable == 'rejected':
      self._rejectOptPoint(traj, info, old)
    else: # e.g. rerun
      pass # nothing to do, just keep moving

  def _collectOptPoint(self, population, fitness, objectiveVal):
    """
      Collects the point (dict) from a realization
      @ In, population, Dataset, container containing the population
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, fitness, xr.DataArray, fitness values at each chromosome of the realization
      @ Out, point, dict, point used in this realization
    """
    optPoints,fit,obj = zip(*[[x,y,z] for x,y,z in sorted(zip(np.atleast_2d(population.data),np.atleast_1d(fitness.data),objectiveVal),reverse=True,key=lambda x: (x[1]))])
    point = dict((var,float(optPoints[0][i])) for i,var in enumerate(self.toBeSampled.keys()))
    if (self.counter>1 and obj[0] < self.bestObjective) or self.counter == 1:
      self.bestPoint = point
      self.bestFitness = fit[0]
      self.bestObjective = obj[0]
    return point

  def _checkAcceptability(self, traj):
    """
      This is an abstract method for all RavenSampled Optimizer, whereas for GA all children are accepted
      @ In, traj, int, identifier
    """
    return # TODO: This method is not needed but it was defined as an abstract

  def checkConvergence(self, traj, new, old):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory to consider
      @ In, new, xr.DataSet, new children realization
      @ In, old, xr.DataArray, old population
      @ Out, any(convs.values()), bool, True of any of the convergence criteria was reached
      @ Out, convs, dict, on the form convs[conv] = bool, where conv is in self._convergenceCriteria
    """
    convs = {}
    for conv in self._convergenceCriteria:
      fName = conv[:1].upper() + conv[1:]
      # get function from lookup
      f = getattr(self, '_checkConv{}'.format(fName))
      # check convergence function
      okay = f(traj,new=new,old=old)
      # store and update
      convs[conv] = okay

    return any(convs.values()), convs

  def _checkConvObjective(self, traj, **kwargs):
    """
      Checks the change in objective for convergence
      @ In, traj, int, trajectory identifier
      @ In, kwargs, dict, dictionary of parameters for convergence criteria
      @ Out, converged, bool, convergence state
    """
    if len(self._optPointHistory[traj]) < 2:
      return False
    o1, _ = self._optPointHistory[traj][-1]
    obj = o1[self._objectiveVar]
    converged = (obj == self._convergenceCriteria['objective'])
    self.raiseADebug(self.convFormat.format(name='objective',
                                            conv=str(converged),
                                            got=obj,
                                            req=self._convergenceCriteria['objective']))
    return converged

  def _checkConvAHDp(self, traj, **kwargs):
    """
      Computes the Average Hausdorff Distance as the termination criteria
      @ In, traj, int, trajectory identifier
      @ In, kwargs, dict, dictionary of parameters for AHDp termination criteria:
            old, np.array, old generation
            new, np.array, new generation
            p, float or integer, Minkowski norm order, (default 3)
      @ Out, converged, bool, convergence state
    """
    old = kwargs['old'].data
    new = datasetToDataArray(kwargs['new'], list(self.toBeSampled)).data
    if ('p' not in kwargs.keys() or kwargs['p'] == None):
      p = 3
    else:
      p = kwargs['p']
    ahdp = self._ahdp(old,new,p)
    self.ahdp = ahdp
    converged = (ahdp <= self._convergenceCriteria['AHDp'])
    self.raiseADebug(self.convFormat.format(name='AHDp',
                                            conv=str(converged),
                                            got=ahdp,
                                            req=self._convergenceCriteria['AHDp']))
    return converged

  def _checkConvAHD(self, traj, **kwargs):
    """
      Computes the Hausdorff Distance as the termination criteria
      @ In, traj, int, trajectory identifier
      @ In, kwargs, dict, dictionary of parameters for AHDp termination criteria:
            old, np.array, old generation
            new, np.array, new generation
      @ Out, converged, bool, convergence state
    """
    old = kwargs['old'].data
    new = datasetToDataArray(kwargs['new'], list(self.toBeSampled)).data
    ahd = self._ahd(old,new)
    self.ahd = ahd
    converged = (ahd < self._convergenceCriteria['AHD'])
    self.raiseADebug(self.convFormat.format(name='AHD',
                                            conv=str(converged),
                                            got=ahd,
                                            req=self._convergenceCriteria['AHD']))
    return converged

  def _ahdp(self,a,b,p):
    """
      p-average Hausdorff Distance for generation convergence
      @ In, a, np.array, old population A
      @ In, b, np.array, new population B
      @ Out, _AHDp, float, average Hausdorff distance
    """
    return max(self._GDp(a,b,p),self._GDp(b,a,p))

  def _GDp(self,a,b,p):
    """
      Modified Generational Distance Indicator
      @ In, a, np.array, old population A
      @ In, b, np.array, new population B
      @ In, p, float, the order of norm
      @ Out, _GDp, float, the modified generational distance $\frac{1}{n_A} \Sigma_{i=1}^{n_A}min_{b \in B} dist(ai,B)$
    """
    s = 0
    n = np.shape(a)[0]
    for i in range(n):
      s += self._popDist(a[i,:],b)**p
    return (1/n * s)**(1/p)

  def _popDist(self,ai,b,q=2):
    """
      Minimum Minkowski distance from a_i to B (nearest point in B)
      @ In, ai, 1d array, the ith chromosome in the generation A
      @ In, b, np.array, population B
      @ In, q, integer, order of the norm
      @ Out, _popDist, float, the minimum distance from ai to B $inf_(\|ai-bj\|_q)**\frac{1}{q}$
    """
    nrm=[]
    for j in range(np.shape(b)[0]):
      nrm.append(np.linalg.norm(ai-b[j,:],q))
    return min(nrm)

  def _ahd(self,a,b):
    """
      Hausdorff Distance for generation convergence
      @ In, a, np.array, old population A
      @ In, b, np.array, new population B
      @ Out, _AHD, float, Hausdorff distance
    """
    return max(self._GD(a,b),self._GD(b,a))

  def _GD(self,a,b):
    """
      Generational Distance Indicator
      @ In, a, np.array, old population A
      @ In, b, np.array, new population B
      @ Out, _GD, float, the generational distance $\frac{1}{n_A} \max_{i \in A}min_{b \in B} dist(ai,B)$
    """
    s = []
    n = np.shape(a)[0]
    for i in range(n):
      s.append(self._popDist(a[i,:],b))
    return max(s)

  def _updateConvergence(self, traj, new, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, new, xr.DataSet, new children
      @ In, old, xr.DataArray, old population
      @ In, acceptable, str, condition of new point
      @ Out, converged, bool, True if converged on ANY criteria
    """
    ## NOTE we have multiple "if acceptable" trees here, as we need to update soln export regardless
    if acceptable == 'accepted':
      self.raiseADebug('Convergence Check for Trajectory {}:'.format(traj))
      # check convergence
      converged, convDict = self.checkConvergence(traj, new, old)
    else:
      converged = False
      convDict = dict((var, False) for var in self._convergenceInfo[traj])
    self._convergenceInfo[traj].update(convDict)
    return converged

  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, int, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """
    # This is not required for the genetic algorithms as it's handled in the probabilistic acceptance criteria
    # But since it is an abstract method it has to exist
    return

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    # This is not required for the genetic algorithms as it's handled in the probabilistic acceptance criteria
    # But since it is an abstract method it has to exist
    return

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
    """
    return

  # * * * * * * * * * * * *
  # Constraint Handling
  def _handleExplicitConstraints(self, point, constraint):
    """
      Computes explicit (i.e. input-based) constraints
      @ In, point, xr.DataArray, the DataArray containing the chromosome (point)
      @ In, constraint, external function, explicit constraint function
      @ out, g, float, the value g_j(x) is the value of the constraint function number j when fed with the chromosome (point)
                if $g_j(x)<0$, then the contraint is violated
    """
    g = self._applyFunctionalConstraints(point, constraint)
    return g

  def _handleImplicitConstraints(self, point, opt,constraint):
    """
      Computes implicit (i.e. output- or output-input-based) constraints
      @ In, point, xr.DataArray, the DataArray containing the chromosome (point)
      @ In, opt, float, the objective value at this chromosome (point)
      @ In, constraint, external function, implicit constraint function
      @ out, g, float,the value g_j(x) is the value of the constraint function number j when fed with the chromosome (point)
                if $g_j(x)<0$, then the contraint is violated
    """
    g = self._checkImpFunctionalConstraints(point, opt, constraint)
    return g

  def _applyFunctionalConstraints(self, point, constraint):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, point, xr.DataArray, the dataArray containing potential point to apply constraints to
      @ In, constraint, external function, constraint function
      @ out, g, float, the value g_j(x) is the value of the constraint function number j when fed with the chromosome (point)
                if $g_j(x)<0$, then the contraint is violated
    """
    # are we violating functional constraints?
    g = self._checkFunctionalConstraints(point, constraint)
    return g

  def _checkFunctionalConstraints(self, point, constraint):
    """
      evaluates the provided constraint at the provided point
      @ In, point, dict, the dictionary containing the chromosome (point)
      @ In, constraint, external function, explicit constraint function
      @ out, g, float, the value g_j(x) is the value of the constraint function number j when fed with the chromosome (point)
                if $g_j(x)<0$, then the contraint is violated
    """
    inputs = dataArrayToDict(point)
    inputs.update(self.constants)
    g = constraint.evaluate('constrain', inputs)
    return g

  def _checkImpFunctionalConstraints(self, point, opt, impConstraint):
    """
      evaluates the provided implicit constraint at the provided point
      @ In, point, dict, the dictionary containing the chromosome (point)
      @ In, opt, dict, the dictionary containing the chromosome (point)
      @ In, impConstraint, external function, implicit constraint function
      @ out, g, float, the value g_j(x, objVar) is the value of the constraint function number j when fed with the chromosome (point)
                if $g_j(x, objVar)<0$, then the contraint is violated
    """
    inputs = dataArrayToDict(point)
    inputs.update(self.constants)
    inputs[self._objectiveVar] = opt
    g = impConstraint.evaluate('impConstrain', inputs)
    return g

  # END constraint handling
  # * * * * * * * * * * * *


  def _addToSolutionExport(self, traj, rlz, acceptable):
    """
      Contributes additional entries to the solution export.
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ Out, toAdd, dict, additional entries
    """
    # meta variables
    toAdd = {'age': 0 if self.popAge==None else self.popAge,
             'batchId':self.batchId,
             'fitness':rlz['fitness'],
             'AHDp':self.ahdp,
             'AHD':self.ahd}

    for var, val in self.constants.items():
      toAdd[var] = val

    toAdd = dict((key, np.atleast_1d(val)) for key, val in toAdd.items())
    for key, val in self._convergenceInfo[traj].items():
      toAdd['conv_{}'.format(key)] = bool(val)
    return toAdd

  def _formatSolutionExportVariableNames(self, acceptable):
    """
      Does magic formatting for variables, based on this class's needs.
      Extend in inheritors as needed.
      @ In, acceptable, set, set of acceptable entries for solution export for this entity
      @ Out, new, set, modified set of acceptable variables with all formatting complete
    """
    # remaking the list is easier than using the existing one
    acceptable = RavenSampled._formatSolutionExportVariableNames(self, acceptable)
    new = []
    while acceptable:
      template = acceptable.pop()
      if '{CONV}' in template:
        new.extend([template.format(CONV=conv) for conv in self._convergenceCriteria])
      elif '{VAR}' in template:
        new.extend([template.format(VAR=var) for var in self.toBeSampled.keys()])
      else:
        new.append(template)
    return set(new)
