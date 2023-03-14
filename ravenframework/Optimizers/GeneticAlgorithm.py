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
# External Modules----------------------------------------------------------------------------------
from collections import deque, defaultdict
import numpy as np
from scipy.special import comb
import xarray as xr
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..utils import mathUtils, InputData, InputTypes, frontUtils
from ..utils.gaUtils import dataArrayToDict, datasetToDataArray
from .RavenSampled import RavenSampled
from .parentSelectors.parentSelectors import returnInstance as parentSelectionReturnInstance
from .parentSelectors.parentSelectors import countConstViolation
from .crossOverOperators.crossovers import returnInstance as crossoversReturnInstance
from .mutators.mutators import returnInstance as mutatorsReturnInstance
from .survivorSelectors.survivorSelectors import returnInstance as survivorSelectionReturnInstance
from .fitness.fitness import returnInstance as fitnessReturnInstance
from .repairOperators.repair import returnInstance as repairReturnInstance
# Internal Modules End------------------------------------------------------------------------------

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
    self.needDenormalized()                                      # the default in all optimizers is to normalize the data which is not the case here
    self.batchId = 0
    self.population = None                                       # panda Dataset container containing the population at the beginning of each generation iteration
    self.popAge = None                                           # population age
    self.fitness = None                                          # population fitness
    self.rank = None                                             # population rank (for Multi-objective optimization only)
    self.constraints = None
    self.constraintsV = None
    self.crowdingDistance = None                                 # population crowding distance (for Multi-objective optimization only)
    self.ahdp = np.NaN                                           # p-Average Hausdorff Distance between populations
    self.ahd  = np.NaN                                           # Hausdorff Distance between populations
    self.bestPoint = None
    self.bestFitness = None
    self.bestObjective = None
    self.multiBestPoint = None
    self.multiBestFitness = None
    self.multiBestObjective = None
    self.multiBestConstraint = None
    self.multiBestRank = None
    self.multiBestCD = None
    self.objectiveVal = None
    self._populationSize = None
    self._parentSelectionType = None
    self._parentSelectionInstance = None
    self._nParents = None
    self._nChildren = None
    self._crossoverType = None
    self._crossoverPoints = None
    self._crossoverProb = None
    self._crossoverInstance = None
    self._mutationType = None
    self._mutationLocs = None
    self._mutationProb = None
    self._mutationInstance = None
    self._survivorSelectionType = None
    self._survivorSelectionInstance = None
    self._fitnessType = None
    self._objCoeff = None
    self._penaltyCoeff = None
    self._fitnessInstance = None
    self._repairInstance = None
    self._canHandleMultiObjective = True

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
                  b. \textbf{\textit{tournamentSelection}},
                  c. \textbf{\textit{rankSelection}}
                  for all methods nParents is computed such that the population size is kept constant.
                  $nChildren = 2 \times {nParents \choose 2} = nParents \times (nParents-1) = popSize$
                  solving for nParents we get:
                  $nParents = ceil(\frac{1 + \sqrt{1+4*popSize}}{2})$
                  This will result in a popSize a little larger than the initial one, these excessive children will be later thrawn away and only the first popSize child will be kept""")
    GAparams.addSub(parentSelection)

    # Reproduction
    reproduction = InputData.parameterInputFactory('reproduction', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the reproduction methods.
                  This accepts subnodes that specifies the types of crossover and mutation.""")
    # 1.  Crossover
    crossover = InputData.parameterInputFactory('crossover', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented crossover mechanisms.
                  This includes: a.    onePointCrossover,
                                 b.    twoPointsCrossover,
                                 c.    uniformCrossover.""")
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
                  This includes: a.    bitFlipMutation,
                                 b.    swapMutation,
                                 c.    scrambleMutation, or
                                 d.    inversionMutation.""")
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
                  This includes: a.    invLinear: $fitness = -a \times obj - b \times \sum_{j=1}^{nConstraint} max(0,-penalty\_j) $.

                                 b.    logistic: $fitness = \frac{1}{1+e^{a\times(obj-b)}}$.

                                 c.    feasibleFirst: $fitness = \left\{\begin{matrix} -obj & g_j(x)\geq 0 \; \forall j \\ -obj_{worst}- \Sigma_{j=1}^{J}<g_j(x)> & otherwise \\ \end{matrix}\right.$""")
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
    for name, descr in cls.convergenceOptions.items():
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
    new['rank'] = 'rank'
    new['CD'] = 'crowding distance'
    new['fitness'] = 'fitness of the current chromosome'
    new['age'] = 'age of current chromosome'
    new['batchId'] = 'Id of the batch to whom the chromosome belongs'
    new['AHDp'] = 'p-Average Hausdorff Distance between populations'
    new['AHD'] = 'Hausdorff Distance between populations'
    new['ConstraintEvaluation_{CONSTRAINT}'] = 'Constraint function evaluation (negative if violating and positive otherwise)'
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
    self._parentSelectionInstance = parentSelectionReturnInstance(self, name=parentSelectionNode.value)
    # reproduction node
    reproductionNode = gaParamsNode.findFirst('reproduction')
    self._nParents = int(np.ceil(1/2 + np.sqrt(1+4*self._populationSize)/2))
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

    # Check if the fitness requested is among the constrained optimization fitnesses
    # Currently, only InvLin and feasibleFirst Fitnesses deal with constrained optimization
    # TODO: @mandd, please explore the possibility to convert the logistic fitness into a constrained optimization fitness.
    if 'Constraint' in self.assemblerObjects and self._fitnessType not in ['invLinear','feasibleFirst','rank_crowding']:
      self.raiseAnError(IOError, f'Currently constrained Genetic Algorithms only support invLinear and feasibleFirst fitnesses, whereas provided fitness is {self._fitnessType}')
    self._objCoeff = fitnessNode.findFirst('a').value if fitnessNode.findFirst('a') is not None else None
    self._penaltyCoeff = fitnessNode.findFirst('b').value if fitnessNode.findFirst('b') is not None else None
    self._fitnessInstance = fitnessReturnInstance(self,name = self._fitnessType)
    self._repairInstance = repairReturnInstance(self,name='replacementRepair')  # currently only replacement repair is implemented.

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
    self.batch = self._populationSize
    if self._populationSize != len(self._initialValues):
      self.raiseAnError(IOError, f'Number of initial values provided for each variable is {len(self._initialValues)}, while the population size is {self._populationSize}')
    for _, init in enumerate(self._initialValues):
      self._submitRun(init, 0, self.getIteration(0) + 1)

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

  ##########################################################################################################
  # Run Methods                                                                                            #
  ##########################################################################################################

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
      self._closeTrajectory(t, 'cancel', 'Currently GA is single trajectory', 0)
    self.incrementIteration(traj)
    info['step'] = self.counter

    # Developer note: each algorithm step is indicated by a number followed by the generation number
    # e.g., '0 @ n-1' refers to step 0 for generation n-1 (i.e., previous generation)
    # for more details refer to GRP-Raven-development/Disceret_opt channel on MS Teams

    # 0 @ n-1: Survivor Selection from previous iteration (children+parents merging from previous generation)
    # 0.1 @ n-1: fitnessCalculation(rlz)
    # perform fitness calculation for newly obtained children (rlz)

    if not self._canHandleMultiObjective or len(self._objectiveVar) == 1:  # This is a single-objective Optimization case
      offSprings = datasetToDataArray(rlz, list(self.toBeSampled))
      objectiveVal = list(np.atleast_1d(rlz[self._objectiveVar[0]].data))

      # collect parameters that the constraints functions need (neglecting the default params such as inputs and objective functions)
      constraintData = {}
      if self._constraintFunctions or self._impConstraintFunctions:
        params = []
        for y in (self._constraintFunctions + self._impConstraintFunctions):
          params += y.parameterNames()
        for p in list(set(params) -set([self._objectiveVar[0]]) -set(list(self.toBeSampled.keys()))):
        # for p in list(set(params) -set([self._objectiveVar]) -set(list(self.toBeSampled.keys()))):
          constraintData[p] = list(np.atleast_1d(rlz[p].data))
      # Compute constraint function g_j(x) for all constraints (j = 1 .. J)
      # and all x's (individuals) in the population
      g0 = np.zeros((np.shape(offSprings)[0],len(self._constraintFunctions)+len(self._impConstraintFunctions)))

      g = xr.DataArray(g0,
                      dims=['chromosome','Constraint'],
                      coords={'chromosome':np.arange(np.shape(offSprings)[0]),
                              'Constraint':[y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]})
      # FIXME The constraint handling is following the structure of the RavenSampled.py,
      #        there are many utility functions that can be simplified and/or merged together
      #        _check, _handle, and _apply, for explicit and implicit constraints.
      #        This can be simplified in the near future in GradientDescent, SimulatedAnnealing, and here in GA
      for index,individual in enumerate(offSprings):
        newOpt = individual
        opt = {self._objectiveVar[0]:objectiveVal[index]}
        for p, v in constraintData.items():
          opt[p] = v[index]

        for constIndex, constraint in enumerate(self._constraintFunctions + self._impConstraintFunctions):
          if constraint in self._constraintFunctions:
            g.data[index, constIndex] = self._handleExplicitConstraints(newOpt, constraint)
          else:
            g.data[index, constIndex] = self._handleImplicitConstraints(newOpt, opt, constraint)

      offSpringFitness = self._fitnessInstance(rlz,
                                               objVar=self._objectiveVar[0],
                                               a=self._objCoeff,
                                               b=self._penaltyCoeff,
                                               penalty=None,
                                               constraintFunction=g,
                                               type=self._minMax)

      self._collectOptPoint(rlz, offSpringFitness, objectiveVal, g)
      self._resolveNewGeneration(traj, rlz, objectiveVal, offSpringFitness, g, info)

    else: # This is a multi-objective Optimization case
      objectiveVal = []
      offSprings = datasetToDataArray(rlz, list(self.toBeSampled))
      for i in range(len(self._objectiveVar)):
        objectiveVal.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))

      # collect parameters that the constraints functions need (neglecting the default params such as inputs and objective functions)
      constraintData = {}
      if self._constraintFunctions or self._impConstraintFunctions:
        params = []
        for y in (self._constraintFunctions + self._impConstraintFunctions):
          params += y.parameterNames()
        for p in list(set(params) -set(self._objectiveVar) -set(list(self.toBeSampled.keys()))):
          constraintData[p] = list(np.atleast_1d(rlz[p].data))
      # Compute constraint function g_j(x) for all constraints (j = 1 .. J)
      # and all x's (individuals) in the population
      g0 = np.zeros((np.shape(offSprings)[0],len(self._constraintFunctions)+len(self._impConstraintFunctions)))

      g = xr.DataArray(g0,
                      dims=['chromosome','Constraint'],
                      coords={'chromosome':np.arange(np.shape(offSprings)[0]),
                              'Constraint':[y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]})

      for index,individual in enumerate(offSprings):
        newOpt = individual
        opt = dict(zip(self._objectiveVar, [item[index] for item in objectiveVal]))
        for p, v in constraintData.items():
          opt[p] = v[index]

        for constIndex, constraint in enumerate(self._constraintFunctions + self._impConstraintFunctions):
          if constraint in self._constraintFunctions:
            g.data[index, constIndex] = self._handleExplicitConstraints(newOpt, constraint)
          else:
            g.data[index, constIndex] = self._handleImplicitConstraints(newOpt, opt, constraint)

      Fitness     = np.zeros((len(offSprings), 1))
      for i in range(len(Fitness)):
        Fitness[i] = countConstViolation(g.data[i])
      Fitness = [item for sublist in Fitness.tolist() for item in sublist]

      Fitness = xr.DataArray(Fitness,
                             dims=['NumOfConstraintViolated'],
                             coords={'NumOfConstraintViolated':np.arange(np.shape(Fitness)[0])})

      # 0.2@ n-1: Survivor selection(rlz)
      # update population container given obtained children

    if self._activeTraj:
      if len(self._objectiveVar) == 1: # If the number of objectives is just 1:
        if self.counter > 1:
          self.population, self.fitness,\
          age,self.objectiveVal = self._survivorSelectionInstance(age=self.popAge,
                                                                  variables=list(self.toBeSampled),
                                                                  population=self.population,
                                                                  fitness=self.fitness,
                                                                  newRlz=rlz,
                                                                  offSpringsFitness=offSpringFitness,
                                                                  popObjectiveVal=self.objectiveVal)
          self.popAge = age
        else:
          self.population = offSprings
          self.fitness = offSpringFitness
          self.objectiveVal = rlz[self._objectiveVar[0]].data

      else: # If the number of objectives is more than 1:
        if self.counter > 1:
          self.population,self.rank, \
          self.popAge,self.crowdingDistance, \
          self.objectiveVal,self.constraints, \
          self.constraintsV                  = self._survivorSelectionInstance(age=self.popAge,
                                                                               variables=list(self.toBeSampled),
                                                                               population=self.population,
                                                                               offsprings=rlz,
                                                                               popObjectiveVal=self.objectiveVal,
                                                                               offObjectiveVal=objectiveVal,
                                                                               popConst = self.constraints,
                                                                               offConst = Fitness,
                                                                               popConstV = self.constraintsV,
                                                                               offConstV = g
                                                                              )



          self._collectOptPointMulti(self.population,
                                     self.rank,
                                     self.crowdingDistance,
                                     self.objectiveVal,
                                     self.constraints,
                                     self.constraintsV)
          self._resolveNewGenerationMulti(traj, rlz, info)

          ##############################################################################
          objs_vals = [list(ele) for ele in list(zip(*self.objectiveVal))]
          import matplotlib.pyplot as plt
          # JY: Visualization: all points - This code block needs to be either deleted or revisited.
          plt.plot(np.array(objs_vals)[:,0], np.array(objs_vals)[:,1],'*')

          # JY: Visualization: optimal points only - This code block needs to be either deleted or revisited.
          # plt.xlim(75,100)
          # plt.ylim(5,20)
          plt.xlim(0,1)
          plt.ylim(0,6)
          plt.title(str('Iteration ' + str(self.counter-1)))

          plt.plot(np.array(list(zip(self._optPointHistory[traj][-1][0]['obj1'], self._optPointHistory[traj][-1][0]['obj2'])))[:,0],
                   np.array(list(zip(self._optPointHistory[traj][-1][0]['obj1'], self._optPointHistory[traj][-1][0]['obj2'])))[:,1],'*')
          for i in range(len(np.array(list(zip(self._optPointHistory[traj][-1][0]['obj1'], self._optPointHistory[traj][-1][0]['obj2'])))[:,0])):
            plt.text(np.array(list(zip(self._optPointHistory[traj][-1][0]['obj1'], self._optPointHistory[traj][-1][0]['obj2'])))[i,0],
                     np.array(list(zip(self._optPointHistory[traj][-1][0]['obj1'], self._optPointHistory[traj][-1][0]['obj2'])))[i,1], str(self.batchId-1))
          # plt.pause()
          ##############################################################################

        else:
          self.population = offSprings
          self.constraints = Fitness
          self.constraintsV = g
          self.rank, self.crowdingDistance = self._fitnessInstance(rlz,
                                                                   objVals = self._objectiveVar
                                                                   )
          self.objectiveVal = []
          for i in range(len(self._objectiveVar)):
            self.objectiveVal.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))

      # 1 @ n: Parent selection from population
      # pair parents together by indexes

      if len(self._objectiveVar) == 1: # If the number of objectives is just 1:
        parents = self._parentSelectionInstance(self.population,
                                                variables=list(self.toBeSampled),
                                                fitness=self.fitness,
                                                nParents=self._nParents)

      else: # This is for a multi-objective Optimization case

        parents = self._parentSelectionInstance(self.population,
                                                variables=list(self.toBeSampled),
                                                nParents=self._nParents,
                                                rank = self.rank,
                                                crowdDistance = self.crowdingDistance,
                                                constraint = self.constraints
                                                )

      # 2 @ n: Crossover from set of parents
      # create childrenCoordinates (x1,...,xM)
      childrenXover = self._crossoverInstance(parents=parents,
                                              variables=list(self.toBeSampled),
                                              crossoverProb=self._crossoverProb,
                                              points=self._crossoverPoints)

      # 3 @ n: Mutation
      # perform random directly on childrenCoordinates
      childrenMutated = self._mutationInstance(offSprings=childrenXover,
                                               distDict=self.distDict,
                                               locs=self._mutationLocs,
                                               mutationProb=self._mutationProb,
                                               variables=list(self.toBeSampled))

      # 4 @ n: repair/replacement
      # repair should only happen if multiple genes in a single chromosome have the same values (),
      # and at the same time the sampling of these genes should be with Out replacement.
      needsRepair = False
      for chrom in range(self._nChildren):
        unique = set(childrenMutated.data[chrom, :])
        if len(childrenMutated.data[chrom,:]) != len(unique):
          for var in self.toBeSampled: # TODO: there must be a smarter way to check if a variables strategy is without replacement
            if (hasattr(self.distDict[var], 'strategy') and self.distDict[var].strategy == 'withoutReplacement'):
              needsRepair = True
              break
      if needsRepair:
        children = self._repairInstance(childrenMutated,variables=list(self.toBeSampled),distInfo=self.distDict)
      else:
        children = childrenMutated

      # keeping the population size constant by ignoring the excessive children
      children = children[:self._populationSize, :]

      daChildren = xr.DataArray(children,
                                dims=['chromosome','Gene'],
                                coords={'chromosome': np.arange(np.shape(children)[0]),
                                        'Gene':list(self.toBeSampled)})

      # 5 @ n: Submit children batch
      # submit children coordinates (x1,...,xm), i.e., self.childrenCoordinates
      for i in range(self.batch):
        newRlz = {}
        for _, var in enumerate(self.toBeSampled.keys()):
          newRlz[var] = float(daChildren.loc[i, var].values)
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
    self.raiseADebug(f'Adding run to queue: {self.denormalizeData(point)} | {info}')
    self._submissionQueue.append((point, info))

  def flush(self):
    """
      Reset Optimizer attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self.population = None
    self.popAge = None
    self.fitness = None
    self.rank = None
    self.crowdingDistance = None
    self.ahdp = np.NaN
    self.ahd = np.NaN
    self.bestPoint = None
    self.bestFitness = None
    self.bestObjective = None
    self.objectiveVal = None
    self.multiBestPoint = None
    self.multiBestFitness = None
    self.multiBestObjective = None
    self.multiBestConstraint = None
    self.multiBestRank = None
    self.multiBestCD = None

  # END queuing Runs
  # * * * * * * * * * * * * * * * *
  def _resolveNewGeneration(self, traj, rlz, objectiveVal, fitness, g, info):
    """
      Store a new Generation after checking convergence
      @ In, traj, int, trajectory for this new point
      @ In, rlz, dict, realized realization
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, fitness, xr.DataArray, fitness values at each chromosome of the realization
      @ In, g, xr.DataArray, the constraint evaluation function
      @ In, info, dict, identifying information about the realization
    """
    self.raiseADebug('*'*80)
    self.raiseADebug(f'Trajectory {traj} iteration {info["step"]} resolving new state ...')
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    acceptable = 'accepted' if self.counter > 1 else 'first'
    old = self.population
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    if converged:
      self._closeTrajectory(traj, 'converge', 'converged', self.bestObjective)
    # NOTE: the solution export needs to be updated BEFORE we run rejectOptPoint or extend the opt
    #       point history.
    if self._writeSteps == 'every':
      for i in range(rlz.sizes['RAVEN_sample_ID']):
        varList = self._solutionExport.getVars('input') + self._solutionExport.getVars('output') + list(self.toBeSampled.keys())
        rlzDict = dict((var,np.atleast_1d(rlz[var].data)[i]) for var in set(varList) if var in rlz.data_vars)
        rlzDict[self._objectiveVar[0]] = np.atleast_1d(rlz[self._objectiveVar[0]].data)[i]
        rlzDict['fitness'] = np.atleast_1d(fitness.data)[i]
        for ind, consName in enumerate(g['Constraint'].values):
          rlzDict['ConstraintEvaluation_'+consName] = g[i,ind]
        self._updateSolutionExport(traj, rlzDict, acceptable, None)
    # decide what to do next
    if acceptable in ['accepted', 'first']:
      # record history
      bestRlz = {}
      bestRlz[self._objectiveVar[0]] = self.bestObjective
      bestRlz['fitness'] = self.bestFitness
      bestRlz.update(self.bestPoint)
      self._optPointHistory[traj].append((bestRlz, info))
    elif acceptable == 'rejected':
      self._rejectOptPoint(traj, info, old)
    else: # e.g. rerun
      pass # nothing to do, just keep moving

  def _resolveNewGenerationMulti(self, traj, rlz, info):
    """
      Store a new Generation after checking convergence
      @ In, traj, int, trajectory for this new point
      @ In, rlz, dict, realized realization
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, fitness, xr.DataArray, fitness values at each chromosome of the realization
      @ In, g, xr.DataArray, the constraint evaluation function
      @ In, info, dict, identifying information about the realization
    """
    self.raiseADebug('*'*80)
    self.raiseADebug(f'Trajectory {traj} iteration {info["step"]} resolving new state ...')
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    acceptable = 'accepted' if self.counter > 1 else 'first'
    old = self.population
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    if converged:
      self._closeTrajectory(traj, 'converge', 'converged', self.bestObjective)
    # NOTE: the solution export needs to be updated BEFORE we run rejectOptPoint or extend the opt
    #       point history.
    objVal = [[] for x in range(len(self.objectiveVal[0]))]
    for i in range(len(self.objectiveVal[0])):
      objVal[i] = [item[i] for item in self.objectiveVal]

    objVal = xr.DataArray(objVal,
                          dims=['chromosome','obj'],
                          coords={'chromosome':np.arange(np.shape(objVal)[0]),
                                  'obj': self._objectiveVar})
    if self._writeSteps == 'every':
      for i in range(rlz.sizes['RAVEN_sample_ID']):
        rlzDict = dict((var,self.population.data[i][j]) for j, var in enumerate(self.population.Gene.data))
        for j in range(len(self._objectiveVar)):
          rlzDict[self._objectiveVar[j]] = objVal.data[i][j]
        rlzDict['rank'] = np.atleast_1d(self.rank.data)[i]
        rlzDict['CD'] = np.atleast_1d(self.crowdingDistance.data)[i]
        rlzDict['fitness'] = np.atleast_1d(self.constraints.data)[i]
        for ind, consName in enumerate([y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]):
          rlzDict['ConstraintEvaluation_'+consName] = self.constraintsV.data[i,ind]
        self._updateSolutionExport(traj, rlzDict, acceptable, None)

    # decide what to do next
    if acceptable in ['accepted', 'first']:
      # record history
      bestRlz = {}
      for i in range(len(self._objectiveVar)):
        bestRlz[self._objectiveVar[i]] = [item[i] for item in self.multiBestObjective]
      bestRlz['fitness'] = self.multiBestFitness
      bestRlz['rank'] = self.multiBestRank
      bestRlz['CD'] = self.multiBestCD
      if len(self.multiBestConstraint) != 0: # No constraints
        for ind, consName in enumerate(self.multiBestConstraint.Constraint):
            bestRlz['ConstraintEvaluation_'+consName.values.tolist()] = self.multiBestConstraint[ind].values
      bestRlz.update(self.multiBestPoint)
      self._optPointHistory[traj].append((bestRlz, info))
    elif acceptable == 'rejected':
      self._rejectOptPoint(traj, info, old)
    else: # e.g. rerun
      pass # nothing to do, just keep moving

  def _collectOptPoint(self, rlz, fitness, objectiveVal, g):
    """
      Collects the point (dict) from a realization
      @ In, population, Dataset, container containing the population
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, fitness, xr.DataArray, fitness values at each chromosome of the realization
      @ Out, point, dict, point used in this realization
    """

    varList = list(self.toBeSampled.keys()) + self._solutionExport.getVars('input') + self._solutionExport.getVars('output')
    varList = set(varList)
    selVars = [var for var in varList if var in rlz.data_vars]
    population = datasetToDataArray(rlz, selVars)
    optPoints,fit,obj,gOfBest = zip(*[[x,y,z,w] for x, y, z,w in sorted(zip(np.atleast_2d(population.data),np.atleast_1d(fitness.data),objectiveVal,np.atleast_2d(g.data)),reverse=True,key=lambda x: (x[1]))])
    point = dict((var,float(optPoints[0][i])) for i, var in enumerate(selVars) if var in rlz.data_vars)
    gOfBest = dict(('ConstraintEvaluation_'+name,float(gOfBest[0][i])) for i, name in enumerate(g.coords['Constraint'].values))
    if (self.counter > 1 and obj[0] <= self.bestObjective and fit[0] >= self.bestFitness) or self.counter == 1:
      point.update(gOfBest)
      self.bestPoint = point
      self.bestFitness = fit[0]
      self.bestObjective = obj[0]

    return point

  def _collectOptPointMulti(self, population, rank, CD, objectiveVal, constraints, constraintsV):
    """
      Collects the point (dict) from a realization
      @ In, population, Dataset, container containing the population
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, rank, xr.DataArray, rank values at each chromosome of the realization
      @ In, crowdingDistance, xr.DataArray, crowdingDistance values at each chromosome of the realization
      @ Out, point, dict, point used in this realization
    """
    objVal = [[] for x in range(len(objectiveVal[0]))]
    for i in range(len(objectiveVal[0])):
      objVal[i] = [item[i] for item in objectiveVal]

    optPointsConsIDX = [i for i, nFit in enumerate(constraints) if nFit == min(constraints)]     #  Find index of chromosome which has smallest numeber of violations among population
    optPointsRankNConsIDX = [i for i, rankValue in enumerate(rank[optPointsConsIDX]) if rankValue == min(rank[optPointsConsIDX])]    #  Find index of chromosome which has smallest numeber of violations among population & smallest rank

    optPoints,optObjVal,optConstraints,optConstraintsV,optRank,optCD = population[optPointsRankNConsIDX], np.array(objVal)[optPointsRankNConsIDX], constraints.data[optPointsRankNConsIDX], constraintsV.data[optPointsRankNConsIDX], rank.data[optPointsRankNConsIDX], CD.data[optPointsRankNConsIDX]

    # Previous ##################################################
    # points,multiFit,rankSorted,cdSorted,objSorted,constSorted = \
    #   zip(*[[a,b,c,d,e,f] for a, b, c, d, e, f in sorted(zip(np.atleast_2d(population.data),np.atleast_1d(constraintsV.data),np.atleast_1d(rank.data),np.atleast_1d(CD.data), objVal, constraints),
    #   reverse=True,key=lambda x: (-x[1], -x[2], x[3]))])
    # optPoints = [points[i] for i, rank in enumerate(rankSorted) if rank == 1 ]
    # optMultiFit = [multiFit[i] for i, rank in enumerate(rankSorted) if rank == 1 ]
    # optObj = [objSorted[i] for i, rank in enumerate(rankSorted) if rank == 1 ]
    # optConst = [constSorted[i] for i, rank in enumerate(rankSorted) if rank == 1 ]
    # optRank = [rankSorted[i] for i, rank in enumerate(rankSorted) if rank == 1 ]
    # optCD = [cdSorted[i] for i, rank in enumerate(rankSorted) if rank == 1 ]
    # if (len(optMultiFit) != len([x for x in optMultiFit if x != 0]) ) :
    #   optPoints = [optPoints[i] for i, nFit in enumerate(optMultiFit) if nFit == 0 ]
    #   optMultiFit = [x for x in optMultiFit if x == 0]
    #   optObj = [optObj[i] for i, nFit in enumerate(optMultiFit) if nFit == 0 ]
    #   optConst = [optConst[i] for i, nFit in enumerate(optMultiFit) if nFit == 0 ]
    #   optRank = [optRank[i] for i, nFit in enumerate(optMultiFit) if nFit == 0 ]
    #   optCD = [optCD[i] for i, nFit in enumerate(optMultiFit) if nFit == 0 ]
    # Previous ##################################################

    optPointsDic = dict((var,np.array(optPoints)[:,i]) for i, var in enumerate(population.Gene.data))
    optConstNew = []
    for i in range(len(optConstraintsV)):
      optConstNew.append(optConstraintsV[i])
    optConstNew = list(map(list, zip(*optConstNew)))
    if (len(optConstNew)) != 0:
      optConstNew = xr.DataArray(optConstNew,
                                dims=['Constraint','Evaluation'],
                                coords={'Constraint':[y.name for y in (self._constraintFunctions + self._impConstraintFunctions)],
                                        'Evaluation':np.arange(np.shape(optConstNew)[1])})

    self.multiBestPoint = optPointsDic
    self.multiBestFitness = optConstraints
    self.multiBestObjective = optObjVal
    self.multiBestConstraint = optConstNew
    self.multiBestRank = optRank
    self.multiBestCD = optCD

    return optPointsDic


  def _checkAcceptability(self, traj):
    """
      This is an abstract method for all RavenSampled Optimizer, whereas for GA all children are accepted
      @ In, traj, int, identifier
    """
    return

  def checkConvergence(self, traj, new, old):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory to consider
      @ In, new, xr.DataSet, new children realization
      @ In, old, xr.DataArray, old population
      @ Out, any(convs.values()), bool, True of any of the convergence criteria was reached
      @ Out, convs, dict, on the form convs[conv] = bool, where conv is in self._convergenceCriteria
    """
    if len(self._objectiveVar) == 1:
      convs = {}
      for conv in self._convergenceCriteria:
        fName = conv[:1].upper() + conv[1:]
        # get function from lookup
        f = getattr(self, f'_checkConv{fName}')
        # check convergence function
        okay = f(traj, new=new, old=old)
        # store and update
        convs[conv] = okay
    else:
      convs = {}
      for conv in self._convergenceCriteria:
        fName = conv[:1].upper() + conv[1:]
        # get function from lookup
        f = getattr(self, f'_checkConv{fName}')
        # check convergence function
        okay = f(traj, new=new, old=old)
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
    if len(self._objectiveVar) == 1: # single objective optimization
      if len(self._optPointHistory[traj]) < 2:
        return False
      o1, _ = self._optPointHistory[traj][-1]
      obj = o1[self._objectiveVar[0]]
      converged = (obj == self._convergenceCriteria['objective'])
      self.raiseADebug(self.convFormat.format(name='objective',
                                              conv=str(converged),
                                              got=obj,
                                              req=self._convergenceCriteria['objective']))
    else:                            # multi objective optimization
      if len(self._optPointHistory[traj]) < 2:
        return False
      o1, _ = self._optPointHistory[traj][-1]
      obj1 = o1[self._objectiveVar[0]]
      obj2 = o1[self._objectiveVar[1]]
      converged = (obj1 == self._convergenceCriteria['objective'] and obj2 == self._convergenceCriteria['objective'])
      # JY: I stopped here. Codeline below needs to be revisited! 01/16/23
      # self.raiseADebug(self.convFormat.format(name='objective',
      #                                         conv=str(converged),
      #                                         got=obj1,
      #                                         req=self._convergenceCriteria['objective']))

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
    if ('p' not in kwargs or kwargs['p'] is None):
      p = 3
    else:
      p = kwargs['p']
    ahdp = self._ahdp(old, new, p)
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

  def _ahdp(self, a, b, p):
    """
      p-average Hausdorff Distance for generation convergence
      @ In, a, np.array, old population A
      @ In, b, np.array, new population B
      @ Out, _AHDp, float, average Hausdorff distance
    """
    return max(self._GDp(a, b, p), self._GDp(b, a, p))

  def _GDp(self, a, b, p):
    r"""
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
    r"""
      Minimum Minkowski distance from a_i to B (nearest point in B)
      @ In, ai, 1d array, the ith chromosome in the generation A
      @ In, b, np.array, population B
      @ In, q, integer, order of the norm
      @ Out, _popDist, float, the minimum distance from ai to B $inf_(\|ai-bj\|_q)**\frac{1}{q}$
    """
    nrm = []
    for j in range(np.shape(b)[0]):
      nrm.append(np.linalg.norm(ai-b[j,:], q))

    return min(nrm)

  def _ahd(self, a, b):
    """
      Hausdorff Distance for generation convergence
      @ In, a, np.array, old population A
      @ In, b, np.array, new population B
      @ Out, _AHD, float, Hausdorff distance
    """
    return max(self._GD(a,b),self._GD(b,a))

  def _GD(self,a,b):
    r"""
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
    # NOTE we have multiple "if acceptable" trees here, as we need to update soln export regardless
    if len(self._objectiveVar) == 1: # single-objective optimization
      if acceptable == 'accepted':
        self.raiseADebug(f'Convergence Check for Trajectory {traj}:')
        # check convergence
        converged, convDict = self.checkConvergence(traj, new, old)
      else:
        converged = False
        convDict = dict((var, False) for var in self._convergenceInfo[traj])
      self._convergenceInfo[traj].update(convDict)
    else: # multi-objective optimization
      if acceptable == 'accepted':
        self.raiseADebug(f'Convergence Check for Trajectory {traj}:')
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
                if $g_j(x)<0$, then the constraint is violated
    """
    return self._applyFunctionalConstraints(point, constraint)

  def _handleImplicitConstraints(self, point, opt,constraint):
    """
      Computes implicit (i.e. output- or output-input-based) constraints
      @ In, point, xr.DataArray, the DataArray containing the chromosome (point)
      @ In, opt, float, the objective value at this chromosome (point)
      @ In, constraint, external function, implicit constraint function
      @ out, g, float,the value g_j(x) is the value of the constraint function number j when fed with the chromosome (point)
                if $g_j(x)<0$, then the constraint is violated
    """
    return self._checkImpFunctionalConstraints(point, opt, constraint)

  def _applyFunctionalConstraints(self, point, constraint):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, point, xr.DataArray, the dataArray containing potential point to apply constraints to
      @ In, constraint, external function, constraint function
      @ out, g, float, the value g_j(x) is the value of the constraint function number j when fed with the chromosome (point)
                if $g_j(x)<0$, then the constraint is violated
    """
    # are we violating functional constraints?
    return self._checkFunctionalConstraints(point, constraint)

  def _checkFunctionalConstraints(self, point, constraint):
    """
      evaluates the provided constraint at the provided point
      @ In, point, dict, the dictionary containing the chromosome (point)
      @ In, constraint, external function, explicit constraint function
      @ out, g, float, the value g_j(x) is the value of the constraint function number j when fed with the chromosome (point)
                if $g_j(x)<0$, then the constraint is violated
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
                if $g_j(x, objVar)<0$, then the constraint is violated
    """
    inputs = dataArrayToDict(point)
    inputs.update(self.constants)
    inputs.update(opt)

    g = impConstraint.evaluate('implicitConstraint', inputs)

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
    toAdd = {'age': 0 if self.popAge is None else self.popAge,
             'batchId': self.batchId,
             'fitness': rlz['fitness'],
             'AHDp': self.ahdp,
             'AHD': self.ahd,
             'rank': 0 if ((type(self._objectiveVar) == list and len(self._objectiveVar) == 1) or type(self._objectiveVar) == str) else rlz['rank'],
             'CD': 0 if ((type(self._objectiveVar) == list and len(self._objectiveVar) == 1) or type(self._objectiveVar) == str) else  rlz['CD']}

    for var, val in self.constants.items():
      toAdd[var] = val

    toAdd = dict((key, np.atleast_1d(val)) for key, val in toAdd.items())
    for key, val in self._convergenceInfo[traj].items():
      toAdd[f'conv_{key}'] = bool(val)

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
        new.extend([template.format(VAR=var) for var in self.toBeSampled])
      elif '{CONSTRAINT}' in template:
        new.extend([template.format(CONSTRAINT=constraint.name) for constraint in self._constraintFunctions + self._impConstraintFunctions])
      else:
        new.append(template)

    return set(new)
