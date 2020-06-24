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
    .. [1]
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
from collections import deque, defaultdict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils, randomUtils, InputData, InputTypes
from .RavenSampled import RavenSampled
from .parentSelectors.parentSelectors import returnInstance as parentSelectionReturnInstance
from .crossOverOperators.crossovers import returnInstance as crossoversReturnInstance
from .mutators.mutators import returnInstance as mutatorsReturnInstance
from .survivorSelectors.survivorSelectors import returnInstance as survivorSelectionReturnInstance
from .fitness.fitness import returnInstance as fitnessReturnInstance
#Internal Modules End--------------------------------------------------------------------------------

class GeneticAlgorithm(RavenSampled):
  """
    This class performs Genetic Algorithm optimization ...
  """
  convergenceOptions = {'objective': r""" provides the desired value for the convergence criterion of the objective function
                        ($\epsilon^{obj}$), i.e., convergence is reached when: $$ |newObjevtive - oldObjective| \le \epsilon^{obj}$$.
                        \default{1e-6}, if no criteria specified"""}
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

    ### TBD ####
    self.population = None # panda Dataset container containing the population at the beginning of each generation iteration

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
                            from the process of natural selection, and like others in the large class of the evolutionary algorithms, it utilizes genetic operations such as selection, crossover, and mutations to avoid being stuck in local minima and hence facilitates
                            finding the global minima. More information can be found in:
                            Holland, John H. "Genetic algorithms." Scientific american 267.1 (1992): 66-73."""

    # GA Params
    GAparams = InputData.parameterInputFactory('GAparams', strictMode=True,
        printPriority=108,
        descr=r""" Genetic Algorithm Parameters.""")
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
        descr=r"""A node containing the criterion based on which the parents are selected. This can be
                  a. a fitness proportionate selection such as Roulette Wheer, Stochastic Universal Sampling,
                  b. Tournament,
                  c. Rank, or
                  d. Random selection""")
    GAparams.addSub(parentSelection)

    # Reproduction
    reproduction = InputData.parameterInputFactory('reproduction', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the reproduction methods.
                  This accepts subnodes that specifies the types of crossover and mutation.""")
    reproduction.addParam("nParents", InputTypes.IntegerType, True)
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
    crossover.addParam("type", InputTypes.StringType, True)
    crossoverPoint = InputData.parameterInputFactory('points', strictMode=True,
        contentType=InputTypes.IntegerListType,
        printPriority=108,
        descr=r""" point/gene(s) at which crossover will occur.""")
    crossover.addSub(crossoverPoint)
    crossoverProbability = InputData.parameterInputFactory('crossoverProb', strictMode=True,
        contentType=InputTypes.FloatType,
        printPriority=108,
        descr=r""" The probability governing the crossover occurance, i.e., the probability that if exceeded crossover will ocur.""")
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
    mutation.addParam("type", InputTypes.StringType, True)
    mutationLocs = InputData.parameterInputFactory('locs', strictMode=True,
        contentType=InputTypes.IntegerListType,
        printPriority=108,
        descr=r""" locations at which mutation will occur.""")
    mutation.addSub(mutationLocs)
    mutationProbability = InputData.parameterInputFactory('mutationProb', strictMode=True,
        contentType=InputTypes.FloatType,
        printPriority=108,
        descr=r""" The probability governing the mutation occurance, i.e., the probability that if exceeded mutation will ocur.""")
    mutation.addSub(mutationProbability)
    reproduction.addSub(mutation)
    GAparams.addSub(reproduction)

    # Survivor Selection
    survivorSelection = InputData.parameterInputFactory('survivorSelection', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented servivor selection mechanisms.
                  This includes: a.    ageBased, or
                                 b.    fitnessBased.""")
    GAparams.addSub(survivorSelection)

    # Fitness
    fitness = InputData.parameterInputFactory('fitness', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented fitness functions.
                  This includes: a.    invLinear.""")
    fitness.addParam("type", InputTypes.StringType, True)
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
    GAparamsNode = paramInput.findFirst('GAparams')
    # populationSize
    populationSizeNode = GAparamsNode.findFirst('populationSize')
    self._populationSize = populationSizeNode.value
    # parent selection
    parentSelectionNode = GAparamsNode.findFirst('parentSelection')
    self._parentSelectionType = parentSelectionNode.value
    self._parentSelectionInstance = parentSelectionReturnInstance(self,name = parentSelectionNode.value)
    # reproduction node
    reproductionNode = GAparamsNode.findFirst('reproduction')
    self._nParents = reproductionNode.parameterValues['nParents']
    # crossover node
    crossoverNode = reproductionNode.findFirst('crossover')
    self._crossoverType = crossoverNode.parameterValues['type']
    self._crossoverPoints = crossoverNode.findFirst('points').value
    self._crossoverProb = crossoverNode.findFirst('crossoverProb').value
    self._crossoverInstance = crossoversReturnInstance(self,name = self._crossoverType)
    # mutation node
    mutationNode = reproductionNode.findFirst('mutation')
    self._mutationType = mutationNode.parameterValues['type']
    self._mutationlocs = mutationNode.findFirst('locs').value
    self._mutationProb = mutationNode.findFirst('mutationProb').value
    self._mutationInstance = mutatorsReturnInstance(self,name = self._mutationType)
    # Survivor selection
    survivorSelectionNode = GAparamsNode.findFirst('survivorSelection')
    self._survivorSelectionType = survivorSelectionNode.value
    self._survivorSelectionInstance = survivorSelectionReturnInstance(self,name = self._survivorSelectionType)
    # Fitness
    fitnessNode = GAparamsNode.findFirst('fitness')
    self._fitnessType = fitnessNode.parameterValues['type']
    self._objCoeff = fitnessNode.findFirst('a').value
    self._penaltyCoeff = fitnessNode.findFirst('b').value
    self._fitnessInstance = fitnessReturnInstance(self,name = self._fitnessType)
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
    # same point is ALWAYS a criterion
    self._convergenceCriteria['samePoint'] = -1 # For simulated Annealing samePoint convergence
                                                # should not be one of the stopping criteria
    # set persistence to 1 if not set
    if self._requiredPersistence is None:
      self.raiseADebug('No persistence given; setting to 1.')
      self._requiredPersistence = 1

    #defaults

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)

    self.info = {}
    for var in self.toBeSampled:
      self.info[var+'_Age'] = None

    for traj, init in enumerate(self._initialValues):
      self._submitRun(init,traj,self.getIteration(traj))

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
  # abstract methods:
  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      This is called by localFinalizeActualSampling, and hence should contain the main skeleton.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization
      @ In, optVal, float, value of objective variable (corrected for min/max)
      @ Out, None
    """
    ## TODO the whole skeleton should be here, this should be calling all classes and _private methods.
    traj = info['traj']
    info['optVal'] = rlz[self._objectiveVar]
    self.incrementIteration(traj)
    info['step'] = self.counter

    # separate population from model evaluations
    # This part is just a dumb emulation of what should be passed by the job handeler batch
    # This part will totally be removed later.
    # population = np.zeros((self._populationSize,len(self.toBeSampled)))
    # obj = np.zeros((self._populationSize))
    fitness = np.zeros((self._populationSize))
    # For now I will assume
    # fitnesses = rlz[self._objectiveVar]*np.random.random(self._populationSize) # All np.random should be replaced with randomUtils.random etc.
    # for var in self.toBeSampled:
    #   chromosome = rlz.copy()
    # chromosome.pop(self._objectiveVar)
    # chromosome = list(chromosome.values())
    # for i in range(self._populationSize):
    #   population[i] = np.random.choice(chromosome,size=len(self.toBeSampled),replace=False)
    # obj = rlz[self._objectiveVar] * randomUtils.random(dim=10,samples=1)

    # model is generating [y1,..,yL] = F(x1,...,xM)
    # population format [y1,..,yL,x1,...,xM,fitness]

    # 5 @ n-1: Survivor Selection from previous iteration (children+parents merging from previous generation)

    # 5.1 @ n-1: fitnessCalculation(rlz)
    # perform fitness calculation for newly obtained children (rlz)
    # childrenCont = self.__fitnessCalculationHandler(rlz,params=paramsDict)
    for i in range(self._populationSize):
      fitness[i] = self._fitnessInstance(rlz,objVar = self._objectiveVar,a=self._objCoeff,b=self._penaltyCoeff,penalty = None)

    # 5.2@ n-1: Survivor selection(rlz)
    # update population container given obtained children
    # self.population = self.__replacementCalculationHandler(parents=self.population,children=childrenCont,params=paramsDict)
    if self.counter > 1:
      # right now these are lists, but this should be changed to xarrays when the realization is ready as an xarray dataset
      population,Fitness,Age = self._survivorSelectionInstance(rlz)
      # This will be added once the rlz is treated as a xarray DataSet
      # for var in self.toBeSampled:
        # self.info[var+'_Age'] = Age[var]

    # 1 @ n: Parent selection from population
    # pair parents together by indexes
    # parentSet = self.__selectionCalculationHandler(parents=self.population,params=paramsDict)
    parents = np.zeros((self._nParents,len(self.toBeSampled)))
    for i in range(self._nParents):
      ind, parents[i] = self._parentSelectionInstance(population=population,fitness=fitness)
      population = np.delete(population, ind, axis=0)

    # 2 @ n: Crossover from set of parents
    # create childrenCoordinates (x1,...,xM)
    # self.childrenCoordinates = self.__crossoverCalculationHandler(parentSet=parentSet,population=self.population,params=paramsDict)
    children = self._crossoverInstance(parents=parents,crossoverProb=self._crossoverProb,points=self._crossoverPoints)

    # 3 @ n: Mutation
    # perform random directly on childrenCoordinates
    # self.__mutationCalculationHandler(children=self.childrenCoordinates,params=paramsDict)
    for i in range(np.shape(children)[0]):
      children[i] = self._mutationInstance(chromosome=children[i],locs = self._mutationlocs, mutationProb=self._mutationProb)
    ## TODO WHAT IF AFTER CROSSOVER AND/OR MUTATION OUR CHROMOSOME NO LONGER SATISFIES THE WITHOUT REPLACEMENT CONSTRAINT

    # 4 @ n: Submit children batch
    # submit children coordinates (x1,...,xm), i.e., self.childrenCoordinates
    # --> how should this be handled? By initialize?


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
    # NOTE: explicit constraints have been checked before this!
    self.raiseADebug('Adding run to queue: {} | {}'.format(self.denormalizeData(point), info))
    self._submissionQueue.append((point, info))
  # END queuing Runs
  # * * * * * * * * * * * * * * * *

  def checkConvergence(self, traj):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory to consider
      @ Out, None
    """
    pass

  def _checkAcceptability(self, traj, opt, optVal):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
    """
    pass

  def _updateConvergence(self, traj, rlz, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, acceptable, str, condition of point
      @ Out, converged, bool, True if converged on ANY criteria
    """
    pass

  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """
  pass

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    pass

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
    """
  pass

  def _applyFunctionalConstraints(self, suggested, previous):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    pass
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
    toAdd = {'PopulationAge': self.popAge,
                }

    for var in self.toBeSampled:
      toAdd[var+'_Age'] = self.info[var+'_Age']

    for var, val in self.constants.items():
      toAdd[var] = val

    toAdd = dict((key, np.atleast_1d(val)) for key, val in toAdd.items())
    for key, val in self._convergenceInfo[traj].items():
      toAdd['conv_{}'.format(key)] = bool(val)
    return toAdd

  # def _updateSolutionExport(self, traj, rlz, acceptable):
  #   """
  #     Stores information to the solution export.
  #     @ In, traj, int, trajectory which should be written
  #     @ In, rlz, dict, collected point
  #     @ In, acceptable, bool, acceptability of opt point
  #     @ Out, None
  #   """
  #   pass

  def __fitnessCalculationHandler(self,children,params):
    # children is a Pandas dataFrame containing N realization of [y1,..,yL,x1,...,xM]
    if params['fitnessType'] == 'fitnessType1':
      pass
      # perform fitness calculation
      # add fitness variable to children dataFrame: [y1,..,yL,x1,...,xM,fitness]
      # children = fitnessType1Calculation(rlz)
    else:
      pass
      # other methods ...
    return children

  def __replacementCalculationHandler(self,parents,children,params):
    # parents and children are two Pandas dataFrame containing realization of [y1,..,yL,x1,...,xM,fitness]
    if params['replacementType'] == 'generational':
      pass
      # the following method remove the parents and leave the children
      # i.e., newPopulation <-- children
      # newPopulation = generationalReplacement(children = self.children)
    else:
      pass
      # other methods ...
      # e.g., newPopulation = mix of parents and children
      #newPopulation = otherReplacement(parents,children,paramsDict)
    return

  def __selectionCalculationHandler(self,parents,params):
    # create a list of pairs of parents: a list of list containing two (or more) parents indexes (e.g., [[2,5],[6,3],...])
    if params['selectionType'] == 'rouletteWheel':
      pass
      # parentSet = stdRouletteSelection(population=parents,params={})
    else:
      pass
      # other methods ...
      # parentSet = otherSelection(population=parents,params={})
    return #parentSet

  def __crossoverCalculationHandler(self,parentSet,population,params):
    if params['crossoverType'] == 'onePointCrossover':
      pass
      # create childrenCoordinates: a panda dataframe
      # childrenCoordinates = onePointCrossover(parents=parentSet,params={})
    elif params['twoPointsCrossover'] == 'twoPointsCrossover':
      pass
      # create childrenCoordinates: a panda dataframe
      # childrenCoordinates = twoPointsCrossover(parents=parentSet,params={})
    else:
      pass
      # other methods ...
      # childrenCoordinates = otherCrossover(parentSet,params={})
    return #childrenCoordinates

  def mutationCalculationHandler(self,children,params):
    # this method does not return anything
    # It simply acts on childrenCoordinates directly
    if params['mutationType'] =='swapMutator':
      pass
      #  mutation(childrenCoordinates, params={})
    elif params['mutationType'] =='scrambleMutator':
      pass
    else:
      pass
      # other methods ...
