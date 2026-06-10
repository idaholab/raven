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
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi, Junyung Kim,
    Joshua Cogliati
  References
    ----------
       [1] Holland, John H. "Genetic algorithms." Scientific American 267.1 (1992): 66-73.
       [2] Z. Michalewicz, "Genetic Algorithms. + Data Structures. = Evolution Programs," Third, Revised and Extended Edition, Springer (1996).
       [3] Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
       [4] Deb, Kalyanmoy. "An efficient constraint handling method for genetic algorithms." Computer methods in applied mechanics and engineering 186.2-4 (2000): 311-338.
                                                                +--------------------------+
                                                                |     AdaptiveSampler      |
                                                                |--------------------------|
                                                                |                          |
                                                                +--------------------------+
                                                                               .
                                                                              /_\
                                                                               |
                                                                               |
                                                                               |
                                                                               |
                                                                               |
                                                                +--------------------------------+
                                                                |           Optimizer            |
                                                                |--------------------------------|
                                                                | _activeTraj                    |
                                                                | _cancelledTraj                 |
                                                                | _constraintFunctions           |
                                                                | _convergedTraj                 |
                                                                | _impConstraintFunctions        |
                                                                | _initSampler                   |
                                                                | _initialValues                 |
                                                                | _initialValuesFromInput        |
                                                                | _minMax                        |
                                                                | _numRepeatSamples              |
                                                                | _objectiveVar                  |
                                                                | _requireSolnExport             |
                                                                | _seed                          |
                                                                | _trajCounter                   |
                                                                | _variableBounds                |
                                                                | assemblerDict                  |
                                                                | metadataKeys                   |
                                                                | optAssemblerList               |
                                                                |--------------------------------|
                                                                | __init__                       |
                                                                | _addTrackingInfo               |
                                                                | _closeTrajectory               |
                                                                | _collectOptPoint               |
                                                                | _initializeInitSampler         |
                                                                | _localGenerateAssembler        |
                                                                | _localWhatDoINeed              |
                                                                | _updateSolutionExport          |
                                                                | amIreadyToProvideAnInput       |
                                                                | checkConvergence               |
                                                                | denormalizeData                |
                                                                | denormalizeVariable            |
                                                                | flush                          |
                                                                | getInputSpecification          |
                                                                | handleInput                    |
                                                                | initialize                     |
                                                                | initializeTrajectory           |
                                                                | localInputAndChecks            |
                                                                | needDenormalized               |
                                                                | normalizeData                  |
                                                                | userManualDescription          |
                                                                +--------------------------------+
                                                                                 .
                                                                                /_\
                                                                                 |
                                                                                 |
                                                                                 |
                                                                                 |
                                                                                 |
                                                                +------------------------------------+
                                                                |            RavenSampled            |
                                                                |------------------------------------|
                                                                | __stepCounter                      |
                                                                | _maxHistLen                        |
                                                                | _optPointHistory                   |
                                                                | _rerunsSinceAccept                 |
                                                                | _stepTracker                       |
                                                                | _submissionQueue                   |
                                                                | _writeSteps                        |
                                                                | batch                              |
                                                                | batchId                            |
                                                                | convFormat                         |
                                                                | inputInfo                          |
                                                                | limit                              |
                                                                | type                               |
                                                                | values                             |
                                                                |------------------------------------|
                                                                | __init__                           |
                                                                | _addToSolutionExport               |
                                                                | _applyBoundaryConstraints          |
                                                                | _applyFunctionalConstraints        |
                                                                | _cancelAssociatedJobs              |
                                                                | _checkAcceptability                |
                                                                | _checkBoundaryConstraints          |
                                                                | _checkForImprovement               |
                                                                | _checkFunctionalConstraints        |
                                                                | _checkImpFunctionalConstraints     |
                                                                | _closeTrajectory                   |
                                                                | _handleExplicitConstraints         |
                                                                | _handleImplicitConstraints         |
                                                                | _initializeStep                    |
                                                                | _rejectOptPoint                    |
                                                                | _resolveNewOptPoint                |
                                                                | _updateConvergence                 |
                                                                | _updatePersistence                 |
                                                                | _updateSolutionExport              |
                                                                | _useRealization                    |
                                                                | amIreadyToProvideAnInput           |
                                                                | checkConvergence                   |
                                                                | finalizeSampler                    |
                                                                | flush                              |
                                                                | getInputSpecification              |
                                                                | getIteration                       |
                                                                | getSolutionExportVariableNames     |
                                                                | handleInput                        |
                                                                | incrementIteration                 |
                                                                | initialize                         |
                                                                | initializeTrajectory               |
                                                                | localFinalizeActualSampling        |
                                                                | localGenerateInput                 |
                                                                +------------------------------------+
                                                                                 .
                                                                                /_\
                                                                                 |
                                                                                 |
                                                                                 |
                                                                                 |
                                                                                 |
                                                                +------------------------------------+
                                                                |          GeneticAlgorithm          |
                                                                |------------------------------------|
                                                                | _acceptHistory                     |
                                                                | _acceptRerun                       |
                                                                | _canHandleMultiObjective           |
                                                                | _convergenceCriteria               |
                                                                | _convergenceInfo                   |
                                                                | _crossoverInstance                 |
                                                                | _crossoverPoints                   |
                                                                | _crossoverProb                     |
                                                                | _crossoverType                     |
                                                                | _expConstr                         |
                                                                | _fitnessInstance                   |
                                                                | _fitnessType                       |
                                                                | _impConstr                         |
                                                                | _kSelection                        |
                                                                | _mutationInstance                  |
                                                                | _mutationLocs                      |
                                                                | _mutationProb                      |
                                                                | _mutationType                      |
                                                                | _nChildren                         |
                                                                | _nParents                          |
                                                                | _numOfConst                        |
                                                                | _objCoeff                          |
                                                                | _objectiveVar                      |
                                                                | _parentSelection                   |
                                                                | _parentSelectionInstance           |
                                                                | _parentSelectionType               |
                                                                | _penaltyCoeff                      |
                                                                | _populationSize                    |
                                                                | _repairInstance                    |
                                                                | _requiredPersistence               |
                                                                | _stepTracker                       |
                                                                | _submissionQueue                   |
                                                                | _survivorSelectionInstance         |
                                                                | _survivorSelectionType             |
                                                                | ahd                                |
                                                                | ahdp                               |
                                                                | batch                              |
                                                                | batchId                            |
                                                                | bestFitVal                        |
                                                                | bestPoint                          |
                                                                | constraintVals                       |
                                                                | convergenceOptions                 |
                                                                | crowdingDistance                   |
                                                                | fitness                            |
                                                                | hdsm                               |
                                                                | multiBestCD                        |
                                                                | multiBestConstraintVals                |
                                                                | multiBestFitVals                   |
                                                                | multiBestMinObjVals                 |
                                                                | multiBestPoint                     |
                                                                | multiBestRank                      |
                                                                | minObjVals                       |
                                                                | popAge                             |
                                                                | population                         |
                                                                | rank                               |
                                                                |------------------------------------|
                                                                | __init__                           |
                                                                | _addToSolutionExport               |
                                                                | _applyFunctionalConstraints        |
                                                                | _checkAcceptability                |
                                                                | _checkConvAHD                      |
                                                                | _checkConvAHDp                     |
                                                                | _checkConvHDSM                     |
                                                                | _checkConvObjective                |
                                                                | _checkForImprovement               |
                                                                | _checkFunctionalConstraints        |
                                                                | _checkImpFunctionalConstraints     |
                                                                | _collectOptPoint                   |
                                                                | _collectOptPointMulti              |
                                                                | _formatSolutionExportVariableNames |
                                                                | _handleExplicitConstraints         |
                                                                | _handleImplicitConstraints         |
                                                                | _rejectOptPoint                    |
                                                                | _resolveNewGeneration              |
                                                                | _submitRun                         |
                                                                | _updateConvergence                 |
                                                                | _updatePersistence                 |
                                                                | _useRealization                    |
                                                                | checkConvergence                   |
                                                                | flush                              |
                                                                | getInputSpecification              |
                                                                | getSolutionExportVariableNames     |
                                                                | handleInput                        |
                                                                | initialize                         |
                                                                | initializeTrajectory               |
                                                                | multiObjectiveConstraintHandling   |
                                                                | needDenormalized                   |
                                                                | singleObjectiveConstraintHandling  |
                                                                +------------------------------------+
"""
# External Modules----------------------------------------------------------------------------------
from collections import deque, defaultdict
import numpy as np
from scipy.special import comb
import xarray as xr
from copy import deepcopy
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..utils import mathUtils, InputData, InputTypes, frontUtils
from ..utils.gaUtils import dataArrayToDict, datasetToDataArray
from .RavenSampled import RavenSampled
from .parentSelectors.parentSelectors import returnInstance as parentSelectionReturnInstance
from .crossOverOperators.crossovers import returnInstance as crossoversReturnInstance
from .mutators.mutators import returnInstance as mutatorsReturnInstance
from .survivorSelectors.survivorSelectors import returnInstance as survivorSelectionReturnInstance
from .survivorSelection import survivorSelection
from .constraintHandling.constraintHandling import constraintHandling
from .fitness.fitness import returnInstance as fitnessReturnInstance
from .repairOperators.repair import returnInstance as repairReturnInstance
# Internal Modules End------------------------------------------------------------------------------

class GeneticAlgorithm(RavenSampled):
  """
    This class performs Genetic Algorithm optimization ...

    The realization used for sampling contains the genes, the
    objectives and other variables. The objectives are changed to all
    be minimization problems internally by multipilying by
    self._objMult[varname].  All the variables in the realization are
    internally normalized to improve the algorithm (methods
    self.normalizeData and self.denormalizeData are used for this)

    The objective variable names are in self._objectiveVar and the
    gene (or chromosome) names are the keys in self.toBeSampled )

  """
  convergenceOptions = {'objective': r""" provides the desired value or values for the convergence criterion of the objective function
                        ($\epsilon^{obj}$). In essence this is solving the inverse problem of finding the design variable
                         at a given objective value, i.e., convergence is reached when: $$ Objective = \epsilon^{obj}$$
                         For multiobjective problems, a comma separated list of objective
                         values should be provided instead of a single value.
                        \default{1e-6}, if no criteria specified.""",
                        'AHDp': r""" provides the desired value for the Average Hausdorff Distance between populations""",
                        'AHD': r""" provides the desired value for the Hausdorff Distance between populations""",
                        'HDSM': r""" provides the desired value for the Hausdorff Distance Similarity Measure between populations.
                                     This convergence criterion is based on a normalized
                                     similarity metric that can be summurized as the normalized Hausdorff distance
                                     (with respect the domain of to population/iterations). The metric is normalized between 0 and 1,
                                     which implies that values closer to 1.0 represents a tighter convergence criterion."""}
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
    self.pop = None                                              # current survivor population P(t)
    self.popAges = None                                          # ages of current survivor population
    self.popFitVals = None                                       # fitness values of current survivor population
    self.popRanks = None                                         # Pareto rank for current survivor population
    self.popCrowdingDist = None                                  # crowding distance for current survivor population
    self.popMinObjVals = None                                    # objective values in RAVEN minimization space
    self.popConstraintVals = None                                # constraint values of current survivor population
    self.popAgesArray = None                                     # ages as an array for reporting
    self.prevPopInputs = None                                    # previous survivor population for convergence checks
    self.population = None                                       # panda Dataset container containing the population at the beginning of each generation iteration
    self.popAge = None                                           # population age
    self.fitVals = None                                          # population fitness
    self.rank = None                                             # population rank (for Multi-objective optimization only)
    self.constraintVals = None                                     # calculated contraints value
    self.crowdingDistance = None                                 # population crowding distance (for Multi-objective optimization only)
    self.ahdp = np.NaN                                           # p-Average Hausdorff Distance between populations
    self.ahd  = np.NaN                                           # Hausdorff Distance between populations
    self.hdsm = np.NaN                                           # Hausdorff Distance Similarity metric between populations
    self.bestPoint = None                                        # the best solution (chromosome) found among population in a specific batchId
    self.bestFitVal = None                                      # fitness value of the best solution found
    self._bestSnapshot = None                                    # cached survivor snapshot for final export alignment
    self.multiBestPoint = {}                                     # the best solutions (chromosomes) found among population in a specific batchId
    self.multiBestFitVals = {}                                   # fitness values of the best solutions found
    self.multiBestMinObjVals = {}                                 # objective values of the best solutions found
    self.multiBestConstraintVals = {}                                # constraint values of the best solutions found
    self.multiBestRank = {}                                      # rank values of the best solutions found
    self.multiBestCD = {}                                        # crowding distance (CD) values of the best solutions found
    self.minObjVals = None                                     # objective values of solutions
    self._populationSize = None                                  # number of population size
    self._parentSelectionType = None                             # type of the parent selection process chosen
    self._parentSelectionInstance = None                         # instance of the parent selection process chosen
    self._nParents = None                                        # number of parents
    self._kSelection = 3                                         # number of chromosomes selected for tournament selection
    self._nChildren = None                                       # number of children
    self._crossoverType = None                                   # type of the crossover process chosen
    self._crossoverPoints = None                                 # point where crossover process will happen
    self._crossoverProb = None                                   # probability of crossover process will happen
    self._crossoverInstance = None                               # instance of the crossover process chosen
    self._mutationType = None                                    # type of the mutation process chosen
    self._mutationLocs = None                                    # point where mutation process will happen
    self._mutationProb = None                                    # probability of mutation process will happen
    self._mutationInstance = None                                # instance of the mutation process chosen
    self._survivorSelectionType = None                           # type of the survivor selection process chosen
    self._survivorSelectionInstance = None                       # instance of the survivor selection process chosen
    self._fitnessType = None                                     # type of the fitness calculation chosen
    self._objCoeff = None                                        # weight coefficients of objectives for fitness calculation
    self._objectiveVar = None                                    # objective variable names
    self._penaltyCoeff = None                                    # weight coefficients corresponding to constraints and objectives for fitness calculation
    self._fitnessInstance = None                                 # instance of fitness
    self._repairInstance = None                                  # instance of repair
    self._canHandleMultiObjective = True                         # boolean indicator whether optimization is a sinlge-objective problem or a multi-objective problem
    self._normalizeFitness = False

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
    objective = specs.popSub('objective')
    objective.description = r"""Name of the objective variable(s) (or ``objective function'') that should be optimized
        (minimized or maximized). It can be a single string or a list of strings if it is a multi-objective problem.
        Note that only genetic algorithm supports multi-objective."""
    specs.addSub(objective)
    implicitConstraint = specs.popSub('ImplicitConstraint')
    implicitConstraint.description = r"""name of \xmlNode{Function} which contains implicit constraints of the Model. From a practical
              point of view, this XML node must contain the name of a function defined in the \xmlNode{Functions}
              block (see Section~\ref{sec:functions}). This external function must contain a method called
              ``implicitConstraint'', which returns a float value which should
        be less than zero when the constraint is violated."""
    specs.addSub(implicitConstraint)
    specs.description = r"""The \xmlNode{GeneticAlgorithm} is a metaheuristic optimization technique inspired by the principles
                            of natural selection and genetics. Introduced by John Holland in the 1960s, GA mimics the process of
                            biological evolution to solve complex optimization and search problems. They operate by maintaining a population of
                            potential solutions represented as arrays of fixed length variables (genes), and each such array is called a chromosome.
                            These solutions undergo iterative refinement through processes such as mutation, crossover, and survivor selection. Mutation involves randomly altering certain genes within
                            individual solutions, introducing diversity into the population and enabling exploration of new regions in the solution space.
                            Crossover, on the other hand, mimics genetic recombination by exchanging genetic material between two parent solutions to create
                            offspring with combined traits. Survivor selection determines which solutions will advance to the next generation based on
                            their fitness—how well they perform in solving the problem at hand. Solutions with higher fitness scores are more likely to
                            survive and reproduce, passing their genetic material to subsequent generations. This iterative process continues
                            until a stopping criterion is met, typically when a satisfactory solution is found or after a predetermined number of generations.
                            More information can be found in:\\\\

                            Holland, John H. ``Genetic algorithms.'' Scientific American 267.1 (1992): 66-73.\\\\

                            Non-dominated Sorting Genetic Algorithm II (NSGA-II) is a variant of GAs designed for multiobjective optimization problems.
                            NSGA-II extends traditional GAs by incorporating a ranking-based approach and crowding distance estimation to maintain a diverse set of
                            non-dominated (Pareto-optimal) solutions. This enables NSGA-II to efficiently explore trade-offs between conflicting objectives,
                            providing decision-makers with a comprehensive view of the problem's solution space. More information about NSGA-II can be found in:\\\\

                            Deb, Kalyanmoy, et al. ``A fast and elitist multiobjective genetic algorithm: NSGA-II.'' IEEE transactions on evolutionary computation 6.2 (2002): 182-197.\\\\

                            Internally, GA and NSGA-II store \texttt{minObjVals} as objective values transformed into RAVEN's minimization convention.
                            Objectives declared as \texttt{min} are unchanged, while objectives declared as \texttt{max} are multiplied by -1 before Pareto ranking,
                            crowding-distance calculation, convergence checks, and fitness calculations. User-facing objective values should be interpreted
                            through the requested minimization or maximization convention when exported.\\
                            The implementation keeps \texttt{minObjVals}, \texttt{externalObjVals}, and \texttt{fitVals} separate: \texttt{minObjVals} drives algorithmic ranking, crowding distance, convergence, and fitness calculation; \texttt{externalObjVals} is used only when converting back to user-facing signs for exports or implicit constraints; and \texttt{fitVals} is the scalar or per-objective fitness container used by parent and survivor selection.\\

                            GA in RAVEN supports for both single and multi-objective optimization problem."""

    # GA Params
    GAparams = InputData.parameterInputFactory('GAparams', strictMode=True,
        printPriority=108,
        descr=r""" """)
    # Population Size
    populationSize = InputData.parameterInputFactory('populationSize', strictMode=True,
        contentType=InputTypes.IntegerType,
        printPriority=108,
        descr=r"""The number of chromosomes in each population.""")
    GAparams.addSub(populationSize)

    #NOTE An indicator saying whather GA will handle constraint hardly or softly will be upgraded later @JunyungKim
    # Parent Selection
    parentSelection = InputData.parameterInputFactory('parentSelection', strictMode=True,
        contentType=InputTypes.makeEnumType('parentSelection','parentSelectionType',['rouletteWheel','tournamentSelection','rankSelection']),
        printPriority=108,
        descr=r"""A node containing the criterion based on which the parents are selected. This can be a fitness proportional selection for all methods.
                  The number of parents (i.e., nParents) is computed such that the population size is kept constant. \\\\
                  $nParents = ceil(\frac{1 + \sqrt{1+4*popSize}}{2})$. \\\\
                  The number of children (i.e., nChildren) is computed by \\\\
                  $nChildren = 2 \times {nParents \choose 2} = nParents \times (nParents-1) = popSize$ \\\\
                  This will result in a popSize a little larger than the initial one, and the excessive children will be later thrawn away and only the first popSize child will be kept. \\\\
                  You can choose three options for parentSelection:
                      \begin{itemize}
                          \item \textit{rouletteWheel} - It assigns probabilities to chromosomes based on their fitness,
                          allowing for selection proportionate to their likelihood of being chosen for reproduction.
                          \item \textit{tournamentSelection} - Chromosomes are randomly chosen from the population to compete in a tournament,
                          and the fittest individual among them is selected for reproduction.
                          \item \textit{rankSelection} - Chromosomes with higher fitness values are selected.
                      \end{itemize}
                  """)
    GAparams.addSub(parentSelection)

    # Reproduction
    reproduction = InputData.parameterInputFactory('reproduction', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the reproduction methods. This accepts subnodes that specifies the types of crossover and mutation. """)
    # 0.  k-selectionNumber of Parents
    kSelection = InputData.parameterInputFactory('kSelection', strictMode=True,
        contentType=InputTypes.IntegerType,
        printPriority=108,
        descr=r"""Number of chromosome selected for tournament selection""")
    reproduction.addSub(kSelection)
    # 1.  Crossover
    crossover = InputData.parameterInputFactory('crossover', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented crossover mechanisms. You can choose one of the crossover options listed below:
                  \begin{itemize}
                    \item \textit{onePointCrossover} - It selects a random crossover point along the chromosome of parent individuals and swapping the genetic material beyond that point to create offspring.
                    \item \textit{twoPointsCrossover} - It selects two random crossover points along the chromosome of parent individuals and swapping the genetic material beyond that point to create offspring.
                    \item \textit{uniformCrossover} - It randomly selects genes from two parent chromosomes with equal probability, creating offspring by exchanging genes at corresponding positions.
                    \item \textit{sbxCrossover} - Simulated Binary Crossover (Deb \& Agrawal, 1995) for real-valued variables; produces offspring distributed around the parents within the variable bounds, controlled by a distribution index. Recommended for continuous multi-objective problems.
                  \end{itemize}""")
    crossover.addParam("type",
                       InputTypes.makeEnumType('crossover','crossoverType',['onePointCrossover','twoPointsCrossover','uniformCrossover','sbxCrossover']),
                       True,
                       descr="type of crossover operation to be used. See the list of options above.")
    crossoverPoint = InputData.parameterInputFactory('points', strictMode=True,
        contentType=InputTypes.IntegerListType,
        printPriority=108,
        descr=r""" point/gene(s) at which crossover will occur.""")
    crossover.addSub(crossoverPoint)
    crossoverProbability = InputData.parameterInputFactory('crossoverProb', strictMode=True,
                                                           contentType=InputTypes.FloatOrStringType,
                                                           printPriority=108,
                                                           descr=r""" The probability governing the crossover step, i.e., the probability that if exceeded crossover will occur.""")
    crossoverProbability.addParam("type", InputTypes.makeEnumType('crossoverProbability','crossoverProbabilityType',['static','adaptive']), False,
                       descr="type of crossover operation to be used (e.g., static,adaptive)")
    crossover.addSub(crossoverProbability)
    reproduction.addSub(crossover)
    # 2.  Mutation
    mutation = InputData.parameterInputFactory('mutation', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented mutation mechanisms. You can choose one of the mutation options listed below:
                \begin{itemize}
                  \item \textit{swapMutator} - It randomly selects two genes within an chromosome and swaps their positions.
                  \item \textit{scrambleMutator} - It randomly selects a subset of genes within an chromosome and shuffles their positions.
                  \item \textit{inversionMutator} - It selects a contiguous subset of genes within an chromosome and reverses their order.
                  \item \textit{bitFlipMutator} - It randomly selects genes within an chromosome and flips their values.
                  \item \textit{randomMutator} - It randomly selects a gene within an chromosome and mutates the gene.
                  \item \textit{polynomialMutator} - Polynomial mutation (Deb \& Goyal, 1996) for real-valued variables; perturbs a gene by a bounded, polynomial-distributed step controlled by a distribution index. Recommended for continuous multi-objective problems, paired with sbxCrossover.
                \end{itemize} """)
    mutation.addParam("type",
                      InputTypes.makeEnumType('mutation','mutationType',['swapMutator','scrambleMutator','inversionMutator','randomMutator','polynomialMutator']),
                      True,
                      descr="type of mutation operation to be used. See the list of options above.")
    mutationLocs = InputData.parameterInputFactory('locs', strictMode=True,
        contentType=InputTypes.IntegerListType,
        printPriority=108,
        descr=r""" locations at which mutation will occur.""")
    mutation.addSub(mutationLocs)
    mutationProbability = InputData.parameterInputFactory('mutationProb', strictMode=True,
        contentType=InputTypes.FloatOrStringType,
        printPriority=108,
        descr=r""" The probability governing the mutation step, i.e., the probability that if exceeded mutation will occur.""")
    mutationProbability.addParam("type", InputTypes.makeEnumType('mutationProbability','mutationProbabilityType',['static','adaptive']), False,
                       descr="type of mutation probability operation to be used (e.g., static, adaptive)")
    mutation.addSub(mutationProbability)
    reproduction.addSub(mutation)
    GAparams.addSub(reproduction)

    # Survivor Selection
    survivorSelection = InputData.parameterInputFactory('survivorSelection', strictMode=True,
        contentType=InputTypes.makeEnumType('survivorSelection','survivalSelectionType',['fitnessBased','ageBased','rankNcrowdingBased']),
        printPriority=108,
        descr=r"""a subnode containing the implemented survivor selection mechanisms. You can choose one of the survivor selection options listed below:
                  \begin{itemize}
                    \item \textit{fitnessBased} - Individuals with higher fitness scores are more likely to be selected to survive and
                    proceed to the next generation. It suppoort only single-objective optimization problem.
                    \item \textit{ageBased} - Individuals are selected for survival based on their age or generation, with older individuals being prioritized
                    for retention. It suppoort only single-objective optimization problem.
                    \item \textit{rankNcrowdingBased} - Individuals with low rank and crowding distance are more likely to be selected to survive and
                    proceed to the next generation. It suppoort only multi-objective optimization problem.
                  \end{itemize}""")
    GAparams.addSub(survivorSelection)

    # Fitness
    fitness = InputData.parameterInputFactory('fitness', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented fitness functions.""")
    fitness.addParam("type", InputTypes.makeEnumType('fitness','fitnessType',['invLinear','feasibleFirst','logistic']),
                     True,
                     descr=r"""You can choose one of the fitness options listed below:
                  \begin{itemize}
                        \item \textit{invLinear} - It assigns fitness values inversely proportional to the individual's objective function values,
                        prioritizing solutions with lower objective function values (i.e., minimization) for selection and reproduction. It suppoort only single-objective optimization problem.\\\\
                        $fitness = -a \times obj - b \times \sum_{j=1}^{nConstraint} max(0,-penalty_{j}) $\\
                        where j represents an index of objects
                        \\

                        \item \textit{logistic} - It applies a logistic function to transform raw objective function values into fitness scores.  It suppoort only single-objective optimization problem.\\\\
                        $fitness = \frac{1}{1+e^{a\times(obj-b)}}$\\
                        \item \textit{feasibleFirst} - It prioritizes solutions that meet constraints by assigning higher fitness scores to feasible solutions,

                        encouraging the evolution of individuals that satisfy the problem's constraints.  It suppoort single-and multi-objective optimization problem.\\\\
                        $fitness = \left\{\begin{matrix} -obj & g_{j}(x)\geq 0 \; \forall j \\ -obj_{worst}- \Sigma_{j=1}^{J}<g_j(x)> & otherwise \\ \end{matrix}\right\}$\\
                  \end{itemize} """)
    objCoeff = InputData.parameterInputFactory('a', strictMode=True,
        contentType=InputTypes.FloatListType,
        printPriority=108,
        descr=r""" a: The weight of objective function(s). \default{list of ones}""")
    fitness.addSub(objCoeff)
    penaltyCoeff = InputData.parameterInputFactory('b', strictMode=True,
        contentType=InputTypes.FloatListType,
        printPriority=108,
        descr=r""" b: The weight of constraint penalty. \default{list of ones}""")
    fitness.addSub(penaltyCoeff)
    scale = InputData.parameterInputFactory('scale', strictMode=False,
        contentType=InputTypes.FloatListType,
        printPriority=108,
        descr=r""" scale: in case of logistic fitness, this is the multiplier of the onjective(s). \default{list of ones}""")
    fitness.addSub(scale)
    shift = InputData.parameterInputFactory('shift', strictMode=False,
        contentType=InputTypes.FloatListType,
        printPriority=108,
        descr=r""" shift: in case of logistic fitness, this is the shift in the exponential function for the onjective(s). \default{list of zeros}""")
    fitness.addSub(shift)
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
      if name != 'objective':
        conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatType,descr=descr,printPriority=108))
      else:
        conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatListType,descr=descr,printPriority=108))

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
    new['rank'] = 'It refers to the sorting of solutions into non-dominated fronts based on their Pareto dominance relationships'
    new['CD'] = 'Crowding Distance measures the density of solutions within each front to guide the selection of diverse individuals for the next generation'
    new['fitness'] = 'fitness of the current chromosome'
    new['age'] = 'age of current chromosome'
    new['batchId'] = 'Id of the batch to whom the chromosome belongs'
    new['AHDp'] = 'p-Average Hausdorff Distance between populations'
    new['AHD'] = 'Hausdorff Distance between populations'
    new['HDSM'] = 'Hausdorff Distance Similarity Measure between populations'
    new['ConstraintEvaluation_{CONSTRAINT}'] = 'Constraint function evaluation (negative if violating and positive otherwise)'
    new['FitnessEvaluation_{OBJ}'] = 'Fitness evaluation of each objective'
    ok.update(new)

    return ok

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    RavenSampled.handleInput(self, paramInput)
    ####################################################################################
    # GAparams                                                                         #
    ####################################################################################
    gaParamsNode = paramInput.findFirst('GAparams')

    ####################################################################################
    # populationSize                                                                   #
    ####################################################################################
    populationSizeNode = gaParamsNode.findFirst('populationSize')
    self._populationSize = populationSizeNode.value

    ####################################################################################
    # parent selection node                                                            #
    ####################################################################################
    parentSelectionNode = gaParamsNode.findFirst('parentSelection')
    self._parentSelectionType = parentSelectionNode.value
    self._parentSelectionInstance = parentSelectionReturnInstance(self, name=parentSelectionNode.value)

    if self._isMultiObjective and self._parentSelectionType != 'tournamentSelection':
      self.raiseAnError(IOError, f'Currently, "tournamentSelection" in the only <parentSelection> mechanism supported by the multi-objective Genetic Algorithms.')

    ####################################################################################
    # reproduction node                                                                #
    ####################################################################################
    reproductionNode = gaParamsNode.findFirst('reproduction')
    self._nParents = int(np.ceil(1/2 + np.sqrt(1+4*self._populationSize)/2))
    self._nChildren = int(2*comb(self._nParents,2))

    ####################################################################################
    # k-Selection node                                                                #
    ####################################################################################
    if reproductionNode.findFirst('kSelection') is not None:
      self._kSelection = reproductionNode.findFirst('kSelection').value

    ####################################################################################
    # crossover node                                                                   #
    ####################################################################################
    crossoverNode = reproductionNode.findFirst('crossover')
    self._crossoverType = crossoverNode.parameterValues['type']
    if crossoverNode.findFirst('points') is None:
      self._crossoverPoints = None
    else:
      self._crossoverPoints = crossoverNode.findFirst('points').value
    self._crossoverProb = crossoverNode.findFirst('crossoverProb').value
    self._crossoverInstance = crossoversReturnInstance(self,name = self._crossoverType)

    ####################################################################################
    # mutation node                                                                    #
    ####################################################################################
    mutationNode = reproductionNode.findFirst('mutation')
    self._mutationType = mutationNode.parameterValues['type']
    if mutationNode.findFirst('locs') is None:
      self._mutationLocs = None
    else:
      self._mutationLocs = mutationNode.findFirst('locs').value
    self._mutationProb = mutationNode.findFirst('mutationProb').value
    self._mutationInstance = mutatorsReturnInstance(self,name = self._mutationType)

    ####################################################################################
    # survivor selection node                                                          #
    ####################################################################################
    survivorSelectionNode = gaParamsNode.findFirst('survivorSelection')
    self._survivorSelectionType = survivorSelectionNode.value
    self._survivorSelectionInstance = survivorSelectionReturnInstance(self,name = self._survivorSelectionType)
    if not self._isMultiObjective and self._survivorSelectionType == 'rankNcrowdingBased':
      self.raiseAnError(IOError, f'(rankNcrowdingBased) in <survivorSelection> only supports Multi-objective Optimization (i.e., number of objectives in <objective> is greater than one).')
    if self._isMultiObjective and self._survivorSelectionType != 'rankNcrowdingBased':
      self.raiseAnError(IOError, f'The only option supported in <survivorSelection> for Multi-objective Optimization is (rankNcrowdingBased).')

    ####################################################################################
    # fitness node                                                                     #
    ####################################################################################
    fitnessNode = gaParamsNode.findFirst('fitness')
    self._fitnessType = fitnessNode.parameterValues['type']
    if self._fitnessType == 'logistic':
      self._scale = fitnessNode.findFirst('scale').value
      self._shift = fitnessNode.findFirst('shift').value
    else:
      self._penaltyCoeff = fitnessNode.findFirst('b').value if fitnessNode.findFirst('b') else None
      self._objCoeff = fitnessNode.findFirst('a').value if fitnessNode.findFirst('a') else None
    ####################################################################################
    # constraint node                                                                  #
    ####################################################################################
    self._expConstr = self.assemblerObjects['Constraint'] if 'Constraint' in self.assemblerObjects else None
    self._impConstr = self.assemblerObjects['ImplicitConstraint'] if 'ImplicitConstraint' in self.assemblerObjects else None
    if self._expConstr is not None and self._impConstr is not None:
      self._numOfConst = len([ele for ele in self._expConstr if ele != 'Functions' if ele !='External']) + len([ele for ele in self._impConstr if ele != 'Functions' if ele !='External'])
    elif self._expConstr is None and self._impConstr is not None:
      self._numOfConst = len([ele for ele in self._impConstr if ele != 'Functions' if ele !='External'])
    elif self._expConstr is not None and self._impConstr is None:
      self._numOfConst = len([ele for ele in self._expConstr if ele != 'Functions' if ele !='External'])
    else:
      self._numOfConst = 0
    if (self._expConstr is not None) and (self._impConstr is not None) and (self._penaltyCoeff is not None):
      if len(self._penaltyCoeff) is not len(self._objectiveVar) * self._numOfConst:
        self.raiseAnError(IOError, f'The number of penaltyCoeff. in <b> should be identical with the number of objective in <objective> and the number of constraints (i.e., <Constraint> and <ImplicitConstraint>)')
    self._fitnessInstance = fitnessReturnInstance(self,name = self._fitnessType)
    self._repairInstance = repairReturnInstance(self,name='replacementRepair')  # currently only replacement repair is implemented.

    ####################################################################################
    # convergence criterion node                                                       #
    ####################################################################################
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

  ######################################################################################
  # Run Methods                                                                        #
  ######################################################################################

  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      This is called by localFinalizeActualSampling, and hence should contain the main skeleton.
      Proper NSGA-II flow with ranking/CD before survivor selection
      @ In, info, dict, identifying information about the realization
      @ In, rlz, xr.Dataset, new batched realizations with EVALUATED objectives
      @ Out, None
                    ┌─────────────────────────────────────────────────────────────┐
                    │  Model evaluates Q(t) objectives [EXPENSIVE - EXTERNAL]     │
                    └──────────────────────┬──────────────────────────────────────┘
                                           │
                                           ↓
                    ┌─────────────────────────────────────────────────────────────┐
                    │  _useRealization receives Q(t) with evaluated objectives    │
                    │                                                             │
                    │  ┌────────────────────────────────────────────────────────┐ │
                    │  │ PHASE 1: Process Q(t) [CHEAP - INTERNAL]               │ │
                    │  │  • Extract inputs, objectives from rlz                 │ │
                    │  │  • Compute constraints for Q(t)                        │ │
                    │  │  • Compute fitness for Q(t)                            │ │
                    │  └────────────────────────────────────────────────────────┘ │
                    │                                                             │
                    │  ┌────────────────────────────────────────────────────────┐ │
                    │  │ PHASE 2: Elitist Selection [CHEAP - INTERNAL]          │ │
                    │  │  • Combine: R(t) = P(t) ∪ Q(t)                         │ │
                    │  │  • Rank R(t) using constraint-domination               │ │
                    │  │  • Compute CD for each front in R(t)                   │ │
                    │  │  • Select best N individuals → P(t+1)                  │ │
                    │  └────────────────────────────────────────────────────────┘ │
                    │                                                             │
                    │  ┌────────────────────────────────────────────────────────┐ │
                    │  │ PHASE 3: Record & Export [CHEAP - INTERNAL]            │ │
                    │  │  • Extract Pareto front (rank 1) from P(t+1)           │ │
                    │  │  • Check convergence (objective, spread, etc.)         │ │
                    │  │  • Export P(t+1) with rank/CD to solution export       │ │
                    │  └────────────────────────────────────────────────────────┘ │
                    │                                                             │
                    │  ┌────────────────────────────────────────────────────────┐ │
                    │  │ PHASE 4: Generate Q(t+1) [CHEAP - INTERNAL]            │ │
                    │  │  • Select parents from P(t+1) using tournament         │ │
                    │  │  • Crossover → offspring                               │ │
                    │  │  • Mutation → offspring                                │ │
                    │  │  • Create Q(t+1) (decision variables only)             │ │
                    │  │  • Submit Q(t+1) to model                              │ │
                    │  └────────────────────────────────────────────────────────┘ │
                    │                                                             │
                    │  ┌────────────────────────────────────────────────────────┐ │
                    │  │ PHASE 5: Store State                                   │ │
                    │  │  • Save P(t+1) as pop for next iteration   │ │
                    │  │  • Save objectives, fitness, rank, CD                  │ │
                    │  └────────────────────────────────────────────────────────┘ │
                    └──────────────────────┬──────────────────────────────────────┘
                                           │
                                           ↓
                    ┌─────────────────────────────────────────────────────────────┐
                    │  Model evaluates Q(t+1) objectives [EXPENSIVE - EXTERNAL]   │
                    └──────────────────────┬──────────────────────────────────────┘
                                           │
                                           ↓
                                     (cycle repeats)
    """
    if self._isMultiObjective:
      self.raiseAnError(IOError, 'GeneticAlgorithm supports only single-objective optimization. '
                        'Use MultiObjectiveGeneticAlgorithm/NSGAII for multi-objective problems.')
    info['step'] = self.counter
    traj = info['traj']
    for t in self._activeTraj[1:]:
      self._closeTrajectory(t, 'cancel', 'Currently GA is single trajectory', 0)
    self.incrementIteration(traj)

    files = self.assemblerDict['Files']
    EQflag = any("EQinput" in sublist for sublist in files)
    if EQflag:
      self._EQcheckfile = files
    else:
      self._EQcheckfile = None

    # ============================================================
    # PART A: Extract and Process Offspring Q(t)
    # ============================================================

    # Step 1: Extract offspring inputs (decision variables)
    offspring = datasetToDataArray(rlz, list(self.toBeSampled))

    # Step 2: Extract offspring objectives in RAVEN minimization space. Maximization
    # objectives have already been multiplied by -1 by RavenSampled.
    offspringMinObjVals = []
    for i in range(len(self._objectiveVar)):
      offspringMinObjVals.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))

    # Step 3: Compute constraints for offspring Q(t)
    offspringConstraintVals = constraintHandling(self, info, rlz, offspring,
                                      offspringMinObjVals, multiObjective=self._isMultiObjective)

    # Step 4: Normalize if requested
    normRlz = deepcopy(rlz)
    if self._normalizeFitness:
      constrVarsList = self._constraintFunctions + self._impConstraintFunctions
      varsToNormalize = []
      for x in constrVarsList:
        varsToNormalize += x.parameterNames()
      varsToNormalize = set(varsToNormalize + self._objectiveVar)

      self.normScores = {}
      for var in varsToNormalize:
        if self._normalizeFitness == "zscore":
          self.normScores[var] = (np.mean(rlz[var].to_dataframe().values),
                                  np.std(rlz[var].to_dataframe().values))
          for i in range(len(rlz[var])):
            normRlz[var][i] = (rlz[var][i] - self.normScores[var][0]) / self.normScores[var][1]
            if np.isnan(normRlz[var][i]):
              normRlz[var][i] = 0.0

      for i in range(len(offspringConstraintVals)):
        for j in range(len(constrVarsList)):
          offspringConstraintVals[i][j] = offspringConstraintVals[i][j] / self.normScores[constrVarsList[j].parameterNames()[0]][1]
          if np.isnan(offspringConstraintVals[i][j]):
            offspringConstraintVals[i][j] = 0.0

    # Step 5: Compute fitness for offspring Q(t)
    offspringFitVals = self._fitnessInstance(normRlz,
                                               objVar=self._objectiveVar,
                                               a=self._objCoeff,
                                               b=self._penaltyCoeff,
                                               penalty=None,
                                               constraintFunction=offspringConstraintVals,
                                               constraintNum=self._numOfConst,
                                               type=self._minMax)

    if self._activeTraj:
      # ============================================================
      # PART B: Combine Populations and Perform Elitist Selection
      # ============================================================

      if self.counter > 1:
        # We have parent population P(t) from previous iteration
        # Combine P(t) ∪ Q(t) → R(t)

        if not self._isMultiObjective:
          # -------------------- SINGLE OBJECTIVE --------------------
          survivorSelection.singleObjSurvivorSelect(self, info, rlz, traj,
                                                     offspring,
                                                     offspringFitVals,
                                                     offspringMinObjVals,
                                                     offspringConstraintVals)
          # Mirror legacy containers for downstream compatibility.
          population = getattr(self, 'population', None)
          fitVals = getattr(self, 'fitVals', None)
          minObjVals = getattr(self, 'minObjVals', None)
          popAge = getattr(self, 'popAge', None)
          constraints = getattr(self, 'constraintVals', None)
          self.pop = population if population is not None else offspring
          self.popFitVals = fitVals if fitVals is not None else offspringFitVals
          self.popMinObjVals = minObjVals if minObjVals is not None else offspringMinObjVals[0]
          self.popAges = popAge if popAge is not None else [0] * len(offspring)
          self.popConstraintVals = constraints if constraints is not None else offspringConstraintVals

        else:
          # -------------------- MULTI-OBJECTIVE --------------------
          # Combine populations first, then rank, then select

          # Combine parent and offspring inputs
          combinedPop = np.vstack([self.pop.data, offspring.data])

          # Combine objectives
          combinedMinObjVals = [self.popMinObjVals[i] + offspringMinObjVals[i]
                            for i in range(len(self._objectiveVar))]

          # Combine ages (increment parent ages, offspring start at 0)
          combinedAges = list(map(lambda x: x+1, self.popAges)) + [0] * len(offspring)

          # Combine fitness values
          popFitValsByObj = [self.popFitVals[key].data.tolist()
                        for key in self.popFitVals.keys()]
          offspringFitValsByObj = [offspringFitVals[key].data.tolist()
                        for key in offspringFitVals.keys()]
          combinedFitValsByObj = np.array([i + j for i, j in zip(popFitValsByObj, offspringFitValsByObj)])
          combinedFitVals = [list(ele) for ele in list(zip(*combinedFitValsByObj))]

          # Combine constraints
          combinedConstraintVals = np.vstack([self.popConstraintVals.data, offspringConstraintVals.data])

          # Step 1: Rank the combined population R(t) using original objective signs
          # and explicit min/max directions. Constraint violations are handled
          # by constrained dominance, not fitness penalties.
          combinedExternalObjValsBySolution = np.array(
              [[self._objMult[obj] * val for obj, val in zip(self._objectiveVar, solution)]
               for solution in zip(*combinedMinObjVals)], dtype=float)
          minMask = np.array([optType == "min" for optType in self._minMax], dtype=bool)
          combinedRanks = frontUtils.rankNonDominatedFrontiers(
              combinedExternalObjValsBySolution,
              constraintVals=combinedConstraintVals,
              minMask=minMask
          )

          # Step 2: Compute crowding distance on the same original objective values.
          combinedCD = frontUtils.crowdingDistance(
              rank=np.array(combinedRanks),
              popSize=len(combinedRanks),
              objectiveValues=combinedExternalObjValsBySolution
          )

          # Step 3: NOW perform survivor selection with rank and CD already computed
          self.pop, self.popRanks, \
          self.popAges, self.popCrowdingDist, \
          self.popMinObjVals, self.popFitVals, \
          self.popConstraintVals = self._survivorSelectionInstance(
                                age=combinedAges,
                                variables=list(self.toBeSampled),
                                combinedPop=combinedPop,
                                combinedRanks=combinedRanks,
                                combinedCD=combinedCD,
                                combinedMinObjVals=combinedMinObjVals,
                                combinedFitVals=combinedFitVals,
                                combinedConstraintVals=combinedConstraintVals,
                                popSize=self._populationSize,
                                objectiveNames=list(self.popFitVals.keys()))
          self.popAge = list(self.popAges)

      else:
        # ============================================================
        # First generation: Q(t) becomes P(t+1) directly
        # ============================================================
        if not self._isMultiObjective:
          self.pop = offspring
          self.popFitVals = offspringFitVals
          self.popMinObjVals = rlz[self._objectiveVar[0]].data
          self.popAges = [0] * len(offspring)
        else:
          # For first generation multi-objective, still need rank and crowding distance.
          currentPopExternalObjValsBySolution = np.array(
              [[self._objMult[obj] * val for obj, val in zip(self._objectiveVar, solution)]
               for solution in zip(*offspringMinObjVals)], dtype=float)
          minMask = np.array([optType == "min" for optType in self._minMax], dtype=bool)

          currentPopRanks = frontUtils.rankNonDominatedFrontiers(
              currentPopExternalObjValsBySolution,
              constraintVals=offspringConstraintVals.data,
              minMask=minMask
          )

          currentPopCD = frontUtils.crowdingDistance(
              rank=np.array(currentPopRanks),
              popSize=len(currentPopRanks),
              objectiveValues=currentPopExternalObjValsBySolution
          )

          # Store as the current population
          self.pop = offspring
          self.popFitVals = offspringFitVals
          self.popMinObjVals = offspringMinObjVals
          self.popAges = [0] * len(offspring)
          self.popRanks = xr.DataArray(currentPopRanks,
                                            dims=['rank'],
                                            coords={'rank': np.arange(len(currentPopRanks))})
          self.popCrowdingDist = xr.DataArray(currentPopCD,
                                         dims=['CrowdingDistance'],
                                         coords={'CrowdingDistance': np.arange(len(currentPopCD))})
          self.popConstraintVals = offspringConstraintVals
          self.popAge = list(self.popAges)

      # ============================================================
      # PART C: Update Ages for Display
      # ============================================================

      self.popAgesArray = np.array(self.popAges)

      # ============================================================
      # PART D: Collect Best Points and Check Convergence
      # ============================================================

      # Initialize prevPopInputs on first iteration
      if not hasattr(self, 'prevPopInputs') or self.prevPopInputs is None:
        self.prevPopInputs = None

      if not self._isMultiObjective:
        # Single-objective: collect single best point
        constraintData = getattr(self, 'popConstraintVals', None)
        if constraintData is None:
          constraintData = offspringConstraintVals

        self._collectOptPoint(rlz,
                              self.popFitVals,
                              self.popMinObjVals,
                              constraintData,
                              population=self.pop)
        self._resolveNewGeneration(traj, rlz, info, self.prevPopInputs,
                                  [self.popMinObjVals], self.popFitVals,
                                  constraintData)
      else:
        # Multi-objective: collect Pareto front (rank 1)
        # Use correct signature for _collectOptPointMulti
        self._collectOptPointMulti(rlz,
                                   self.pop,
                                   self.popRanks,
                                   self.popCrowdingDist,
                                   self.popMinObjVals,
                                   self.popFitVals,
                                   self.popConstraintVals)
        # Multi-objective version with ranks and CD
        self._resolveNewGeneration(traj,
                                   rlz,
                                   info,
                                   self.prevPopInputs,
                                   self.popMinObjVals,  # minObjVals
                                   self.popFitVals,   # fitVals
                                   self.popConstraintVals,        # constraintVals
                                   self.popRanks,     # ranks
                                   self.popCrowdingDist)        # CD

      # ============================================================
      # PART E: Parent Selection from P(t+1)
      # ============================================================

      parents = self._parentSelectionInstance(self.pop,
                                              variables=list(self.toBeSampled),
                                              popFitVals=self.popFitVals,
                                              kSelection=self._kSelection,
                                              nParents=self._nParents,
                                              rank=self.popRanks if self._isMultiObjective else None,
                                              crowdDistance=self.popCrowdingDist if self._isMultiObjective else None,
                                              objVar=self._objectiveVar,
                                              isMultiObjective=self._isMultiObjective)

      # ============================================================
      # PART F: Reproduction (Crossover and Mutation)
      # ============================================================

      # Crossover
      childrenXover = self._crossoverInstance(parents=parents,
                                              variables=list(self.toBeSampled),
                                              crossoverProb=self._crossoverProb,
                                              points=self._crossoverPoints,
                                              distDict=self.distDict,
                                              EQfiles=self._EQcheckfile)

      # Mutation
      childrenMutated = self._mutationInstance(offspring=childrenXover,
                                               distDict=self.distDict,
                                               locs=self._mutationLocs,
                                               mutationProb=self._mutationProb,
                                               variables=list(self.toBeSampled))

      # ============================================================
      # PART G: Repair (if needed)
      # ============================================================

      needsRepair = False
      for chrom in range(min(self._nChildren, len(childrenMutated))):
        unique = set(childrenMutated.data[chrom, :])
        if len(childrenMutated.data[chrom,:]) != len(unique):
          for var in self.toBeSampled:
            if (hasattr(self.distDict[var], 'strategy') and
                self.distDict[var].strategy == 'withoutReplacement'):
              needsRepair = True
              break

      if needsRepair:
        children = self._repairInstance(childrenMutated, variables=list(self.toBeSampled),
                                       distInfo=self.distDict)
      else:
        children = childrenMutated

      # Truncate to population size
      children = children[:self._populationSize, :]

      daChildren = xr.DataArray(children,
                                dims=['chromosome','Gene'],
                                coords={'chromosome': np.arange(np.shape(children)[0]),
                                        'Gene': list(self.toBeSampled)})

      # ============================================================
      # PART H: Submit Children for Evaluation
      # ============================================================

      for i in range(self.batch):
        newRlz = {}
        for _, var in enumerate(self.toBeSampled.keys()):
          newRlz[var] = float(daChildren.loc[i, var].values)
        self._submitRun(newRlz, traj, self.getIteration(traj))

    # ============================================================
    # PART I: Save P(t+1) as P(t) for Next Iteration
    # ============================================================

    self.prevPopInputs = deepcopy(self.pop)

    # ============================================================
    # PART I: Save P(t+1) as P(t) for Next Iteration
    # ============================================================

    self.prevPopInputs = deepcopy(self.pop)

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
    self.raiseADebug(f'Adding run to queue: {self.denormalizeData(point)} | {info}')
    self._submissionQueue.append((point, info))

  def flush(self):
    """
      Reset Optimizer attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    # Use new naming convention
    self.pop = None
    self.popAges = None
    self.popFitVals = None
    self.popRanks = None
    self.popCrowdingDist = None
    self.popMinObjVals = None
    self.popConstraintVals = None
    self.popAgesArray = None

    # Keep old names for backward compatibility (if needed)
    self.population = None
    self.popAge = None
    self.fitVals = None
    self.rank = None
    self.crowdingDistance = None
    self.minObjVals = None
    self.constraintVals = None
    self._bestSnapshot = None

    self.ahdp = np.NaN
    self.ahd = np.NaN
    self.hdsm = np.NaN
    self.bestPoint = None
    self.bestFitVal = None
    self.multiBestPoint = None
    self.multiBestFitVals = None
    self.multiBestMinObjVals = None
    self.multiBestConstraintVals = None
    self.multiBestRank = None
    self.multiBestCD = None

  # END queuing Runs
  # * * * * * * * * * * * * * * * *

  def _resolveNewGeneration(self, traj, rlz, info, pastPop, minObjVals, fitVals, constraintVals, ranks=None, CD=None):
    """
      Store a new Generation after checking convergence
      @ In, traj, int, trajectory for this new point
      @ In, rlz, dict, realized realization
      @ In, pastPop, previous population (for convergence checking)
      @ In, minObjVals, list, minimization-space objective values at each chromosome of the realization
      @ In, fitVals, xr.DataArray, fitness values at each chromosome of the realization
      @ In, constraintVals, xr.DataArray, the constraint evaluation function
      @ In, info, dict, identifying information about the realization
      @ In, ranks, xr.DataArray, optional, ranks for multi-objective
      @ In, CD, xr.DataArray, optional, crowding distance for multi-objective
    """
    self.raiseADebug(f'Trajectory {traj} iteration {info["step"]} resolving new Generation (population) ...')
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    acceptable = 'accepted' if self.counter > 1 else 'first'
    old = pastPop  # Use pastPop parameter instead of self.population
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    if converged:
      self._closeTrajectory(traj, 'converge', 'converged', self.multiBestMinObjVals)
    # NOTE: the solution export needs to be updated BEFORE we run rejectOptPoint or extend the opt
    #       point history.

    if self._writeSteps == 'every':
      self.raiseADebug("### rlz.sizes['RAVEN_sample_ID'] = {}".format(rlz.sizes['RAVEN_sample_ID']))
      for i in range(rlz.sizes['RAVEN_sample_ID']):
        if self._isMultiObjective:
          # Use pop instead of self.population
          rlzDict = self.pop.isel(chromosome=i).to_series().to_dict()
          for j in range(len(self._objectiveVar)):
             # Use popMinObjVals instead of self.minObjVals
             rlzDict[self._objectiveVar[j]] = self.popMinObjVals[j][i]
          rlzDict['batchId'] = self.batchId
          # Use popRanks instead of self.rank
          rlzDict['rank'] = np.atleast_1d(ranks.data)[i] if ranks is not None else np.atleast_1d(self.popRanks.data)[i]
          # Use popCrowdingDist instead of self.crowdingDistance
          rlzDict['CD'] = np.atleast_1d(CD.data)[i] if CD is not None else np.atleast_1d(self.popCrowdingDist.data)[i]
          if self.popAges is not None:
            rlzDict['age'] = self.popAges[i]
          # Use popFitVals instead of self.fitVals
          for ind, fitName in enumerate(list(fitVals.keys() if isinstance(fitVals, dict) else self.popFitVals.keys())):
            rlzDict['FitnessEvaluation_'+fitName] = (fitVals if isinstance(fitVals, dict) else self.popFitVals)[fitName].data[i]
          # Use popConstraintVals instead of self.constraintVals
          for ind, consName in enumerate([y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]):
            rlzDict['ConstraintEvaluation_'+consName] = constraintVals.data[i,ind]
        else:
          varList = self._solutionExport.getVars('input') + self._solutionExport.getVars('output') + list(self.toBeSampled.keys())
          rlzDict = dict((var,np.atleast_1d(rlz[var].data)[i]) for var in set(varList) if var in rlz.data_vars)
          # Override sampled variables with the actual survivor values
          if hasattr(self, 'pop') and self.pop is not None:
            survInputs = self.pop.isel(chromosome=i)
            for var in self.toBeSampled.keys():
              if 'Gene' in survInputs.coords:
                rlzDict[var] = float(survInputs.sel(Gene=var).item())
              else:
                rlzDict[var] = float(survInputs.loc[var].item())
          # Override objective values with survivor objectives
          survivorObjs = np.asarray(self.popMinObjVals)
          if survivorObjs.ndim == 0:
            survivorObjs = np.asarray([survivorObjs])
          if survivorObjs.ndim == 1:
            rlzDict[self._objectiveVar[0]] = float(survivorObjs[i]) if survivorObjs.size > i else rlzDict.get(self._objectiveVar[0])
          else:
            for j, objName in enumerate(self._objectiveVar):
              rlzDict[objName] = float(survivorObjs[j, i]) if survivorObjs.shape[1] > i else rlzDict.get(objName)
          # Survivor fitness (single objective has a single fitness value)
          fitnessNames = list(self.popFitVals.keys()) if isinstance(self.popFitVals, xr.Dataset) else []
          if fitnessNames:
            rlzDict['fitness'] = float(self.popFitVals[fitnessNames[0]].data[i])
          elif isinstance(self.popFitVals, dict):
            firstKey = next(iter(self.popFitVals))
            rlzDict['fitness'] = float(self.popFitVals[firstKey].data[i])
          # Track survivor age and batchId if available
          if self.popAges is not None:
            rlzDict['age'] = self.popAges[i]
          rlzDict['batchId'] = self.batchId
          # Constraints
          if hasattr(constraintVals, 'coords') and 'Constraint' in constraintVals.coords:
            for ind, consName in enumerate(constraintVals['Constraint'].values):
              rlzDict['ConstraintEvaluation_'+consName] = constraintVals.data[i,ind]
        self._updateSolutionExport(traj, rlzDict, acceptable, None)

    # decide what to do next
    if acceptable in ['accepted', 'first']:
      # record history
      bestRlz = {}
      if self._isMultiObjective:
        varList = self._solutionExport.getVars('input') + self._solutionExport.getVars('output') + list(self.toBeSampled.keys())
        bestRlz = dict((var,np.atleast_1d(self.multiBestPoint[var])) for var in set(varList) if var in list(self.toBeSampled.keys()))
        for i in range(len(self._objectiveVar)):
          bestRlz[self._objectiveVar[i]] = [item[i] for item in self.multiBestMinObjVals]
        bestRlz['rank'] = self.multiBestRank
        bestRlz['CD'] = self.multiBestCD
        if len(self.multiBestConstraintVals) != 0: # No constraints
          for ind, consName in enumerate(self.multiBestConstraintVals.Constraint):
              bestRlz['ConstraintEvaluation_'+consName.values.tolist()] = self.multiBestConstraintVals[ind].values
        for ind, fitName in enumerate(list(self.multiBestFitVals.keys())):
            bestRlz['FitnessEvaluation_'+ fitName] = self.multiBestFitVals[fitName].data
        bestRlz.update(self.multiBestPoint)
      else:
        bestRlz[self._objectiveVar[0]] = self.multiBestMinObjVals[0]
        bestRlz['fitness'] = self.bestFitVal
        bestRlz.update(self.bestPoint)
      self._optPointHistory[traj].append((bestRlz, info))

  def _collectOptPoint(self, rlz, fitVals, minObjVals, constraintVals, population=None):
    """
      Collects the point (dict) from a realization
      @ In, rlz, xr.Dataset, realization data
      @ In, fitVals, xr.Dataset, fitness values at each chromosome of the realization
      @ In, minObjVals, list, minimization-space objective values at each chromosome of the realization
      @ In, constraintVals, xr.DataArray, constraint evaluation
      @ In, population, xr.DataArray, optional survivor population aligned with fitVals/minObjVals lists
      @ Out, point, dict, point used in this realization
    """
    selVars = list(self.toBeSampled.keys())
    # Draw the best chromosome information from the survivor population
    # instead of the raw evaluation batch so objective/fitness remain aligned.
    if population is not None:
      try:
        popArray = population.sel(Gene=selVars)
      except Exception:
        popArray = population
    else:
      popArray = datasetToDataArray(rlz, selVars)
    popArray = popArray.transpose('chromosome', 'Gene')
    geneNames = list(popArray.coords['Gene'].values)
    popMatrix = np.asarray(popArray.data, dtype=float)

    if fitVals is None:
      self.raiseAnError(RuntimeError, 'Fitness container is None while collecting optimal point.')

    objNames = self._objectiveVar if isinstance(self._objectiveVar, (list, tuple)) else [self._objectiveVar]

    if isinstance(fitVals, xr.Dataset):
      fitValsScalar = np.asarray(fitVals[objNames[0]].data, dtype=float)
    elif isinstance(fitVals, xr.DataArray):
      fitValsScalar = np.asarray(fitVals.data, dtype=float)
    elif isinstance(fitVals, dict):
      fitData = fitVals[objNames[0]]
      fitValsScalar = np.asarray(fitData.data if hasattr(fitData, 'data') else fitData, dtype=float)
    else:
      fitValsScalar = np.asarray(fitVals, dtype=float)
    fitValsScalar = np.atleast_1d(fitValsScalar)

    if constraintVals is None:
      constraintValsArray = np.zeros((popMatrix.shape[0], 0))
      constraintNames = []
    else:
      constraintValsArray = np.atleast_2d(constraintVals.data)
      constraintNames = constraintVals.coords['Constraint'].values if 'Constraint' in constraintVals.coords else []
    bestIdx = int(np.argmax(fitValsScalar))

    point = {gene: float(popMatrix[bestIdx, idx]) for idx, gene in enumerate(geneNames)}

    # Capture any additional variables (typically model outputs) associated with the best chromosome.
    extraVars = []
    if hasattr(self, '_solutionExport') and self._solutionExport is not None:
      extraVars = [var for var in self._solutionExport.getVars('output') if var not in point]
    def matchRealizationIndex():
      """
      matchRealizationIndex method.
      @ Out, None.
      """
      if not isinstance(rlz, xr.Dataset):
        return None
      try:
        genesForMatch = [gene for gene in geneNames if gene in rlz.data_vars]
        if not genesForMatch:
          return None
        rlzGeneArray = datasetToDataArray(rlz, genesForMatch)
      except Exception:
        return None
      rlzMatrix = np.asarray(rlzGeneArray.data, dtype=float)
      target = np.asarray([point[gene] for gene in rlzGeneArray.coords['Gene'].values], dtype=float)
      if rlzMatrix.ndim != 2 or target.size != rlzMatrix.shape[1]:
        return None
      matches = np.where(np.all(np.isclose(rlzMatrix, target[np.newaxis, :], rtol=1e-9, atol=1e-12), axis=1))[0]
      return int(matches[0]) if matches.size else None
    bestRlzIdx = matchRealizationIndex()
    candidateIdx = bestRlzIdx if bestRlzIdx is not None else bestIdx
    if extraVars and candidateIdx is not None:
      for var in extraVars:
        if var not in rlz.data_vars:
          continue
        data = rlz[var].data
        array = np.asarray(data)
        if array.ndim == 0:
          value = array.item()
        else:
          if candidateIdx >= array.shape[0]:
            continue
          value = np.take(array, candidateIdx, axis=0)
          if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
          elif isinstance(value, np.generic):
            value = value.item()
        point[var] = value

    constraintValsOfBest = {}
    if constraintValsArray.shape[1] > 0:
      for ind, consName in enumerate(constraintNames):
        constraintValsOfBest[f'ConstraintEvaluation_{consName}'] = float(constraintValsArray[bestIdx, ind])

    objectiveArray = np.asarray(minObjVals, dtype=float)
    if objectiveArray.ndim == 1:
      currentObj = float(objectiveArray[bestIdx])
    else:
      objectiveArray = np.atleast_2d(objectiveArray)
      if objectiveArray.shape[0] == 1:
        currentObj = float(objectiveArray[0, bestIdx])
      else:
        currentObj = float(objectiveArray[:, bestIdx][0])

    currentFit = float(fitValsScalar[bestIdx])

    if self.counter == 1:
      point.update(constraintValsOfBest)
      point['fitness'] = currentFit
      self.bestPoint = point
      self.bestFitVal = currentFit
      self.multiBestMinObjVals = np.array([currentObj])
      snapshot = {var: point[var] for var in geneNames}
      if hasattr(self, '_solutionExport') and self._solutionExport is not None:
        for outVar in self._solutionExport.getVars('output'):
          if outVar in point:
            snapshot[outVar] = point[outVar]
      snapshot.update(constraintValsOfBest)
      snapshot['fitness'] = currentFit
      snapshot['objective'] = currentObj
      snapshot['batchId'] = self.batchId
      if self.popAges is not None and len(self.popAges) > bestIdx:
        snapshot['age'] = self.popAges[bestIdx]
      self._bestSnapshot = snapshot.copy()
    elif currentObj <= self.multiBestMinObjVals[0] and currentFit >= self.bestFitVal:
      point.update(constraintValsOfBest)
      point['fitness'] = currentFit
      self.bestPoint = point
      self.bestFitVal = currentFit
      self.multiBestMinObjVals = np.array([currentObj])
      snapshot = {var: point[var] for var in geneNames}
      if hasattr(self, '_solutionExport') and self._solutionExport is not None:
        for outVar in self._solutionExport.getVars('output'):
          if outVar in point:
            snapshot[outVar] = point[outVar]
      snapshot.update(constraintValsOfBest)
      snapshot['fitness'] = currentFit
      snapshot['objective'] = currentObj
      snapshot['batchId'] = self.batchId
      if self.popAges is not None and len(self.popAges) > bestIdx:
        snapshot['age'] = self.popAges[bestIdx]
      self._bestSnapshot = snapshot.copy()

    return point

  def _collectOptPointMulti(self, rlz, population, rank, CD, minObjVals, fitVals, constraintVals):
    """
      Collects the point (dict) from a realization
      @ In, population, Dataset, container containing the population
      @ In, rank, xr.DataArray, rank values at each chromosome of the realization
      @ In, CD (crowdingDistance), xr.DataArray, crowdingDistance values at each chromosome of the realization
      @ In, minObjVals, list, minimization-space objective values at each chromosome of the realization
      @ In, fitVals, dict, population fitness values
      @ In, constraintVals, xr.DataArray, calculated contraints value
      @ Out, point, dict, point used in this realization
    """
    rankOneIDX = np.where(rank.data == 1)[0].tolist()
    optPoints = population[rankOneIDX]
    optMinObjVals = np.array(minObjVals)[:,rankOneIDX].T
    count = 0
    for i in list(fitVals.keys()):
      data = fitVals[i][rankOneIDX]
      if count == 0:
        fitSet = data.to_dataset(name = i)
      else:
        fitSet[i] = data
      count = count + 1
    optConstraintVals = constraintVals.data[rankOneIDX]
    optRank = rank.data[rankOneIDX]
    optCD = CD.data[rankOneIDX]

    optPointsDic = dict((var,np.array(optPoints)[:,i]) for i, var in enumerate(population.Gene.data))
    optConstNew = [list(y) for y in zip(*optConstraintVals)]
    if len(optConstNew) > 0:
      optConstNew = xr.DataArray(optConstNew,
                            dims=['Constraint','Evaluation'],
                            coords={'Constraint': [y.name for y in (self._constraintFunctions + self._impConstraintFunctions)],
                                    'Evaluation': np.arange(np.shape(optConstNew)[1])})

    self.multiBestPoint = optPointsDic
    self.multiBestFitVals = fitSet
    self.multiBestMinObjVals = optMinObjVals
    self.multiBestConstraintVals = optConstNew
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
      Checks whether the objective(s) have reached the user-specified target value(s).
      Unlike GradientDescent/SimulatedAnnealing (whose <objective> is a change tolerance),
      the GA <objective> criterion is a goal / inverse-problem target: convergence is declared
      when each objective exactly equals its requested value (see the <objective> manual entry).
      @ In, traj, int, trajectory identifier
      @ In, kwargs, dict, dictionary of parameters for convergence criteria
      @ Out, converged, bool, convergence state
    """
    # _optPointHistory is used to check that we run the algorithm one step
    if len(self._optPointHistory[traj]) < 2:
      return False
    # An alternative was to use:
    # o1, _ = self._optPointHistory[traj][-1]
    # but that will only search the "best points" so is slower at finding
    # one that matches the objective.
    o1 = kwargs['new']
    for j in range(len(np.atleast_1d(o1[self._objectiveVar[0]]))):
      converged = True
      bestObjective = []
      for i,objVar in enumerate(self._objectiveVar):
        currentObj = np.atleast_1d(o1[objVar])[j]*self._objMult[objVar]
        bestObjective.append(currentObj*self._objMult[objVar])
        converged = (currentObj == self._convergenceCriteria['objective'][i]) and converged
      if converged:
        self.multiBestMinObjVals = np.array([bestObjective])
        return converged
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
    ahdp = mathUtils.averageHausdorffDistanceP(old, new, p)
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
    ahd = mathUtils.averageHausdorffDistance(old, new)
    self.ahd = ahd
    converged = (ahd < self._convergenceCriteria['AHD'])
    self.raiseADebug(self.convFormat.format(name='AHD',
                                            conv=str(converged),
                                            got=ahd,
                                            req=self._convergenceCriteria['AHD']))

    return converged

  def _checkConvHDSM(self, traj, **kwargs):
    """
      Computes the Hausdorff Distance Similarity Metric as the termination criteria
      @ In, traj, int, trajectory identifier
      @ In, kwargs, dict, dictionary of parameters for SAHDp termination criteria:
            old, np.array, old generation
            new, np.array, new generation
      @ Out, converged, bool, convergence state
    """
    old = kwargs['old'].data
    new = datasetToDataArray(kwargs['new'], list(self.toBeSampled)).data
    self.hdsm = mathUtils.hausdorffDistanceSimilarityMeasure(old, new)
    converged = (self.hdsm >= self._convergenceCriteria['HDSM'])
    self.raiseADebug(self.convFormat.format(name='HDSM',
                                            conv=str(converged),
                                            got= self.hdsm,
                                            req=self._convergenceCriteria['HDSM']))

    return converged


  def _checkConvSpread(self, traj, **kwargs):
    """
    Checks convergence based on spread (diversity) metric from Deb et al. (2002).
    @ In, traj, int, trajectory identifier
    @ In, kwargs, dict, parameters
    @ Out, converged, bool, convergence state
    """
    # Need at least rank-1 front
    if not hasattr(self, 'popRanks'):
        return False

    rank1Indices = np.where(self.popRanks.data == 1)[0]
    if len(rank1Indices) < 3:
        return False  # Need at least 3 points for meaningful spread

    # Extract rank-1 objective values
    front = []
    for idx in rank1Indices:
        point = [self.popMinObjVals[j][idx] for j in range(len(self._objectiveVar))]
        front.append(point)

    spread = self._computeSpread(front)

    converged = spread < self._convergenceCriteria.get('spread', 0.5)

    self.raiseADebug(self.convFormat.format(
        name='Spread',
        conv=str(converged),
        got=spread,
        req=self._convergenceCriteria.get('spread', 0.5)
    ))

    return converged

  def _computeSpread(self, front):
    """
    Compute spread metric (Δ) from Deb et al. (2002) NSGA-II paper.

    Δ = (d_f + d_l + Σ|d_i - d̄|) / (d_f + d_l + (N-1)d̄)

    @ In, front, list of lists, Pareto front points
    @ Out, spread, float, spread value (0 = perfect distribution)
    """
    n = len(front)
    if n < 2:
        return 0.0

    numObj = len(front[0])

    # Calculate Euclidean distances between consecutive solutions
    distances = []

    for obj in range(numObj):
        # Sort front by this objective
        sortedIndices = sorted(range(n), key=lambda i: front[i][obj])
        sortedFront = [front[i] for i in sortedIndices]

        # Distance between consecutive points
        for i in range(len(sortedFront) - 1):
            dist = np.linalg.norm(np.array(sortedFront[i+1]) - np.array(sortedFront[i]))
            distances.append(dist)

    if len(distances) == 0:
        return 0.0

    # Extreme distances (distance to boundary solutions)
    # For simplicity, use distance from first to ideal and last to nadir
    ideal = [min(p[i] for p in front) for i in range(numObj)]
    nadir = [max(p[i] for p in front) for i in range(numObj)]

    sortedByFirstObj = sorted(front, key=lambda p: p[0])
    distFirst = np.linalg.norm(np.array(sortedByFirstObj[0]) - np.array(ideal))
    distLast = np.linalg.norm(np.array(sortedByFirstObj[-1]) - np.array(nadir))

    # Mean distance
    meanDist = np.mean(distances)

    if meanDist == 0:
        return 0.0

    # Spread calculation
    numerator = distFirst + distLast + sum(abs(d - meanDist) for d in distances)
    denominator = distFirst + distLast + (len(distances)) * meanDist

    if denominator == 0:
        return 0.0

    spread = numerator / denominator

    return spread
  def _checkConvMaxSpread(self, traj, **kwargs):
    """
    Checks convergence based on maximum spread stabilization.
    @ In, traj, int, trajectory identifier
    @ In, kwargs, dict, parameters
    @ Out, converged, bool, convergence state
    """
    if len(self._optPointHistory[traj]) < 2:
        return False

    # Current front
    rank1Indices = np.where(self.popRanks.data == 1)[0]
    currentFront = []
    for idx in rank1Indices:
        point = [self.popMinObjVals[j][idx] for j in range(len(self._objectiveVar))]
        currentFront.append(point)

    # Previous front
    prevOpt, _ = self._optPointHistory[traj][-2]
    prevRank1 = np.where(np.array(prevOpt['rank']) == 1)[0]
    prevFront = []
    for idx in prevRank1:
        point = [prevOpt[self._objectiveVar[j]][idx] for j in range(len(self._objectiveVar))]
        prevFront.append(point)

    # Compute MS
    currentMS = self._computeMaxSpread(currentFront)
    prevMS = self._computeMaxSpread(prevFront)

    # Check relative change
    if prevMS == 0:
        relChange = float('inf')
    else:
        relChange = abs(currentMS - prevMS) / prevMS

    converged = relChange < self._convergenceCriteria.get('maxSpread', 0.05)

    self.raiseADebug(self.convFormat.format(
        name='MaxSpread',
        conv=str(converged),
        got=relChange,
        req=self._convergenceCriteria.get('maxSpread', 0.05)
    ))

    return converged

  def _computeMaxSpread(self, front):
    """
    Compute maximum spread metric.
    @ In, front, list of lists, Pareto front
    @ Out, ms, float, maximum spread
    """
    if len(front) < 2:
        return 0.0

    numObj = len(front[0])
    ranges = []

    for obj in range(numObj):
        objValues = [p[obj] for p in front]
        ranges.append(max(objValues) - min(objValues))

    ms = np.sqrt(sum(r**2 for r in ranges))

    return ms

  def _checkConvRank1Ratio(self, traj, **kwargs):
    """
    Checks convergence based on percentage of population in rank 1.
    @ In, traj, int, trajectory identifier
    @ In, kwargs, dict, parameters
    @ Out, converged, bool, convergence state
    """
    if not hasattr(self, 'popRanks'):
        return False

    # Count rank-1 solutions
    rank1Count = np.sum(self.popRanks.data == 1)
    ratio = rank1Count / self._populationSize

    # Track history
    if not hasattr(self, '_rank1History'):
        self._rank1History = {}
    if traj not in self._rank1History:
        self._rank1History[traj] = []
    self._rank1History[traj].append(ratio)

    # Converged if ratio high and stable
    threshold = self._convergenceCriteria.get('rank1Ratio', 0.5)
    stableGenerations = 3  # Require stability

    if len(self._rank1History[traj]) < stableGenerations:
        converged = False
    else:
        recentRatios = self._rank1History[traj][-stableGenerations:]
        allAboveThreshold = all(r >= threshold for r in recentRatios)
        variation = max(recentRatios) - min(recentRatios)
        converged = allAboveThreshold and variation < 0.1

    self.raiseADebug(self.convFormat.format(
        name='Rank1Ratio',
        conv=str(converged),
        got=ratio,
        req=threshold
    ))

    return converged

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
    if acceptable == 'accepted':
      self.raiseADebug(f'Convergence Check for Trajectory {traj}:')
      # check convergence
      converged, convDict = self.checkConvergence(traj, new, old)
    else:
      converged = False
      convDict = dict((var, False) for var in self._convergenceInfo[traj])
    self._convergenceInfo[traj].update(convDict)
    return converged

  def _updatePersistence(self, traj, converged, minObjVal):
    """
      Update persistence tracking state variables
      @ In, traj, int, identifier
      @ In, converged, bool, convergence check result
      @ In, minObjVal, float, new minimization-space objective value
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

  ###############################
  # Constraint Handling         #
  ###############################
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
  ###############################
  # END constraint handling     #
  ###############################
  def _updateSolutionExport(self, traj, rlz, acceptable, rejectReason):
    """
      Ensure solution export rows stay synchronized after the GA refactor.
      In particular, the single-objective 'final' row needs the same source
      values used for every accepted iteration to keep objective/fitness aligned.
    """
    if not self._isMultiObjective and acceptable == 'final':
      rlz = self._composeFinalRealization(rlz)
    super(GeneticAlgorithm, self)._updateSolutionExport(traj, rlz, acceptable, rejectReason)

  def _composeFinalRealization(self, rlz):
    """
      Build the realization used for the final solution export so that the
      objective, fitness, decision variables, and constraint metrics match the
      aligned values written for intermediary accepted iterations.
    """
    finalRlz = dict(rlz)
    # carry over the best decision variables and constraint evaluations
    bestIdx = self._matchBestChromosomeIndex()
    if bestIdx is None:
      bestIdx = self._inferBestIndexFromObjective()

    if self._bestSnapshot:
      for var in self.toBeSampled:
        if var in self._bestSnapshot:
          finalRlz[var] = self._bestSnapshot[var]
      for key, val in self._bestSnapshot.items():
        if key.startswith('ConstraintEvaluation_'):
          finalRlz[key] = val
      if hasattr(self, '_solutionExport') and self._solutionExport is not None:
        for outVar in self._solutionExport.getVars('output'):
          if outVar in self._bestSnapshot:
            finalRlz[outVar] = self._bestSnapshot[outVar]
      if 'objective' in self._bestSnapshot:
        finalRlz[self._objectiveVar[0]] = self._bestSnapshot['objective']
      if 'fitness' in self._bestSnapshot:
        finalRlz['fitness'] = self._bestSnapshot['fitness']
      if 'age' in self._bestSnapshot:
        finalRlz['age'] = self._bestSnapshot['age']
      if 'batchId' in self._bestSnapshot:
        finalRlz['batchId'] = self._bestSnapshot['batchId']
    elif isinstance(self.bestPoint, dict):
      for key, val in self.bestPoint.items():
        if key.startswith('ConstraintEvaluation_'):
          finalRlz[key] = val

    # objective value and fitness are stored separately from the population
    if self.multiBestMinObjVals is not None and 'objective' not in (self._bestSnapshot or {}):
      finalRlz[self._objectiveVar[0]] = float(np.atleast_1d(self.multiBestMinObjVals)[0])
    if self.bestFitVal is not None and 'fitness' not in (self._bestSnapshot or {}):
      finalRlz['fitness'] = float(np.atleast_1d(self.bestFitVal)[0])

    # Overwrite decision variables using the survivor record to keep them
    # consistent with the recorded objective/fitness.
    if bestIdx is not None and hasattr(self, 'pop') and self.pop is not None:
      for var in self.toBeSampled:
        try:
          arr = np.asarray(self.pop.sel(Gene=var))
          finalRlz[var] = float(np.atleast_1d(arr)[bestIdx])
        except Exception:
          if var in self.bestPoint:
            finalRlz[var] = self.bestPoint[var]
    elif isinstance(self.bestPoint, dict):
      for var in self.toBeSampled:
        if var in self.bestPoint:
          finalRlz[var] = self.bestPoint[var]

    # include survivor age information if we can match the stored best point
    if bestIdx is not None and self.popAges and 'age' not in (self._bestSnapshot or {}):
      finalRlz['age'] = self.popAges[bestIdx]
    elif isinstance(self.popAge, list) and self.popAge:
      finalRlz['age'] = self.popAge[0]
    else:
      finalRlz['age'] = 0

    finalRlz['batchId'] = self.batchId
    return finalRlz

  def _matchBestChromosomeIndex(self):
    """
      Locate the index of the stored best chromosome within the current mating
      population so we can recover metadata such as age for the final export.
    """
    if not hasattr(self, 'pop') or self.pop is None:
      return None
    genes = list(self.toBeSampled.keys())
    try:
      popDA = self.pop.sel(Gene=genes).transpose('chromosome', 'Gene')
    except Exception:
      popDA = self.pop.transpose('chromosome', 'Gene')
    popMatrix = np.asarray(popDA)
    if popMatrix.ndim != 2 or not popMatrix.size:
      return None
    targetVals = []
    for gene in genes:
      if gene not in self.bestPoint:
        return None
      targetVals.append(float(self.bestPoint[gene]))
    target = np.asarray(targetVals, dtype=float)
    if np.any(np.isnan(target)):
      return None
    matches = np.isclose(popMatrix, target[np.newaxis, :], rtol=1e-9, atol=1e-12)
    hit = np.where(np.all(matches, axis=1))[0]
    return int(hit[0]) if hit.size else None

  def _inferBestIndexFromObjective(self):
    """
      Fallback helper used when the stored best-point keys no longer match the
      survivor population. Selects the chromosome with the minimal objective
      value from the current mating population.
    """
    if self.popMinObjVals is None:
      return None
    try:
      objValues = np.asarray(self.popMinObjVals, dtype=float)
    except Exception:
      return None
    if objValues.ndim == 1 and objValues.size:
      return int(np.argmin(objValues))
    if objValues.ndim > 1 and objValues.size:
      return int(np.argmin(objValues[0]))
    return None

  def _addToSolutionExport(self, traj, rlz, acceptable):
    """
      Contributes additional entries to the solution export.
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ Out, toAdd, dict, additional entries
    """
    # meta variables
    ageVal = rlz.get('age', None)
    if ageVal is None:
      if self.popAges is not None and len(np.atleast_1d(self.popAges)) > 0:
        ageVal = np.atleast_1d(self.popAges)[0]
      else:
        ageVal = 0
    toAdd = {'age': ageVal,
             'batchId': self.batchId,
             'AHDp': self.ahdp,
             'AHD': self.ahd,
             'HDSM': self.hdsm
             }

    if self._isMultiObjective:
      toAdd['rank'] = rlz['rank']
      toAdd['CD'] = rlz['CD']

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

    acceptable = super(RavenSampled, self)._formatSolutionExportVariableNames(acceptable)
    new = []
    while acceptable:
      template = acceptable.pop()
      if '{CONV}' in template:
        new.extend([template.format(CONV=conv) for conv in self._convergenceCriteria])
      elif '{VAR}' in template:
        new.extend([template.format(VAR=var) for var in self.toBeSampled])
      elif '{OBJ}' in template:
        new.extend([template.format(OBJ=obj) for obj in self._objectiveVar])
      elif '{CONSTRAINT}' in template:
        new.extend([template.format(CONSTRAINT=constraint.name) for constraint in self._constraintFunctions + self._impConstraintFunctions])
      else:
        new.append(template)

    return set(new)
