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
                                                                | bestFitness                        |
                                                                | bestPoint                          |
                                                                | constraintsV                       |
                                                                | convergenceOptions                 |
                                                                | crowdingDistance                   |
                                                                | fitness                            |
                                                                | hdsm                               |
                                                                | multiBestCD                        |
                                                                | multiBestConstraint                |
                                                                | multiBestFitness                   |
                                                                | multiBestObjective                 |
                                                                | multiBestPoint                     |
                                                                | multiBestRank                      |
                                                                | objectiveVal                       |
                                                                | popAge                             |
                                                                | population                         |
                                                                | rank                               |
                                                                |------------------------------------|
                                                                | _GD                                |
                                                                | _GDp                               |
                                                                | __init__                           |
                                                                | _addToSolutionExport               |
                                                                | _ahd                               |
                                                                | _ahdp                              |
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
                                                                | _envelopeSize                      |
                                                                | _formatSolutionExportVariableNames |
                                                                | _handleExplicitConstraints         |
                                                                | _handleImplicitConstraints         |
                                                                | _hdsm                              |
                                                                | _popDist                           |
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
# from .survivorSelection import survivorSelection as survivorSelectionProcess
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
  ##TODO: Explore MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) or
  # PESA-II (Pareto Envelope-Based Selection Algorithm II)
  # These algorithms can offer better performance and robustness in certain scenarios
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
    self.constraintsV = None                                     # calculated contraints value
    self.crowdingDistance = None                                 # population crowding distance (for Multi-objective optimization only)
    self.ahdp = np.NaN                                           # p-Average Hausdorff Distance between populations
    self.ahd  = np.NaN                                           # Hausdorff Distance between populations
    self.hdsm = np.NaN                                           # Hausdorff Distance Similarity metric between populations
    self.bestPoint = None                                        # the best solution (chromosome) found among population in a specific batchId
    self.bestFitness = None                                      # fitness value of the best solution found
    self._bestSnapshot = None                                    # cached survivor snapshot for final export alignment
    self.multiBestPoint = {}                                     # the best solutions (chromosomes) found among population in a specific batchId
    self.multiBestFitness = {}                                   # fitness values of the best solutions found
    self.multiBestObjective = {}                                 # objective values of the best solutions found
    self.multiBestConstraint = {}                                # constraint values of the best solutions found
    self.multiBestRank = {}                                      # rank values of the best solutions found
    self.multiBestCD = {}                                        # crowding distance (CD) values of the best solutions found
    self.objectiveVal = None                                     # objective values of solutions
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
                  \end{itemize}""")
    crossover.addParam("type",
                       InputTypes.makeEnumType('crossover','crossoverType',['onePointCrossover','twoPointsCrossover','uniformCrossover']),
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
                \end{itemize} """)
    mutation.addParam("type",
                      InputTypes.makeEnumType('mutation','mutationType',['swapMutator','scrambleMutator','inversionMutator','randomMutator']),
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

  ## TODO: We have to estimate the max number of unique chromosomes and make sure population size doesn't exceed that number. Or should it?
  # def _useRealization(self, info, rlz):
  #   """
  #     Used to feedback the collected runs into actionable items within the sampler.
  #     This is called by localFinalizeActualSampling, and hence should contain the main skeleton.
  #     @ In, info, dict, identifying information about the realization
  #     @ In, rlz, xr.Dataset, new batched realizations
  #     @ Out, None
  #   """
  #   info['step'] = self.counter
  #   traj = info['traj']
  #   for t in self._activeTraj[1:]:
  #     self._closeTrajectory(t, 'cancel', 'Currently GA is single trajectory', 0)
  #   self.incrementIteration(traj)

  #   population = datasetToDataArray(rlz, list(self.toBeSampled))

  #   objectiveVal = []
  #   for i in range(len(self._objectiveVar)):
  #     objectiveVal.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))

  #   # 1. Check constraint violations and calculate the constraint function g (<0 if the constraint is violated)
  #   g = constraintHandling(self, info, rlz, population, objectiveVal, multiObjective=self._isMultiObjective)

  #   # 2. Compute fitness for the offspring
  #   populationFitness = self._fitnessInstance(rlz,
  #                                            objVar=self._objectiveVar,
  #                                            a=self._objCoeff,
  #                                            b=self._penaltyCoeff,
  #                                            penalty=None,
  #                                            constraintFunction=g,
  #                                            constraintNum=self._numOfConst,
  #                                            type=self._minMax)

  #   # Single-objective post-processing (if needed)
  #   if not self._isMultiObjective:
  #       self._collectOptPoint(rlz, populationFitness, objectiveVal[0], g)
  #       self._resolveNewGeneration(traj, rlz, info, objectiveVal[0], populationFitness, g)

  #   # 3. Survivor selection
  #   if self._activeTraj:

  #     survivorSelection =  survivorSelectionProcess.multiObjSurvivorSelect if self._isMultiObjective else  survivorSelectionProcess.singleObjSurvivorSelect
  #     survivorSelection(self, info, rlz, traj, population, populationFitness, objectiveVal, g)
  #     if self._isMultiObjective:
  #       if self.counter <= 1:
  #         # offspringObjsVals for Rank and CD calculation
  #         fitVal = datasetToDataArray(self.fitness, self._objectiveVar).data
  #         offspringFitVals = fitVal.tolist()
  #         # 4. Compute the rank of offspring
  #         offSpringRank = frontUtils.rankNonDominatedFrontiers(np.array(offspringFitVals), isFitness=True)
  #         self.rank = xr.DataArray(offSpringRank,
  #                                      dims=['rank'],
  #                                      coords={'rank': np.arange(np.shape(offSpringRank)[0])})
  #         # 5. Compute the crowding distance of offspring
  #         offSpringCD = frontUtils.crowdingDistance(rank=offSpringRank,
  #                                                             popSize=len(offSpringRank),
  #                                                             fitness=np.array(offspringFitVals))
  #         self.crowdingDistance = xr.DataArray(offSpringCD,
  #                                              dims=['CrowdingDistance'],
  #                                              coords={'CrowdingDistance': np.arange(np.shape(offSpringCD)[0])})
  #         self.objectiveVal = []
  #         for i in range(len(self._objectiveVar)):
  #           self.objectiveVal.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))
  #       self._collectOptPointMulti(self.population,
  #                                  self.rank,
  #                                  self.crowdingDistance,
  #                                  self.objectiveVal,
  #                                  self.fitness,
  #                                  self.constraintsV)
  #       self._resolveNewGeneration(traj, rlz, info)


  #     # 6. Parent selection from population
  #     parents = self._parentSelectionInstance(self.population,
  #                                             variables=list(self.toBeSampled),
  #                                             fitness=self.fitness,
  #                                             kSelection=self._kSelection,
  #                                             nParents=self._nParents,
  #                                             rank=self.rank,
  #                                             crowdDistance=self.crowdingDistance,
  #                                             objVar=self._objectiveVar,
  #                                             isMultiObjective = self._isMultiObjective,
  #                                             )

  #     # 7. Reproduction
  #     # 7.1 Crossover
  #     childrenXover = self._crossoverInstance(parents=parents,
  #                                             variables=list(self.toBeSampled),
  #                                             crossoverProb=self._crossoverProb,
  #                                             points=self._crossoverPoints)

  #     # 7.2 Mutation
  #     childrenMutated = self._mutationInstance(offSprings=childrenXover,
  #                                              distDict=self.distDict,
  #                                              locs=self._mutationLocs,
  #                                              mutationProb=self._mutationProb,
  #                                              variables=list(self.toBeSampled))

  #     # 8. repair/replacement
  #     # Repair should only happen if multiple genes in a single chromosome have the same values (),
  #     # and at the same time the sampling of these genes should be with Out replacement.
  #     needsRepair = False
  #     for chrom in range(self._nChildren):
  #       unique = set(childrenMutated.data[chrom, :])
  #       if len(childrenMutated.data[chrom,:]) != len(unique):
  #         for var in self.toBeSampled: # TODO: there must be a smarter way to check if a variables strategy is without replacement
  #           if (hasattr(self.distDict[var], 'strategy') and self.distDict[var].strategy == 'withoutReplacement'):
  #             needsRepair = True
  #             break
  #     if needsRepair:
  #       children = self._repairInstance(childrenMutated,variables=list(self.toBeSampled),distInfo=self.distDict)
  #     else:
  #       children = childrenMutated

  #     # keeping the population size constant by ignoring the excessive children
  #     children = children[:self._populationSize, :]
  #     daChildren = xr.DataArray(children,
  #                               dims=['chromosome','Gene'],
  #                               coords={'chromosome': np.arange(np.shape(children)[0]),
  #                                       'Gene':list(self.toBeSampled)})

  #     # 9. Submit children batch
    #     # Submit children coordinates (x1,...,xm), i.e., self.childrenCoordinates
    #     for i in range(self.batch):
    #       newRlz = {}
    #       for _, var in enumerate(self.toBeSampled.keys()):
    #         newRlz[var] = float(daChildren.loc[i, var].values)
    #       self._submitRun(newRlz, traj, self.getIteration(traj))

  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      This is called by localFinalizeActualSampling, and hence should contain the main skeleton.
      FIXED: Proper NSGA-II flow with ranking/CD before survivor selection
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
                    │  │  • Check convergence (hypervolume, etc.)               │ │
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
                    │  │  • Save P(t+1) as matingPopInputs for next iteration   │ │
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
    currentPopInputs = datasetToDataArray(rlz, list(self.toBeSampled))

    # Step 2: Extract offspring objectives (already evaluated by model)
    currentPop_objvals = []
    for i in range(len(self._objectiveVar)):
      currentPop_objvals.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))

    # Step 3: Compute constraints for offspring Q(t)
    currentPop_g = constraintHandling(self, info, rlz, currentPopInputs,
                                      currentPop_objvals, multiObjective=self._isMultiObjective)

    # Step 4: Normalize if requested
    norm_rlz = deepcopy(rlz)
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
            norm_rlz[var][i] = (rlz[var][i] - self.normScores[var][0]) / self.normScores[var][1]
            if np.isnan(norm_rlz[var][i]):
              norm_rlz[var][i] = 0.0

      for i in range(len(currentPop_g)):
        for j in range(len(constrVarsList)):
          currentPop_g[i][j] = currentPop_g[i][j] / self.normScores[constrVarsList[j].parameterNames()[0]][1]
          if np.isnan(currentPop_g[i][j]):
            currentPop_g[i][j] = 0.0

    # Step 5: Compute fitness for offspring Q(t)
    currentPopFitness = self._fitnessInstance(norm_rlz,
                                               objVar=self._objectiveVar,
                                               a=self._objCoeff,
                                               b=self._penaltyCoeff,
                                               penalty=None,
                                               constraintFunction=currentPop_g,
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
                                                     currentPopInputs,
                                                     currentPopFitness,
                                                     currentPop_objvals,
                                                     currentPop_g)
          # Mirror legacy containers for downstream compatibility.
          population = getattr(self, 'population', None)
          fitness = getattr(self, 'fitness', None)
          objective_val = getattr(self, 'objectiveVal', None)
          pop_age = getattr(self, 'popAge', None)
          constraints = getattr(self, 'constraintsV', None)
          self.matingPopInputs = population if population is not None else currentPopInputs
          self.matingPopFitness = fitness if fitness is not None else currentPopFitness
          self.matingPopObjVals = objective_val if objective_val is not None else currentPop_objvals[0]
          self.matingPopAges = pop_age if pop_age is not None else [0] * len(currentPopInputs)
          self.matingPop_g = constraints if constraints is not None else currentPop_g

        else:
          # -------------------- MULTI-OBJECTIVE --------------------
          # FIXED: Combine populations first, then rank, then select

          # Combine parent and offspring inputs
          combinedInputs = np.vstack([self.matingPopInputs.data, currentPopInputs.data])

          # Combine objectives
          combinedObjVals = [self.matingPopObjVals[i] + currentPop_objvals[i]
                            for i in range(len(self._objectiveVar))]

          # Combine ages (increment parent ages, offspring start at 0)
          combinedAges = list(map(lambda x: x+1, self.matingPopAges)) + [0] * len(currentPopInputs)

          # Combine fitness values
          popFitArray = [self.matingPopFitness[key].data.tolist()
                        for key in self.matingPopFitness.keys()]
          offFitArray = [currentPopFitness[key].data.tolist()
                        for key in currentPopFitness.keys()]
          combinedFitness = np.array([i + j for i, j in zip(popFitArray, offFitArray)])
          combinedFitnessPairs = [list(ele) for ele in list(zip(*combinedFitness))]

          # Combine constraints
          combinedConstraints = np.vstack([self.matingPop_g.data, currentPop_g.data])

          # FIXED Step 1: Rank the COMBINED population R(t)
          combinedRanks = frontUtils.rankNonDominatedFrontiers(
              np.array(combinedFitnessPairs),
              isFitness=True
          )

          # FIXED Step 2: Compute CD for the COMBINED population R(t)
          combinedCD = frontUtils.crowdingDistance(
              rank=np.array(combinedRanks),
              popSize=len(combinedRanks),
              fitness=np.array(combinedFitnessPairs)
          )

          # FIXED Step 3: NOW perform survivor selection with rank and CD already computed
          self.matingPopInputs, self.matingPopRanks, \
          self.matingPopAges, self.matingPopCD, \
          self.matingPopObjVals, self.matingPopFitness, \
          self.matingPop_g = self._survivorSelectionInstance(
                                age=combinedAges,
                                variables=list(self.toBeSampled),
                                combinedInputs=combinedInputs,
                                combinedRanks=combinedRanks,
                                combinedCD=combinedCD,
                                combinedObjectives=combinedObjVals,
                                combinedFitness=combinedFitnessPairs,
                                combinedConstraints=combinedConstraints,
                                popSize=self._populationSize,
                                objectiveNames=list(self.matingPopFitness.keys()))
          self.popAge = list(self.matingPopAges)

      else:
        # ============================================================
        # First generation: Q(t) becomes P(t+1) directly
        # ============================================================
        if not self._isMultiObjective:
          self.matingPopInputs = currentPopInputs
          self.matingPopFitness = currentPopFitness
          self.matingPopObjVals = rlz[self._objectiveVar[0]].data
          self.matingPopAges = [0] * len(currentPopInputs)
        else:
          # For first generation multi-objective, still need to rank
          currentPop_fitsbysoln = datasetToDataArray(currentPopFitness,
                                                     self._objectiveVar).data.tolist()

          # Rank first generation
          currentPopRanks = frontUtils.rankNonDominatedFrontiers(
              np.array(currentPop_fitsbysoln),
              isFitness=True
          )

          # Compute crowding distance for first generation
          currentPopCD = frontUtils.crowdingDistance(
              rank=np.array(currentPopRanks),
              popSize=len(currentPopRanks),
              fitness=np.array(currentPop_fitsbysoln)
          )

          # Store as mating population
          self.matingPopInputs = currentPopInputs
          self.matingPopFitness = currentPopFitness
          self.matingPopObjVals = currentPop_objvals
          self.matingPopAges = [0] * len(currentPopInputs)
          self.matingPopRanks = xr.DataArray(currentPopRanks,
                                            dims=['rank'],
                                            coords={'rank': np.arange(len(currentPopRanks))})
          self.matingPopCD = xr.DataArray(currentPopCD,
                                         dims=['CrowdingDistance'],
                                         coords={'CrowdingDistance': np.arange(len(currentPopCD))})
          self.matingPop_g = currentPop_g
          self.popAge = list(self.matingPopAges)

      # ============================================================
      # PART C: Update Ages for Display
      # ============================================================

      self.currentPop_ages = np.array(self.matingPopAges)

      # ============================================================
      # PART D: Collect Best Points and Check Convergence
      # ============================================================

      # FIXED: Initialize prevPop_inputs on first iteration
      if not hasattr(self, 'prevPop_inputs') or self.prevPop_inputs is None:
        self.prevPop_inputs = None

      if not self._isMultiObjective:
        # Single-objective: collect single best point
        constraint_data = self.matingPop_g if hasattr(self, 'matingPop_g') else currentPop_g

        self._collectOptPoint(rlz,
                              self.matingPopFitness,
                              self.matingPopObjVals,
                              constraint_data,
                              population=self.matingPopInputs)
        self._resolveNewGeneration(traj, rlz, info, self.prevPop_inputs,
                                  [self.matingPopObjVals], self.matingPopFitness,
                                  constraint_data)
      else:
        # Multi-objective: collect Pareto front (rank 1)
        # FIXED: Use correct signature for _collectOptPointMulti
        # def _collectOptPointMulti(self, rlz, population, rank, CD, objVal, fitness, constraintsV)
        self._collectOptPointMulti(rlz,
                                   self.matingPopInputs,
                                   self.matingPopRanks,
                                   self.matingPopCD,
                                   self.matingPopObjVals,
                                   self.matingPopFitness,
                                   self.matingPop_g)
        # Multi-objective version with ranks and CD
        self._resolveNewGeneration(traj,
                                   rlz,
                                   info,
                                   self.prevPop_inputs,
                                   self.matingPopObjVals,  # objectiveVal
                                   self.matingPopFitness,   # fitness
                                   self.matingPop_g,        # g
                                   self.matingPopRanks,     # ranks
                                   self.matingPopCD)        # CD

      # ============================================================
      # PART E: Parent Selection from P(t+1)
      # ============================================================

      parents = self._parentSelectionInstance(self.matingPopInputs,
                                              variables=list(self.toBeSampled),
                                              fitness=self.matingPopFitness,
                                              kSelection=self._kSelection,
                                              nParents=self._nParents,
                                              rank=self.matingPopRanks if self._isMultiObjective else None,
                                              crowdDistance=self.matingPopCD if self._isMultiObjective else None,
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
                                              EQfiles=self._EQcheckfile)

      # Mutation
      childrenMutated = self._mutationInstance(offSprings=childrenXover,
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

    self.prevPop_inputs = deepcopy(self.matingPopInputs)

    # ============================================================
    # PART I: Save P(t+1) as P(t) for Next Iteration
    # ============================================================

    self.prevPop_inputs = deepcopy(self.matingPopInputs)

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
    self.matingPopInputs = None
    self.matingPopAges = None
    self.matingPopFitness = None
    self.matingPopRanks = None
    self.matingPopCD = None
    self.matingPopObjVals = None
    self.matingPop_g = None
    self.currentPop_ages = None

    # Keep old names for backward compatibility (if needed)
    self.population = None
    self.popAge = None
    self.fitness = None
    self.rank = None
    self.crowdingDistance = None
    self.objectiveVal = None
    self.constraintsV = None
    self._bestSnapshot = None

    self.ahdp = np.NaN
    self.ahd = np.NaN
    self.hdsm = np.NaN
    self.bestPoint = None
    self.bestFitness = None
    self.multiBestPoint = None
    self.multiBestFitness = None
    self.multiBestObjective = None
    self.multiBestConstraint = None
    self.multiBestRank = None
    self.multiBestCD = None

  # END queuing Runs
  # * * * * * * * * * * * * * * * *

  def _resolveNewGeneration(self, traj, rlz, info, pastPop, objectiveVal, fitness, g, ranks=None, CD=None):
    """
      Store a new Generation after checking convergence
      @ In, traj, int, trajectory for this new point
      @ In, rlz, dict, realized realization
      @ In, pastPop, previous population (for convergence checking)
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, fitness, xr.DataArray, fitness values at each chromosome of the realization
      @ In, g, xr.DataArray, the constraint evaluation function
      @ In, info, dict, identifying information about the realization
      @ In, ranks, xr.DataArray, optional, ranks for multi-objective
      @ In, CD, xr.DataArray, optional, crowding distance for multi-objective
    """
    self.raiseADebug(f'Trajectory {traj} iteration {info["step"]} resolving new Generation (population) ...')
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    acceptable = 'accepted' if self.counter > 1 else 'first'
    old = pastPop  # FIXED: Use pastPop parameter instead of self.population
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    if converged:
      self._closeTrajectory(traj, 'converge', 'converged', self.multiBestObjective)
    # NOTE: the solution export needs to be updated BEFORE we run rejectOptPoint or extend the opt
    #       point history.

    if self._writeSteps == 'every':
      self.raiseADebug("### rlz.sizes['RAVEN_sample_ID'] = {}".format(rlz.sizes['RAVEN_sample_ID']))
      for i in range(rlz.sizes['RAVEN_sample_ID']):
        if self._isMultiObjective:
          # FIXED: Use matingPopInputs instead of self.population
          rlzDict = self.matingPopInputs.isel(chromosome=i).to_series().to_dict()
          for j in range(len(self._objectiveVar)):
             # FIXED: Use matingPopObjVals instead of self.objectiveVal
             rlzDict[self._objectiveVar[j]] = self.matingPopObjVals[j][i]
          rlzDict['batchId'] = self.batchId
          # FIXED: Use matingPopRanks instead of self.rank
          rlzDict['rank'] = np.atleast_1d(ranks.data)[i] if ranks is not None else np.atleast_1d(self.matingPopRanks.data)[i]
          # FIXED: Use matingPopCD instead of self.crowdingDistance
          rlzDict['CD'] = np.atleast_1d(CD.data)[i] if CD is not None else np.atleast_1d(self.matingPopCD.data)[i]
          if self.matingPopAges is not None:
            rlzDict['age'] = self.matingPopAges[i]
          # FIXED: Use matingPopFitness instead of self.fitness
          for ind, fitName in enumerate(list(fitness.keys() if isinstance(fitness, dict) else self.matingPopFitness.keys())):
            rlzDict['FitnessEvaluation_'+fitName] = (fitness if isinstance(fitness, dict) else self.matingPopFitness)[fitName].data[i]
          # FIXED: Use matingPop_g instead of self.constraintsV
          for ind, consName in enumerate([y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]):
            rlzDict['ConstraintEvaluation_'+consName] = g.data[i,ind]
        else:
          varList = self._solutionExport.getVars('input') + self._solutionExport.getVars('output') + list(self.toBeSampled.keys())
          rlzDict = dict((var,np.atleast_1d(rlz[var].data)[i]) for var in set(varList) if var in rlz.data_vars)
          # Override sampled variables with the actual survivor values
          if hasattr(self, 'matingPopInputs') and self.matingPopInputs is not None:
            survInputs = self.matingPopInputs.isel(chromosome=i)
            for var in self.toBeSampled.keys():
              if 'Gene' in survInputs.coords:
                rlzDict[var] = float(survInputs.sel(Gene=var).item())
              else:
                rlzDict[var] = float(survInputs.loc[var].item())
          # Override objective values with survivor objectives
          survivorObjs = np.asarray(self.matingPopObjVals)
          if survivorObjs.ndim == 0:
            survivorObjs = np.asarray([survivorObjs])
          if survivorObjs.ndim == 1:
            rlzDict[self._objectiveVar[0]] = float(survivorObjs[i]) if survivorObjs.size > i else rlzDict.get(self._objectiveVar[0])
          else:
            for j, objName in enumerate(self._objectiveVar):
              rlzDict[objName] = float(survivorObjs[j, i]) if survivorObjs.shape[1] > i else rlzDict.get(objName)
          # Survivor fitness (single objective has a single fitness variable)
          fitnessNames = list(self.matingPopFitness.keys()) if isinstance(self.matingPopFitness, xr.Dataset) else []
          if fitnessNames:
            rlzDict['fitness'] = float(self.matingPopFitness[fitnessNames[0]].data[i])
          elif isinstance(self.matingPopFitness, dict):
            firstKey = next(iter(self.matingPopFitness))
            rlzDict['fitness'] = float(self.matingPopFitness[firstKey].data[i])
          # Track survivor age and batchId if available
          if self.matingPopAges is not None:
            rlzDict['age'] = self.matingPopAges[i]
          rlzDict['batchId'] = self.batchId
          # Constraints
          if hasattr(g, 'coords') and 'Constraint' in g.coords:
            for ind, consName in enumerate(g['Constraint'].values):
              rlzDict['ConstraintEvaluation_'+consName] = g.data[i,ind]
        self._updateSolutionExport(traj, rlzDict, acceptable, None)

    # decide what to do next
    if acceptable in ['accepted', 'first']:
      # record history
      bestRlz = {}
      if self._isMultiObjective:
        varList = self._solutionExport.getVars('input') + self._solutionExport.getVars('output') + list(self.toBeSampled.keys())
        bestRlz = dict((var,np.atleast_1d(self.multiBestPoint[var])) for var in set(varList) if var in list(self.toBeSampled.keys()))
        for i in range(len(self._objectiveVar)):
          bestRlz[self._objectiveVar[i]] = [item[i] for item in self.multiBestObjective]
        bestRlz['rank'] = self.multiBestRank
        bestRlz['CD'] = self.multiBestCD
        if len(self.multiBestConstraint) != 0: # No constraints
          for ind, consName in enumerate(self.multiBestConstraint.Constraint):
              bestRlz['ConstraintEvaluation_'+consName.values.tolist()] = self.multiBestConstraint[ind].values
        for ind, fitName in enumerate(list(self.multiBestFitness.keys())):
            bestRlz['FitnessEvaluation_'+ fitName] = self.multiBestFitness[fitName].data
        bestRlz.update(self.multiBestPoint)
      else:
        bestRlz[self._objectiveVar[0]] = self.multiBestObjective[0]
        bestRlz['fitness'] = self.bestFitness
        bestRlz.update(self.bestPoint)
      self._optPointHistory[traj].append((bestRlz, info))

  def _collectOptPoint(self, rlz, fitness, objectiveVal, g, population=None):
    """
      Collects the point (dict) from a realization
      @ In, rlz, xr.Dataset, realization data
      @ In, fitness, xr.Dataset, fitness values at each chromosome of the realization
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, g, xr.DataArray, constraint evaluation
      @ In, population, xr.DataArray, optional survivor population aligned with fitness/objective lists
      @ Out, point, dict, point used in this realization
    """
    selVars = list(self.toBeSampled.keys())
    # FIXED: Draw the best chromosome information from the survivor population
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

    if fitness is None:
      self.raiseAnError(RuntimeError, 'Fitness container is None while collecting optimal point.')

    objNames = self._objectiveVar if isinstance(self._objectiveVar, (list, tuple)) else [self._objectiveVar]

    if isinstance(fitness, xr.Dataset):
      fitnessScalar = np.asarray(fitness[objNames[0]].data, dtype=float)
    elif isinstance(fitness, xr.DataArray):
      fitnessScalar = np.asarray(fitness.data, dtype=float)
    elif isinstance(fitness, dict):
      fitData = fitness[objNames[0]]
      fitnessScalar = np.asarray(fitData.data if hasattr(fitData, 'data') else fitData, dtype=float)
    else:
      fitnessScalar = np.asarray(fitness, dtype=float)
    fitnessScalar = np.atleast_1d(fitnessScalar)

    gValues = np.atleast_2d(g.data)
    bestIdx = int(np.argmax(fitnessScalar))

    point = {gene: float(popMatrix[bestIdx, idx]) for idx, gene in enumerate(geneNames)}

    # Capture any additional variables (typically model outputs) associated with the best chromosome.
    extraVars = []
    if hasattr(self, '_solutionExport') and self._solutionExport is not None:
      extraVars = [var for var in self._solutionExport.getVars('output') if var not in point]
    def _match_realization_index():
      """
      _match_realization_index method.
      @ Out, None.
      """
      if not isinstance(rlz, xr.Dataset):
        return None
      try:
        genes_for_match = [gene for gene in geneNames if gene in rlz.data_vars]
        if not genes_for_match:
          return None
        rlzGeneArray = datasetToDataArray(rlz, genes_for_match)
      except Exception:
        return None
      rlzMatrix = np.asarray(rlzGeneArray.data, dtype=float)
      target = np.asarray([point[gene] for gene in rlzGeneArray.coords['Gene'].values], dtype=float)
      if rlzMatrix.ndim != 2 or target.size != rlzMatrix.shape[1]:
        return None
      matches = np.where(np.all(np.isclose(rlzMatrix, target[np.newaxis, :], rtol=1e-9, atol=1e-12), axis=1))[0]
      return int(matches[0]) if matches.size else None
    best_rlz_idx = _match_realization_index()
    candidate_idx = best_rlz_idx if best_rlz_idx is not None else bestIdx
    if extraVars and candidate_idx is not None:
      for var in extraVars:
        if var not in rlz.data_vars:
          continue
        data = rlz[var].data
        array = np.asarray(data)
        if array.ndim == 0:
          value = array.item()
        else:
          if candidate_idx >= array.shape[0]:
            continue
          value = np.take(array, candidate_idx, axis=0)
          if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
          elif isinstance(value, np.generic):
            value = value.item()
        point[var] = value

    gOfBest = {}
    if gValues.shape[1] > 0:
      for ind, consName in enumerate(g.coords['Constraint'].values):
        gOfBest[f'ConstraintEvaluation_{consName}'] = float(gValues[bestIdx, ind])

    objectiveArray = np.asarray(objectiveVal, dtype=float)
    if objectiveArray.ndim == 1:
      currentObj = float(objectiveArray[bestIdx])
    else:
      objectiveArray = np.atleast_2d(objectiveArray)
      if objectiveArray.shape[0] == 1:
        currentObj = float(objectiveArray[0, bestIdx])
      else:
        currentObj = float(objectiveArray[:, bestIdx][0])

    currentFit = float(fitnessScalar[bestIdx])

    if self.counter == 1:
      point.update(gOfBest)
      point['fitness'] = currentFit
      self.bestPoint = point
      self.bestFitness = currentFit
      self.multiBestObjective = np.array([currentObj])
      snapshot = {var: point[var] for var in geneNames}
      if hasattr(self, '_solutionExport') and self._solutionExport is not None:
        for outVar in self._solutionExport.getVars('output'):
          if outVar in point:
            snapshot[outVar] = point[outVar]
      snapshot.update(gOfBest)
      snapshot['fitness'] = currentFit
      snapshot['objective'] = currentObj
      snapshot['batchId'] = self.batchId
      if self.matingPopAges is not None and len(self.matingPopAges) > bestIdx:
        snapshot['age'] = self.matingPopAges[bestIdx]
      self._bestSnapshot = snapshot.copy()
    elif currentObj <= self.multiBestObjective[0] and currentFit >= self.bestFitness:
      point.update(gOfBest)
      point['fitness'] = currentFit
      self.bestPoint = point
      self.bestFitness = currentFit
      self.multiBestObjective = np.array([currentObj])
      snapshot = {var: point[var] for var in geneNames}
      if hasattr(self, '_solutionExport') and self._solutionExport is not None:
        for outVar in self._solutionExport.getVars('output'):
          if outVar in point:
            snapshot[outVar] = point[outVar]
      snapshot.update(gOfBest)
      snapshot['fitness'] = currentFit
      snapshot['objective'] = currentObj
      snapshot['batchId'] = self.batchId
      if self.matingPopAges is not None and len(self.matingPopAges) > bestIdx:
        snapshot['age'] = self.matingPopAges[bestIdx]
      self._bestSnapshot = snapshot.copy()

    return point

  def _collectOptPointMulti(self, rlz, population, rank, CD, objVal, fitness, constraintsV):
    """
      Collects the point (dict) from a realization
      @ In, population, Dataset, container containing the population
      @ In, rank, xr.DataArray, rank values at each chromosome of the realization
      @ In, CD (crowdingDistance), xr.DataArray, crowdingDistance values at each chromosome of the realization
      @ In, objVal, list, objective values at each chromosome of the realization
      @ In, fitness, dict, population fitness
      @ In, constraintsV, xr.DataArray, calculated contraints value
      @ Out, point, dict, point used in this realization
    """
    rankOneIDX = np.where(rank.data == 1)[0].tolist()
    optPoints = population[rankOneIDX]
    optObjVal = np.array(objVal)[:,rankOneIDX].T
    count = 0
    for i in list(fitness.keys()):
      data = fitness[i][rankOneIDX]
      if count == 0:
        fitSet = data.to_dataset(name = i)
      else:
        fitSet[i] = data
      count = count + 1
    optConstraintsV = constraintsV.data[rankOneIDX]
    optRank = rank.data[rankOneIDX]
    optCD = CD.data[rankOneIDX]

    optPointsDic = dict((var,np.array(optPoints)[:,i]) for i, var in enumerate(population.Gene.data))
    optConstNew = [list(y) for y in zip(*optConstraintsV)]
    if len(optConstNew) > 0:
      optConstNew = xr.DataArray(optConstNew,
                            dims=['Constraint','Evaluation'],
                            coords={'Constraint': [y.name for y in (self._constraintFunctions + self._impConstraintFunctions)],
                                    'Evaluation': np.arange(np.shape(optConstNew)[1])})

    self.multiBestPoint = optPointsDic
    self.multiBestFitness = fitSet
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
        self.multiBestObjective = np.array([bestObjective])
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
    self.hdsm = self._hdsm(old, new)
    converged = (self.hdsm >= self._convergenceCriteria['HDSM'])
    self.raiseADebug(self.convFormat.format(name='HDSM',
                                            conv=str(converged),
                                            got= self.hdsm,
                                            req=self._convergenceCriteria['HDSM']))

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
    """
    _GDp method.
    @ In, a, object, TODO.
    @ In, b, object, TODO.
    @ In, p, object, TODO.
    @ Out, None.
    """
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
    """
    _popDist method.
    @ In, ai, object, TODO.
    @ In, b, object, TODO.
    @ In, q, object, TODO.
    @ Out, None.
    """
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
    """
    _GD method.
    @ In, a, object, TODO.
    @ In, b, object, TODO.
    @ Out, None.
    """
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

  def _envelopeSize(self,a,b):
    """
    _envelopeSize method.
    @ In, a, object, TODO.
    @ In, b, object, TODO.
    @ Out, None.
    """
    r"""
      Compute hyper diagonal of envelope containing old and new population
      @ In, a, np.array, old population A
      @ In, b, np.array, new population B
      @ Out, _GD, float, the generational distance $\frac{1}{n_A} \max_{i \in A}min_{b \in B} dist(ai,B)$
    """
    aLenght = np.abs(np.amax(a, axis=0) -  np.amin(a, axis=0))
    bLenght = np.abs(np.amax(b, axis=0) -  np.amin(b, axis=0))
    sides = np.amax(np.stack([aLenght, bLenght], axis=0), axis=0).tolist()
    hyperDiagonal = mathUtils.hyperdiagonal(sides)
    return hyperDiagonal

  def _hdsm(self, a, b):
    """
      Hausdorff Distance Similarity Measure for generation convergence
      @ In, a, np.array, old population A
      @ In, b, np.array, new population B
      @ Out, _hdsm, float, average Hausdorff distance
    """
    normFactor = self._envelopeSize(a, b)
    ahd = self._ahd(a,b)
    if mathUtils.compareFloats(ahd, 0.0, 1e-14):
      return 1.
    if mathUtils.compareFloats(normFactor, 0.0, 1e-14):
      # the envelope has a zero size (=> populations are
      # composed by the same genes (all the same numbers
      # => minimum == maximum within the population
      return 1.
    return  1. - ahd / normFactor

  def _checkConvHypervolume(self, traj, **kwargs):
      """
      Checks convergence based on relative hypervolume improvement.
      @ In, traj, int, trajectory identifier
      @ In, kwargs, dict, must contain 'new' and 'old' populations
      @ Out, converged, bool, convergence state
      """
      if len(self._optPointHistory[traj]) < 2:
          return False

      # Extract current Pareto front (rank 1)
      rank1_indices = np.where(self.matingPopRanks.data == 1)[0]
      if len(rank1_indices) == 0:
          return False

      current_front = []
      for idx in rank1_indices:
          point = [self.matingPopObjVals[j][idx] for j in range(len(self._objectiveVar))]
          current_front.append(point)

      # Extract previous Pareto front from history
      prev_opt, _ = self._optPointHistory[traj][-2] if len(self._optPointHistory[traj]) >= 2 else (None, None)
      if prev_opt is None:
          return False

      prev_rank1_indices = np.where(np.array(prev_opt['rank']) == 1)[0]
      prev_front = []
      for idx in prev_rank1_indices:
          point = [prev_opt[self._objectiveVar[j]][idx] for j in range(len(self._objectiveVar))]
          prev_front.append(point)

      # Calculate hypervolumes
      # Reference point: slightly worse than nadir
      all_points = current_front + prev_front
      nadir = [max(p[i] for p in all_points) for i in range(len(self._objectiveVar))]
      reference = [n * 1.1 for n in nadir]

      current_hv = self._computeHypervolume(current_front, reference)
      prev_hv = self._computeHypervolume(prev_front, reference)

      # Store for tracking
      if not hasattr(self, '_hvHistory'):
          self._hvHistory = {}
      if traj not in self._hvHistory:
          self._hvHistory[traj] = []
      self._hvHistory[traj].append(current_hv)

      # Check relative improvement
      if prev_hv == 0:
          rel_improvement = float('inf')
      else:
          rel_improvement = abs(current_hv - prev_hv) / prev_hv

      converged = rel_improvement < self._convergenceCriteria.get('hypervolume', 0.01)

      self.raiseADebug(self.convFormat.format(
          name='Hypervolume',
          conv=str(converged),
          got=rel_improvement,
          req=self._convergenceCriteria.get('hypervolume', 0.01)
      ))

      return converged

  def _computeHypervolume(self, front, reference):
      """
      Compute hypervolume indicator for a Pareto front.
      Uses WFG algorithm (Walking Fish Group).

      @ In, front, list of lists, Pareto front points
      @ In, reference, list, reference point (must be dominated by all points)
      @ Out, hv, float, hypervolume value
      """
      if not front:
          return 0.0

      n_objectives = len(front[0])

      # 2D case: use efficient algorithm
      if n_objectives == 2:
          return self._hypervolume2D(front, reference)

      # 3D case: use 3D algorithm
      elif n_objectives == 3:
          return self._hypervolume3D(front, reference)

      # Higher dimensions: use recursive WFG
      else:
          return self._hypervolumeWFG(front, reference)

  def _hypervolume2D(self, front, reference):
      """
      Efficient 2D hypervolume calculation.
      @ In, front, list of lists, 2D points
      @ In, reference, list, reference point [r1, r2]
      @ Out, hv, float, hypervolume
      """
      # Sort by first objective
      sorted_front = sorted(front, key=lambda p: p[0])

      hv = 0.0
      prev_x = reference[0]

      for point in sorted_front:
          width = prev_x - point[0]
          height = reference[1] - point[1]
          hv += width * height
          prev_x = point[0]

      return hv

  def _hypervolume3D(self, front, reference):
      """
      Efficient 3D hypervolume calculation.
      @ In, front, list of lists, 3D points
      @ In, reference, list, reference point
      @ Out, hv, float, hypervolume
      """
      # Sort by first objective
      sorted_front = sorted(front, key=lambda p: p[0])

      hv = 0.0

      for i, point in enumerate(sorted_front):
          # Slice for this point
          x_extent = reference[0] - point[0]

          # Project to 2D for remaining objectives
          remaining_front = [p[1:] for p in sorted_front[:i+1]]
          remaining_ref = reference[1:]

          # 2D hypervolume for this slice
          slice_hv = self._hypervolume2D(remaining_front, remaining_ref)

          hv += x_extent * slice_hv

      return hv

  def _hypervolumeWFG(self, front, reference):
      """
      WFG algorithm for n-dimensional hypervolume (n > 3).
      @ In, front, list of lists, n-D points
      @ In, reference, list, reference point
      @ Out, hv, float, hypervolume
      """
      # For simplicity, use approximation for high dimensions
      # Or implement full WFG algorithm
      # Here we use a simplified recursive approach

      if len(reference) == 1:
          return reference[0] - min(p[0] for p in front)

      # Sort by last objective
      sorted_front = sorted(front, key=lambda p: p[-1])

      hv = 0.0
      for i, point in enumerate(sorted_front):
          # Project to lower dimension
          lower_dim_front = [p[:-1] for p in sorted_front[:i+1]]
          lower_dim_ref = reference[:-1]

          # Recursive call
          lower_hv = self._hypervolumeWFG(lower_dim_front, lower_dim_ref)

          # Add contribution
          height = reference[-1] - point[-1]
          hv += height * lower_hv

      return hv

  def _checkConvSpread(self, traj, **kwargs):
    """
    Checks convergence based on spread (diversity) metric from Deb et al. (2002).
    @ In, traj, int, trajectory identifier
    @ In, kwargs, dict, parameters
    @ Out, converged, bool, convergence state
    """
    # Need at least rank-1 front
    if not hasattr(self, 'matingPopRanks'):
        return False

    rank1_indices = np.where(self.matingPopRanks.data == 1)[0]
    if len(rank1_indices) < 3:
        return False  # Need at least 3 points for meaningful spread

    # Extract rank-1 objective values
    front = []
    for idx in rank1_indices:
        point = [self.matingPopObjVals[j][idx] for j in range(len(self._objectiveVar))]
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

    n_obj = len(front[0])

    # Calculate Euclidean distances between consecutive solutions
    distances = []

    for obj in range(n_obj):
        # Sort front by this objective
        sorted_indices = sorted(range(n), key=lambda i: front[i][obj])
        sorted_front = [front[i] for i in sorted_indices]

        # Distance between consecutive points
        for i in range(len(sorted_front) - 1):
            dist = np.linalg.norm(np.array(sorted_front[i+1]) - np.array(sorted_front[i]))
            distances.append(dist)

    if len(distances) == 0:
        return 0.0

    # Extreme distances (distance to boundary solutions)
    # For simplicity, use distance from first to ideal and last to nadir
    ideal = [min(p[i] for p in front) for i in range(n_obj)]
    nadir = [max(p[i] for p in front) for i in range(n_obj)]

    sorted_by_first_obj = sorted(front, key=lambda p: p[0])
    d_f = np.linalg.norm(np.array(sorted_by_first_obj[0]) - np.array(ideal))
    d_l = np.linalg.norm(np.array(sorted_by_first_obj[-1]) - np.array(nadir))

    # Mean distance
    d_mean = np.mean(distances)

    if d_mean == 0:
        return 0.0

    # Spread calculation
    numerator = d_f + d_l + sum(abs(d - d_mean) for d in distances)
    denominator = d_f + d_l + (len(distances)) * d_mean

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
    rank1_indices = np.where(self.matingPopRanks.data == 1)[0]
    current_front = []
    for idx in rank1_indices:
        point = [self.matingPopObjVals[j][idx] for j in range(len(self._objectiveVar))]
        current_front.append(point)

    # Previous front
    prev_opt, _ = self._optPointHistory[traj][-2]
    prev_rank1 = np.where(np.array(prev_opt['rank']) == 1)[0]
    prev_front = []
    for idx in prev_rank1:
        point = [prev_opt[self._objectiveVar[j]][idx] for j in range(len(self._objectiveVar))]
        prev_front.append(point)

    # Compute MS
    current_ms = self._computeMaxSpread(current_front)
    prev_ms = self._computeMaxSpread(prev_front)

    # Check relative change
    if prev_ms == 0:
        rel_change = float('inf')
    else:
        rel_change = abs(current_ms - prev_ms) / prev_ms

    converged = rel_change < self._convergenceCriteria.get('maxSpread', 0.05)

    self.raiseADebug(self.convFormat.format(
        name='MaxSpread',
        conv=str(converged),
        got=rel_change,
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

    n_obj = len(front[0])
    ranges = []

    for obj in range(n_obj):
        obj_values = [p[obj] for p in front]
        ranges.append(max(obj_values) - min(obj_values))

    ms = np.sqrt(sum(r**2 for r in ranges))

    return ms

  def _checkConvRank1Ratio(self, traj, **kwargs):
    """
    Checks convergence based on percentage of population in rank 1.
    @ In, traj, int, trajectory identifier
    @ In, kwargs, dict, parameters
    @ Out, converged, bool, convergence state
    """
    if not hasattr(self, 'matingPopRanks'):
        return False

    # Count rank-1 solutions
    rank1_count = np.sum(self.matingPopRanks.data == 1)
    ratio = rank1_count / self._populationSize

    # Track history
    if not hasattr(self, '_rank1History'):
        self._rank1History = {}
    if traj not in self._rank1History:
        self._rank1History[traj] = []
    self._rank1History[traj].append(ratio)

    # Converged if ratio high and stable
    threshold = self._convergenceCriteria.get('rank1Ratio', 0.5)
    stable_generations = 3  # Require stability

    if len(self._rank1History[traj]) < stable_generations:
        converged = False
    else:
        recent_ratios = self._rank1History[traj][-stable_generations:]
        all_above_threshold = all(r >= threshold for r in recent_ratios)
        variation = max(recent_ratios) - min(recent_ratios)
        converged = all_above_threshold and variation < 0.1

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
    final_rlz = dict(rlz)
    # carry over the best decision variables and constraint evaluations
    best_idx = self._matchBestChromosomeIndex()
    if best_idx is None:
      best_idx = self._inferBestIndexFromObjective()

    if self._bestSnapshot:
      for var in self.toBeSampled:
        if var in self._bestSnapshot:
          final_rlz[var] = self._bestSnapshot[var]
      for key, val in self._bestSnapshot.items():
        if key.startswith('ConstraintEvaluation_'):
          final_rlz[key] = val
      if hasattr(self, '_solutionExport') and self._solutionExport is not None:
        for outVar in self._solutionExport.getVars('output'):
          if outVar in self._bestSnapshot:
            final_rlz[outVar] = self._bestSnapshot[outVar]
      if 'objective' in self._bestSnapshot:
        final_rlz[self._objectiveVar[0]] = self._bestSnapshot['objective']
      if 'fitness' in self._bestSnapshot:
        final_rlz['fitness'] = self._bestSnapshot['fitness']
      if 'age' in self._bestSnapshot:
        final_rlz['age'] = self._bestSnapshot['age']
      if 'batchId' in self._bestSnapshot:
        final_rlz['batchId'] = self._bestSnapshot['batchId']
    elif isinstance(self.bestPoint, dict):
      for key, val in self.bestPoint.items():
        if key.startswith('ConstraintEvaluation_'):
          final_rlz[key] = val

    # objective value and fitness are stored separately from the population
    if self.multiBestObjective is not None and 'objective' not in (self._bestSnapshot or {}):
      final_rlz[self._objectiveVar[0]] = float(np.atleast_1d(self.multiBestObjective)[0])
    if self.bestFitness is not None and 'fitness' not in (self._bestSnapshot or {}):
      final_rlz['fitness'] = float(np.atleast_1d(self.bestFitness)[0])

    # Overwrite decision variables using the survivor record to keep them
    # consistent with the recorded objective/fitness.
    if best_idx is not None and hasattr(self, 'matingPopInputs') and self.matingPopInputs is not None:
      for var in self.toBeSampled:
        try:
          arr = np.asarray(self.matingPopInputs.sel(Gene=var))
          final_rlz[var] = float(np.atleast_1d(arr)[best_idx])
        except Exception:
          if var in self.bestPoint:
            final_rlz[var] = self.bestPoint[var]
    elif isinstance(self.bestPoint, dict):
      for var in self.toBeSampled:
        if var in self.bestPoint:
          final_rlz[var] = self.bestPoint[var]

    # include survivor age information if we can match the stored best point
    if best_idx is not None and self.matingPopAges and 'age' not in (self._bestSnapshot or {}):
      final_rlz['age'] = self.matingPopAges[best_idx]
    elif isinstance(self.popAge, list) and self.popAge:
      final_rlz['age'] = self.popAge[0]
    else:
      final_rlz['age'] = 0

    final_rlz['batchId'] = self.batchId
    return final_rlz

  def _matchBestChromosomeIndex(self):
    """
      Locate the index of the stored best chromosome within the current mating
      population so we can recover metadata such as age for the final export.
    """
    if not hasattr(self, 'matingPopInputs') or self.matingPopInputs is None:
      return None
    genes = list(self.toBeSampled.keys())
    try:
      pop_da = self.matingPopInputs.sel(Gene=genes).transpose('chromosome', 'Gene')
    except Exception:
      pop_da = self.matingPopInputs.transpose('chromosome', 'Gene')
    pop_matrix = np.asarray(pop_da)
    if pop_matrix.ndim != 2 or not pop_matrix.size:
      return None
    target_vals = []
    for gene in genes:
      if gene not in self.bestPoint:
        return None
      target_vals.append(float(self.bestPoint[gene]))
    target = np.asarray(target_vals, dtype=float)
    if np.any(np.isnan(target)):
      return None
    matches = np.isclose(pop_matrix, target[np.newaxis, :], rtol=1e-9, atol=1e-12)
    hit = np.where(np.all(matches, axis=1))[0]
    return int(hit[0]) if hit.size else None

  def _inferBestIndexFromObjective(self):
    """
      Fallback helper used when the stored best-point keys no longer match the
      survivor population. Selects the chromosome with the minimal objective
      value from the current mating population.
    """
    if self.matingPopObjVals is None:
      return None
    try:
      obj_values = np.asarray(self.matingPopObjVals, dtype=float)
    except Exception:
      return None
    if obj_values.ndim == 1 and obj_values.size:
      return int(np.argmin(obj_values))
    if obj_values.ndim > 1 and obj_values.size:
      return int(np.argmin(obj_values[0]))
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
      if self.matingPopAges is not None and len(np.atleast_1d(self.matingPopAges)) > 0:
        ageVal = np.atleast_1d(self.matingPopAges)[0]
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
