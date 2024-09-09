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
  Updated Sepember,17,2023
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi, Junyung Kim
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
                                                                | _collectOptValue               |
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
                                                                | bestObjective                      |
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
                                                                | _resolveNewGenerationMulti         |
                                                                | _solutionExportUtilityUpdate       |
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
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..utils import mathUtils, InputData, InputTypes, frontUtils
from ..utils.gaUtils import dataArrayToDict, datasetToDataArray
from .RavenSampled import RavenSampled
from .parentSelectors.parentSelectors import returnInstance as parentSelectionReturnInstance
from .crossOverOperators.crossovers import returnInstance as crossoversReturnInstance
from .mutators.mutators import returnInstance as mutatorsReturnInstance
from .survivorSelectors.survivorSelectors import returnInstance as survivorSelectionReturnInstance
from .survivorSelection import survivorSelection as survivorSelectionProcess
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
    self.bestObjective = None                                    # objective value of the best solution found
    self.multiBestPoint = None                                   # the best solutions (chromosomes) found among population in a specific batchId
    self.multiBestFitness = None                                 # fitness values of the best solutions found
    self.multiBestObjective = None                               # objective values of the best solutions found
    self.multiBestConstraint = None                              # constraint values of the best solutions found
    self.multiBestRank = None                                    # rank values of the best solutions found
    self.multiBestCD = None                                      # crowding distance (CD) values of the best solutions found
    self.objectiveVal = None                                     # objective values of solutions
    self._populationSize = None                                  # number of population size
    self._parentSelectionType = None                             # type of the parent selection process chosen
    self._parentSelectionInstance = None                         # instance of the parent selection process chosen
    self._nParents = None                                        # number of parents
    self._kSelection = None                                      # number of chromosomes selected for tournament selection
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
    specs.description = r"""The \xmlNode{GeneticAlgorithm} is a metaheuristic optimization technique inspired by the principles
                            of natural selection and genetics. Introduced by John Holland in the 1960s, GA mimics the process of
                            biological evolution to solve complex optimization and search problems. They operate by maintaining a population of
                            potential solutions represented as as arrays of fixed length variables (genes), and each such array is called a chromosome.
                            These solutions undergo iterative refinement through processes such as mutation, crossover, and survivor selection. Mutation involves randomly altering certain genes within
                            individual solutions, introducing diversity into the population and enabling exploration of new regions in the solution space.
                            Crossover, on the other hand, mimics genetic recombination by exchanging genetic material between two parent solutions to create
                            offspring with combined traits. Survivor selection determines which solutions will advance to the next generation based on
                            their fitness—how well they perform in solving the problem at hand. Solutions with higher fitness scores are more likely to
                            survive and reproduce, passing their genetic material to subsequent generations. This iterative process continues
                            until a stopping criterion is met, typically when a satisfactory solution is found or after a predetermined number of generations.
                            More information can be found in:\\\\

                            Holland, John H. "Genetic algorithms." Scientific American 267.1 (1992): 66-73.\\\\

                            Non-dominated Sorting Genetic Algorithm II (NSGA-II) is a variant of GAs designed for multiobjective optimization problems.
                            NSGA-II extends traditional GAs by incorporating a ranking-based approach and crowding distance estimation to maintain a diverse set of
                            non-dominated (Pareto-optimal) solutions. This enables NSGA-II to efficiently explore trade-offs between conflicting objectives,
                            providing decision-makers with a comprehensive view of the problem's solution space. More information about NSGA-II can be found in:\\\\

                            Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE transactions on evolutionary computation 6.2 (2002): 182-197.\\\\

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
    # # Constraint Handling
    # constraintHandling = InputData.parameterInputFactory('constraintHandling', strictMode=True,
    #     contentType=InputTypes.StringType,
    #     printPriority=108,
    #     descr=r"""a node indicating whether GA will handle constraints hardly or softly.""")
    # GAparams.addSub(constraintHandling)

    # Parent Selection
    parentSelection = InputData.parameterInputFactory('parentSelection', strictMode=True,
        contentType=InputTypes.StringType,
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
    crossover.addParam("type", InputTypes.StringType, True,
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
    mutation.addParam("type", InputTypes.StringType, True,
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
        contentType=InputTypes.StringType,
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
        descr=r"""a subnode containing the implemented fitness functions.You can choose one of the fitness options listed below:
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
                        $fitness = \left\{\begin{matrix} -obj & g_{j}(x)\geq 0 \; \forall j \\ -obj_{worst}- \Sigma_{j=1}^{J}<g_j(x)> & otherwise \\ \end{matrix}\right$\\
                  \end{itemize} """)
    fitness.addParam("type", InputTypes.StringType, True,
                     descr=r"""[invLin, logistic, feasibleFirst]""")
    objCoeff = InputData.parameterInputFactory('a', strictMode=True,
        contentType=InputTypes.FloatListType,
        printPriority=108,
        descr=r""" a: coefficient of objective function.""")
    fitness.addSub(objCoeff)
    penaltyCoeff = InputData.parameterInputFactory('b', strictMode=True,
        contentType=InputTypes.FloatListType,
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
    new['rank'] = 'It refers to the sorting of solutions into non-dominated fronts based on their Pareto dominance relationships'
    new['CD'] = 'It measures the density of solutions within each front to guide the selection of diverse individuals for the next generation'
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

    if len(self._objectiveVar) >=2 and self._parentSelectionType != 'tournamentSelection':
      self.raiseAnError(IOError, f'tournamentSelection in <parentSelection> is a sole mechanism supportive in multi-objective optimization.')

    ####################################################################################
    # reproduction node                                                                #
    ####################################################################################
    reproductionNode = gaParamsNode.findFirst('reproduction')
    self._nParents = int(np.ceil(1/2 + np.sqrt(1+4*self._populationSize)/2))
    self._nChildren = int(2*comb(self._nParents,2))

    ####################################################################################
    # k-Selection node                                                                #
    ####################################################################################
    if reproductionNode.findFirst('kSelection') is None:
      self._kSelection = 3 # Default value is set to 3.
    else:
      self._kSelection = reproductionNode.findFirst('kSelection').value

    ####################################################################################
    # crossover node                                                                   #
    ####################################################################################
    crossoverNode = reproductionNode.findFirst('crossover')
    self._crossoverType = crossoverNode.parameterValues['type']
    if self._crossoverType not in ['onePointCrossover','twoPointsCrossover','uniformCrossover']:
      self.raiseAnError(IOError, f'Currently constrained Genetic Algorithms only support onePointCrossover, twoPointsCrossover and uniformCrossover as a crossover, whereas provided crossover is {self._crossoverType}')
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
    if self._mutationType not in ['swapMutator','scrambleMutator','inversionMutator','bitFlipMutator','randomMutator']:
      self.raiseAnError(IOError, f'Currently constrained Genetic Algorithms only support swapMutator, scrambleMutator, inversionMutator, bitFlipMutator, and randomMutator as a mutator, whereas provided mutator is {self._mutationType}')
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
    if self._survivorSelectionType not in ['ageBased','fitnessBased','rankNcrowdingBased']:
      self.raiseAnError(IOError, f'Currently constrained Genetic Algorithms only support ageBased, fitnessBased, and rankNcrowdingBased as a survivorSelector, whereas provided survivorSelector is {self._survivorSelectionType}')
    if len(self._objectiveVar) == 1 and self._survivorSelectionType == 'rankNcrowdingBased':
      self.raiseAnError(IOError, f'(rankNcrowdingBased) in <survivorSelection> only supports when the number of objective in <objective> is bigger than two. ')
    if len(self._objectiveVar) > 1 and self._survivorSelectionType != 'rankNcrowdingBased':
      self.raiseAnError(IOError, f'The only option supported in <survivorSelection> for Multi-objective Optimization is (rankNcrowdingBased).')

    ####################################################################################
    # fitness node                                                                     #
    ####################################################################################
    fitnessNode = gaParamsNode.findFirst('fitness')
    self._fitnessType = fitnessNode.parameterValues['type']

    ####################################################################################
    # constraint node                                                                  #
    ####################################################################################
    # TODO: @mandd, please explore the possibility to convert the logistic fitness into a constrained optimization fitness.
    if 'Constraint' in self.assemblerObjects and self._fitnessType not in ['invLinear','logistic', 'feasibleFirst']:
      self.raiseAnError(IOError, f'Currently constrained Genetic Algorithms only support invLinear, logistic, and feasibleFirst as a fitness, whereas provided fitness is {self._fitnessType}')
    self._expConstr = self.assemblerObjects['Constraint'] if 'Constraint' in self.assemblerObjects else None
    self._impConstr = self.assemblerObjects['ImplicitConstraint'] if 'ImplicitConstraint' in self.assemblerObjects else None
    if self._expConstr != None and self._impConstr != None:
      self._numOfConst = len([ele for ele in self._expConstr if ele != 'Functions' if ele !='External']) + len([ele for ele in self._impConstr if ele != 'Functions' if ele !='External'])
    elif self._expConstr == None and self._impConstr != None:
      self._numOfConst = len([ele for ele in self._impConstr if ele != 'Functions' if ele !='External'])
    elif self._expConstr != None and self._impConstr == None:
      self._numOfConst = len([ele for ele in self._expConstr if ele != 'Functions' if ele !='External'])
    else:
      self._numOfConst = 0
    if (self._expConstr != None) and (self._impConstr != None) and (self._penaltyCoeff != None):
      if len(self._penaltyCoeff) != len(self._objectiveVar) * self._numOfConst:
        self.raiseAnError(IOError, f'The number of penaltyCoeff. in <b> should be identical with the number of objective in <objective> and the number of constraints (i.e., <Constraint> and <ImplicitConstraint>)')
    else:
      pass
    self._objCoeff = fitnessNode.findFirst('a').value if fitnessNode.findFirst('a') is not None else None
    #NOTE the code lines below are for 'feasibleFirst' temperarily. It will be generalized for invLinear as well.
    if self._fitnessType == 'feasibleFirst':
      if self._numOfConst != 0 and fitnessNode.findFirst('b') is not None:
        self._penaltyCoeff = fitnessNode.findFirst('b').value
        self._objCoeff = fitnessNode.findFirst('a').value
      elif self._numOfConst == 0 and fitnessNode.findFirst('b') is not None:
        self.raiseAnError(IOError, f'The number of constraints used are 0 but there are penalty coefficieints')
      elif self._numOfConst != 0 and fitnessNode.findFirst('b') is None:
        self._penaltyCoeff = [1] * self._numOfConst * len(self._objectiveVar) #list(np.repeat(1, self._numOfConst * len(self._objectiveVar))) #NOTE if penaltyCoeff is not provided, then assume they are all 1.
        self._objCoeff = fitnessNode.findFirst('a').value if fitnessNode.findFirst('a') is not None else [1] * len(self._objectiveVar) #list(np.repeat(
      else:
        self._penaltyCoeff = [0] * len(self._objectiveVar) #list(np.repeat(0, len(self._objectiveVar)))
        self._objCoeff = [1] * len(self._objectiveVar)
    else:
      self._penaltyCoeff = fitnessNode.findFirst('b').value if fitnessNode.findFirst('b') is not None else None
      self._objCoeff = fitnessNode.findFirst('a').value if fitnessNode.findFirst('a') is not None else None
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

  def singleObjectiveConstraintHandling(self, info, rlz):
    """
      This function handles the constraints for a single objective optimization.
      @ In, info, dict, dictionary containing information about the run
      @ In, rlz, dict, dictionary containing the results of the run
      @ Out, None
    """
    traj = info['traj']
    for t in self._activeTraj[1:]:
      self._closeTrajectory(t, 'cancel', 'Currently GA is single trajectory', 0)
    self.incrementIteration(traj)

    offSprings = datasetToDataArray(rlz, list(self.toBeSampled))
    objectiveVal = list(np.atleast_1d(rlz[self._objectiveVar[0]].data))

    # Collect parameters that the constraints functions need (neglecting the default params such as inputs and objective functions)
    constraintData = {}
    if self._constraintFunctions or self._impConstraintFunctions:
      params = []
      for y in (self._constraintFunctions + self._impConstraintFunctions):
        params += y.parameterNames()
      for p in list(set(params) -set([self._objectiveVar[0]]) -set(list(self.toBeSampled.keys()))):
        constraintData[p] = list(np.atleast_1d(rlz[p].data))
    # Compute constraint function g_j(x) for all constraints (j = 1 .. J) and all x's (individuals) in the population
    g0 = np.zeros((np.shape(offSprings)[0],len(self._constraintFunctions)+len(self._impConstraintFunctions)))

    g = xr.DataArray(g0,
                      dims=['chromosome','Constraint'],
                      coords={'chromosome':np.arange(np.shape(offSprings)[0]),
                              'Constraint':[y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]})
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
                                              constraintNum = self._numOfConst,
                                              type=self._minMax)

    self._collectOptPoint(rlz, offSpringFitness, objectiveVal, g)
    self._resolveNewGeneration(traj, rlz, objectiveVal, offSpringFitness, g, info)
    return traj, g, objectiveVal, offSprings, offSpringFitness

  def multiObjectiveConstraintHandling(self, info, rlz):
    """
      This function handles the constraints for a multi-objective optimization.
      @ In, info, dict, dictionary containing information about the run
      @ In, rlz, dict, dictionary containing the results of the run
      @ Out, None
    """
    traj = info['traj']
    for t in self._activeTraj[1:]:
      self._closeTrajectory(t, 'cancel', 'Currently GA is single trajectory', 0)
    self.incrementIteration(traj)

    objectiveVal = []
    offSprings = datasetToDataArray(rlz, list(self.toBeSampled))
    for i in range(len(self._objectiveVar)):
      objectiveVal.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))

    # Collect parameters that the constraints functions need (neglecting the default params such as inputs and objective functions)
    constraintData = {}
    if self._constraintFunctions or self._impConstraintFunctions:
      params = []
      for y in (self._constraintFunctions + self._impConstraintFunctions):
        params += y.parameterNames()
      for p in list(set(params) -set(self._objectiveVar) -set(list(self.toBeSampled.keys()))):
        constraintData[p] = list(np.atleast_1d(rlz[p].data))
    # Compute constraint function g_j(x) for all constraints (j = 1 .. J) and all x's (individuals) in the population
    g0 = np.zeros((np.shape(offSprings)[0],len(self._constraintFunctions)+len(self._impConstraintFunctions)))

    g = xr.DataArray(g0,
                      dims=['chromosome','Constraint'],
                      coords={'chromosome':np.arange(np.shape(offSprings)[0]),
                              'Constraint':[y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]})

    for index,individual in enumerate(offSprings):
      newOpt = individual
      objOpt = dict(zip(self._objectiveVar,
                        list(map(lambda x:-1 if x=="max" else 1 , self._minMax))))
      opt = dict(zip(self._objectiveVar, [item[index] for item in objectiveVal]))
      opt = {k: objOpt[k]*opt[k] for k in opt}
      for p, v in constraintData.items():
        opt[p] = v[index]

      for constIndex, constraint in enumerate(self._constraintFunctions + self._impConstraintFunctions):
        if constraint in self._constraintFunctions:
          g.data[index, constIndex] = self._handleExplicitConstraints(newOpt, constraint)
        else:
          g.data[index, constIndex] = self._handleImplicitConstraints(newOpt, opt, constraint)

    offSpringFitness = self._fitnessInstance(rlz,
                                             objVar=self._objectiveVar,
                                             a=self._objCoeff,
                                             b=self._penaltyCoeff,
                                             constraintFunction=g,
                                             constraintNum = self._numOfConst,
                                             type = self._minMax)
    return traj, g, objectiveVal, offSprings, offSpringFitness

  #########################################################################################################
  # Run Methods                                                                                           #
  #########################################################################################################

  #########################################################################################################
  # Developer note:
  # Each algorithm step is indicated by a number followed by the generation number
  # e.g., '0 @ n-1' refers to step 0 for generation n-1 (i.e., previous generation)
  # for more details refer to GRP-Raven-development/Disceret_opt channel on MS Teams.
  #########################################################################################################

  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      This is called by localFinalizeActualSampling, and hence should contain the main skeleton.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, xr.Dataset, new batched realizations
      @ Out, None
    """

    info['step'] = self.counter

    # 0 @ n-1: Survivor Selection from previous iteration (children+parents merging from previous generation)
    # 0.1 @ n-1: fitnessCalculation(rlz): Perform fitness calculation for newly obtained children (rlz)

    objInd = int(len(self._objectiveVar)>1) + 1 #if len(self._objectiveVar) == 1 else 2
    constraintFuncs: dict = {1: GeneticAlgorithm.singleObjectiveConstraintHandling, 2: GeneticAlgorithm.multiObjectiveConstraintHandling}
    const = constraintFuncs.get(objInd, GeneticAlgorithm.singleObjectiveConstraintHandling)
    traj, g, objectiveVal, offSprings, offSpringFitness = const(self, info, rlz)


    # 0.2@ n-1: Survivor selection(rlz): Update population container given obtained children
    if self._activeTraj:
      survivorSelectionFuncs: dict = {1: survivorSelectionProcess.singleObjSurvivorSelect, 2: survivorSelectionProcess.multiObjSurvivorSelect}
      survivorSelection = survivorSelectionFuncs.get(objInd, survivorSelectionProcess.singleObjSurvivorSelect)
      survivorSelection(self, info, rlz, traj, offSprings, offSpringFitness, objectiveVal, g)

      # 1 @ n: Parent selection from population
      # Pair parents together by indexes
      parents = self._parentSelectionInstance(self.population,
                                              variables=list(self.toBeSampled),
                                              fitness = self.fitness,
                                              kSelection = self._kSelection,
                                              nParents=self._nParents,
                                              rank = self.rank,
                                              crowdDistance = self.crowdingDistance,
                                              objVal = self._objectiveVar
                                              )

      # 2 @ n: Crossover from set of parents
      # Create childrenCoordinates (x1,...,xM)
      childrenXover = self._crossoverInstance(parents=parents,
                                              variables=list(self.toBeSampled),
                                              crossoverProb=self._crossoverProb,
                                              points=self._crossoverPoints)

      # 3 @ n: Mutation
      # Perform random directly on childrenCoordinates
      childrenMutated = self._mutationInstance(offSprings=childrenXover,
                                               distDict=self.distDict,
                                               locs=self._mutationLocs,
                                               mutationProb=self._mutationProb,
                                               variables=list(self.toBeSampled))

      # 4 @ n: repair/replacement
      # Repair should only happen if multiple genes in a single chromosome have the same values (),
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
      # Submit children coordinates (x1,...,xm), i.e., self.childrenCoordinates
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
    self.hdsm = np.NaN
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

  def _solutionExportUtilityUpdate(self, traj, rlz, fitness, g, acceptable):
    """
      Utility method to update the solution export
      @ In, traj, int, trajectory for this new point
      @ In, rlz, dict, realized realization
      @ In, fitness, xr.DataArray, fitness values at each chromosome of the realization
      @ In, g, xr.DataArray, the constraint evaluation function
      @ In, acceptable, str, 'accetable' status (i.e. first, accepted, rejected, final)
      @ Out, None
    """
    for i in range(rlz.sizes['RAVEN_sample_ID']):
      varList = self._solutionExport.getVars('input') + self._solutionExport.getVars('output') + list(self.toBeSampled.keys())
      rlzDict = dict((var,np.atleast_1d(rlz[var].data)[i]) for var in set(varList) if var in rlz.data_vars)
      rlzDict[self._objectiveVar] = np.atleast_1d(rlz[self._objectiveVar].data)[i]
      rlzDict['fitness'] = np.atleast_1d(fitness.data)[i]
      for ind, consName in enumerate(g['Constraint'].values):
        rlzDict['ConstraintEvaluation_'+consName] = g[i,ind]
      self._updateSolutionExport(traj, rlzDict, acceptable, None)

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
        rlzDict['fitness'] = np.atleast_1d(fitness.to_array()[:,i])
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
    self.raiseADebug(f'Trajectory {traj} iteration {info["step"]} resolving new state ...')
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    acceptable = 'accepted' if self.counter > 1 else 'first'
    old = self.population
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    if converged:
      self._closeTrajectory(traj, 'converge', 'converged', self.multiBestObjective)
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
      self.raiseADebug("### rlz.sizes['RAVEN_sample_ID'] = {}".format(rlz.sizes['RAVEN_sample_ID']))
      self.raiseADebug("### self.population.shape is {}".format(self.population.shape))
      for i in range(rlz.sizes['RAVEN_sample_ID']):
        varList = self._solutionExport.getVars('input') + self._solutionExport.getVars('output') + list(self.toBeSampled.keys())
        # rlzDict = dict((var,np.atleast_1d(rlz[var].data)[i]) for var in set(varList) if var in rlz.data_vars)
        rlzDict = dict((var,self.population.data[i][j]) for j, var in enumerate(self.population.Gene.data))
        rlzDict.update(dict((var,objVal.data[i][j]) for j, var in enumerate(objVal.obj.data)))
        rlzDict['batchId'] = rlz['batchId'].data[i]
        for j in range(len(self._objectiveVar)):
          rlzDict[self._objectiveVar[j]] = objVal.data[i][j]
        rlzDict['rank'] = np.atleast_1d(self.rank.data)[i]
        rlzDict['CD'] = np.atleast_1d(self.crowdingDistance.data)[i]
        for ind, fitName in enumerate(list(self.fitness.keys())):
          rlzDict['FitnessEvaluation_'+fitName] = self.fitness[fitName].data[i]
        for ind, consName in enumerate([y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]):
          rlzDict['ConstraintEvaluation_'+consName] = self.constraintsV.data[i,ind]
        self._updateSolutionExport(traj, rlzDict, acceptable, None)

    # decide what to do next
    if acceptable in ['accepted', 'first']:
      # record history
      bestRlz = {}
      varList = self._solutionExport.getVars('input') + self._solutionExport.getVars('output') + list(self.toBeSampled.keys())
      bestRlz = dict((var,np.atleast_1d(rlz[var].data)) for var in set(varList) if var in rlz.data_vars)
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
    if self._fitnessType == 'hardConstraint':
      optPoints,fit,obj,gOfBest = zip(*[[x,y,z,w] for x, y, z,w in sorted(zip(np.atleast_2d(population.data),datasetToDataArray(fitness, self._objectiveVar).data,objectiveVal,np.atleast_2d(g.data)),reverse=True,key=lambda x: (x[1],-x[2]))])
    else:
      optPoints,fit,obj,gOfBest = zip(*[[x,y,z,w] for x, y, z,w in sorted(zip(np.atleast_2d(population.data),datasetToDataArray(fitness, self._objectiveVar).data,objectiveVal,np.atleast_2d(g.data)),reverse=True,key=lambda x: (x[1]))])
    point = dict((var,float(optPoints[0][i])) for i, var in enumerate(selVars) if var in rlz.data_vars)
    gOfBest = dict(('ConstraintEvaluation_'+name,float(gOfBest[0][i])) for i, name in enumerate(g.coords['Constraint'].values))
    if (self.counter > 1 and obj[0] <= self.bestObjective and fit[0] >= self.bestFitness) or self.counter == 1:
      point.update(gOfBest)
      self.bestPoint = point
      self.bestFitness = fit[0]
      self.bestObjective = obj[0]

    return point

  def _collectOptPointMulti(self, population, rank, CD, objVal, fitness, constraintsV):
    """
      Collects the point (dict) from a realization
      @ In, population, Dataset, container containing the population
      @ In, objectiveVal, list, objective values at each chromosome of the realization
      @ In, rank, xr.DataArray, rank values at each chromosome of the realization
      @ In, crowdingDistance, xr.DataArray, crowdingDistance values at each chromosome of the realization
      @ Out, point, dict, point used in this realization
    """
    rankOneIDX = [i for i, rankValue in enumerate(rank.data) if rankValue == 1]
    optPoints = population[rankOneIDX]
    optObjVal = np.array([list(ele) for ele in list(zip(*objVal))])[rankOneIDX]
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
    self.multiBestFitness = fitSet
    self.multiBestObjective = optObjVal
    self.multiBestConstraint = optConstNew
    self.multiBestRank = optRank
    self.multiBestCD = optCD

    return #optPointsDic


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
    if len(self._objectiveVar) == 1: # This is for a single-objective Optimization case.
      if len(self._optPointHistory[traj]) < 2:
        return False
      o1, _ = self._optPointHistory[traj][-1]
      obj = o1[self._objectiveVar[0]]
      converged = (obj == self._convergenceCriteria['objective'])
      self.raiseADebug(self.convFormat.format(name='objective',
                                              conv=str(converged),
                                              got=obj,
                                              req=self._convergenceCriteria['objective']))
    else:                            # This is for a multi-objective Optimization case.
      if len(self._optPointHistory[traj]) < 2:
        return False
      o1, _ = self._optPointHistory[traj][-1]
      obj1 = o1[self._objectiveVar[0]]
      obj2 = o1[self._objectiveVar[1]]
      converged = (obj1 == self._convergenceCriteria['objective'] and obj2 == self._convergenceCriteria['objective'])
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

  def _envelopeSize(self,a,b):
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
    if len(self._objectiveVar) == 1: # This is for a single-objective Optimization case.
      if acceptable == 'accepted':
        self.raiseADebug(f'Convergence Check for Trajectory {traj}:')
        # check convergence
        converged, convDict = self.checkConvergence(traj, new, old)
      else:
        converged = False
        convDict = dict((var, False) for var in self._convergenceInfo[traj])
      self._convergenceInfo[traj].update(convDict)
    else: # This is for a multi-objective Optimization case.
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
             'AHDp': self.ahdp,
             'AHD': self.ahd,
             'rank': 0 if ((type(self._objectiveVar) == list and len(self._objectiveVar) == 1) or type(self._objectiveVar) == str) else rlz['rank'],
             'CD': 0 if ((type(self._objectiveVar) == list and len(self._objectiveVar) == 1) or type(self._objectiveVar) == str) else  rlz['CD'],
             'HDSM': self.hdsm
             }


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
      elif '{OBJ}' in template:
        new.extend([template.format(OBJ=obj) for obj in self._objectiveVar])
      elif '{CONSTRAINT}' in template:
        new.extend([template.format(CONSTRAINT=constraint.name) for constraint in self._constraintFunctions + self._impConstraintFunctions])
      else:
        new.append(template)

    return set(new)
