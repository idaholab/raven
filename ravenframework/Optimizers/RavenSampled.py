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
  Base class for Optimizers using RAVEN's internal sampling mechanics.

  Created 2020-01
  @author: talbpaul
"""
# for future compatibility with Python 3------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
# End compatibility block for Python 3--------------------------------------------------------------

# External Modules----------------------------------------------------------------------------------
import abc
import ast
from collections import deque
import copy
import datetime
import h5py
import json
import math
import os
import numpy as np
import xarray as xr
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..utils import InputData, InputTypes
from .Optimizer import Optimizer
# Internal Modules End------------------------------------------------------------------------------

# Sentinel strings used to represent non-JSON-native float values in checkpoint state.
_NAN_SENTINEL    = '__checkpoint_nan__'
_POSINF_SENTINEL = '__checkpoint_inf__'
_NEGINF_SENTINEL = '__checkpoint_neginf__'

def _encodeCheckpointState(obj):
  """
    Recursively converts an optimizer state dict to a JSON-safe representation.
    Handles numpy scalars/arrays, xarray objects, deques, sets, and tuples.
    NaN and Inf float values are replaced with sentinel strings.
    All dict keys are converted to strings (required by JSON).
    @ In, obj, any, Python object to encode
    @ Out, obj, any, JSON-serializable representation
  """
  if isinstance(obj, float):
    if math.isnan(obj):  return _NAN_SENTINEL
    if math.isinf(obj):  return _POSINF_SENTINEL if obj > 0 else _NEGINF_SENTINEL
    return obj
  if isinstance(obj, np.floating):
    return _encodeCheckpointState(float(obj))
  if isinstance(obj, np.integer):
    return int(obj)
  if isinstance(obj, np.bool_):
    return bool(obj)
  if isinstance(obj, np.ndarray):
    return {'__type__': 'ndarray', 'dtype': str(obj.dtype), 'data': obj.tolist()}
  if isinstance(obj, xr.DataArray):
    return {'__type__': 'DataArray', 'data': _encodeCheckpointState(obj.to_dict())}
  if isinstance(obj, xr.Dataset):
    return {'__type__': 'Dataset', 'data': _encodeCheckpointState(obj.to_dict())}
  if isinstance(obj, deque):
    return {'__type__': 'deque', 'maxlen': obj.maxlen,
            'data': [_encodeCheckpointState(i) for i in obj]}
  if isinstance(obj, set):
    return {'__type__': 'set', 'data': [_encodeCheckpointState(i) for i in obj]}
  if isinstance(obj, tuple):
    return {'__type__': 'tuple', 'data': [_encodeCheckpointState(i) for i in obj]}
  if isinstance(obj, dict):
    # JSON requires string keys; callers must restore original key types on decode.
    return {str(k): _encodeCheckpointState(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_encodeCheckpointState(i) for i in obj]
  return obj  # str, bool, int, None pass through unchanged

def _decodeCheckpointState(obj):
  """
    Recursively restores a state dict encoded by _encodeCheckpointState.
    @ In, obj, any, JSON-decoded Python object
    @ Out, obj, any, restored Python object with original types
  """
  if isinstance(obj, str):
    if obj == _NAN_SENTINEL:    return float('nan')
    if obj == _POSINF_SENTINEL: return float('inf')
    if obj == _NEGINF_SENTINEL: return float('-inf')
    return obj
  if isinstance(obj, list):
    return [_decodeCheckpointState(i) for i in obj]
  if isinstance(obj, dict):
    if '__type__' not in obj:
      return {k: _decodeCheckpointState(v) for k, v in obj.items()}
    t = obj['__type__']
    if t == 'ndarray':
      return np.array(obj['data'], dtype=obj['dtype'])
    if t == 'deque':
      return deque((_decodeCheckpointState(i) for i in obj['data']), maxlen=obj['maxlen'])
    if t == 'set':
      def _toHashable(x):
        if isinstance(x, list): return tuple(_toHashable(i) for i in x)
        return _decodeCheckpointState(x)
      return set(_toHashable(i) for i in obj['data'])
    if t == 'tuple':
      return tuple(_decodeCheckpointState(i) for i in obj['data'])
    if t == 'DataArray':
      return xr.DataArray.from_dict(_decodeCheckpointState(obj['data']))
    if t == 'Dataset':
      return xr.Dataset.from_dict(_decodeCheckpointState(obj['data']))
    # Unknown tagged type — return as plain decoded dict
    return {k: _decodeCheckpointState(v) for k, v in obj.items()}
  return obj


class RavenSampled(Optimizer):
  """
    Base class for Optimizers using RAVEN's internal sampling mechanics.
    Handles the following:
     - Maintain queue for required realizations
     - Label and retrieve realizations given labels
     - Manage sign flipping for maximization problems
     - Establish API for convergence checking
     - Establish API to extend labels for particular implementations
     - Implements constraint checking
     - Implements model evaluation limitations
     - Implements rejection strategy (?)
     - Implements convergence persistence
     - Establish API for iterative sample output to solution export
     - Implements specific sampling methods from Sampler (when not present in Optimizer)
  """
  # * * * * * * * * * * * * * * * *
  # Convergence Checks
  # Note these names need to be formatted according to checkConvergence check!
  convFormat = ' ... {name:^12s}: {conv:5s}, {got:1.2e} / {req:1.2e}'

  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(RavenSampled, cls).getInputSpecification()
    specs.description = 'Base class for Optimizers whose iterative sampling is performed through RAVEN.'
    # initialization: add sampling-based options
    init = specs.getSub('samplerInit')
    limit = InputData.parameterInputFactory('limit', contentType=InputTypes.IntegerType,
        printPriority=100,
        descr=r"""limits the number of Model evaluations that may be performed as part of this optimization.
              For example, a limit of 100 means at most 100 total Model evaluations may be performed.""")
    whenSolnExpEnum = InputTypes.makeEnumType('whenWriteEnum', 'whenWriteType', ['final', 'every'])
    write = InputData.parameterInputFactory('writeSteps', contentType=whenSolnExpEnum,
        printPriority=100,
        descr=r"""delineates when the \xmlNode{SolutionExport} DataObject should be written to. In case
              of \xmlString{final}, only the final optimal solution for each trajectory will be written.
              In case of \xmlString{every}, the \xmlNode{SolutionExport} will be updated with each iteration
              of the Optimizer.""")
    init.addSub(limit)
    init.addSub(write)

    writeCheckpoint = InputData.parameterInputFactory('writeCheckpoint',
        contentType=InputTypes.BoolType,
        printPriority=105,
        descr=r"""Enables writing of optimizer state checkpoints after each completed iteration.
              When \xmlString{True}, the optimizer writes a \texttt{.ravenrst} restart file that
              can later be provided to \xmlNode{restartFrom} to resume an interrupted run. The
              optional \xmlAttr{file} attribute sets the checkpoint file path; if omitted, the
              file is named \texttt{<optimizerName>.ravenrst} in the working directory. The
              optional \xmlAttr{interval} attribute controls how many completed iterations occur
              between writes (default: 1).""")
    writeCheckpoint.addParam('file', InputTypes.StringType, required=False,
        descr=r"""Path for the checkpoint output file. Defaults to \texttt{<optimizerName>.ravenrst}.""")
    writeCheckpoint.addParam('interval', InputTypes.IntegerType, required=False,
        descr=r"""Number of completed iterations between checkpoint writes. Default: 1.""")
    init.addSub(writeCheckpoint)

    restartFrom = InputData.parameterInputFactory('restartFrom',
        contentType=InputTypes.StringType,
        printPriority=106,
        descr=r"""Path to an existing \texttt{.ravenrst} checkpoint file from which the optimizer
              should resume. The optimizer validates that the settings in the restart file are
              compatible with the current XML configuration before restoring state. See
              \xmlNode{writeCheckpoint} for producing restart files.""")
    init.addSub(restartFrom)

    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, ok, list(str), list of acceptable variable names
    """
    ok = super(RavenSampled, cls).getSolutionExportVariableNames()
    ok.update({'trajID': 'integer identifier for different optimization starting locations and paths',
               'iteration': 'integer identifying which iteration (or step, or generation) a trajectory is on',
               'accepted': 'string acceptance status of the potential optimal point (algorithm dependent)',
               'rejectReason':'description of reject reason, \'noImprovement\' means rejected the new optimization point for no improvement from last point, \'implicitConstraintsViolation\' means rejected by implicit constraints violation, return None if the point is accepted',
               '{VAR}': r'any variable from the \xmlNode{TargetEvaluation} input or output; gives the value of that variable at the optimal candidate for this iteration.',
               'modelRuns': 'integer identifying the number of times the model is evaluated up to the current step'
              })

    return ok

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    Optimizer.__init__(self)
    # Instance Variable Initialization
    # public
    self.limit = None  # max samples
    self.type = 'Sampled Optimizer'  # type
    self.batch = 1  # batch size: 1 means no batching (default)
    self.batchId = 0  # Id of each batch of evaluations
    # _protected
    self._writeSteps = 'final'  # when steps should be written
    self._submissionQueue = deque()  # TODO change to Queue.Queue if multithreading samples
    self._stepTracker = {}  # action tracking: what is collected, what needs collecting?
    self._optPointHistory = {}  # a dictionary of deque's by traj (-1 is most recent)
    self._maxHistLen = 2  # FIXME who should set this?
    self._rerunsSinceAccept = {} # by traj, how long since our last accepted point
    self._evaluatedSubmissionKeys = set()  # Points we already evaluated; used to quickly detect duplicates.
    self._evaluatedSubmissionData = {}  # Saved result for each evaluated point; duplicates reuse this result.
    self._deduplicatedSubmissions = []  # Duplicate runs we skipped now and will be restored later from cached results.
    # __private
    self.__stepCounter = {}  # tracks the "generation" or "iteration" of each trajectory -> iteration is defined by inheritor
    # additional methods
    # # register adaptive sample identification criteria
    self.registerIdentifier('step')  # the step within the action
    self._finals = []                # A list of unique final points
    #These objective multipliers are used so that the objective can always
    # appear as a minimization problem internally.
    # This multipiles by -1 to turn a maximization problem to a minimization
    # problem.
    self._objMult = {} #max will be -1, min will be 1
    self._objMultArray = np.array([])
    # Checkpoint / restart
    self._writeCheckpointEnabled = False   # whether to write a checkpoint file after each iteration
    self._checkpointFile = None            # path for the checkpoint output file
    self._checkpointInterval = 1           # write checkpoint every N completed iterations
    self._restartFromFile = None           # path to a .ravenrst file to resume from
    self._checkpointRestored = False       # True after a successful checkpoint restore
    self.__checkpointIterCount = 0         # iteration counter for interval-based checkpoint writing


  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    Optimizer.handleInput(self, paramInput)
    # samplerInit
    init = paramInput.findFirst('samplerInit')
    if init is not None:
      # limit
      limit = init.findFirst('limit')
      if limit is not None:
        self.limit = limit.value
      # writeSteps
      writeSteps = init.findFirst('writeSteps')
      if writeSteps is not None:
        self._writeSteps = writeSteps.value
      # writeCheckpoint
      writeCheckpointNode = init.findFirst('writeCheckpoint')
      if writeCheckpointNode is not None and writeCheckpointNode.value:
        self._writeCheckpointEnabled = True
        fileAttr = writeCheckpointNode.parameterValues.get('file', None)
        if fileAttr is not None:
          self._checkpointFile = fileAttr
        intervalAttr = writeCheckpointNode.parameterValues.get('interval', None)
        if intervalAttr is not None:
          self._checkpointInterval = int(intervalAttr)
      # restartFrom
      restartFromNode = init.findFirst('restartFrom')
      if restartFromNode is not None:
        self._restartFromFile = restartFromNode.value
    # additional checks
    if self.limit is None:
      self.raiseAnError(IOError, 'A <limit> is required for any RavenSampled Optimizer!')
    self._objMultArray = np.ones(len(self._objectiveVar))
    for i in range(len(self._objectiveVar)):
      if self._minMax[i] == 'max':
        self._objMult[self._objectiveVar[i]] = -1
        self._objMultArray[i] = -1
      else:
        self._objMult[self._objectiveVar[i]] = 1


  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    Optimizer.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    self.batch = 1
    self.batchId = 0
    # Set default checkpoint file path now that self.name is available
    if self._writeCheckpointEnabled and self._checkpointFile is None:
      self._checkpointFile = f'{self.name}.ravenrst'
    # If a restart file was provided, restore optimizer state (assembler objects are live at this point)
    if self._restartFromFile is not None:
      self._restoreFromCheckpoint()

  ###############
  # Run Methods #
  ###############
  @abc.abstractmethod
  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization (corrected for min-max)
      @ Out, None
    """

  @abc.abstractmethod
  def checkConvergence(self, traj, new, old):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory identifier
      @ In, new, dict, new opt point
      @ In, old, dict, previous opt point
    """

  @abc.abstractmethod
  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """

  def _initializeStep(self, traj):
    """
      Initializes a new step in the optimization process.
      @ In, traj, int, the trajectory of interest
      @ Out, None
    """
    self._stepTracker[traj] = {'opt': None}  # add entries in inheritors as needed

  def amIreadyToProvideAnInput(self):
    """
      This is a method that should be called from any user of the optimizer before requiring the generation of a new input.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of model evaluation, convergence criteria met, etc.
      @ In, None
      @ Out, ready, bool, indicating the readiness of the optimizer to generate a new input.
    """
    # if any trajectories are still active, we're ready to provide an input
    ready = Optimizer.amIreadyToProvideAnInput(self)

    # This guard is checked in amIreadyToProvideAnInput, before localFinalizeActualSampling is called.
    # If all realizations are deduplicated, localFinalizeActualSampling (and therefore _useRealization)
    # is not called, so _submissionQueue remains empty. In that case, MultiRun would stop early
    # (see line 247 of MultiRun.py), so we replay deduplicated realizations through _useRealization.
    if ready and len(self._submissionQueue) == 0 and self._deduplicatedSubmissions and len(self._prefixToIdentifiers) == 0:
      self.raiseADebug('No queued runs remain; restoring deduplicated submissions from cache.')
      restoredInfo, restoredRlz = self._restoreDeduplicatedSubmissions()
      if restoredRlz is not None:
        self._useRealization(restoredInfo, restoredRlz)
        if self._writeCheckpointEnabled:
          self._writeCheckpoint()

    # we're not ready yet if we don't have anything in queue
    ready = ready and len(self._submissionQueue) != 0
    return ready

  def localGenerateInput(self, model, inp):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, inp, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    runsToGenerate = min(self.batch, len(self._submissionQueue))
    if self.batch > 1:
      self.inputInfo['batchMode'] = True
      batchData = []
      self.batchId += 1
    else:
      self.inputInfo['batchMode'] = False
    for _ in range(runsToGenerate):
      inputInfo = {'SampledVarsPb':{}, 'batchMode':self.inputInfo['batchMode']}  # ,'prefix': str(self.batchId)+'_'+str(i)
      if self.counter == self.limit + 1:
        break
      # get point from stack
      point, info = self._submissionQueue.popleft()
      point = self.denormalizeData(point)
      # assign a tracking prefix
      # prefix = inputInfo['prefix']
      prefix = self.inputInfo['prefix']
      inputInfo['prefix'] = prefix
      # register the point tracking information
      self._registerSample(prefix, info)
      # build the point in the way the Sampler expects
      for var in self.toBeSampled:  # , val in point.items():
        val = point[var] if isinstance(point[var], float) else np.atleast_1d(point[var].data)[0]
        self.values[var] = val  # TODO should be np.atleast_1d?
        ptProb = self.distDict[var].pdf(val)
        # sampler-required meta information # TODO should we not require this?
        inputInfo[f'ProbabilityWeight-{var}'] = ptProb
        inputInfo['SampledVarsPb'][var] = ptProb
      inputInfo['ProbabilityWeight'] = 1  # TODO assume all weight 1? Not well-distributed samples
      inputInfo['PointProbability'] = np.prod([x for x in inputInfo['SampledVarsPb'].values()])
      inputInfo['SamplerType'] = self.type
      if self.inputInfo['batchMode']:
        inputInfo['SampledVars'] = self.values
        inputInfo['batchId'] = self.batchId
        batchData.append(copy.deepcopy(inputInfo))
      else:
        inputInfo['SampledVars'] = self.values
        inputInfo['batchId'] = self.batchId
        self.inputInfo.update(inputInfo)
    if self.batch > 1:
      self.inputInfo['batchInfo'] = {'nRuns': len(batchData), 'batchRealizations': batchData, 'batchId': str('gen_' + str(self.batchId))}

  # @profile
  def localFinalizeActualSampling(self, jobObject, model, myInput):
    """
      Runs after each sample is collected from the JobHandler.
      @ In, jobObject, Runner instance, job runner entity
      @ In, model, Model instance, RAVEN model that was run
      @ In, myInput, list, generated inputs for run
      @ Out, None
    """
    Optimizer.localFinalizeActualSampling(self, jobObject, model, myInput)
    # TODO should this be an Optimizer class action instead of Sampled?
    # collect finished job
    prefix = jobObject.getMetadata()['prefix']
    # If we're not looking for the prefix, don't bother with using it
    # this usually happens if we've cancelled the run but it's already done
    if not self.stillLookingForPrefix(prefix):
      return
    # FIXME implicit constraints probable should be handled here too
    # get information and realization, and update trajectories
    info = self.getIdentifierFromPrefix(prefix, pop=True)
    if self.batch == 1:
      _, rlz = self._targetEvaluation.realization(matchDict={'prefix': prefix}, asDataSet=False)
    else:
      # NOTE if here, then rlz is actually a xr.Dataset, NOT a dictionary!!
      _, rlz = self._targetEvaluation.realization(matchDict={'batchId': self.batchId}, asDataSet=True, first=False)
    # _, full = self._targetEvaluation.realization(matchDict={'prefix': prefix}, asDataSet=False)
    if self._targetEvaluation.isEmpty:
      self.raiseAnError(RuntimeError, f'Expected to find entry with prefix "{prefix}" in TargetEvaluation "{self._targetEvaluation.name}", but it is empty!')
    _, full = self._targetEvaluation.realization(matchDict={'prefix': prefix})
    if full is None:
      self.raiseAnError(RuntimeError, f'Expected to find entry with prefix "{prefix}" in TargetEvaluation! Found: {self._targetEvaluation.getVarValues("prefix")}')
    # trim down opt point to the useful parts
    # TODO making a new dict might be costly, maybe worth just passing whole point?
    # # testing suggests no big deal on smaller problem
    # the sign of the objective function is flipped in case we do maximization
    # so get the correct-signed value into the realization

    for objVar in self._objectiveVar:
      rlz[objVar] *= self._objMult[objVar]
    # TODO FIXME let normalizeData work on an xr.DataSet (batch) not just a dictionary!
    # NOTE:
    # Previously we called _useRealization(info, rlz) directly here.
    # With dedup enabled, skipped duplicates never produce a fresh model result, so we must:
    #  1) normalize and cache the current realization by submission key,
    #  2) restore any pending deduplicated submissions from cache,
    #  3) forward one merged/restored payload to _useRealization.
    # This keeps optimizer state transitions identical for both evaluated and deduplicated points.
    rlz = self.normalizeData(rlz)
    self._cacheEvaluatedSubmissionPoints(rlz)
    restoredInfo, restoredRlz = self._restoreDeduplicatedSubmissions(info, rlz)
    if restoredRlz is not None:
      self._useRealization(restoredInfo, restoredRlz)
      if self._writeCheckpointEnabled:
        self._writeCheckpoint()

  ###########################
  # Checkpoint / Restart    #
  ###########################

  def _getCheckpointSettings(self):
    """
      Returns a dict of critical configuration settings to embed in the checkpoint for validation
      on restart. Subclasses should call super() and add algorithm-specific settings.
      @ In, None
      @ Out, settings, dict, configuration settings required for restart compatibility checks
    """
    return {
      'variables':     sorted(self.toBeSampled.keys()),
      'objectiveVars': self._objectiveVar,
      'minMax':        self._minMax,
    }

  def _getCheckpointState(self):
    """
      Returns a dict of runtime state variables to be persisted in the checkpoint.
      Subclasses should call super() and add algorithm-specific state.
      @ In, None
      @ Out, state, dict, serializable optimizer runtime state
    """
    return {
      'counter':                  self.counter,
      'batchId':                  self.batchId,
      '__stepCounter':            self.__stepCounter,
      '_stepTracker':             self._stepTracker,
      '_optPointHistory':         self._optPointHistory,
      '_rerunsSinceAccept':       self._rerunsSinceAccept,
      '_activeTraj':              self._activeTraj,
      '_cancelledTraj':           self._cancelledTraj,
      '_convergedTraj':           self._convergedTraj,
      '_trajCounter':             self._trajCounter,
      '_submissionQueue':         list(self._submissionQueue),
      '_evaluatedSubmissionKeys': self._evaluatedSubmissionKeys,
      '_evaluatedSubmissionData': self._evaluatedSubmissionData,
    }

  def _restoreCheckpointState(self, state):
    """
      Restores runtime state variables from a checkpoint state dict.
      Subclasses should call super() and restore algorithm-specific state.
      @ In, state, dict, state dict previously produced by _getCheckpointState
      @ Out, None
    """
    self.counter                  = state['counter']
    self.batchId                  = state['batchId']
    self._trajCounter             = state['_trajCounter']
    # Trajectory-indexed dicts have int keys serialized as strings by JSON; restore them.
    self.__stepCounter            = {int(k): v for k, v in state['__stepCounter'].items()}
    self._stepTracker             = {int(k): v for k, v in state['_stepTracker'].items()}
    self._optPointHistory         = {int(k): v for k, v in state['_optPointHistory'].items()}
    self._rerunsSinceAccept       = {int(k): v for k, v in state['_rerunsSinceAccept'].items()}
    self._cancelledTraj           = {int(k): v for k, v in state['_cancelledTraj'].items()}
    self._convergedTraj           = {int(k): v for k, v in state['_convergedTraj'].items()}
    self._activeTraj              = state['_activeTraj']
    self._submissionQueue         = deque(state['_submissionQueue'])
    self._evaluatedSubmissionKeys = state['_evaluatedSubmissionKeys']
    # Keys are tuple-of-tuples serialized as strings; restore via ast.literal_eval.
    self._evaluatedSubmissionData = {ast.literal_eval(k): v
                                     for k, v in state['_evaluatedSubmissionData'].items()}

  def _validateCheckpoint(self, checkpoint):
    """
      Validates that a loaded checkpoint is compatible with the current XML configuration.
      Raises an error for incompatible settings; issues a warning for recoverable differences.
      Subclasses should call super() and add algorithm-specific validation.
      @ In, checkpoint, dict, loaded checkpoint dict
      @ Out, None
    """
    currentVersion = '1.0' # Currently supported checkpoint version #NOTE: update this if the checkpoint format is changed.
    ckptVersion = checkpoint.get('version', '0.0')
    if ckptVersion != currentVersion:
      self.raiseAWarning(f'Restart file version "{ckptVersion}" may differ from the current '
                        f'version {currentVersion}; compatibility is not guaranteed.')
    ckptType = checkpoint.get('optimizerType', 'unknown')
    if ckptType != self.__class__.__name__:
      self.raiseAnError(IOError,
          f'Restart file was written by optimizer type "{ckptType}" but the current optimizer '
          f'is "{self.__class__.__name__}". Restart files cannot be used across different optimizer types.')
    ckptName = checkpoint.get('optimizerName', '')
    if ckptName != self.name:
      self.raiseAWarning(f'Restart file optimizer name "{ckptName}" does not match the current '
                        f'optimizer name "{self.name}". Proceeding with restore.')
    settings = checkpoint.get('settings', {})
    ckptVars = settings.get('variables', [])
    currentVars = sorted(self.toBeSampled.keys())
    if ckptVars != currentVars:
      self.raiseAnError(IOError,
          f'Restart file variable set {ckptVars} does not match the current variable set {currentVars}.')
    ckptObjs = settings.get('objectiveVars', [])
    if ckptObjs != self._objectiveVar:
      self.raiseAnError(IOError,
          f'Restart file objective variables {ckptObjs} do not match the current objective '
          f'variables {self._objectiveVar}.')
    ckptMinMax = settings.get('minMax', [])
    if ckptMinMax != self._minMax:
      self.raiseAnError(IOError,
          f'Restart file min/max settings {ckptMinMax} do not match the current settings {self._minMax}.')

  def _writeCheckpoint(self):
    """
      Serializes the current optimizer state to the checkpoint file. Respects the configured
      write interval; writes are skipped until the interval is reached.
      @ In, None
      @ Out, None
    """
    self.__checkpointIterCount += 1
    if self.__checkpointIterCount % self._checkpointInterval != 0:
      self.raiseADebug(f'Checkpoint write skipped '
                       f'({self.__checkpointIterCount % self._checkpointInterval} / {self._checkpointInterval} intervals elapsed).')
      return
    generation = self.getIteration(self._activeTraj[0]) if self._activeTraj else 0
    self.raiseAMessage(f'Writing checkpoint for generation {generation} to "{self._checkpointFile}" ...')
    state      = self._getCheckpointState()
    settings   = self._getCheckpointSettings()
    # Collect SolutionExport rows for continuous output on restart
    rows = []
    if self._solutionExport is not None:
      try:
        ds = self._solutionExport.asDataset()
        if ds is not None and len(ds.data_vars) > 0:
          nSamples = ds.dims.get('RAVEN_sample_ID', 0)
          for i in range(nSamples):
            row = {}
            for var in ds.data_vars:
              vals = ds[var].values
              val  = vals[i] if vals.ndim > 0 else vals.item()
              row[var] = val.item() if hasattr(val, 'item') else val
            rows.append(row)
        self.raiseADebug(f'Collected {len(rows)} SolutionExport row(s) for checkpoint.')
      except Exception as err:
        self.raiseAWarning(f'Could not save SolutionExport data to checkpoint: {err}')
    with h5py.File(self._checkpointFile, 'w') as hf:
      # Root attributes: human-readable metadata
      hf.attrs['version']       = '1.0'
      hf.attrs['optimizerType'] = self.__class__.__name__
      hf.attrs['optimizerName'] = self.name
      hf.attrs['generation']    = generation
      hf.attrs['timestamp']     = datetime.datetime.now().isoformat()
      # Settings blob (JSON) used by _validateCheckpoint on restart
      hf.create_dataset('settings', data=np.bytes_(json.dumps(settings)))
      # Full optimizer state encoded to a JSON-safe dict, stored as a single bytes dataset.
      # All numpy/xarray/deque/set/tuple types are handled by _encodeCheckpointState.
      hf.create_dataset('state', data=np.bytes_(json.dumps(_encodeCheckpointState(state))))
      # SolutionExport: columnar datasets with gzip compression for numerical variables.
      seGrp = hf.create_group('solutionExport')
      seGrp.attrs['nRows'] = len(rows)
      if rows:
        varData = {}
        for row in rows:
          for var, val in row.items():
            varData.setdefault(var, []).append(val)
        for var, vals in varData.items():
          if isinstance(vals[0], str):
            seGrp.create_dataset(var, data=np.array(vals, dtype=object),
                                 dtype=h5py.string_dtype())
          else:
            seGrp.create_dataset(var, data=np.array(vals),
                                 compression='gzip', compression_opts=4)
    self.raiseAMessage(f'Checkpoint written to "{self._checkpointFile}" '
                       f'(generation {generation}, {len(rows)} SolutionExport row(s) saved).')

  def _restoreFromCheckpoint(self):
    """
      Loads a checkpoint file, validates it against the current configuration, and restores
      optimizer state. Also pre-populates the SolutionExport DataObject with historical rows
      so the output file appears continuous.
      @ In, None
      @ Out, None
    """
    if not os.path.exists(self._restartFromFile):
      self.raiseAnError(IOError, f'Restart file "{self._restartFromFile}" not found.')
    self.raiseAMessage(f'Loading restart file "{self._restartFromFile}" ...')
    rows = []
    with h5py.File(self._restartFromFile, 'r') as hf:
      # Build a metadata dict from root attributes for use by _validateCheckpoint
      checkpoint = {
        'version':       hf.attrs.get('version', '0.0'),
        'optimizerType': hf.attrs.get('optimizerType', 'unknown'),
        'optimizerName': hf.attrs.get('optimizerName', ''),
        'generation':    hf.attrs.get('generation', 'unknown'),
        'settings':      json.loads(hf['settings'][()]),
      }
      self.raiseAMessage(f'Restart file metadata: optimizer "{checkpoint["optimizerName"]}" '
                         f'(type: {checkpoint["optimizerType"]}, version: {checkpoint["version"]}, '
                         f'generation: {checkpoint["generation"]}).')
      # Validate settings before touching any runtime state
      self._validateCheckpoint(checkpoint)
      self.raiseAMessage('Restart file validated successfully against current configuration.')
      # Decode and restore the full optimizer state
      self.raiseAMessage('Restoring optimizer state ...')
      state = _decodeCheckpointState(json.loads(hf['state'][()]))
      self._restoreCheckpointState(state)
      self.raiseAMessage(f'Optimizer state restored: counter={self.counter}, batchId={self.batchId}, '
                         f'active trajectories={self._activeTraj}, '
                         f'converged trajectories={list(self._convergedTraj.keys())}.')
      # Reconstruct SolutionExport rows from the columnar datasets
      seGrp = hf.get('solutionExport')
      if seGrp is not None and seGrp.attrs.get('nRows', 0) > 0 and self._solutionExport is not None:
        nRows    = int(seGrp.attrs['nRows'])
        varNames = list(seGrp.keys())
        self.raiseAMessage(f'Reading {nRows} SolutionExport row(s) from restart file ...')
        for i in range(nRows):
          row = {}
          for var in varNames:
            val = seGrp[var][i]
            if isinstance(val, bytes):
              val = val.decode('utf-8')
            row[var] = np.atleast_1d(val)
          rows.append(row)
      elif self._solutionExport is not None:
        self.raiseADebug('No SolutionExport data found in restart file.')
    if rows and self._solutionExport is not None:
      try:
        for row in rows:
          self._solutionExport.addRealization(row)
        self.raiseAMessage(f'Restored {len(rows)} SolutionExport row(s) from restart file.')
      except Exception as err:
        self.raiseAWarning(f'Could not restore SolutionExport data from restart file: {err}')
    self._checkpointRestored = True
    gen = checkpoint.get('generation', 'unknown')
    self.raiseAMessage(f'Successfully resumed optimizer "{self.name}" from restart file '
                      f'"{self._restartFromFile}" (last completed generation: {gen}).')

  def finalizeSampler(self, failedRuns): #!TODO: is this unused??
    """
      Last tasks to perform before Step is finished.
      @ In, failedRuns, list, runs that failed as part of this sampling
      @ Out, None
    """
    # get and print the best trajectory obtained
    bestValue = None
    bestTraj = None
    bestPoint = None


    # check converged trajectories
    self.raiseAMessage('*' * 80)
    self.raiseAMessage('Optimizer Final Results:')
    self.raiseADebug('')
    self.raiseADebug(' - Trajectory Results:')
    self.raiseADebug('  TRAJ   STATUS    VALUE')
    statusTemplate = '   {traj:2d}  {status:^11s}  {val}'
    templateNoValue = '   {traj:2d}  {status:^11s}'
    # Define the template for the values
    valueTemplate = '{val}'

    # print cancelled traj
    for traj, info in self._cancelledTraj.items():
      val = info['value']
      status = info['reason']
      self.raiseADebug(statusTemplate.format(status=status, traj=traj, val=self._objMultArray * val))
    # check converged traj
    for traj, info in self._convergedTraj.items():
      opt = self._optPointHistory[traj][-1][0]
      val = info['value']

      # Format the values in the array
      formattedValues = np.vectorize(lambda v: valueTemplate.format(val=v))(self._objMultArray*val)

      # Combine the formatted values into a single string with appropriate spacing
      formattedValuesString = '\n'.join(['   '.join(row) for row in formattedValues])

      # Raise debug message for the entire formatted string
      self.raiseADebug(templateNoValue.format(status='converged', traj=traj)+formattedValuesString.format(formattedValues))
      if bestValue is None or val < bestValue:
        bestTraj = traj
        bestValue = val

    if bestValue is not None:
      traj = bestTraj
    else:
      # further check active unfinished trajectories
      # FIXME why should there be any active, unfinished trajectories when we're cleaning up sampler?
      # FIXME why only 0?? what if it's other trajectories that are active and unfinished?
      self.raiseAWarning("No bestValue found in the trajectories, "+
                         "this may indicate problems finding a solution. "+
                         "Arbitarily defaulting to using trajectory 0.")
      traj = 0
      bestTraj = traj
    # sanity check: if there's no history (we never got any answers) then report rather than crash
    if len(self._optPointHistory[traj]) == 0:
      self.raiseAnError(RuntimeError, f'There is no optimization history for traj {traj}! ' +
                        'Perhaps the Model failed?')

    ## If any solution in the history has a higher objective value than what is found in ._optPointHistory, add it to opt. NOTE: this only searches the most recent iteration of the optimizer.
    opt = self._optPointHistory[traj][-1][0]

    objectiveVars = []
    optKeys = list(opt)
    # Map objective variable names to keys in the optimization history.
    for obj in np.atleast_1d(self._objectiveVar):
      if obj in optKeys:
        objectiveVars.append(obj)
      else:
        matches = [var for var in optKeys if obj.lower() in var.lower()]
        objectiveVars.extend(matches)
    # Deduplicate while preserving order
    objectiveVars = list(dict.fromkeys(objectiveVars))

    self._solutionExport.asDataset() #empty _collector into _data
    ## Search the rest of the history
    if hasattr(self,"_sampledHistoryInfo"):
      bestInHistory = {}
      # find best solns from the history
      for soln1 in self._sampledHistoryInfo:
          if not bestInHistory:
              bestInHistory[soln1] = self._sampledHistoryInfo[soln1]
          else:
              for soln2 in copy.deepcopy(bestInHistory):
                  if (self._sampledHistoryInfo[soln1] > bestInHistory[soln2]).all():
                      del bestInHistory[soln2] #replace previous entry with better one
                      bestInHistory[soln1] = self._sampledHistoryInfo[soln1]
                  elif (self._sampledHistoryInfo[soln1] > bestInHistory[soln2]).any():
                      bestInHistory[soln1] = self._sampledHistoryInfo[soln1]
      # try to add best solns to list of optimal solns
      for soln in bestInHistory:
        if (bestInHistory[soln] > [max(opt[key]) for key in objectiveVars]).any(): #if any objective value in soln is greater than all corresponding objective values in opt, this is True.
            for indx in range(len(self._solutionExport._data[list(self._solutionExport._data)[0]])): #search the history data for the inputs in soln
              if np.all(np.array(soln) == [self._solutionExport._data[key][indx].item() for key in self.toBeSampled]):
                # add soln values from history data to opt
                for key in list(opt):
                  try:
                    if key in self._objectiveVar:
                      minMaxC = {'max':-1, 'min':1} #convert objective scales to the expected format in 'opt'
                      opt[key] = np.append(opt[key], self._solutionExport._data[key][indx].item() * minMaxC[self._minMax[self._objectiveVar.index(key)]])
                    else:
                      opt[key] = np.append(opt[key], self._solutionExport._data[key][indx].item())
                  except KeyError:
                    opt[key] = np.append(opt[key], None) #If the key is missing from solutionExport, it isn't in the addRealization and won't be used anyway.
                break # NOTE: it shouldn't be possible, but this would fail silently if soln isn't found in the history data

    #Note: bestTraj == traj
    for i in range(len(np.atleast_1d(opt[self._objectiveVar[0]]))):
      optElm = {key: np.atleast_1d(opt[key])[i] for key in opt}
      bestOpt = self.denormalizeData(optElm)
      bestPoint = dict((var, bestOpt[var]) for var in self.toBeSampled)

      val = optElm[self._objectiveVar[0]]
      self.raiseADebug(statusTemplate.format(status='active', traj=traj, val=self._objMultArray * val))
      self.raiseADebug('')
      self.raiseAMessage(' - Final Optimal Point:')
      finalTemplate = '    {name}  {value}'
      self.raiseAMessage(finalTemplate.format(name=self._objectiveVar, value=self._objMultArray * val))
      self.raiseAMessage(finalTemplate.format(name='trajID', value=bestTraj))
      for var, val in bestPoint.items():
        self.raiseAMessage(finalTemplate.format(name=var, value=val))
      self.raiseAMessage('*' * 80)
      # write final best solution to soln export
      if bestPoint not in self._finals:
          self._updateSolutionExport(bestTraj, self.normalizeData(bestOpt), 'final', 'None')
          self._finals.append(bestPoint)


  def flush(self):
    """
      Reset Optimizer attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self._stepTracker = {}
    self._optPointHistory = {}
    self._rerunsSinceAccept = {}
    self.__stepCounter = {}
    self._submissionQueue = deque()
    self._evaluatedSubmissionKeys = set()
    self._evaluatedSubmissionData = {}
    self._deduplicatedSubmissions = []

  ###################
  # Utility Methods #
  ###################
  def _makeSubmissionKey(self, point):
    """
      Creates a hashable key from sampled variable values in a point.
      @ In, point, dict, sampled variable values for a single realization
      @ Out, key, tuple, hashable representation of the sampled point
    """
    key = []
    for var in sorted(self.toBeSampled.keys()):
      val = point[var]
      if hasattr(val, 'values'):
        val = val.values
      if hasattr(val, 'data'):
        val = val.data
      arr = np.asarray(val).reshape(-1)
      key.append(tuple(int(x) for x in arr))
    return tuple(key)

  def _cacheEvaluatedSubmissionPoints(self, rlz):
    """
      Caches evaluated points so later duplicate submissions can be skipped.
      @ In, rlz, dict or xr.Dataset, normalized realization(s)
      @ Out, None
    """
    if not self._deduplication:
      return
    if isinstance(rlz, dict):
      key = self._makeSubmissionKey(rlz)
      self._evaluatedSubmissionKeys.add(key)
      self._evaluatedSubmissionData[key] = copy.deepcopy(rlz)
      return
    if 'RAVEN_sample_ID' not in rlz.sizes:
      return
    for i in range(rlz.sizes['RAVEN_sample_ID']):
      point = {var: np.atleast_1d(rlz[var].data)[i] for var in self.toBeSampled}
      key = self._makeSubmissionKey(point)
      self._evaluatedSubmissionKeys.add(key)
      cached = rlz.isel({'RAVEN_sample_ID': [i]}).copy(deep=True)
      self._evaluatedSubmissionData[key] = cached

  def _recordDeduplicatedSubmission(self, key, info):
    """
      Records a duplicate submission for dedup processing in finalize.
      @ In, key, tuple, hashable key for sampled point
      @ In, info, dict, run tracking information
      @ Out, recorded, bool, True if duplicate was recorded
    """
    self.raiseADebug(f'Recording duplicate run for dedup processing: {info}')
    self._deduplicatedSubmissions.append((key, copy.deepcopy(info)))
    return True

  def _overwriteBatchId(self, rlz):
    """
      Overwrite batchId in restored realizations with the current optimizer batch.
      @ In, rlz, dict or xr.Dataset, realization(s) to update
      @ Out, updated, dict or xr.Dataset, realization(s) with updated batchId
    """
    if rlz is None:
      return None
    if isinstance(rlz, dict):
      updated = copy.deepcopy(rlz)
      updated['batchId'] = self.batchId
      return updated
    if isinstance(rlz, xr.Dataset):
      updated = rlz.copy(deep=True)
      sampleDim = None
      if 'RAVEN_sample_ID' in updated.sizes:
        sampleDim = 'RAVEN_sample_ID'
      elif len(updated.sizes) == 1:
        sampleDim = next(iter(updated.sizes))
      if sampleDim is None:
        updated['batchId'] = self.batchId
      else:
        updated['batchId'] = xr.DataArray(np.full(updated.sizes[sampleDim], self.batchId), dims=[sampleDim])
      return updated
    return rlz

  def _restoreDeduplicatedSubmissions(self, info=None, rlz=None):
    """
      Restores deduplicated submissions by combining cached realizations
      with an optional freshly evaluated realization.
      This method only prepares merged/restored realizations; it does not call _useRealization.
      @ In, info, dict, optional, tracking information for rlz
      @ In, rlz, dict or xr.Dataset, optional, freshly evaluated normalized realization
      @ Out, restoredInfo, dict or None, tracking information for restored rlz
      @ Out, restoredRlz, dict or xr.Dataset or None, realization ready for _useRealization
    """
    pendingDedup = self._deduplicatedSubmissions
    self._deduplicatedSubmissions = []
    datasetRestores = []
    dictRestores = []
    for key, dedupInfo in pendingDedup:
      cached = self._evaluatedSubmissionData.get(key)
      if isinstance(cached, xr.Dataset):
        datasetRestores.append((key, copy.deepcopy(dedupInfo), copy.deepcopy(cached)))
      elif isinstance(cached, dict):
        dictRestores.append((key, copy.deepcopy(dedupInfo), copy.deepcopy(cached)))

    if isinstance(rlz, xr.Dataset):
      mergedPieces = [rlz]
      mergedInfo = copy.deepcopy(info)
      if datasetRestores:
        mergedPieces.extend([cached for _, _, cached in datasetRestores])
      if dictRestores:
        self._deduplicatedSubmissions.extend([(key, dedupInfo) for key, dedupInfo, _ in dictRestores])
      return mergedInfo, self._overwriteBatchId(xr.concat(mergedPieces, dim='RAVEN_sample_ID'))

    if rlz is not None:
      if datasetRestores:
        self._deduplicatedSubmissions.extend([(key, dedupInfo) for key, dedupInfo, _ in datasetRestores])
      if dictRestores:
        self._deduplicatedSubmissions.extend([(key, dedupInfo) for key, dedupInfo, _ in dictRestores])
      return copy.deepcopy(info), self._overwriteBatchId(rlz)

    if dictRestores:
      key, dedupInfo, cached = dictRestores[0]
      if len(dictRestores) > 1:
        self._deduplicatedSubmissions.extend([(k, i) for k, i, _ in dictRestores[1:]])
      if datasetRestores:
        self._deduplicatedSubmissions.extend([(k, i) for k, i, _ in datasetRestores])
      return dedupInfo, self._overwriteBatchId(cached)

    if datasetRestores:
      merged = xr.concat([cached for _, _, cached in datasetRestores], dim='RAVEN_sample_ID')
      return datasetRestores[0][1], self._overwriteBatchId(merged)

    return None, None

  def _queueSubmission(self, point, info):
    """
      Adds a run to the submission queue, skipping duplicates when deduplication is active.
      @ In, point, dict, normalized point to submit
      @ In, info, dict, run tracking information
      @ Out, queued, bool, True if point was queued
    """
    if self._deduplication:
      key = self._makeSubmissionKey(point)
      if key in self._evaluatedSubmissionKeys:
        self.raiseADebug(f'Skipping duplicate run: {self.denormalizeData(point)} | {info}')
        self._recordDeduplicatedSubmission(key, info)
        return False
    self.raiseADebug(f'Adding run to queue: {self.denormalizeData(point)} | {info}')
    self._submissionQueue.append((point, info))
    return True

  def incrementIteration(self, traj):
    """
      Increments the "generation" or "iteration" of an optimization algorithm.
      The definition of generation is algorithm-specific; this is a utility for tracking only.
      @ In, traj, int, identifier for trajectory
      @ Out, None
    """
    self.__stepCounter[traj] += 1

  def getIteration(self, traj):
    """
      Provides the "generation" or "iteration" of an optimization algorithm.
      The definition of generation is algorithm-specific; this is a utility for tracking only.
      @ In, traj, int, identifier for trajectory
      @ Out, counter, int, iteration of the trajectory
    """
    return self.__stepCounter[traj]

  # * * * * * * * * * * * *
  # Constraint Handling
  def _handleExplicitConstraints(self, proposed, previous, pointType):
    """
      Considers all explicit (i.e. input-based) constraints
      @ In, proposed, dict, NORMALIZED sample opt point
      @ In, previous, dict, NORMALIZED previous opt point
      @ In, pointType, string, type of point to handle constraints for
      @ Out, normed, dict, suggested NORMALIZED constraint-handled point
      @ Out, modded, bool, whether point was modified or not
    """
    denormed = self.denormalizeData(proposed)
    # check and fix boundaries
    denormed, boundaryModded = self._applyBoundaryConstraints(denormed)
    normed = self.normalizeData(denormed)
    # fix functionals
    normed, funcModded = self._applyFunctionalConstraints(normed, previous)
    modded = boundaryModded or funcModded

    return normed, modded

  def _checkFunctionalConstraints(self, point):
    """
      Checks that provided point does not violate functional constraints
      @ In, point, dict, suggested point to submit (denormalized)
      @ Out, allOkay, bool, False if violations found else True
    """
    allOkay = True
    inputs = dict(point)
    inputs.update(self.constants)
    for constraint in self._constraintFunctions:
      okay = constraint.evaluate('constrain', inputs)
      if not okay:
        self.raiseADebug(f'Functional constraint "{constraint.name}" was violated!')
        self.raiseADebug(' ... point:', point)
      allOkay *= okay

    return bool(allOkay)

  def _applyBoundaryConstraints(self, point):
    """
      Checks and fixes boundary constraints of variables in "point" -> DENORMED point expected!
      @ In, point, dict, potential point against which to check
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    # TODO should some of this go into the parent Optimizer class, such as the boundary acquiring?
    modded = False
    for var in self.toBeSampled:
      dist = self.distDict[var]
      val = point[var]
      lower = dist.lowerBound
      upper = dist.upperBound
      if val < lower:
        self.raiseADebug(f' BOUNDARY VIOLATION "{var}" suggested value: {val:1.3e} lower bound: {lower:1.3e} under by {lower - val:1.3e}')
        self.raiseADebug(f' ... -> for point {point}')
        point[var] = lower
        modded = True
      elif val > upper:
        self.raiseADebug(f' BOUNDARY VIOLATION "{var}" suggested value: {val:1.3e} upper bound: {upper:1.3e} over by {val - upper:1.3e}')
        self.raiseADebug(f' ... -> for point {point}')
        point[var] = upper
        modded = True

    return point, modded

  def _checkBoundaryConstraints(self, point):
    """
      Checks (NOT fixes) boundary constraints of variables in "point" -> DENORMED point expected!
      @ In, point, dict, potential point against which to check
      @ Out, okay, bool, True if no constraints violated
    """
    okay = True
    for var in self.toBeSampled:
      dist = self.distDict[var]
      val = point[var]
      lower = dist.lowerBound
      upper = dist.upperBound
      if val < lower or val > upper:
        okay = False
        break
    return okay

  @abc.abstractmethod
  def _applyFunctionalConstraints(self, suggested, previous):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """

  def _handleImplicitConstraints(self, previous):
    """
      Considers all implicit constraints
      @ In, previous, dict, NORMALIZED previous opt point
      @ Out, accept, bool, whether point was satisfied implicit constraints
    """
    normed = copy.deepcopy(previous)
    oldVal = normed[self._objectiveVar[0]]
    normed.pop(self._objectiveVar[0], oldVal)
    denormed = self.denormalizeData(normed)
    denormed[self._objectiveVar[0]] = oldVal
    accept = self._checkImpFunctionalConstraints(denormed)

    return accept

  def _checkImpFunctionalConstraints(self, previous):
    """
      Checks that provided point does not violate implicit functional constraints
      @ In, previous, dict, previous opt point (denormalized)
      @ Out, allOkay, bool, False if violations found else True
    """
    allOkay = True
    inputs = dict(previous)
    for impConstraint in self._impConstraintFunctions:
      okay = impConstraint.evaluate('implicitConstraint', inputs)
      if not okay:
        self.raiseADebug(f'Implicit constraint "{impConstraint.name}" was violated!')
        self.raiseADebug(' ... point:', previous)
      allOkay *= okay

    return bool(allOkay)

  # END constraint handling
  # * * * * * * * * * * * *

  # * * * * * * * * * * * * * * * *
  # Resolving potential opt points
  def _resolveNewOptPoint(self, traj, rlz, optVal, info):
    """
      Consider and store a new optimal point
      @ In, traj, int, trajectory for this new point
      @ In, info, dict, identifying information about the realization
      @ In, rlz, xr.DataSet, batched realizations
      @ In, optVal, list of floats, values of objective variable
    """
    self.raiseADebug('*' * 80)
    self.raiseADebug(f'Trajectory {traj} iteration {info["step"]} resolving new opt point ...')
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    # FIXME check implicit constraints? Function call, - Jia
    acceptable, old, rejectReason = self._checkAcceptability(traj, rlz, optVal, info)
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    # we only want to update persistence if we've accepted a new point.
    # We don't want rejected points to count against our convergence.
    if acceptable in ['accepted']:
      self._updatePersistence(traj, converged, optVal)
    # NOTE: the solution export needs to be updated BEFORE we run rejectOptPoint or extend the opt
    #       point history.
    if self._writeSteps == 'every':
      self._updateSolutionExport(traj, rlz, acceptable, rejectReason)
    self.raiseADebug('*' * 80)
    # decide what to do next
    if acceptable in ['accepted', 'first']:
      # record history
      self._optPointHistory[traj].append((rlz, info))
      self._rerunsSinceAccept[traj] = 0
      # nothing else to do but wait for the grad points to be collected
    elif acceptable == 'rejected':
      self._rejectOptPoint(traj, info, old)
    elif acceptable == 'rerun':
      # update the most recently obtained opt value for the rerun point
      # NOTE we do this because if we got "lucky" in an opt point evaluation, we can get stuck
      #      there even as we rerun and discover that original value is not reliable.
      # so use successive reruns to update the average
      # note 1 rerun means 2 total values to work with, so use this in averaging update
      # TODO could we ever use old rerun gradients to inform the gradient direction as well?
      self._rerunsSinceAccept[traj] += 1
      N = self._rerunsSinceAccept[traj] + 1
      oldVal = self._optPointHistory[traj][-1][0][self._objectiveVar[0]]
      newAvg = ((N-1)*oldVal + optVal) / N
      self._optPointHistory[traj][-1][0][self._objectiveVar[0]] = newAvg
    else:
      self.raiseAnError(f'Unrecognized acceptability: "{acceptable}"')

  # support methods for _resolveNewOptPoint
  @abc.abstractmethod
  def _checkAcceptability(self, traj, opt, optVal):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
      @ Out, rejectReason, str, reject reason of opt point, or return None if accepted
    """

  @abc.abstractmethod
  def _updateConvergence(self, traj, new, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, new, dict, new point
      @ In, old, dict, old point
      @ In, acceptable, str, condition of new point
    """

  @abc.abstractmethod
  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """

  @abc.abstractmethod
  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
    """

  def _updateSolutionExport(self, traj, rlz, acceptable, rejectReason):
    """
      Stores information to the solution export.
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ In, rejectReason, str, reject reason of opt point, or return None if accepted
      @ Out, None
    """
    # make a holder for the realization that will go to the solutionExport
    toExport = {}
    # add some meta information
    toExport.update({'iteration': self.getIteration(traj),
                     'trajID': traj,
                     'accepted': acceptable,
                     'rejectReason': rejectReason,
                     'modelRuns': self.counter
                    })
    # add the contents of the realization
    toExport.update(dict((var, rlz[var]) for var in rlz))

    # optimal point input and output spaces
    for objVar in self._objectiveVar:
      objValue = rlz[objVar]*self._objMult[objVar]
      toExport[objVar] = objValue
    toExport.update(self.denormalizeData(dict((var, rlz[var]) for var in self.toBeSampled)))
    # constants and functions
    toExport.update(self.constants)
    toExport.update(dict((var, rlz[var]) for var in self.dependentSample if var in rlz))
    # additional from inheritors
    toExport.update(self._addToSolutionExport(traj, rlz, acceptable))
    # check for anything else that solution export wants that rlz might provide
    for var in self._solutionExport.getVars():
      if var not in toExport and var in rlz:
        toExport[var] = rlz[var]
    # formatting
    toExport = dict((var, np.atleast_1d(val)) for var, val in toExport.items())
    # Force solutionExport to expect all the vars we want to give it.
    for key in toExport.keys():
      if key not in self._solutionExport.vars:
        self._solutionExport.addedVars.append(key)
    self._solutionExport.addedVars = list(set(self._solutionExport.addedVars))
    # Write solution data to solutionExport
    self._solutionExport.addRealization(toExport)

  def _addToSolutionExport(self, traj, rlz, acceptable):
    """
      Contributes additional entries to the solution export.
      Should be used by inheritors instead of overloading updateSolutionExport
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ Out, toAdd, dict, additional entries
    """
    return {}

  # END resolving potential opt points
  # * * * * * * * * * * * * * * * *

  def _cancelAssociatedJobs(self, traj, step=None):
    """
      Queues jobs to be cancelled based on opt run
      @ In, traj, int, trajectory identifier
      @ In, step, int, optional, iteration identifier (unused if not provided)
      @ Out, None
    """
    # generic tracking info: we want this trajectory, this step, all purposes
    ginfo = {'traj': traj}
    if step is not None:
      ginfo['step'] = step
    # remove them from the submission queue
    toRemove = []
    # NOTE use a queue lock here if taking samples in multithreading (not currently true)
    for point, info in self._submissionQueue:
      if all(item in info.items() for item in ginfo.items()):
        toRemove.append((point, info))
    for x in toRemove:
      try:
        self._submissionQueue.remove(x)
      except ValueError:
        pass  # it must have been submitted since we flagged it for removal
    # get prefixes of already-submitted jobs; get all matches, and pop them so we don't track them anymore
    prefixes = self.getPrefixFromIdentifier(ginfo, getAll=True, pop=True)
    self.raiseADebug(f'Canceling grad jobs for traj "{traj}" iteration "{"all" if step is None else step}":', prefixes)
    self._jobsToEnd.extend(prefixes)

  def initializeTrajectory(self, traj=None):
    """
      Sets up a new trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, trajectory number
    """
    traj = Optimizer.initializeTrajectory(self, traj=traj)
    self._optPointHistory[traj] = deque(maxlen=self._maxHistLen)
    self.__stepCounter[traj] = -1  # allows 0-based counting
    self._rerunsSinceAccept[traj] = 0
    self._initializeStep(traj)

    return traj

  def _closeTrajectory(self, traj, action, reason, value):
    """
      Removes a trajectory from active space.
      @ In, traj, int, trajectory identifier
      @ In, action, str, method in which to close ('converge' or 'cancel')
      @ In, reason, str, reason for closure
      @ In, value, float, opt value obtained
      @ Out, None
    """
    Optimizer._closeTrajectory(self, traj, action, reason, value)
    # kill jobs associated with trajectory
    self._cancelAssociatedJobs(traj)
