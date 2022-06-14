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
  Step size manipulations based on gradient history

  Created 2020-01
  @author: zhoujia, alfoa
"""
# for future compatibility with Python 3------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
# End compatibility block for Python 3--------------------------------------------------------------

# External Modules----------------------------------------------------------------------------------
import numpy as np
from scipy.optimize import minpack2
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ...utils import mathUtils, randomUtils
from .StepManipulator import StepManipulator
from . import NoConstraintResolutionFound, NoMoreStepsNeeded
# Internal Modules End------------------------------------------------------------------------------

class ConjugateGradient(StepManipulator):
  """
    Changes step size depending on history of gradients
  """
  requiredInformation = ['gradientHist', 'prevStepSize']
  optionalInformation = ['recommend']

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
    specs = super(ConjugateGradient, cls).getInputSpecification()

    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, ok, list(str), list of acceptable variable names
    """
    ok = super(ConjugateGradient, cls).getSolutionExportVariableNames()
    ok['CG_task'] = 'for ConjugateGradient, current task of line search. FD suggests continuing the search, and CONV indicates the line search converged and will pivot.'

    return ok

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    StepManipulator.__init__(self)
    ## Instance Variable Initialization
    # public
    self.needsAccessToAcceptance = True # if True, then this stepManip may need to modify opt point acceptance criteria
    # _protected
    self._persistence = None     # consecutive line search converges until acceptance
    # __private
    # additional methods
    self._minRotationAngle = 2.0 # how close to perpendicular should we try rotating towards?
    self._numRandomPerp = 10     # how many random perpendiculars should we try rotating towards?
    self._growth = None
    self._shrink = None

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    StepManipulator.handleInput(self, specs)
    growth = specs.findFirst('growthFactor')
    if growth is not None:
      self._growth = growth.value
    shrink = specs.findFirst('shrinkFactor')
    if shrink is not None:
      self._shrink = shrink.value

  def initialize(self, optVars, persistence=None, **kwargs):
    """
      initializes this object
      @ In, optVars, list(str), list of optimization variables (e.g. input space)
      @ In, persistence, integer, optional, successive converges required to consider total convergence
      @ In, kwargs, dict, additional unused arguments
      @ Out, None
    """
    StepManipulator.initialize(self, optVars, **kwargs)
    self._persistence = persistence

  ###############
  # Run Methods #
  ###############
  def initialStepSize(self, numOptVars=None, scaling=1.0, **kwargs):
    """
      Provides an initial step size
      @ In, numOptVars, int, number of optimization variables
      @ In, scaling, float, optional, scaling factor
    """
    return mathUtils.hyperdiagonal(np.ones(numOptVars) * scaling) * self._initialStepScaling

  def step(self, prevOpt, gradientHist=None, prevStepSize=None, objVar=None, **kwargs):
    """
      calculates the step size and direction to take
      @ In, prevOpt, dict, previous opt point
      @ In, gradientHist, deque, optional, if not given then none available; list of gradient dictionaries with 0 being oldest; versors
      @ In, prevStepSize, deque, optional, if not given then none available; list of float step sizes
      @ In, recommend, str, optional, override to 'grow' or 'shrink' step size
      @ In, kwargs, dict, keyword-based specifics as required by individual step sizers
      @ Out, newOpt, dict, new opt point
      @ Out, stepSize, float, new step size
    """
    # Conjugate Gradient does line searches along consecutive gradient estimations
    # with gradient estimations updated by more than just local estimation.
    # For conjugate gradient notations, see https://en.wikipedia.org/wiki/iNonlinear_conjugate_gradient_method
    # For line search notations, see github.com/scipy/scipy/blob/master/scipy/optimize/minpack2/dcsrch.f

    # We start from an opt point, then find the gradient direction.
    #   from there we line search for the best point along that grad direction
    #   During this time, we store the original opt point from were we found the gradient,
    #     as well as the best point found so far along the line search. -> optPointHist
    # In the grad hist, we store only gradients around best-in-line points historically
    # In the step hist, we store line search information

    lastStepInfo = prevStepSize[-1]['info']
    if lastStepInfo is None:
      # this MUST MEAN that this is the very very first step in this algorithm
      # note that we use binary strings because that's what scipy gives us
      lastStepInfo = {'task': b'START',
                      'fortranParams': {'iSave': np.zeros((2,), dtype=np.intc),
                                        'dSave': np.zeros((13,), dtype=float),
                                       }
                     }
    lineSearchTask = lastStepInfo['task']

    # get the gradient
    curGradMag = gradientHist[-1][0]
    curGrad = np.array(list(gradientHist[-1][1][v] for v in self._optVars)) * curGradMag

    # get an array of the current optimal point
    curPoint = np.array(list(prevOpt[v] for v in self._optVars))
    # get the current opt point objective value
    curObjVal = prevOpt[objVar]

    # if we're starting a new line search because we found a minimum along the previous line search
    # NOTE this only gets called the first time ever for each trajectory, because of how we start
    #      new line searches under the task == 'CONVERGE' switch below
    if lineSearchTask == b'START':
      lastStepInfo = self._startLineSearch(lastStepInfo, curPoint, curObjVal, curGrad, curGradMag)
    else: # if lineSearchTask is anything except "start"
      # store some indicative information about the gradient (scalar product of current gradient and
      # the search vector, also "derPhi" in literature)
      lastStepInfo['line']['objDerivative'] = np.dot(curGrad, lastStepInfo['searchVector']) # derPhi1

    # update the line search information, and get the next task (and step size if relevant)
    stepSize, task = self._lineSearchStep(lastStepInfo, curObjVal)

    # take actions depending on the task
    if task.startswith(b'FG'):
      # continue line search
      lastStepInfo['prev task'] = 'FG'
    elif task.startswith(b'CONV'):
      # local minimum reached, so pivot into new line search
      lastStepInfo['persistence'] = 0
      lastStepInfo['prev task'] = 'CONV'
      lastStepInfo = self._startLineSearch(lastStepInfo, curPoint, curObjVal, curGrad, curGradMag)
      stepSize, task = self._lineSearchStep(lastStepInfo, curObjVal)
    elif task.startswith((b'WARN', b'ERROR')):
      if task.startswith(b'WARN'):
        lastStepInfo['prev task'] = 'WARN'
        msg = task[9:].decode().lower()
        print(f'ConjugateGradient WARNING: "{msg}"')
      elif task.startswith(b'ERROR'):
        lastStepInfo['prev task'] = 'ERROR'
        print('ConjugateGradient ERROR: Not able to continue line search!')
      lastStepInfo['persistence'] += 1
      if lastStepInfo['persistence'] >= self._persistence:
        raise NoMoreStepsNeeded
    else:
      self.raiseAnError(RuntimeError, f'Unrecognized "task" return from scipy.optimize.minpack2: "{task}"')

    lastStepInfo['stepSize'] = stepSize
    lastStepInfo['task'] = task

    currentPivot = lastStepInfo['pivot']['point']
    newPivot = currentPivot - stepSize * lastStepInfo['pivot']['gradient']
    newOpt = dict((var, newPivot[v]) for v, var in enumerate(self._optVars))

    return newOpt, stepSize, lastStepInfo

  def fixConstraintViolations(self, proposed, previous, fixInfo):
    """
      Given constraint violations, update the desired optimal point to consider.
      @ In, proposed, dict, proposed new optimal point
      @ In, previous, dict, previous optimal point
      @ In, fixInfo, dict, contains record of progress in fixing search including but not limited to angles, perpendiculars, counters, and step sizes
      @ Out, proposed, new proposed point
      @ Out, stepSize, float, new step size taken
      @ Out, fixInfo, dict, updated fixing info
    """
    # TODO this is copied from GradientHistory; it should be updated for the ConjugateGradient when
    #      we know how we want to do this
    # DESIGN
    # While not okay:
    # 1. See if cutting the step will fix it.
    # 2. If not, try rotating towards a random perpendicular. Repeat 1.
    # 3. If not, try a new random perpendicular. Repeat 1. Repeat N times.
    # TODO should this be specific to step manipulators, or something else?
    # TODO updating opt point in place! Is this safe?
    minStepSize = fixInfo['minStepSize']
    stepVector = dict((var, proposed[var] - previous[var]) for var in self._optVars)
    stepDistance, stepDirection, _ = mathUtils.calculateMagnitudeAndVersor(list(stepVector.values()))
    if 'originalStepSize' not in fixInfo:
      fixInfo['originalStepSize'] = stepDistance
    if 'perpDir' in fixInfo:
      perpDir = fixInfo['perpDir']
    # if not done cutting step, start cutting
    if stepDistance > minStepSize:
      # cut step again
      stepSize = 0.5 * stepDistance # TODO user option?
      for v, var in enumerate(stepVector):
        proposed[var] = previous[var] + stepSize * stepDirection[v]
      print(' ... cutting step ...')

      return proposed, stepSize, fixInfo
    else:
      # rotate vector and restore full step size
      stepSize = fixInfo['originalStepSize']
      # store original direction
      if 'originalDirection' not in fixInfo:
        fixInfo['originalDirection'] = np.atleast_1d(stepDirection)
      # if this isn't the first time, check if there's angle left to rotate through; reset if not
      if 'perpDir' in fixInfo:
        ang = mathUtils.angleBetweenVectors(stepDirection, fixInfo['perpDir'])
        print(f' ... trying angle: {ang}')
        if ang < self._minRotationAngle:
          del fixInfo['perpDir']

      if 'perpDir' not in fixInfo:
        # find perpendicular vector
        perp = randomUtils.randomPerpendicularVector(fixInfo['originalDirection'])
        # NOTE we could return to point format, but no reason to
        # normalize perpendicular to versor and resize
        rotations = fixInfo.get('numRotations', 0)
        if rotations > self._numRandomPerp:
          raise NoConstraintResolutionFound
        _, perpDir, _ = mathUtils.calculateMagnitudeAndVersor(perp)
        fixInfo['perpDir'] = perpDir
        fixInfo['numRotations'] = rotations + 1
      # END fixing perpendicular direction
      # rotate vector halfway towards perpendicular
      perpDir = fixInfo['perpDir']

      # rotate towards selected perpendicular
      splitVector = {} # vector that evenly divides stepDirection and perp
      for v, var in enumerate(self._optVars):
        splitVector[var] = stepDirection[v] + perpDir[v]
        #splitVector[var] = - stepDirection[v] + perpDir[v]
      _, splitDir, _ = mathUtils.calculateMagnitudeAndVersor(list(splitVector.values()))
      for v, var in enumerate(self._optVars):
        proposed[var] = previous[var] + stepSize * splitDir[v]
      print(' ... rotating step ...')

    return proposed, stepSize, fixInfo

  def needDenormalized(self):
    """
      Determines if this algorithm needs denormalized input spaces
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    return True

  def updateSolutionExport(self, stepHistory):
    """
      Prints information to the solution export.
      @ In, stepHistory, list, (magnitude, versor, info) for each step entry
      @ Out, info, dict, realization of data to go in the solutionExport object
    """
    lastStepInfo = stepHistory[-1]['info']
    if lastStepInfo is not None:
      task = lastStepInfo['prev task']
      info = {'CG_task': task}
    else:
      info = {'CG_task': 'START'}

    return info

  def trajIsFollowing(self, traj, opt, info, dataObject, followers, tolerance):
    """
      Determines if the current trajectory is following another trajectory.
      @ In, traj, int, integer identifier for trajectory that needs to be checked
      @ In, opt, dict, DENORMALIZED most recent optimal point for trajectory
      @ In, info, dict, additional information about optimal point
      @ In, dataObject, DataObject.DataSet, data collected through optimization so far (SolutionExport)
      @ In, followers, list(int), trajectories that are following traj currently
      @ In, tolerance, float, termination distance (in scaled space)
      @ Out, found, int, trajectory that traj is following (or None)
    """
    if followers is None:
      followers = []
    # we define a trajectory as following if its current opt point is sufficiently near other opt
    # points from other trajectories
    matchDict = dict((var, opt[var]) for var in self._optVars)
    # only look in accepted points #TODO would there be value in looking at others?
    matchDict['CG_task'] = 'CONV'
    # only look at other trajectories that this trajectory hasn't killed
    noMatchDict = {'trajID': [traj] + followers}

    _, found = dataObject.realization(matchDict=matchDict, noMatchDict=noMatchDict, tol=tolerance)
    if found is not None:
      return found['trajID']

    return None

  ###################
  # Utility Methods #
  ###################

  def _polakRibierePowellStep(self, prevGrad, curGrad, gradDotProduct, searchVector):
    """
      Update the search vector (magnitude and direction) for conjugate gradient
      @ In, prevGrad, np.array, previous gradient in order of sampled variables
      @ In, curGrad, np.array, current gradient in order of sampled variables
      @ In, gradDorProduct, float, scalar product of the current grad with itself
      @ In, searchVector, np.array, ongoing search vector (not unit vector)
      @ Out, searchVectorMag, float, magnitude of new search vector
      @ Out, searchVector, dict, search vector
    """
    deltaGradient = curGrad - prevGrad
    gain = max(0, np.dot(deltaGradient, curGrad) / gradDotProduct)
    searchVector = -curGrad + gain * searchVector
    searchVectorMag = mathUtils.calculateMultivectorMagnitude(searchVector)

    return searchVectorMag, searchVector

  def modifyAcceptance(self, oldPoint, oldVal, newPoint, newVal):
    """
      Allows modification of acceptance criteria.
      Note this is only called if self.needsAccessToAcceptance is True.
      @ In, oldPoint, dict, old opt point
      @ In, oldVal, float, old objective function value
      @ In, newPoint, dict, new opt point
      @ In, newVal, float, new objective function value
      @ In, info, dict, identifying information about evaluation
      @ Out, accept, boolean, whether we store the point
    """
    # Because in ConjugateGradient we use all the line search information,
    # we "accept" all points from the Optimizer's standpoint, and allow
    # the step manipulator to use the information.

    return 'accepted'

  def _startLineSearch(self, lastStepInfo, curPoint, curObjVal, curGrad, curGradMag):
    """
      Begins a new line search.
      @ In, lastStepInfo, dict, information about the last step taken
      @ In, curPoint, dict, current most-recent collected potential opt point
      @ In, curObjVal, float, objective value at curPoint
      @ In, curGrad, dict, magnitude-and-vector gradient estimate
      @ In, curGradMag, float, magnitude of curGrad
      @ Out, lastStepInfo, dict, modified with new line search information
    """
    # use the previous pivots to update the conjugate gradient
    # first the objective value
    # then the conjugate gradient

    # since we've accepted a pivot, we need to store the old pivot and set up the new one
    # first grab the savable params
    pivot = lastStepInfo.pop('pivot', None)
    if pivot is None:
      # ONLY RUN ONCE per trajectory! First time ever initialization of line step search
      # use the current gradient to back-guess the would-be previous objective value
      prevObjVal = curObjVal + curGradMag / 2 # oldOldFVal
      # magnitude of the search vector first time is just the gradient magnitude
      searchVectorMag = curGradMag
      # search direction at the start is the opposite direction of the initial gradient
      # note this is not great naming
      searchVector = curGrad * -1 # pk
      gradDotProduct = np.dot(curGrad, curGrad) # delta_k
    else:
      # LITERALLY every time except the first for each traj
      lastStepInfo['previous pivot'] = pivot
      prevObjVal = lastStepInfo['previous pivot']['objVal'] # oldFVal
      prevGrad = lastStepInfo['previous pivot']['gradient']
      gradDotProduct = lastStepInfo['gradDotProduct']
      searchVectorMag, searchVector = self._polakRibierePowellStep(prevGrad, curGrad, gradDotProduct, lastStepInfo['searchVector'])
    pivotObjDerivative = np.dot(searchVector, curGrad) # derPhi_0
    stepSize = min(1.0, 1.01 * 2 * (curObjVal - prevObjVal) / pivotObjDerivative)
    # comments are the literature equivalents for each variable name
    lastStepInfo.update({'pivot': {'point': curPoint,                   # x_0
                                   'objVal': curObjVal,                 # phi_0
                                   'gradient': curGrad,                 # gf_k
                                   'objDerivative': pivotObjDerivative, # derPhi_0
                                  }, # w.r.t. pivot vals
                         'line': {'point': curPoint,                    # x_k
                                 'objDerivative': pivotObjDerivative,   # derPhi_1
                                 }, # w.r.t. line search
                         'gradDotProduct': gradDotProduct,              # delta_k
                         'searchVector': searchVector,                  # p_k
                         'searchVectorMag': searchVectorMag,            # gNorm
                         'stepSize': stepSize,                          # alpha
                         'task': b'START',
                         'persistence': 0,
                         })

    return lastStepInfo

  def _lineSearchStep(self, lastStepInfo, curObjVal):
    """
      Determine the next action to take in the line search process
      @ In, lastStepInfo, dict, dictionary of past and present relevant information
      @ In, curObjVal, float, current objective value obtained during line search
      @ Out, stepSize, float, new suggested step size
      @ Out, task, binary string, task of line search
    """
    # determine next task in line search
    stepSize = lastStepInfo['stepSize']
    lineObjDerivative = lastStepInfo['line']['objDerivative']
    task = lastStepInfo['task']
    iSave = lastStepInfo['fortranParams']['iSave']
    dSave = lastStepInfo['fortranParams']['dSave']
    stepSize, _, _, task = minpack2.dcsrch(stepSize, curObjVal, lineObjDerivative,
                                           ftol=1e-4, gtol=0.4, xtol=1e-14,
                                           task=task, stpmin=1e-100, stpmax=1e100,
                                           isave=iSave, dsave=dSave)

    return stepSize, task

  def flush(self):
    """
      Reset ConjugateGradient attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self._persistence = None
