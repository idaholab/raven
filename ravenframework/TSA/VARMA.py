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
  Vector AutoRegressive Moving Average time series analysis
"""
import os
import copy
import itertools
import numpy as np
import scipy as sp

from .. import Distributions

from ..utils import InputData, InputTypes, randomUtils, xmlUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer, TimeSeriesTransformer


class VARMA(TimeSeriesGenerator, TimeSeriesCharacterizer, TimeSeriesTransformer):
  r"""
    AutoRegressive Moving Average time series analyzer algorithm
  """
  # class attribute
  ## define the clusterable features for this trainer.
  _features = ['ar',
               'ma',
               'cov',
               'const']
  _acceptsMissingValues = True
  _isStochastic = True

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(VARMA, cls).getInputSpecification()
    specs.name = 'varma'
    specs.description = r"""characterizes the vector-valued signal using Auto-Regressive and Moving
        Average coefficients to stochastically fit the training signal.
        The VARMA representation has the following form:
        \begin{equation*}
          A_t = \sum_{i=1}^P \phi_i A_{t-i} + \epsilon_t + \sum_{j=1}^Q \theta_j \epsilon_{t-j},
        \end{equation*}
        where $t$ indicates a discrete time step, $\phi$ are the signal lag (or auto-regressive)
        coefficients, $P$ is the number of signal lag terms to consider, $\epsilon$ is a random noise
        term, $\theta$ are the noise lag (or moving average) coefficients, and $Q$ is the number of
        noise lag terms to consider. For signal $A_t$ which is a $k \times 1$ vector, each $\phi_i$
        and $\theta_j$ are $k \times k$ matrices, and $\epsilon_t$ is characterized by the
        $k \times k$ covariance matrix $\Sigma$. The VARMA algorithms are developed in RAVEN using the
        \texttt{statsmodels} Python library."""
    specs.addSub(InputData.parameterInputFactory('P', contentType=InputTypes.IntegerType,
                 descr=r"""the number of terms in the AutoRegressive term to retain in the
                       regression; typically represented as $P$ in literature."""))
    specs.addSub(InputData.parameterInputFactory('Q', contentType=InputTypes.IntegerType,
                 descr=r"""the number of terms in the Moving Average term to retain in the
                       regression; typically represented as $Q$ in literature."""))
    return specs

  #
  # API Methods
  #
  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    # general infrastructure
    super().__init__(*args, **kwargs)

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['P'] = spec.findFirst('P').value
    settings['Q'] = spec.findFirst('Q').value
    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'engine' not in settings:
      settings['engine'] = randomUtils.newRNG()
    return settings

  def fit(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ Out, params, dict, characteristic parameters
    """
    # lazy statsmodels import
    import statsmodels.api

    P = settings['P']
    Q = settings['Q']
    seed = settings['seed']
    if seed is not None:
      randomUtils.randomSeed(seed, engine=settings['engine'], seedBoth=True)

    model = statsmodels.api.tsa.VARMAX(endog=signal, order=(P, Q))
    results = model.fit(disp=False, maxiter=1000)

    lenHist, numVars = signal.shape
    # train multivariate normal distributions using covariances, keep it around so we can control the RNG
    ## it appears "measurement" always has 0 covariance, and so is all zeros (see generate method)
    ## all the noise comes from the stateful properties
    stateDist = self._trainMultivariateNormal(numVars, np.zeros(numVars), model.ssm['state_cov'])

    # Solve for the mean and covariance of the state vector of the VARMAX model. This is used later
    # to sample initial state values for the model in a way that we can control.
    initMean, initCov = self._solveStateDistribution(model.ssm['transition'],
                                                     model.ssm['state_intercept'],
                                                     model.ssm['state_cov'],
                                                     model.ssm['selection'])
    initDist = self._trainMultivariateNormal(len(initMean), initMean, initCov)

    # NOTE (@j-bryan, 2023-09-07) A transformed covariance matrix is stored to help with interpolation.
    # Covariance matrices should be interpolated using geometric mean interpolation
    #   C_interp(t) = C_1^{1/2} (C_1^{-1/2} C_2 C_1^{-1/2})^t C_1^{1/2},       (1)
    # as derived from the optimal transport between two multivariate normal distributions [1].
    # However, interpolation in the SupervisedLearning.ROMCollection.Interpolation class is done on
    # a parameter-by-parameter basis. The above interpolation can be re-expressed as
    #   C_interp(t) = exp(t*log(C_1) + (1-t)*log(C_2)),                        (2)
    # where exp and log are the matrix exponential and logarithm, respectively. This is the approach
    # is equivalent to element-wise interpolation of the log-transformed covariance matrices. Therefore,
    # by storing the log-transformed covariance matrix, we can use the existing interpolation infrastructure
    # to interpolate the covariance matrices. However, it seems this approach may introduce some numerical error
    # relative to Eq. (1).
    #
    # [1] Bhatia, R., & Holbrook, J. (2006). Riemannian geometry and matrix geometric means. Linear
    #     algebra and its applications, 413(2-3), 594-618.

    params = {}
    params['VARMA'] = {'model': model,
                       'targets': targets,
                       'const': results.params[model._params_trend],
                       'ar': results.params[model._params_ar],
                       'ma': results.params[model._params_ma],
                       'cov': self._covToTransformedFlat(model.ssm['state_cov']),
                       'initDist': initDist,
                       'noiseDist': stateDist,
                       'resid': results.resid}
    return params

  def getResidual(self, initial, params, pivot, settings):
    """
      Removes trained signal from data and find residual
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, residual, np.array, reduced signal shaped [pivotValues, targets]
    """
    residual = params['VARMA']['resid']
    return residual

  def getComposite(self, initial, params, pivot, settings):
    """
      Combines two component signals to form a composite signal. This is essentially the inverse
      operation of the getResidual method.
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, composite, np.array, resulting composite signal
    """
    # Add a generated ARMA signal to the initial signal.
    synthetic = self.generate(params, pivot, settings)
    composite = initial + synthetic
    return composite

  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, settings for this ROM
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    model = params['VARMA']['model']
    numSamples = len(pivot)
    numVars = model.k_endog

    # sample measure, state shocks
    measureShocks = np.zeros([numSamples, numVars])

    ## state shocks come from sampling multivariate, with CROW
    noiseDist = params['VARMA']['noiseDist']
    stateShocks = np.array([noiseDist.rvs() for _ in range(numSamples)])
    # pick an initial point by sampling multinormal distribution
    initDist = params['VARMA']['initDist']
    init = np.array(initDist.rvs())

    # The covariance parameters are the Cholesky decomposition of the covariance matrix.
    cov = np.linalg.cholesky(self._transformedFlatToCov(params['VARMA']['cov']))

    # Load model params
    modelParams = np.r_[params['VARMA']['const'],
                        params['VARMA']['ar'],
                        params['VARMA']['ma'],
                        cov[np.tril_indices_from(cov)]]

    synthetic = model.simulate(modelParams,
                               numSamples,
                               initial_state=init,
                               measurement_shocks=measureShocks,
                               state_shocks=stateShocks)
    return synthetic

  def getParamNames(self, settings):
    """
      Return list of expected variable names based on the parameters
      @ In, settings, dict, training parameters for this algorithm
      @ Out, names, list, string list of names
    """
    names = []
    base = f'{self.name}'

    for i, target in enumerate(settings['target']):
      # constant and variance
      names.append(f'{base}__constant__{target}')
      names.append(f'{base}__variance__{target}')

      # covariances
      for j, target2 in enumerate(settings['target']):
        if j >= i:
          continue
        names.append(f'{base}__covariance__{target}_{target2}')

      # AR/MA coefficients
      for p in range(settings['P']):
        for j, target2 in enumerate(settings['target']):
          names.append(f'{base}__AR__{p}__{target}_{target2}')
      for q in range(settings['Q']):
        for j, target2 in enumerate(settings['target']):
          names.append(f'{base}__MA__{q}__{target}_{target2}')
    return names

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    rlz = {}
    paramNames = params['VARMA']['model'].param_names
    modelParams = np.r_[params['VARMA']['const'],
                        params['VARMA']['ar'],
                        params['VARMA']['ma'],
                        params['VARMA']['cov']]
    targetNames = params['VARMA']['targets']

    for name, value in zip(paramNames, modelParams):
      parts = name.split('.')  # e.g. 'intercept.y1', 'sqrt.var.y1', 'L1.y1.y1', 'L1.e(y1).y2', 'sqrt.cov.y1.y2'
      if parts[0] == 'intercept':
        target = targetNames[int(parts[1][1:]) - 1]
        rlz[f'{self.name}__constant__{target}'] = value
      elif parts[1] == 'var':  # variance (diagonal)
        target = targetNames[int(parts[2][1:]) - 1]
        rlz[f'{self.name}__variance__{target}'] = value ** 2
      elif parts[1] == 'cov':  # covariance (off-diagonal)
        target1 = targetNames[int(parts[2][1:]) - 1]
        target2 = targetNames[int(parts[3][1:]) - 1]
        rlz[f'{self.name}__covariance__{target1}_{target2}'] = value ** 2
      elif parts[0].startswith('L') and parts[1].startswith('y'):  # AR coeff
        target1 = targetNames[int(parts[1][1:]) - 1]
        target2 = targetNames[int(parts[2][1:]) - 1]
        rlz[f'{self.name}__AR__{parts[0][1:]}__{target1}_{target2}'] = value
      elif parts[0].startswith('L') and parts[1].startswith('e'):  # MA coeff
        target1 = targetNames[int(parts[1][3:-1]) - 1]
        target2 = targetNames[int(parts[2][1:]) - 1]
        rlz[f'{self.name}__MA__{parts[0][1:]}__{target1}_{target2}'] = value
      else:
        raise ValueError(f'Unrecognized parameter name "{name}"!')

    return rlz

  # clustering
  def getClusteringValues(self, nameTemplate: str, requests: list, params: dict) -> dict:
    """
      Provide the characteristic parameters of this ROM for clustering with other ROMs
      @ In, nameTemplate, str, formatting string template for clusterable params (target, metric id)
      @ In, requests, list, list of requested attributes from this ROM
      @ In, params, dict, parameters from training this ROM
      @ Out, features, dict, params as {paramName: value}
    """
    # nameTemplate convention:
    # -> target is the trained variable (e.g. Signal, Temperature)
    # -> metric is the algorithm used (e.g. Fourier, ARMA)
    # -> id is the subspecific characteristic ID (e.g. sin, AR_0)
    # nameTemplate = "{target}|{metric}|{id}"
    features = {}
    data = params['VARMA']
    model = data['model']
    numVars = len(data['targets'])

    # combinations of targets
    # Covariance matrix only has the lower triangle
    # AR and MA coefficients are full matrices
    covCombinations = itertools.combinations(data['targets'], 2)
    armaCombinations = itertools.product(data['targets'], data['targets'])

    for req in requests:
      paramValues = data[req]
      if req == 'const':
        for i, target in enumerate(data['targets']):
          features[nameTemplate.format(target=target, metric='VARMA', id='const')] = paramValues[i]
      elif req == 'cov':
        for targets, value in zip(covCombinations, paramValues):
          features[nameTemplate.format(target=targets[0], metric='VARMA', id=f'cov_{targets[1]}')] = value
      elif req == 'ar':  # ordered by target1, then lag, then target2
        for p in range(model.k_ar):
          for targets, value in zip(armaCombinations, paramValues[p*numVars*numVars:(p+1)*numVars*numVars]):
            features[nameTemplate.format(target=targets[0], metric='VARMA', id=f'ar_{p+1}_{targets[1]}')] = value
      elif req == 'ma':  # ordered by target1, then lag, then target2
        for q in range(model.k_ma):
          for targets, value in zip(armaCombinations, paramValues[q*numVars*numVars:(q+1)*numVars*numVars]):
            features[nameTemplate.format(target=targets[0], metric='VARMA', id=f'ma_{q+1}_{targets[1]}')] = value
    return features

  def setClusteringValues(self, fromCluster, params):
    """
      Interpret returned clustering settings as settings for this algorithm.
      Acts somewhat as the inverse of getClusteringValues.
      @ In, fromCluster, list(tuple), (target, identifier, values) to interpret as settings
      @ In, params, dict, trained parameter settings
      @ Out, params, dict, updated parameter settings
    """
    targets = params['VARMA']['targets']
    arOrder = params['VARMA']['model'].k_ar
    maOrder = params['VARMA']['model'].k_ma
    numVars = len(targets)

    for target, identifier, value in fromCluster:
      value = float(value)
      targetIndex = targets.index(target)
      idSplit = identifier.split('_')
      if idSplit[0] == 'const':
        params['VARMA']['const'][targetIndex] = value
      elif idSplit[0] == 'cov':
        target2Index = targets.index(idSplit[1])
        params['VARMA']['cov'][targetIndex * (numVars - 1) + target2Index] = value
      elif idSplit[0] == 'ar':
        lag = int(idSplit[1])
        target2Index = targets.index(idSplit[2])
        params['VARMA']['ar'][targetIndex * numVars * arOrder + (lag - 1) * numVars + target2Index] = value
      elif idSplit[0] == 'ma':
        lag = int(idSplit[1])
        target2Index = targets.index(idSplit[2])
        params['VARMA']['ma'][targetIndex * numVars * maOrder + (lag - 1) * numVars + target2Index] = value

    transition, stateIntercept, stateCov, selection = self._buildStateSpaceMatrices(params['VARMA'])
    initMean, initCov = self._solveStateDistribution(transition, stateIntercept, stateCov, selection)
    initDist = self._trainMultivariateNormal(numVars, initMean, initCov)
    params['VARMA']['initDist'] = initDist
    noiseDist = self._trainMultivariateNormal(numVars, np.zeros(numVars), stateCov)
    params['VARMA']['noiseDist'] = noiseDist

    return params

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.characterize
      @ Out, None
    """
    # Before we start parsing values from the params dictionary, we should convert transform the
    # covariance parameters back into the original (non-log) space.
    cov = self._transformedFlatToCov(params['VARMA']['cov'])
    params['VARMA']['cov'] = cov[np.tril_indices_from(cov)]  # Only keep the unique values

    # We can leverage the getParamsAsVars method to parse the model parameters into a flat dictionary
    # that's easier to work with.
    parsedParams = self.getParamsAsVars(params)

    # Under the <VARMA> base node, we'll organize the parameter values by parameter type, then by target.
    constant = xmlUtils.newNode('constant')
    covariance = xmlUtils.newNode('covariance')
    ar = xmlUtils.newNode('AR')
    ma = xmlUtils.newNode('MA')

    for name, value in parsedParams.items():
      parts = name.split('__')
      # The format of name depends a bit on which parameter it's for. Here are the possibilities:
      #    Constant: VARMA__constant__<target>
      #    Variance: VARMA__variance__<target>
      #    Covariance: VARMA__covariance__<target1>_<target2>
      #    AR: VARMA__AR__<lag>__<target1>_<target2>
      #    MA: VARMA__MA__<lag>__<target1>_<target2>
      # We want to write the XML by grouping by parameter type, not by target. Given the number of
      # cross-target parameters, I think this makes more sense.
      if parts[1] == 'constant':
        constant.append(xmlUtils.newNode(parts[2], text=f'{value}'))
      elif parts[1] == 'variance':
        covariance.append(xmlUtils.newNode(f'var_{parts[2]}', text=f'{value}'))
      elif parts[1] == 'covariance':
        covariance.append(xmlUtils.newNode(f'cov_{parts[2]}', text=f'{value}'))
      elif parts[1] == 'AR':
        ar.append(xmlUtils.newNode(f'Lag{parts[2]}_{parts[3]}', text=f'{value}'))
      elif parts[1] == 'MA':
        ma.append(xmlUtils.newNode(f'Lag{parts[2]}_{parts[3]}', text=f'{value}'))
      else:
        raise ValueError(f'Unrecognized parameter name "{name}"!')

    writeTo.append(constant)
    writeTo.append(covariance)
    # Each of the AR and MA nodes may or may not have children, so we'll only append them if they do.
    if len(ar) > 0:
      writeTo.append(ar)
    if len(ma) > 0:
      writeTo.append(ma)

  # Internal utility methods
  def _solveStateDistribution(self, transition, stateIntercept, stateCov, selection):
    """
      Determines the steady state mean vector and covariance matrix of a state space model
        x_{t+1} = T x_t + R w_t + c
      where x is the state vector, T is the transition matrix, R is the selection matrix,
      w is the noise vector (w ~ N(0, Q) for state covariance matrix Q), and c is the state
      intercept vector.

      @ In, transition, np.array, transition matrix (T)
      @ In, stateIntercept, np.array, state intercept vector (c)
      @ In, stateCov, np.array, state covariance matrix (Q)
      @ In, selection, np.array, selection matrix (R)
      @ Out, mean, np.array, steady state mean vector
      @ Out, cov, np.array, steady state covariance matrix
    """
    # The mean vector (m) solves the linear system (I - T) m = c
    mean = np.linalg.solve(np.eye(transition.shape[0]) - transition, stateIntercept)
    # The covariance matrix (C) solves the discrete Lyapunov equation C = T C T' + R Q R'
    cov = sp.linalg.solve_discrete_lyapunov(transition, selection @ stateCov @ selection.T)
    return mean, cov

  def _trainMultivariateNormal(self, dim, means, cov):
    """
      Trains multivariate normal distribution for future sampling
      @ In, dim, int, number of dimensions
      @ In, means, np.array, distribution mean
      @ In, cov, np.ndarray, dim x dim matrix of covariance terms
      @ Out, dist, Distributions.MultivariateNormal, distribution
    """
    dist = Distributions.MultivariateNormal()
    dist.method = 'pca'
    dist.dimension = dim
    dist.rank = dim
    dist.mu = means
    dist.covariance = np.ravel(cov)
    dist.initializeDistribution()
    dist.workingDir = os.getcwd()
    return dist

  def _buildStateSpaceMatrices(self, params):
    """
      Builds the state space matrices for the ARMA model. Specifically, the transition, state intercept,
      state covariance, and selection matrices are built.

      @ In, params, dict, dictionary of trained model parameters
      @ Out, transition, np.array, transition matrix
      @ Out, stateIntercept, np.array, state intercept vector
      @ Out, stateCov, np.array, state covariance matrix
      @ Out, selection, np.array, selection matrix
    """
    # The state vector has dimension max(P, Q + 1)
    numVars = len(params['targets'])
    P = int(len(params['ar']) / numVars ** 2)
    Q = int(len(params['ma']) / numVars ** 2)
    dim = numVars * (P + Q)

    # The lower rows handle transferring lagged state values, and that doesn't change with parameters.
    # Copying the old transition matrix will save us some work and copy over those rows.
    transition = copy.deepcopy(params['model'].ssm['transition'])
    # Upper left block is the AR matrix. Replace with new values.
    if P > 0:
      transition[:numVars, :P*numVars] = np.hstack(params['ar'].reshape((P, numVars, numVars)))
    # Upper right block is the MA matrix. Replace with new values.
    if Q > 0:
      transition[:numVars, P*numVars:] = np.hstack(params['ma'].reshape((Q, numVars, numVars)))

    # The state intercept vector is zero except for the first numVars elements, which are the constant terms.
    stateIntercept = np.zeros(dim)
    stateIntercept[:numVars] = params['const']
    # The state covariance matrix needs to be rebuilt from the transformed flat representation.
    stateCov = self._transformedFlatToCov(params['cov'])

    # The selection matrix shouldn't need to change just because the model parameters have changed, so we
    # can copy the matrix from the existing model.
    selection = params['model'].ssm['selection']

    return transition, stateIntercept, stateCov, selection

  def _covToTransformedFlat(self, cov):
    """
      Transforms a full covariance matrix into a flattened representation of the unique
      values of the matrix logarithm of the covariance matrix. Valid covariance matrices
      are symmetric and positive semi-definite (PSD), so the logarithm is well-defined for all
      valid covariance matrices. Furthermore, the logarithm of a PSD matrix is Hermitian, so
      only the lower triangular portion of the matrix is needed to uniquely represent the
      covariance matrix.

      This is the inverse of _transformedFlatToCov.

      @ In, cov, np.array, covariance matrix
      @ Out, flat, np.array, flattened and transformed covariance matrix
    """
    # Logarithm of the covariance matrix
    log = sp.linalg.logm(cov)
    # Extract the lower triangular portion of the matrix
    flat = log[np.tril_indices_from(log)]
    return flat

  def _transformedFlatToCov(self, flat):
    """
      Transforms a flattened representation of the unique values of the matrix logarithm of the
      covariance matrix into a full covariance matrix in the original space.

      This is the inverse of _covToTransformedFlat.

      @ In, flat, np.array, flattened and transformed covariance matrix
      @ Out, cov, np.array, covariance matrix
    """
    numVars = int(np.sqrt(2 * len(flat) + 0.25) - 0.5)  # inverse of n(n+1)/2 = len(flat)
    # Create a matrix of zeros
    cov = np.zeros((numVars, numVars))
    # Fill in the lower triangular portion of the matrix
    cov[np.tril_indices(numVars)] = flat
    # Fill in the upper triangular portion of the matrix by copying the lower triangular portion
    cov[np.triu_indices(numVars, k=1)] = cov[np.tril_indices(numVars, k=-1)]
    # Exponentiate the matrix to get the covariance matrix
    cov = sp.linalg.expm(cov)
    return cov
