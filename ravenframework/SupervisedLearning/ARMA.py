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
  Created on May 8, 2018

  @author: talbpaul, wangc
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for ARMA (Autoregressive Moving Average) ROM
"""
#External Modules------------------------------------------------------------------------------------
import copy
import collections
from ..utils import importerUtils
from ..utils.utils import findCrowModule
statsmodels = importerUtils.importModuleLazy("statsmodels", globals())
import numpy as np
import functools
from scipy.linalg import solve_discrete_lyapunov
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import rv_histogram

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import randomUtils, xmlUtils, mathUtils, utils
from ..utils import InputTypes, InputData
from .. import Distributions
from .SupervisedLearning import SupervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class ARMA(SupervisedLearning):
  r"""
    Autoregressive Moving Average model for time series analysis. First train then evaluate.
    Specify a Fourier node in input file if detrending by Fourier series is needed.
    Time series Y: Y = X + \sum_{i}\sum_k [\delta_ki1*sin(2pi*k/basePeriod_i)+\delta_ki2*cos(2pi*k/basePeriod_i)]
    ARMA series X: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
  """
  # class attribute
  ## define the clusterable features for this ROM.
  _clusterableFeatures = {'global':['miu'],
                          'fourier': ['equal','shorter'],
                          #FIXME shorter fourier intepolation\\
                          'arma': ['sigma', 'p', 'q'],
                          # NO CDF
                          'peak': ['probability', 'mean', 'sigma', 'index'],
                          }

  info = {'problemtype':'regression', 'normalize':True}

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.description = r"""The \xmlString{ARMA} ROM is based on an autoregressive moving average time series model with
                        Fourier signal processing, sometimes referred to as a FARMA.
                        ARMA is a type of time dependent model that characterizes the autocorrelation between time series data. The mathematic description of ARMA is given as
                        \begin{equation*}
                        x_t = \sum_{i=1}^p\phi_ix_{t-i}+\alpha_t+\sum_{j=1}^q\theta_j\alpha_{t-j},
                        \end{equation*}
                        where $x$ is a vector of dimension $n$, and $\phi_i$ and $\theta_j$ are both $n$ by $n$ matrices. When $q=0$, the above is
                        autoregressive (AR); when $p=0$, the above is moving average (MA).
                        When
                        training an ARMA, the input needs to be a synchronized HistorySet. For unsynchronized data, use PostProcessor methods to
                        synchronize the data before training an ARMA.
                        The ARMA model implemented allows an option to use Fourier series to detrend the time series before fitting to ARMA model to
                        train. The Fourier trend will be stored in the trained ARMA model for data generation. The following equation
                        describes the detrending
                        process.
                        \begin{equation*}
                        \begin{aligned}
                        x_t &= y_t - \sum_m\left\{a_m\sin(2\pi f_mt)+b_m\cos(2\pi f_mt)\right\}  \\
                        &=y_t - \sum_m\ c_m\sin(2\pi f_mt+\phi_m)
                        \end{aligned}
                        \end{equation*}
                        where $1/f_m$ is defined by the user parameter \xmlNode{Fourier}. \nb $a_m$ and $b_m$ will be calculated then transformed to
                        $c_m$ and $\phi$. The $c_m$ will be stored as \xmlString{amplitude}, and $\phi$ will be stored as \xmlString{phase}.
                        By default, each target in the training will be considered independent and have an unique ARMA for each
                        target.  Correlated targets can be specified through the \xmlNode{correlate} node, at which point
                        the correlated targets will be trained together using a vector ARMA (or VARMA). Due to limitations in
                        the VARMA, in order to seed samples the VARMA must be trained with the node \xmlNode{seed}, which acts
                        independently from the global random seed used by other RAVEN entities.
                        Both the ARMA and VARMA make use of the \texttt{statsmodels} python package.
                        In order to use this Reduced Order Model, the \xmlNode{ROM} attribute
                        \xmlAttr{subType} needs to be \xmlString{ARMA}.
                        """
    specs.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType,
                                                 descr=r"""defines the pivot variable (e.g., time) that is non-decreasing in
                                                 the input HistorySet.""", default='time'))
    specs.addSub(InputData.parameterInputFactory("correlate", contentType=InputTypes.StringListType,
                                                 descr=r"""indicates the listed variables
                                                   should be considered as influencing each other, and trained together instead of independently.  This node
                                                   can only be listed once, so all variables that are desired for correlation should be included.  \nb The
                                                   correlated VARMA takes notably longer to train than the independent ARMAs for the same number of targets.""",
                                                   default=None))
    specs.addSub(InputData.parameterInputFactory("seed", contentType=InputTypes.IntegerType,
                                                 descr=r"""provides seed for VARMA and ARMA sampling.
                                                  Must be provided before training. If no seed is assigned,
                                                  then a random number will be used.""", default=None))
    specs.addSub(InputData.parameterInputFactory("reseedCopies", contentType=InputTypes.BoolType,
                                                 descr=r"""if \xmlString{True} then whenever the ARMA is loaded from file, a
                                                  random reseeding will be performed to ensure different histories. \nb If
                                                  reproducible histories are desired for an ARMA loaded from file,
                                                  \xmlNode{reseedCopies} should be set to \xmlString{False}, and in the
                                                  \xmlNode{RunInfo} block \xmlNode{batchSize} needs to be 1
                                                  and \xmlNode{internalParallel} should be
                                                   \xmlString{False} for RAVEN runs sampling the trained ARMA model.
                                                  If \xmlNode{InternalParallel} is \xmlString{True} and the ARMA has
                                                  \xmlNode{reseedCopies} as \xmlString{False}, an identical ARMA history
                                                  will always be provided regardless of how many samples are taken.
                                                  If \xmlNode{InternalParallel} is \xmlString{False} and \xmlNode{batchSize}
                                                  is more than 1, it is not possible to guarantee the order of RNG usage by
                                                  the separate processes, so it is not possible to guarantee reproducible
                                                  histories are generated.""", default=True))
    specs.addSub(InputData.parameterInputFactory("P", contentType=InputTypes.IntegerType,
                                                 descr=r"""defines the value of $p$.""", default=3))
    specs.addSub(InputData.parameterInputFactory("Q", contentType=InputTypes.IntegerType,
                                                 descr=r"""defines the value of $q$.""", default=3))
    specs.addSub(InputData.parameterInputFactory("Fourier", contentType=InputTypes.IntegerListType,
                                                 descr=r"""must be positive integers. This defines the
                                                   based period that will be used for Fourier detrending, i.e., this
                                                   field defines $1/f_m$ in the above equation.
                                                   When this filed is not specified, the ARMA considers no Fourier detrend.""",
                                                   default=None))

    peaks = InputData.parameterInputFactory("Peaks", contentType=InputTypes.StringType,
                                                 descr=r"""designed to estimate the peaks in signals that repeat with some frequency,
                                                 often in periodic data.""", default=None)
    peaks.addParam("target", InputTypes.StringType, required=True,
                  descr=r"""defines the name of one target (besides the
                        pivot parameter) expected to have periodic peaks.""")
    peaks.addParam("threshold", InputTypes.FloatType, required=True,
                  descr=r"""user-defined minimum required
                        height of peaks (absolute value).""")
    peaks.addParam("period", InputTypes.FloatType, required=True,
                  descr=r"""user-defined expected period for target variable.""")
    nbin= InputData.parameterInputFactory('nbin',contentType=InputTypes.IntegerType, default=5)
    window = InputData.parameterInputFactory("window", contentType=InputTypes.FloatListType,
                                                 descr=r"""lists the window of time within each period in which a peak should be discovered.
                                                 The text of this node is the upper and lower boundary of this
                                                 window \emph{relative to} the start of the period, separated by a comma.
                                                 User can define the lower bound to be a negative
                                                 number if the window passes through one side of one period. For example, if the period is 24
                                                 hours, the window can be -2,2 which is equivalent to 22, 2.""")
    window.addParam("width", InputTypes.FloatType, required=True,
                  descr=r"""The user defined  width of peaks in that window. The width is in the unit of the signal as well.""")
    peaks.addSub(nbin)
    peaks.addSub(window)
    specs.addSub(peaks)
    specs.addSub(InputData.parameterInputFactory("preserveInputCDF", contentType=InputTypes.BoolType,
                                                 descr=r"""enables a final transform on sampled
                                                   data coercing it to have the same distribution as the original data. If \xmlString{True}, then every
                                                   sample generated by this ARMA after training will have a distribution of values that conforms within
                                                   numerical accuracy to the original data. This is especially useful when variance is desired not to stretch
                                                   the most extreme events (high or low signal values), but instead the sequence of events throughout this
                                                   history. For example, this transform can preserve the load duration curve for a load signal.""", default=False))
    specificFourier = InputData.parameterInputFactory("SpecificFourier", contentType=InputTypes.StringType,
                                                 descr=r"""provides a means to specify different Fourier
                                                   decomposition for different target variables.  Values given in the subnodes of this node will supercede
                                                   the defaults set by the  \xmlNode{Fourier} and \xmlNode{FourierOrder} nodes.""", default=None)
    specificFourier.addParam("variables", InputTypes.StringListType, required=True,
                  descr=r"""lists the variables to whom
                    the \xmlNode{SpecificFourier} parameters will apply.""")
    specificFourier.addSub(InputData.parameterInputFactory("periods", contentType=InputTypes.IntegerListType,
                                                 descr=r"""lists the (fundamental)
                                                   periodic wavelength of the Fourier decomposition for these variables,
                                                   as in the \xmlNode{Fourier} general node.""", default='no-default'))
    specs.addSub(specificFourier)

    multicycle = InputData.parameterInputFactory("Multicycle", contentType=InputTypes.StringType,
                                                 descr=r"""indicates that each sample of the ARMA should yield
                                                   multiple sequential samples. For example, if an ARMA model is trained to produce a year's worth of data,
                                                   enabling \xmlNode{Multicycle} causes it to produce several successive years of data. Multicycle sampling
                                                   is independent of ROM training, and only changes how samples of the ARMA are created.
                                                   \nb The output of a multicycle ARMA must be stored in a \xmlNode{DataSet}, as the targets will depend
                                                   on both the \xmlNode{pivotParameter} as well as the cycle, \xmlString{Cycle}. The cycle is a second
                                                   \xmlNode{Index} that all targets should depend on, with variable name \xmlString{Cycle}.""", default=None)
    multicycle.addSub(InputData.parameterInputFactory("cycles", contentType=InputTypes.IntegerType,
                                                 descr=r"""the number of cycles the ARMA should produce
                                                   each time it yields a sample.""", default='no-default'))
    growth = InputData.parameterInputFactory("growth", contentType=InputTypes.FloatType,
                                                 descr=r"""if provided then the histories produced by
                                                   the ARMA will be increased by the growth factor for successive cycles. This node can be added
                                                   multiple times with different settings for different targets.
                                                   The text of this node is the growth factor in percentage. Some examples are in
                                                   Table~\ref{tab:arma multicycle growth}, where \emph{Growth factor} is the value used in the RAVEN
                                                   input and \emph{Scaling factor} is the value by which the history will be multiplied.
                                                   \begin{table}[h!]
                                                     \centering
                                                     \begin{tabular}{r c l}
                                                       Growth factor & Scaling factor & Description \\ \hline
                                                       50 & 1.5 & growing by 50\% each cycle \\
                                                       -50 & 0.5 & shrinking by 50\% each cycle \\
                                                       150 & 2.5 & growing by 150\% each cycle \\
                                                     \end{tabular}
                                                     \caption{ARMA Growth Factor Examples}
                                                     \label{tab:arma multicycle growth}
                                                   \end{table}""", default=None)
    growth.addParam("targets", InputTypes.StringListType, required=True,
                  descr=r"""lists the targets
                    in this ARMA that this growth factor should apply to.""")
    growth.addParam('start_index', InputTypes.IntegerType)
    growth.addParam('end_index', InputTypes.IntegerType)
    growthEnumType = InputTypes.makeEnumType('growth', 'armaGrowthType', ['exponential', 'linear'])
    growth.addParam("mode", growthEnumType, required=True,
                  descr=r"""either \xmlString{linear} or
                    \xmlString{exponential}, determines the manner in which the growth factor is applied.
                    If \xmlString{linear}, then the scaling factor is $(1+y\cdot g/100)$;
                    if \xmlString{exponential}, then the scaling factor is $(1+g/100)^y$;
                    where $y$ is the cycle after the first and $g$ is the provided scaling factor.""")
    multicycle.addSub(growth)
    specs.addSub(multicycle)

    specs.addSub(InputData.parameterInputFactory("nyquistScalar", contentType=InputTypes.IntegerType, default=1))
    ### ARMA zero filter
    zeroFilt = InputData.parameterInputFactory('ZeroFilter', contentType=InputTypes.StringType,
                                               descr=r"""turns on \emph{zero filtering}
                                                 for the listed targets. Zero filtering is a very specific algorithm, and should not be used without
                                                 understanding its application.  When zero filtering is enabled, the ARMA will remove all the values from
                                                 the training data equal to zero for the target, then train on the remaining data (including Fourier detrending
                                                 if applicable). If the target is set as correlated to another target, the second target will be treated as
                                                 two distinct series: one containing times in which the original target is zero, and one in the remaining
                                                 times. The results from separated ARMAs are recombined after sampling. This can be a methodology for
                                                 treating histories with long zero-value segments punctuated periodically by peaks.""", default=None)
    zeroFilt.addParam('tol', InputTypes.FloatType)
    specs.addSub(zeroFilt)
    ### ARMA out truncation
    outTrunc = InputData.parameterInputFactory("outTruncation", contentType=InputTypes.StringListType,
                                                 descr=r"""defines whether and how output
                                                   time series are limited in domain. This node has one attribute, \xmlAttr{domain}, whose value can be
                                                   \xmlString{positive} or \xmlString{negative}. The value of this node contains the list of targets to whom
                                                   this domain limitation should be applied. In the event a negative value is discovered in a target whose
                                                   domain is strictly positive, the absolute value of the original negative value will be used instead, and
                                                   similarly for the negative domain.""", default=None)
    domainEnumType = InputTypes.makeEnumType('domain', 'truncateDomainType', ['positive', 'negative'])
    outTrunc.addParam('domain', domainEnumType, True)
    specs.addSub(outTrunc)
    return specs

  ### INHERITED METHODS ###
  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, kwargs: an arbitrary dictionary of keywords and values
    """
    # general infrastructure
    super().__init__()
    self.printTag = 'ARMA'
    self._dynamicHandling  = True # This ROM is able to manage the time-series on its own.
    # training storage
    self.trainingData      = {} # holds normalized ('norm') and original ('raw') training data, by target
    self.cdfParams         = {} # dictionary of fitted CDF parameters, by target
    self.armaResult        = {} # dictionary of assorted useful arma information, by target
    self.correlations      = [] # list of correlated variables
    self.fourierResults    = {} # dictionary of Fourier results, by target
    # training parameters
    self.fourierParams     = {} # dict of Fourier training params, by target (if requested, otherwise not present)

    self.outTruncation     = {'positive':set(), 'negative':set()} # store truncation requests
    self.pivotParameterValues = None  # In here we store the values of the pivot parameter (e.g. Time)
    self._trainingCDF      = {} # if preserveInputCDF, these CDFs are scipy.stats.rv_histogram objects for the training data
    self.zeroFilterTarget  = None # target for whom zeros should be filtered out
    self.zeroFilterTol     = None # tolerance for zerofiltering to be considered zero, set below
    self._masks            = collections.defaultdict(dict)   # dictionay of masks, including zeroFilterMask(where zero), notZeroFilterMask(Where non zero), and maskPeakRes.
    self._minBins          = 20   # min number of bins to use in determining distributions, eventually can be user option, for now developer's pick
    #peaks
    self.peaks             = {} # dictionary of peaks information, by target
    # signal storage
    self._signalStorage    = collections.defaultdict(dict) # various signals obtained in the training process
    # multicycle ---> NOTE that cycles are usually years!
    self.multicycle = False # if True, then multiple cycles per sample are going to be taken
    self.numCycles = None # if self.multicycle, this is the number of cycles per sample
    self.growthFactors = collections.defaultdict(list) # by target, this is how to scale the signal over successive cycles


  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['nyquistScalar', 'P', 'Q', 'reseedCopies', 'pivotParameter',
                                                            'seed', 'preserveInputCDF', 'correlate'])
    assert(not notFound)
    self.nyquistScalar     = settings.get('nyquistScalar')
    self.P                 = settings.get('P') # autoregressive lag
    self.Q                 = settings.get('Q') # moving average lag
    # data manipulation
    self.reseedCopies      = settings.get('reseedCopies') # reseed unless explicitly asked not to
    self.pivotParameterID  = settings.get('pivotParameter')
    self.seed              = settings.get('seed')
    self.preserveInputCDF  = settings.get('preserveInputCDF') # if True, then CDF of the training data will be imposed on the final sampled signal
    # get seed if provided
    ## FIXME only applies to VARMA sampling right now, since it has to be sampled through Numpy!
    ## see note under "check for correlation" below.
    if self.seed is None:
      self.seed = randomUtils.randomIntegers(0,4294967295,self)
    else:
      self.seed = int(self.seed)
    self.normEngine = Distributions.factory.returnInstance('Normal')
    self.normEngine.mean = 0.0
    self.normEngine.sigma = 1.0
    self.normEngine.upperBoundUsed = False
    self.normEngine.lowerBoundUsed = False
    self.normEngine.initializeDistribution()

    self.setEngine(randomUtils.newRNG(), seed=self.seed)

    # FIXME set the numpy seed
      ## we have to do this because VARMA.simulate does not accept a random number generator,
      ## but instead uses numpy directly.  As a result, for now, we have to seed numpy.
      ## Because we use our RNG to set the seed, though, it should follow the global seed still.
    self.raiseADebug('Setting ARMA seed to',self.seed)
    randomUtils.randomSeed(self.seed,engine=self.randomEng)
    # check for correlation
    correlated = settings.get('correlate')
    if correlated is not None:
      np.random.seed(self.seed)
      corVars = correlated
      for var in corVars:
        if var not in self.target:
          self.raiseAnError(IOError,'Variable "{}" requested in "correlate" but not found among the targets!'.format(var))
      # NOTE: someday, this could be expanded to include multiple sets of correlated variables.
      self.correlations = correlated

    # check if the pivotParameter is among the targetValues
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")

    # can only handle one scaling input currently
    if len(self.features) != 1:
      self.raiseAnError(IOError,"The ARMA can only currently handle a single feature, which scales the outputs!")

    multicycleNode = paramInput.findFirst('Multicycle')
    if multicycleNode is not None:
      self.setMulticycleParams(multicycleNode)

    for child in paramInput.subparts:
      # read truncation requests (really value limits, not truncation)
      if child.getName() == 'outTruncation':
        # did the user request positive or negative?
        domain = child.parameterValues['domain']
        # if a recognized request, store it for later
        if domain in self.outTruncation:
          self.outTruncation[domain] = self.outTruncation[domain] | set(child.value)
        # if unrecognized, error out
        else:
          self.raiseAnError(IOError,'Unrecognized "domain" for "outTruncation"! Was expecting "positive" '+\
                                    'or "negative" but got "{}"'.format(domain))
      # additional info for zerofilter
      elif child.getName() == 'ZeroFilter':
        self.zeroFilterTarget = child.value
        if self.zeroFilterTarget not in self.target:
          self.raiseAnError(IOError,'Requested zero filtering for "{}" but not found among targets!'.format(self.zeroFilterTarget))
        self.zeroFilterTol = child.parameterValues.get('tol', 1e-16)
      # read SPECIFIC parameters for Fourier detrending
      elif child.getName() == 'SpecificFourier':
        # clear old information
        periods = None
        # what variables share this Fourier?
        variables = child.parameterValues['variables']
        # check for variables that aren't targets
        missing = set(variables) - set(self.target)
        if len(missing):
          self.raiseAnError(IOError,
                            'Requested SpecificFourier for variables {} but not found among targets!'.format(missing))
        # record requested Fourier periods
        for cchild in child.subparts:
          if cchild.getName() == 'periods':
            periods = cchild.value
        # set these params for each variable
        for v in variables:
          self.raiseADebug('recording specific Fourier settings for "{}"'.format(v))
          if v in self.fourierParams:
            self.raiseAWarning('Fourier params for "{}" were specified multiple times! Using first values ...'
                               .format(v))
            continue
          self.fourierParams[v] = sorted(periods, reverse=True) # Must be largest to smallest!
      elif child.getName() == 'Peaks':
        # read peaks information for each target
        peak={}
        # creat an empty list for each target
        threshold = child.parameterValues['threshold']
        peak['threshold']=threshold
        # read the threshold for the peaks and store it in the dict
        period = child.parameterValues['period']
        peak['period']=period
        # read the period for the peaks and store it in the dict
        windows=[]
        nbin=5
        # creat an empty list to store the windows' information
        for cchild in child.subparts:

          if cchild.getName() == 'window':
            tempDict={}
            window = cchild.value
            width = cchild.parameterValues['width']
            tempDict['window']=window
            tempDict['width']=width
            # for each window in the windows, we create a dictionary. Then store the
            # peak's width, the index of stating point and ending point in time unit
            windows.append(tempDict)
          elif cchild.getName() == 'nbin':
            nbin=cchild.value
        peak['windows']=windows
        peak['nbin']=nbin

        target = child.parameterValues['target']
        # target is the key to reach each peak information
        self.peaks[target]=peak
    # read GENERAL parameters for Fourier detrending
    ## these apply to everyone without SpecificFourier nodes
    ## use basePeriods to check if Fourier node present
    basePeriods = paramInput.findFirst('Fourier')
    if basePeriods is not None:
      # read periods
      basePeriods = basePeriods.value
      if len(set(basePeriods)) != len(basePeriods):
        self.raiseAnError(IOError,'Some <Fourier> periods have been listed multiple times!')
      # set to any variable that doesn't already have a specific one
      for v in set(self.target) - set(self.fourierParams.keys()):
        self.raiseADebug('setting general Fourier settings for "{}"'.format(v))
        self.fourierParams[v] = sorted(basePeriods, reverse=True) # Must be largest to smallest!

  def __getstate__(self):
    """
      Obtains state of object for pickling.
      @ In, None
      @ Out, d, dict, stateful dictionary
    """
    d = super().__getstate__()
    return d

  def __setstate__(self, d):
    """
      Sets state of object from pickling.
      @ In, d, dict, stateful dictionary
      @ Out, None
    """
    rngCounts = d.pop('crow_rng_counts', None)
    super().__setstate__(d)

    try:
      self.randomEng
    except AttributeError:  # catches where ARMA was pickled without saving the RNG
      self.setEngine(randomUtils.newRNG(), seed=self.seed, count=rngCounts)

    if self.reseedCopies:
      randd = np.random.randint(1, 2e9)
      self.reseed(randd)

  def setMulticycleParams(self, node):
    """
      Sets multicycle parameters in an object-oriented sense
      @ In, node, InputData, input specs (starting with 'multicycle' node)
      @ Out, None
    """
    self.multicycle = True
    self.numCycles = 0 # minimum
    # clear existing parameters
    self.growthFactors = collections.defaultdict(list)
    growthNodes = node.findAll('growth')
    numCyclesNode = node.findFirst('cycles')
    # if <cycles> given, then we use that as the baseline default duration range(0, cycles) (not inclusive)
    if numCyclesNode is not None:
      defaultIndices = [0, numCyclesNode.value - 1]
    else:
      defaultIndices = [None, None]
    # read in settings from each <growth> node
    ## NOTE that each target may have multiple <growth> nodes.
    checkOverlap = collections.defaultdict(set)
    for gNode in growthNodes:
      targets = gNode.parameterValues['targets']
      # sanity check ...
      for target in targets:
        if target not in self.target:
          self.raiseAnError(IOError, 'Growth parameters were given for "{t}" but "{t}" is not '.format(t=target),
                            'among the targets of this ROM!')
      settings = {'mode': gNode.parameterValues['mode'],
                  'start': gNode.parameterValues.get('start_index', defaultIndices[0]),
                  'end': gNode.parameterValues.get('end_index', defaultIndices[1]),
                  'value': gNode.value}
      # check that a valid index set has been supplied
      if settings['start'] is None:
        self.raiseAnError(IOError, 'No start index for Multicycle <growth> attribute "start_index" ' +
                          'for targets {} was specified, '.format(gNode.parameterValues['targets'])+
                          'and no default <cycles> given!')
      if settings['end'] is None:
        self.raiseAnError(IOError, 'No end index for Multicycle <growth> attribute "end_index" ' +
                          'for targets {} was specified, '.format(gNode.parameterValues['targets'])+
                          'and no default <cycles> given!')
      self.numCycles = max(self.numCycles, settings['end']+1)
      # check for overlapping coverage
      newCoverage = range(settings['start'], settings['end']+1)
      settings['range'] = newCoverage
      # store results by target
      for target in gNode.parameterValues['targets']:
        for existing in self.growthFactors[target]:
          overlap = range(max(existing['range'].start, newCoverage.start),
                          min(existing['range'].stop-1, newCoverage.stop-1) + 1)
          if overlap:
            self.raiseAnError(IOError, 'Target "{}" has overlapping growth factors for cycles with index'.format(target),
                               ' {} to {} (inclusive)!'.format(overlap.start, overlap.stop - 1))
        self.growthFactors[target].append(settings)
    else:
      self.numCycles = numCyclesNode.value

  def setAdditionalParams(self, params):
    """
      Sets parameters aside from initialization, such as during deserialization.
      @ In, params, dict, parameters to set (dependent on ROM)
      @ Out, None
    """
    # reseeding is taken care of in the supervisedLearning base class of this method
    SupervisedLearning.setAdditionalParams(self, params)
    paramInput = params['paramInput']
    # multicycle; note that myNode is "multicycleNode" not a node that I own
    myNode = paramInput.findFirst('Multicycle')
    if myNode:
      self.setMulticycleParams(myNode)

  def _train(self, featureVals, targetVals):
    """
      Perform training on input database stored in featureVals.

      @ In, featureVals, array, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], an array of time series data
    """
    self.raiseADebug('Training...')
    # obtain pivot parameter
    self.raiseADebug('... gathering pivot values ...')
    self.pivotParameterValues = targetVals[:,:,self.target.index(self.pivotParameterID)]
    # NOTE: someday, this ARMA could be expanded to take Fourier signals in time on the TypicalHistory,
    #   and then use several realizations of the target to train an ND ARMA that captures not only
    #   the mean and variance in time, but the mean, variance, skewness, and kurtosis over time and realizations.
    #   In this way, outliers in the training data could be captured with significantly more representation.
    if len(self.pivotParameterValues) > 1:
      self.raiseAnError(Exception,self.printTag +" does not handle multiple histories data yet! # histories: "+str(len(self.pivotParameterValues)))
    self.pivotParameterValues.shape = (self.pivotParameterValues.size,)
    targetVals = np.delete(targetVals,self.target.index(self.pivotParameterID),2)[0]
    # targetVals now has shape (1, # time samples, # targets)
    self.target.pop(self.target.index(self.pivotParameterID))

    # prep the correlation data structure
    correlationData = np.zeros([len(self.pivotParameterValues),len(self.correlations)])

    for t,target in enumerate(self.target):
      timeSeriesData = targetVals[:,t]
      self._signalStorage[target]['original'] = copy.deepcopy(timeSeriesData)
      # if we're enforcing the training CDF, we should store it now
      if self.preserveInputCDF:
        self._trainingCDF[target] = mathUtils.trainEmpiricalFunction(timeSeriesData, minBins=self._minBins)
      # if this target governs the zero filter, extract it now
      if target == self.zeroFilterTarget:
      # # if we're removing Fourier signal, do that now.
        if 'notZeroFilterMask' not in self._masks[target]:
          self._masks[target]['zeroFilterMask']= self._trainZeroRemoval(timeSeriesData,tol=self.zeroFilterTol) # where zeros or less than zeros are
          self._masks[target]['notZeroFilterMask'] = np.logical_not(self._masks[target]['zeroFilterMask']) # where data are
          # if correlated, then all the correlated variables share the same masks
          if target in self.correlations:
            for cor in self.correlations:
              if cor == target:
                continue
              self._masks[cor]['zeroFilterMask'] = self._masks[target]['zeroFilterMask']
              self._masks[cor]['notZeroFilterMask'] = self._masks[target]['notZeroFilterMask']
      # if we're removing Fourier signal, do that now.

      if target in self.peaks:
        peakResults=self._trainPeak(timeSeriesData,windowDict=self.peaks[target])
        self.peaks[target].update(peakResults)
        if target not in self._masks.keys():
          self._masks[target] = {}
        self._masks[target]['maskPeakRes']= peakResults['mask']
      # Make a full mask

      if target in self.fourierParams:
        # Make a full mask
        fullMask = np.ones(len(timeSeriesData), dtype=bool)
        if target in self._masks.keys():
          fullMask = self._combineMask(self._masks[target])

        self.raiseADebug('... analyzing Fourier signal  for target "{}" ...'.format(target))
        self.fourierResults[target] = self._trainFourier(self.pivotParameterValues,
                                                         self.fourierParams[target],
                                                         timeSeriesData,
                                                         masks=fullMask,  # In future, a consolidated masking system for multiple signal processors can be implemented.
                                                         target=target)
        self._signalStorage[target]['fourier'] = copy.deepcopy(self.fourierResults[target]['predict'])
        timeSeriesData -= self.fourierResults[target]['predict']
        self._signalStorage[target]['nofourier'] = copy.deepcopy(timeSeriesData)
      # zero filter application
      ## find the mask for the requested target where values are nonzero
      if target == self.zeroFilterTarget:
        # artifically force signal to 0 post-fourier subtraction where it should be zero
        zfMask= self._masks[target]['zeroFilterMask']
        targetVals[:,t][zfMask] = 0.0
        self._signalStorage[target]['zerofilter'] = copy.deepcopy(timeSeriesData)

    # Transform data to obatain normal distrbuted series. See
    # J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
    # Applied Energy, 87(2010) 843-855
    #
    # Kernel density estimation has also been tried for estimating the CDF of the data but with little practical
    # benefit over using the empirical CDF. See RAVEN Theory Manual for more discussion.
    for t,target in enumerate(self.target):
      # if target correlated with the zero-filter target, truncate the training material now?
      timeSeriesData = targetVals[:,t]
      self.raiseADebug('... analyzing ARMA properties for target "{}" ...'.format(target))
      self.cdfParams[target] = self._trainCDF(timeSeriesData, binOps=2)
      # normalize data
      normed = self._normalizeThroughCDF(timeSeriesData, self.cdfParams[target])
      self._signalStorage[target]['gaussianed'] = copy.deepcopy(normed[:])
      # check if this target is part of a correlation set, or standing alone
      if target in self.correlations:
        # store the data and train it separately in a moment
        ## keep data in order of self.correlations
        correlationData[:,self.correlations.index(target)] = normed
        self.raiseADebug('... ... saving to train with other correlated variables.')
      else:
        # go ahead and train it now
        ## if using zero filtering and target is the zero-filtered, only train on the masked part
        self.raiseADebug('... ... training "{}"...'.format(target))
        if target == self.zeroFilterTarget:
          fullMask = self._combineMask(self._masks[target])
        else:
          fullMask = np.ones(len(timeSeriesData), dtype=bool)
        self.armaResult[target] = self._trainARMA(normed,masks=fullMask)
        self.raiseADebug('... ... finished training target "{}"'.format(target))

    # now handle the training of the correlated armas
    if len(self.correlations):
      self.raiseADebug('... ... training correlated: {} ...'.format(self.correlations))
      # if zero filtering, then all the correlation data gets split
      if self.zeroFilterTarget in self.correlations:
        # split data into the zero-filtered and non-zero filtered
        notZeroFilterMask = self._masks[self.zeroFilterTarget]['notZeroFilterMask']
        zeroFilterMask = self._masks[self.zeroFilterTarget]['zeroFilterMask']
        unzeroed = correlationData[notZeroFilterMask]
        zeroed = correlationData[zeroFilterMask]
        ## throw out the part that's all zeros (axis 1, row corresponding to filter target)
        #print('mask:', self._masks[self.zeroFilterTarget]['zeroFilterMask'])
        zeroed = np.delete(zeroed, self.correlations.index(self.zeroFilterTarget), 1)
        self.raiseADebug('... ... ... training unzeroed ...')
        unzVarma, unzNoise, unzInit = self._trainVARMA(unzeroed)
        self.raiseADebug('... ... ... training zeroed ...')
        ## the VAR fails if only 1 variable is non-constant, so we need to decide whether "zeroed" is actually an ARMA
        ## -> instead of a VARMA
        if zeroed.shape[1] == 1:
          # then actually train an ARMA instead
          zVarma = self._trainARMA(zeroed,masks=None)
          zNoise = None # NOTE this is used to check whether an ARMA was trained later!
          zInit = None
        else:
          zVarma, zNoise, zInit = self._trainVARMA(zeroed)
        self.varmaResult = (unzVarma, zVarma) # NOTE how for zero-filtering we split the results up
        self.varmaNoise = (unzNoise, zNoise)
        self.varmaInit = (unzInit, zInit)
      else:
        varma, noiseDist, initDist = self._trainVARMA(correlationData)
        # FUTURE if extending to multiple VARMA per training, these will need to be dictionaries
        self.varmaResult = (varma,)
        self.varmaNoise = (noiseDist,)
        self.varmaInit = (initDist,)

  def __evaluateLocal__(self, featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    if self.multicycle:
      ## create storage for the sampled result
      finalResult = dict((target, np.zeros((self.numCycles, len(self.pivotParameterValues)))) for target in self.target if target != self.pivotParameterID)
      finalResult[self.pivotParameterID] = self.pivotParameterValues
      cycles = np.arange(self.numCycles)
      # calculate scaling factors for targets
      scaling = {}
      for target in (t for t in self.target if t != self.pivotParameterID):
        scaling[target] = self._evaluateScales(self.growthFactors[target], cycles)
      # create synthetic history for each cycle
      for y in cycles:
        self.raiseADebug('Evaluating cycle', y)
        vals = copy.deepcopy(featureVals) # without deepcopy, the vals are modified in-place -> why should this matter?
        result = self._evaluateCycle(vals)
        for target, value in ((t, v) for (t, v) in result.items() if t != self.pivotParameterID): #, growthInfos in self.growthFactors.items():
          finalResult[target][y][:] = value # [:] is a size checker
      # apply growth factors
      for target in (t for t in finalResult if t != self.pivotParameterID):
        scaling = self._evaluateScales(self.growthFactors[target], cycles)
        finalResult[target][:] = (finalResult[target].T * scaling).T # -> people say this is as fast as any way to multiply columns by a vector of scalars
      # high-dimensional indexing information
      finalResult['Cycle'] = cycles
      finalResult['_indexMap'] = dict((target, ['Cycle', self.pivotParameterID]) for target in self.target if target != self.pivotParameterID)
      return finalResult
    else:
      return self._evaluateCycle(featureVals)

  def _evaluateScales(self, growthInfos, cycles):
    """
      @ In, growthInfo, dictionary of growth value for each target
      @ In, cycle, int, cycle index in multicycle
      @ Out, scale, float, scaling factor for each cycle
    """
    scales = np.ones(len(cycles))
    for y, cycle in enumerate(cycles):
      old = scales[y-1] if y > 0 else 1
      for growthInfo in growthInfos:
        if cycle in growthInfo['range']:
          mode = growthInfo['mode']
          growth = growthInfo['value'] / 100
          scales[y] = (old * (1 + growth)) if mode == 'exponential' else (old + growth)
          break
      else:
        scales[y] = old
    return scales

  def _evaluateCycle(self, featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, returnEvaluation, dict, dictionary of values for each target (and pivot parameter)
    """
    if featureVals.size > 1:
      self.raiseAnError(ValueError, 'The input feature for ARMA for evaluation cannot have size greater than 1. ')

    # Instantiate a normal distribution for time series synthesis (noise part)
    # TODO USE THIS, but first retrofix rvs on norm to take "size=") for number of results

    # make sure pivot value is in return object
    returnEvaluation = {self.pivotParameterID:self.pivotParameterValues}

    # TODO when we have output printing for ROMs, the distinct signals here could be outputs!
    # leaving "debuggFile" as examples of this, in comments
    #debuggFile = open('signal_bases.csv','w')
    #debuggFile.writelines('Time,'+','.join(str(x) for x in self.pivotParameterValues)+'\n')
    correlatedSample = None
    for target in self.target:
      # start with the random gaussian signal
      if target in self.correlations:
        # where is target in correlated data
        corrIndex = self.correlations.index(target)
        # check if we have zero-filtering in play here
        if len(self.varmaResult) > 1:
          # where would the filter be in the index lineup had we included it in the zeroed varma?
          filterTargetIndex = self.correlations.index(self.zeroFilterTarget)
          # if so, we need to sample both VARMAs
          # have we already taken the correlated sample yet?
          if correlatedSample is None:
            # if not, take the samples now
            unzeroedSample = self._generateVARMASignal(self.varmaResult[0],
                                                       numSamples=self._masks[target]['notZeroFilterMask'].sum(),
                                                       randEngine=self.normEngine.rvs,
                                                       rvsIndex=0)
            ## zero sampling is dependent on whether the trained model is a VARMA or ARMA
            if self.varmaNoise[1] is not None:
              zeroedSample = self._generateVARMASignal(self.varmaResult[1],
                                                       numSamples=self._masks[target]['zeroFilterMask'].sum(),
                                                       randEngine=self.normEngine.rvs,
                                                       rvsIndex=1)
            else:
              result = self.varmaResult[1]
              sample = self._generateARMASignal(result,
                                                numSamples = self._masks[target]['zeroFilterMask'].sum(),
                                                randEngine = self.randomEng)
              zeroedSample = np.zeros((self._masks[target]['zeroFilterMask'].sum(),1))
              zeroedSample[:, 0] = sample
            correlatedSample = True # placeholder, signifies we've sampled the correlated distribution
          # reconstruct base signal from samples
          ## initialize
          signal = np.zeros(len(self.pivotParameterValues))
          ## first the data from the non-zero portions of the original signal
          signal[self._masks[self.zeroFilterTarget]['notZeroFilterMask']] = unzeroedSample[:,corrIndex]
          ## then the data from the zero portions (if the filter target, don't bother because they're zero anyway)
          if target != self.zeroFilterTarget:
            # fix offset since we didn't include zero-filter target in zeroed correlated arma
            indexOffset = 0 if corrIndex < filterTargetIndex else -1
            signal[self._masks[self.zeroFilterTarget]['zeroFilterMask']] = zeroedSample[:,corrIndex+indexOffset]
        # if no zero-filtering (but still correlated):
        else:
          ## check if sample taken yet
          if correlatedSample is None:
            ## if not, do so now
            correlatedSample = self._generateVARMASignal(self.varmaResult[0],
                                                         numSamples = len(self.pivotParameterValues),
                                                         randEngine = self.normEngine.rvs,
                                                         rvsIndex = 0)
          # take base signal from sample
          signal = correlatedSample[:,self.correlations.index(target)]
      # if NOT correlated
      else:
        result = self.armaResult[target] # ARMAResults object
        # generate baseline ARMA + noise
        # are we zero-filtering?
        if target == self.zeroFilterTarget:
          sample = self._generateARMASignal(result,
                                            numSamples = self._masks[target]['notZeroFilterMask'].sum(),
                                            randEngine = self.randomEng)

          ## if so, then expand result into signal space (functionally, put back in all the zeros)
          signal = np.zeros(len(self.pivotParameterValues))
          signal[self._masks[target]['notZeroFilterMask']] = sample
        else:
          ## if not, no extra work to be done here!
          sample = self._generateARMASignal(result,
                                            numSamples = len(self.pivotParameterValues),
                                            randEngine = self.randomEng)
          signal = sample
      # END creating base signal
      # DEBUG adding arbitrary variables for debugging, TODO find a more elegant way, leaving these here as markers
      #returnEvaluation[target+'_0base'] = copy.copy(signal)
      # denoise
      signal = self._denormalizeThroughCDF(signal, self.cdfParams[target])
      # DEBUG adding arbitrary variables
      #returnEvaluation[target+'_1denorm'] = copy.copy(signal)
      #debuggFile.writelines('signal_arma,'+','.join(str(x) for x in signal)+'\n')

      # Add fourier trends
      if target in self.fourierParams:
        signal += self.fourierResults[target]['predict']
        # DEBUG adding arbitrary variables
        #returnEvaluation[target+'_2fourier'] = copy.copy(signal)
        #debuggFile.writelines('signal_fourier,'+','.join(str(x) for x in self.fourierResults[target]['predict'])+'\n')
      if target in self.peaks:
        signal = self._transformBackPeaks(signal,windowDict=self.peaks[target])
        #debuggFile.writelines('signal_peak,'+','.join(str(x) for x in signal)+'\n')

      # if enforcing the training data CDF, apply that transform now
      if self.preserveInputCDF:
        signal = self._transformThroughInputCDF(signal, self._trainingCDF[target])

      # Re-zero out zero filter target's zero regions
      if target == self.zeroFilterTarget:
        # DEBUG adding arbitrary variables
        #returnEvaluation[target+'_3zerofilter'] = copy.copy(signal)
        signal[self._masks[target]['zeroFilterMask']] = 0.0

      # Domain limitations
      for domain,requests in self.outTruncation.items():
        if target in requests:
          if domain == 'positive':
            signal = np.absolute(signal)
          elif domain == 'negative':
            signal = -np.absolute(signal)
        # DEBUG adding arbitrary variables
        #returnEvaluation[target+'_4truncated'] = copy.copy(signal)

      # store results
      ## FIXME this is ASSUMING the input to ARMA is only ever a single scaling factor.
      signal *= featureVals[0]
      # DEBUG adding arbitrary variables
      #returnEvaluation[target+'_5scaled'] = copy.copy(signal)

      # sanity check on the signal
      assert(signal.size == returnEvaluation[self.pivotParameterID].size)
      #debuggFile.writelines('final,'+','.join(str(x) for x in signal)+'\n')
      returnEvaluation[target] = signal
    # END for target in targets
    return returnEvaluation

  def reseed(self, seed):
    """
      Used to set the underlying random seed.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    #self.raiseADebug('Reseeding ARMA with seed "{}"'.format(seed))
    randomUtils.randomSeed(seed, engine=self.randomEng)
    self.seed = seed

  ### UTILITY METHODS ###
  def _computeNumberOfBins(self, data, binOps=None):
    """
      Uses the Freedman-Diaconis rule for histogram binning
      -> For relatively few samples, this can cause unnatural flat-lining on low, top end of CDF
      @ In, data, np.array, data to bin
      @ Out, n, integer, number of bins
    """
    # leverage the math utils implementation
    n, _ = mathUtils.numBinsDraconis(data, low=self._minBins, alternateOkay=True,binOps=binOps)
    return n

  def _denormalizeThroughCDF(self, data, params):
    """
      Normalizes "data" using a Gaussian normal plus CDF of data
      @ In, data, np.array, data to normalize with
      @ In, params, dict, CDF parameters (as obtained by "generateCDF")
      @ Out, normed, np.array, normalized data
    """
    denormed = self.normEngine.cdf(data)
    denormed = self._sampleICDF(denormed, params)
    return denormed

  def _generateARMASignal(self, model, numSamples=None,randEngine=None):
    """
      Generates a synthetic history from fitted parameters.
      @ In, model, statsmodels.tsa.arima_model.ARMAResults, fitted ARMA such as otained from _trainARMA
      @ In, numSamples, int, optional, number of samples to take (default to pivotParameters length)
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ Out, hist, np.array(float), synthetic ARMA signal
    """
    if numSamples is None:
      numSamples =  len(self.pivotParameterValues)
    if randEngine is None:
      randEngine=self.randomEng
    import statsmodels.api
    hist = statsmodels.tsa.arima_process.arma_generate_sample(ar=model.polyar,
                                                              ma=model.polyma,
                                                              nsample=numSamples,
                                                              distrvs=functools.partial(randomUtils.randomNormal,engine=randEngine),
                                                              # functool.partial provide the random number generator as a function
                                                              # with normal distribution and take engine as the positional arguments keywords.
                                                              scale=model.sigma,
                                                              burnin=2*max(self.P,self.Q)) # @alfoa, 2020
    return hist

  def _generateFourierSignal(self, pivots, periods):
    """
      Generate fourier signal as specified by the input file
      @ In, pivots, np.array, pivot values (e.g. time)
      @ In, periods, list, list of Fourier periods (1/frequency)
      @ Out, fourier, array, shape = [n_timeStep, n_basePeriod]
    """
    fourier = np.zeros((pivots.size, 2*len(periods))) # sin, cos for each period
    for p, period in enumerate(periods):
      hist = 2. * np.pi / period * pivots
      fourier[:, 2 * p] = np.sin(hist)
      fourier[:, 2 * p + 1] = np.cos(hist)
    return fourier

  def _generateVARMASignal(self, model, numSamples=None, randEngine=None, rvsIndex=None):
    """
      Generates a set of correlated synthetic histories from fitted parameters.
      @ In, model, statsmodels.tsa.statespace.VARMAX, fitted VARMA such as otained from _trainVARMA
      @ In, numSamples, int, optional, number of samples to take (default to pivotParameters length)
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ In, rvsIndex, int, optional, if provided then will take from list of varmaNoise and varmaInit distributions
      @ Out, hist, np.array(float), synthetic ARMA signal
    """
    if numSamples is None:
      numSamples = len(self.pivotParameterValues)
    # sample measure, state shocks
    ## TODO it appears that measure shock always has a 0 variance multivariate normal, so just create it
    numVariables = len(self.correlations)
    if rvsIndex == 1:
      # TODO implicit; this indicates that we're sampling ZEROED correlated variables,
      # -> so the dimensionality is actually one less (since we don't train the VARMA coupled to the all-zeroes variable)
      numVariables -= 1
    measureShocks = np.zeros([numSamples, numVariables])
    ## state shocks come from sampling multivariate
    noiseDist = self.varmaNoise
    initDist = self.varmaInit
    if rvsIndex is not None:
      noiseDist = noiseDist[rvsIndex]
      initDist = initDist[rvsIndex]
    # with NUMPY:
    mean = noiseDist.mu
    cov = noiseDist.covariance.reshape([len(mean)]*2)
    stateShocks = np.random.multivariate_normal(mean, cov, numSamples)
    # with CROW:
    #stateShocks = np.array([noiseDist.rvs() for _ in range(numSamples)])
    # pick an intial by sampling multinormal distribution
    init = np.array(initDist.rvs())
    obs, states = model.ssm.simulate(numSamples,
                                     initial_state=init,
                                     measurement_shocks=measureShocks,
                                     state_shocks=stateShocks)
    # add zeros back in for zeroed variable, if necessary? FIXME -> looks like no, this is done later in _evaluateCycle
    return obs

  def _interpolateDist(self, x, y, Xlow, Xhigh, Ylow, Yhigh, inMask):
    """
      Interplotes values for samples "x" to get dependent values "y" given bins
      @ In, x, np.array, sampled points (independent var)
      @ In, y, np.array, sampled points (dependent var)
      @ In, Xlow, np.array, left-nearest neighbor in empirical distribution for each x
      @ In, Xhigh, np.array, right-nearest neighbor in empirical distribution for each x
      @ In, Ylow, np.array, value at left-nearest neighbor in empirical distribution for each x
      @ In, Yhigh, np.array, value at right-nearest neighbor in empirical distribution for each x
      @ In, inMask, np.array, boolean mask in "y" where the distribution values apply
      @ Out, y, np.array, same "y" but with values inserted
    """
    # treat potential divide-by-zeroes specially
    ## mask
    divZero = Xlow == Xhigh
    ## careful when using double masks
    zMask=[a[divZero] for a in np.where(inMask)]
    y[tuple(zMask)] =  0.5*(Yhigh[divZero] + Ylow[divZero])
    # interpolate all other points as y = low + slope*frac
    ## mask
    okay = np.logical_not(divZero)
    ## empirical CDF change in y, x
    dy = Yhigh[okay] - Ylow[okay]
    dx = Xhigh[okay] - Xlow[okay]
    ## distance from x to low is fraction through dx
    frac = x[inMask][okay] - Xlow[okay]
    ## careful when using double masks
    ## Adding tuple to the mask for future warning
    # FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    okayMask=[a[okay] for a in np.where(inMask)]
    y[tuple(okayMask)] = Ylow[okay] + dy/dx * frac
    return y

  def _normalizeThroughCDF(self, data, params):
    """
      Normalizes "data" using a Gaussian normal plus CDF of data
      @ In, data, np.array, data to normalize with
      @ In, params, dict, CDF parameters (as obtained by "generateCDF")
      @ Out, normed, np.array, normalized data
    """
    normed = self._sampleCDF(data, params)
    normed = self.normEngine.ppf(normed)
    return normed

  def _sampleCDF(self, x, params):
    """
      Samples the CDF defined in 'params' to get values
      @ In, x, float, value at which to sample inverse CDF
      @ In, params, dict, CDF parameters (as constructed by "_trainCDF")
      @ Out, y, float, value of inverse CDF at x
    """
    # TODO could this be covered by an empirical distribution from Distributions?
    # set up I/O
    x = np.atleast_1d(x)
    y = np.zeros(x.shape)
    # create masks for data outside range (above, below), inside range of empirical CDF
    belowMask = x <= params['bins'][0]
    aboveMask = x >= params['bins'][-1]
    inMask = np.logical_and(np.logical_not(belowMask), np.logical_not(aboveMask))
    # outside CDF set to min, max CDF values
    y[belowMask] = params['cdf'][0]
    y[aboveMask] = params['cdf'][-1]
    # for points in the CDF linearly interpolate between empirical entries
    ## get indices where points should be inserted (gives higher value)
    indices = np.searchsorted(params['bins'],x[inMask])
    Xlow = params['bins'][indices-1]
    Ylow = params['cdf'][indices-1]
    Xhigh = params['bins'][indices]
    Yhigh = params['cdf'][indices]
    y = self._interpolateDist(x,y,Xlow,Xhigh,Ylow,Yhigh,inMask)
    # numerical errors can happen due to not-sharp 0 and 1 in empirical cdf
    ## also, when Crow dist is asked for ppf(1) it returns sys.max (similar for ppf(0))
    y[y >= 1.0] = 1.0 - np.finfo(float).eps
    y[y <= 0.0] = np.finfo(float).eps
    return y

  def _sampleICDF(self, x, params):
    """
      Samples the inverse CDF defined in 'params' to get values
      @ In, x, float, value at which to sample inverse CDF
      @ In, params, dict, CDF parameters (as constructed by "_trainCDF")
      @ Out, y, float, value of inverse CDF at x
   """
    # TODO could this be covered by an empirical distribution from Distributions?
    # set up I/O
    x = np.atleast_1d(x)
    y = np.zeros(x.shape)
    # create masks for data outside range (above, below), inside range of empirical CDF
    belowMask = x <= params['cdf'][0]
    aboveMask = x >= params['cdf'][-1]
    inMask = np.logical_and(np.logical_not(belowMask), np.logical_not(aboveMask))
    # outside CDF set to min, max CDF values
    y[belowMask] = params['bins'][0]
    y[aboveMask] = params['bins'][-1]
    # for points in the CDF linearly interpolate between empirical entries
    ## get indices where points should be inserted (gives higher value)
    indices = np.searchsorted(params['cdf'],x[inMask])
    Xlow = params['cdf'][indices-1]
    Ylow = params['bins'][indices-1]
    Xhigh = params['cdf'][indices]
    Yhigh = params['bins'][indices]
    y = self._interpolateDist(x,y,Xlow,Xhigh,Ylow,Yhigh,inMask)
    return y

  def _trainARMA(self, data, masks=None):
    r"""
      Fit ARMA model: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
      @ In, data, np.array(float), data on which to train
      @ In, masks, np.array, optional, boolean mask where is the signal should be train by ARMA
      @ Out, results, statsmodels.tsa.arima_model.ARMAResults, fitted ARMA
    """
    if masks is not None:
      # Setting masked values to NaN instead of removing them preserves the correct lag between unmasked values, leading
      # to a more accurate fit in the case that values not at the ends of the array are being masked.
      data[~masks] = np.nan
    import statsmodels.api
    results = statsmodels.tsa.arima.model.ARIMA(data, order=(self.P, 0, self.Q), trend='c').fit()
    # The ARIMAResults object here can cause problems with ray when running in parallel. Dropping it
    # in exchange for the armaResultsProxy class avoids the issue while preserving what we really
    # care out from the ARIMAResults object.
    results = armaResultsProxy(results.polynomial_ar,
                               results.polynomial_ma,
                               np.sqrt(results.params[results.param_names.index('sigma2')]))
    return results

  def _trainCDF(self, data, binOps=None):
    """
      Constructs a CDF from the given data
      @ In, data, np.array(float), values to fit to
      @ Out, params, dict, essential parameters for CDF
    """
    # caluclate number of bins
    # binOps=Length or value
    nBins = self._computeNumberOfBins(data,binOps=binOps)
    # construct histogram
    counts, edges = np.histogram(data, bins = nBins, density = False)
    counts = np.array(counts) / float(len(data))
    # numerical CDF, normalizing to 0..1
    cdf = np.cumsum(counts)
    # set lowest value as first entry,
    ## from Jun implementation, min of CDF set to starting point for ?numerical issues?
    #cdf = np.insert(cdf, 0, cdf[0]) # Jun
    cdf = np.insert(cdf, 0, 0) # trying something else
    # store parameters
    # TODO FIXME WORKING also add the max, min, counts
    # miu sigma of data and counts edges
    params = {'bins': edges,
              'counts':counts,
              'pdf' : counts * nBins,
              'cdf' : cdf,
              'lens' : len(data)}
              #'binSearch':neighbors.NearestNeighbors(n_neighbors=2).fit([[b] for b in edges]),
              #'cdfSearch':neighbors.NearestNeighbors(n_neighbors=2).fit([[c] for c in cdf])}
    return params

  def _trainPeak(self, timeSeriesData, windowDict):
    """
      Generate peaks results from each target data
      @ In, timeSeriesData, np.array, list of values for the dependent variable (signal to take fourier from)
      @ In, windowDict, dict, dictionary for specefic target peaks
      @ Out, peakResults, dict, results of this training in keys 'period', 'windows', 'groupWin', 'mask', 'rangeWindow'
    """
    peakResults={}
    deltaT=self.pivotParameterValues[-1]-self.pivotParameterValues[0]
    deltaT=deltaT/(len(self.pivotParameterValues)-1)
    # change the peak information in self.peak from time unit into index by divided the timestep
    # deltaT is the time step calculated by (ending point - stating point in time)/(len(time)-1)
    peakResults['period']=int(round(windowDict['period']/deltaT))
    windows=[]
    for i in range(len(windowDict['windows'])):
      window={}
      a = windowDict['windows'][i]['window'][0]
      b = windowDict['windows'][i]['window'][1]
      window['window']=[int(round(windowDict['windows'][i]['window'][0]/deltaT)),int(round(windowDict['windows'][i]['window'][1]/deltaT))]
      window['width']=int(round(windowDict['windows'][i]['width']/deltaT))
      windows.append(window)
    peakResults['windows']=windows
    peakResults['threshold']=windowDict['threshold']
    groupWin , maskPeakRes=self._peakGroupWindow(timeSeriesData, windowDict = peakResults )
    peakResults['groupWin']=groupWin
    peakResults['mask']=maskPeakRes
    peakResults['nbin']=windowDict['nbin']
    rangeWindow = self.rangeWindow(windowDict=peakResults)
    peakResults['rangeWindow']=rangeWindow
    return peakResults

  def _trainFourier(self, pivotValues, periods, values, masks=None,target=None):
    """
      Perform fitting of Fourier series on self.timeSeriesDatabase
      @ In, pivotValues, np.array, list of values for the independent variable (e.g. time)
      @ In, periods, list, list of the base periods
      @ In, values, np.array, list of values for the dependent variable (signal to take fourier from)
      @ In, masks, np.array, optional, boolean mask where is the signal should be train by Fourier
      @ In, target, string, optional, target of the training
      @ Out, fourierResult, dict, results of this training in keys 'residues', 'fourierSet', 'predict', 'regression'
    """
    import sklearn.linear_model
    # XXX fix for no order
    if masks is None:
      masks = np.ones(len(values), dtype=bool)

    fourierSignalsFull = self._generateFourierSignal(pivotValues, periods)
    # fourierSignals dimensions, for each key (base):
    #   0: length of history
    #   1: evaluations, in order and flattened:
    #                 0:   sin(2pi*t/period[0]),
    #                 1:   cos(2pi*t/period[0]),
    #                 2:   sin(2pi*t/period[1]),
    #                 3:   cos(2pi*t/period[1]), ...
    fourierEngine = sklearn.linear_model.LinearRegression(normalize=False)
    fourierSignals = fourierSignalsFull[masks, :]
    values = values[masks]
    # check collinearity
    condNumber = np.linalg.cond(fourierSignals)
    if condNumber  > 30:
      self.raiseADebug('Fourier fitting condition number is {:1.1e}!'.format(condNumber),
                       ' Calculating iteratively instead of all-at-once.')
      # fourierSignals has shape (H, 2F) where H is history len and F is number of Fourier periods
      ## Fourier periods are in order from largest period to smallest, with sin then cos for each:
      ## [S0, C0, S1, C1, ..., SN, CN]
      H, F2 = fourierSignals.shape
      signalToFit = copy.deepcopy(values[:])
      intercept = 0
      coeffs = np.zeros(F2)
      for fn in range(F2):
        fSignal = fourierSignals[:,fn]
        eng = sklearn.linear_model.LinearRegression(normalize=False)
        eng.fit(fSignal.reshape(H,1), signalToFit)
        thisIntercept = eng.intercept_
        thisCoeff = eng.coef_[0]
        coeffs[fn] = thisCoeff
        intercept += thisIntercept
        # remove this signal from the signal to fit
        thisSignal = thisIntercept + thisCoeff * fSignal
        signalToFit -= thisSignal
    else:
      self.raiseADebug('Fourier fitting condition number is {:1.1e}.'.format(condNumber),
                       ' Calculating all Fourier coefficients at once.')
      fourierEngine.fit(fourierSignals, values)
      intercept = fourierEngine.intercept_
      coeffs = fourierEngine.coef_

    # get coefficient map for A*sin(ft) + B*cos(ft)
    waveCoefMap = collections.defaultdict(dict) # {period: {sin:#, cos:#}}
    for c, coef in enumerate(coeffs):
      period = periods[c//2]
      waveform = 'sin' if c % 2 == 0 else 'cos'
      waveCoefMap[period][waveform] = coef
    # convert to C*sin(ft + s)
    ## since we use fitting to get A and B, the magnitudes can be deceiving.
    ## this conversion makes "C" a useful value to know the contribution from a period
    coefMap = {}
    signal=np.ones(len(pivotValues)) * intercept
    for period, coefs in waveCoefMap.items():
      A = coefs['sin']
      B = coefs['cos']
      C, s = mathUtils.convertSinCosToSinPhase(A, B)
      coefMap[period] = {'amplitude': C, 'phase': s}
      signal += mathUtils.evalFourier(period,C,s,pivotValues)
    # re-add zero-filtered
    if target == self.zeroFilterTarget:
      signal[self._masks[target]['zeroFilterMask']] = 0.0

    # store results
    fourierResult = {'regression': {'intercept':intercept,
                                    'coeffs'   :coefMap,
                                    'periods'  :periods},
                     'predict': signal}
    return fourierResult

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
    return dist

  def _trainVARMA(self, data):
    """
      Train correlated ARMA model on white noise ARMA, with Fourier already removed
      @ In, data, np.array(np.array(float)), data on which to train with shape (# pivot values, # targets)
      @ Out, results, statsmodels.tsa.arima_model.ARMAResults, fitted VARMA
      @ Out, stateDist, Distributions.MultivariateNormal, MVN from which VARMA noise is taken
      @ Out, initDist, Distributions.MultivariateNormal, MVN from which VARMA initial state is taken
    """
    import statsmodels.api
    model = statsmodels.api.tsa.VARMAX(endog=data, order=(self.P, self.Q))
    self.raiseADebug('... ... ... fitting VARMA ...')
    results = model.fit(disp=False, maxiter=1000)
    lenHist, numVars = data.shape
    # train multivariate normal distributions using covariances, keep it around so we can control the RNG
    ## it appears "measurement" always has 0 covariance, and so is all zeros (see _generateVARMASignal)
    ## all the noise comes from the stateful properties
    stateDist = self._trainMultivariateNormal(numVars, np.zeros(numVars),model.ssm['state_cov'])
    # train initial state sampler
    ## Used to pick an initial state for the VARMA by sampling from the multivariate normal noise
    #    and using the AR and MA initial conditions.  Implemented so we can control the RNG internally.
    #    Implementation taken directly from statsmodels.tsa.statespace.kalman_filter.KalmanFilter.simulate
    ## get mean
    smoother = model.ssm
    mean = np.linalg.solve(np.eye(smoother.k_states) - smoother['transition',:,:,0],
                           smoother['state_intercept',:,0])
    ## get covariance
    r = smoother['selection',:,:,0]
    q = smoother['state_cov',:,:,0]
    selCov = r.dot(q).dot(r.T)
    cov = solve_discrete_lyapunov(smoother['transition',:,:,0], selCov)
    # FIXME it appears this is always resulting in a lowest-value initial state.  Why?
    initDist = self._trainMultivariateNormal(len(mean),mean,cov)
    # NOTE: uncomment this line to get a printed summary of a lot of information about the fitting.
    # self.raiseADebug('VARMA model training summary:\n',results.summary())
    return model, stateDist, initDist

  def _trainZeroRemoval(self, data, tol=1e-10):
    """
      A test for SOLAR GHI data.
      @ In, data, np.array, original signal
      @ In, tol, float, optional, tolerance below which to consider 0
      @ Out, mask, np.ndarray(bool), mask where zeros occur
    """
    # where should the data be truncated?
    mask = data < tol
    return mask

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained! Cannot write to DataObject.')
    rlz = {}
    # set up pivot parameter index
    pivotID = self.pivotParameterID
    pivotVals = self.pivotParameterValues
    rlz[self.pivotParameterID] = self.pivotParameterValues
    # set up sample counter ID
    ## ASSUMPTION: data object is EMPTY!
    if writeTo.size > 0:
      self.raiseAnError(ValueError,'Target data object has "{}" entries, but require an empty object to write ROM to!'.format(writeTo.size))
    counterID = writeTo.sampleTag
    counterVals = np.array([0])
    # Training signals
    for target, signals in self._signalStorage.items():
      for name, signal in signals.items():
        varName = '{}_{}'.format(target,name)
        writeTo.addVariable(varName, np.array([]), classify='meta', indices=[pivotID])
        rlz[varName] = signal
    # add realization
    writeTo.addRealization(rlz)

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused (kept for compatability)
      @ In, skip, list, optional, unused (kept for compatability)
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError, 'ROM is not yet trained! Cannot write to DataObject.')
    root = writeTo.getRoot()
    # - multicycle, if any
    if self.multicycle:
      myNode = xmlUtils.newNode('Multicycle')
      myNode.append(xmlUtils.newNode('num_cycles', text=self.numCycles))
      gNode = xmlUtils.newNode('growth_factors')
      for target, infos in self.growthFactors.items():
        for info in infos:
          tag = target
          attrib = {'mode': info['mode'], 'start_index': info['start'], 'end_index': info['end']}
          text = '{}'.format(info['value'])
          gNode.append(xmlUtils.newNode(tag, attrib=attrib, text=text))
      myNode.append(gNode)
      root.append(myNode)
    # - Fourier coefficients (by period, waveform)
    for target, fourier in self.fourierResults.items():
      targetNode = root.find(target)
      if targetNode is None:
        targetNode = xmlUtils.newNode(target)
        root.append(targetNode)
      fourierNode = xmlUtils.newNode('Fourier')
      targetNode.append(fourierNode)
      fourierNode.append(xmlUtils.newNode('SignalIntercept', text='{:1.9e}'.format(float(fourier['regression']['intercept']))))
      for period in fourier['regression']['periods']:
        periodNode = xmlUtils.newNode('period', text='{:1.9e}'.format(period))
        fourierNode.append(periodNode)
        periodNode.append(xmlUtils.newNode('frequency', text='{:1.9e}'.format(1.0/period)))
        for stat, value in sorted(list(fourier['regression']['coeffs'][period].items()), key=lambda x:x[0]):
          periodNode.append(xmlUtils.newNode(stat, text='{:1.9e}'.format(value)))
    # - ARMA std
    for target, arma in self.armaResult.items():
      targetNode = root.find(target)
      if targetNode is None:
        targetNode = xmlUtils.newNode(target)
        root.append(targetNode)
      armaNode = xmlUtils.newNode('ARMA_params')
      targetNode.append(armaNode)
      armaNode.append(xmlUtils.newNode('std', text=arma.sigma))
      # TODO covariances, P and Q, etc
    for target,peakInfo in self.peaks.items():
      targetNode = root.find(target)
      if targetNode is None:
        targetNode = xmlUtils.newNode(target)
        root.append(targetNode)
      peakNode = xmlUtils.newNode('Peak_params')
      targetNode.append(peakNode)
      if 'groupWin' in peakInfo.keys():
        for group in peakInfo['groupWin']:
          groupnode=xmlUtils.newNode('peak')
          groupnode.append(xmlUtils.newNode('Amplitude', text='{}'.format(np.array(group['Amp']).mean())))
          groupnode.append(xmlUtils.newNode('Index', text='{}'.format(np.array(group['Ind']).mean())))
          peakNode.append(groupnode)

  def _transformThroughInputCDF(self, signal, originalDist, weights=None):
    """
      Transforms a signal through the original distribution
      @ In, signal, np.array(float), signal to transform
      @ In, originalDist, scipy.stats.rv_histogram, distribution to transform through
      @ In, weights, np.array(float), weighting for samples (assumed uniform if not given)
      @ Out, new, np.array, new signal after transformation
    """
    # first build a histogram object of the sampled data
    dist, hist = mathUtils.trainEmpiricalFunction(signal, minBins=self._minBins, weights=weights)
    # transform data through CDFs
    new = originalDist[0].ppf(dist.cdf(signal))
    return new

  def _combineMask(self,masks):
    """
    Combine different masks, remove zerofiletermask and combine other masks
    @ In, masks, dictionay, dictionary of all the mask need to be combined
    @ Out, combMask, np.ndarray(bool) or None, one mask contain all the False in the masks
    """
    if masks == None:
      combMask = None
    else:
      woZFMask = copy.copy(masks)
      rmZFMask = woZFMask.pop("zeroFilterMask", None)
      if len(woZFMask) ==0:
        combMask= None
      else:
        combMask = True
        for key, val in woZFMask.items():
          combMask = np.logical_and(combMask, val)
    return combMask

  ### Segmenting and Clustering ###
  def checkRequestedClusterFeatures(self, request):
    """
      Takes the user-requested features (sometimes "all") and interprets them for this ROM.
      @ In, request, dict(list), as from ROMColletion.Cluster._extrapolateRequestedClusterFeatures
      @ Out, interpreted, dict(list), interpreted features
    """
    if request is None:
      # since no special requests were made, we cluster on EV ER Y THING
      return self._clusterableFeatures
    # otherwise we have to unpack the values as known to this ROM
    interpreted = collections.defaultdict(list)
    # create containers for unrecognized entries so we can report them all at once, bc WFFTU
    unrecognizedSets = []
    unrecognizedFeatures = collections.defaultdict(list)
    for featureSet, featureList in request.items():
      if featureSet not in self._clusterableFeatures:
        unrecognizedSets.append(featureSet)
        continue
      subClusterable = self._clusterableFeatures[featureSet]
      # if all the subfeatures of this featureSet were requested, take them now
      if featureList == 'all':
        interpreted[featureSet] = subClusterable
        continue
      # otherwise loop over the requests
      for feature in featureList:
        if feature not in subClusterable:
          unrecognizedFeatures[featureSet].append(feature)
        else:
          interpreted[featureSet].append(feature)

    # if anything wasn't recognized, print it so the user can fix it
    ## print all of them because WE FIGHT FOR THE USERS
    if unrecognizedSets or unrecognizedFeatures:
      self.raiseAWarning('Problems in clusterFeatures!', verbosity='silent')
      if unrecognizedSets:
        self.raiseAWarning(' -> unrecognized clusterFeatures base feature requests: {}'.format(unrecognizedSets), verbosity='silent')
      if unrecognizedFeatures:
        for key, vals in unrecognizedFeatures.items():
          self.raiseAWarning(' -> unrecognized clusterFeatures feature "{}" requests: {}'.format(key, vals), verbosity='silent')
      self.raiseAnError(IOError, 'Invalid clusterFeatures input! See messages above for details.')

    return interpreted

  def isClusterable(self):
    """
      Allows ROM to declare whether it has methods for clustring. Default is no.
      @ In, None
      @ Out, isClusterable, bool, if True then has clustering mechanics.
    """
    # clustering methods have been added
    return True

  def _getMeanFromGlobal(self, settings, pickers, targets=None):
    """
      Derives segment means from global trends
      @ In, settings, dict, as per getGlobalRomSegmentSettings
      @ In, pickers, list(slice), picks portion of signal of interest
      @ In, targets, list, optional, targets to include (default is all)
      @ Out, results, list(dict), mean for each target per picker
    """
    if 'long Fourier signal' not in settings:
      return []
    if isinstance(pickers, slice):
      pickers = [pickers]
    if targets == None:
      targets = settings['long Fourier signal'].keys()
    results = [] # one per "pickers"
    for picker in pickers:
      res = dict((target, signal['predict'][picker].mean()) for target, signal in settings['long Fourier signal'].items())
      results.append(res)
    return results

  def getLocalRomClusterFeatures(self, featureTemplate, settings, request, picker=None, **kwargs):
    """
      Provides metrics aka features on which clustering compatibility can be measured.
      This is called on LOCAL subsegment ROMs, not on the GLOBAL template ROM
      @ In, featureTemplate, str, format for feature inclusion
      @ In, settings, dict, as per getGlobalRomSegmentSettings
      @ In, request, dict(list) or None, requested features to cluster on (by featureSet)
      @ In, picker, slice, indexer for segmenting data
      @ In, kwargs, dict, arbitrary keyword arguments
      @ Out, features, dict, {target_metric: np.array(floats)} features to cluster on
    """
    # algorithm for providing Fourier series and ARMA white noise variance and #TODO covariance
    features = self.getFundamentalFeatures(request, featureTemplate=featureTemplate)
    # segment means
    # since we've already detrended globally, get the means from that (if present)
    if 'long Fourier signal' in settings:
      assert picker is not None
      results = self._getMeanFromGlobal(settings, picker)
      for target, mean in results[0].items():
        feature = featureTemplate.format(target=target, metric="global", id="mean")
        features[feature] = mean
    return features

  def getFundamentalFeatures(self, requestedFeatures, featureTemplate=None):
    """
      Collect the fundamental parameters for this ROM
      Used for writing XML, interpolating, clustering, etc
      @ In, requestedFeatures, dict(list), featureSet and features to collect (may be None)
      @ In, featureTemplate, str, templated string for naming features (probably leave as None)
      @ Out, features, dict,
    """
    assert self.amITrained
    if featureTemplate is None:
      featureTemplate = '{target}|{metric}|{id}' # TODO this kind of has to be the format currently
    features = {}

    # include Fourier if available
    # TODO if not requestedFeatures or 'Fourier' in requestedFeatures: # TODO propagate requestedFeatures throughout method
    for target, fourier in self.fourierResults.items():
      feature = featureTemplate.format(target=target, metric='Fourier', id='fittingIntercept')
      features[feature] = fourier['regression']['intercept']
      for period in fourier['regression']['periods']:
        # amp, phase
        amp = fourier['regression']['coeffs'][period]['amplitude']
        phase = fourier['regression']['coeffs'][period]['phase']
        # rather than use amp, phase as properties, use sine and cosine coeffs
        ## this mitigates the cyclic nature of the phase causing undesirable clustering
        sinAmp = amp * np.cos(phase)
        cosAmp = amp * np.sin(phase)
        ID = '{}_{}'.format(period, 'sineAmp')
        feature = featureTemplate.format(target=target, metric='Fourier', id=ID)
        features[feature] = sinAmp
        ID = '{}_{}'.format(period, 'cosineAmp')
        feature = featureTemplate.format(target=target, metric='Fourier', id=ID)
        features[feature] = cosAmp

    # ARMA (not varma)
    for target, arma in self.armaResult.items():
      # sigma
      feature = featureTemplate.format(target=target, metric='arma', id='std')
      features[feature] = arma.sigma
      # autoregression
      for p, val in enumerate(-arma.polyar[1:]):  # The AR coefficients are stored in polynomial form here (flipped sign and with a term in the zero position of the array for lag=0)
        feature = featureTemplate.format(target=target, metric='arma', id='AR_{}'.format(p))
        features[feature] = val
      # moving average
      for q, val in enumerate(arma.polyma[1:]):  # keep only the terms for lag>0
        feature = featureTemplate.format(target=target, metric='arma', id='MA_{}'.format(q))
        features[feature] = val
      for target, cdfParam in self.cdfParams.items():
        lenthOfData = cdfParam['lens']
        feature = featureTemplate.format(target=target, metric='arma', id='len')
        features[feature] = lenthOfData
        for e, edge in enumerate(cdfParam['bins']):
          feature = featureTemplate.format(target=target, metric='arma', id='bin_{}'.format(e))
          features[feature] = edge
        for c, count in enumerate(cdfParam['counts']):
          feature = featureTemplate.format(target=target, metric='arma', id='counts_{}'.format(c))
          features[feature] = count

    # CDF preservation
    for target, cdf in self._trainingCDF.items():
      _, (counts, edges) = cdf
      for c, count in enumerate(counts):
        feature = featureTemplate.format(target=target, metric='cdf', id='counts_{}'.format(c))
        features[feature] = count
      for e, edge in enumerate(edges):
        feature = featureTemplate.format(target=target, metric='cdf', id='edges_{}'.format(e))
        features[feature] = edge

    # Peaks
    for target, peak in self.peaks.items():
      nBin = self.peaks[target]['nbin']
      period = self.peaks[target]['period']
      if 'groupWin' in peak.keys() and 'rangeWindow' in peak.keys():
        for g , group in enumerate(peak['groupWin']):
          ## prbExit
          # g is the group of the peaks probExist is the exist probability for this type of peak
          lenWin=min(len(peak['rangeWindow'][g]['bg']),len(peak['rangeWindow'][g]['end']))
          ## This might be used in the future.
          # ID = 'gp_{}_lenWin'.format(g)
          # feature = featureTemplate.format(target=target, metric='peak', id=ID)
          # features[feature] = lenWin

          # prbability if this peak exist
          prbExist = len(group['Ind'])/lenWin
          ID = 'gp_{}_probExist'.format(g)
          feature = featureTemplate.format(target=target, metric='peak', id=ID)
          features[feature] = prbExist

          ## IND
          #most probabble index
          if len(group['Ind']):
            modeInd = stats.mode(group['Ind'])[0][0]
          else:
            modeInd = 0
          ID = 'gp_{}_modeInd'.format(g)
          feature = featureTemplate.format(target=target, metric='peak', id=ID)
          features[feature] = modeInd
          # index distribution
          if peak['rangeWindow'][g]['end'][0]>peak['rangeWindow'][g]['bg'][0]:
            indBins=np.arange(peak['rangeWindow'][g]['end'][0]-peak['rangeWindow'][g]['bg'][0]-1)+1
          else:
            indBins=np.arange(peak['rangeWindow'][g]['end'][0]-peak['rangeWindow'][g]['bg'][0]-1+period)+1
          indCounts, _ = np.histogram(group['Ind'], bins=indBins, density=False)
          for c, count in enumerate(indCounts):
            feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_ind {}'.format(g,c))
            features[feature] = count

          ## AMP
          #mean
          if len(group['Amp']):
            if np.isnan((group['Amp'][0])):
              meanAmp = np.mean(self._signalStorage[target]['original'])
            else:
              meanAmp = np.mean(group['Amp'])

            feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_meanAmp'.format(g))
            features[feature] = meanAmp

          else:
            meanAmp = np.mean(self._signalStorage[target]['original'])
            feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_meanAmp'.format(g))
            features[feature] = meanAmp

          ##std
          if len(group['Amp']) > 1:
            stdAmp = rv_histogram(np.histogram(group['Amp'])).std()
            feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_stdAmp'.format(g))
            features[feature] = stdAmp
          else:
            stdAmp = 0
            feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_stdAmp'.format(g))
            features[feature] = stdAmp

          if len(group['Amp']):
            if np.isnan((group['Amp'][0])):
              maxAmp=max(self._signalStorage[target]['original'])
              feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_maxAmp'.format(g))
              features[feature] = maxAmp
              minAmp=min(self._signalStorage[target]['original'])
              feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_minAmp'.format(g))
              features[feature] = minAmp
            else:
              maxAmp=max(group['Amp'])
              feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_maxAmp'.format(g))
              features[feature] = maxAmp
              minAmp=min(group['Amp'])
              feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_minAmp'.format(g))
              features[feature] = minAmp
            ## distribution on the Amp
            if np.isnan((group['Amp'][0])):
              ampCounts, _ = np.histogram([], range=(minAmp,maxAmp),density = False)
            else:
              ampCounts, _ = np.histogram(group['Amp'], bins = nBin,density = False)
            for c, count in enumerate(ampCounts):
              feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_amp {}'.format(g,c))
              features[feature] = count
          else:
            maxAmp=max(self._signalStorage[target]['original'])
            feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_maxAmp'.format(g))
            features[feature] = maxAmp
            minAmp=min(self._signalStorage[target]['original'])
            feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_minAmp'.format(g))
            features[feature] = minAmp
            ## distribution on the Amp
            ampCounts, _ = np.histogram(group['Amp'], bins = nBin,density = False)
            for c, count in enumerate(ampCounts):
              feature = featureTemplate.format(target=target, metric='peak', id='gp_{}_amp {}'.format(g,c))
              features[feature] = count

    # Remove features that were not requested, if selective.
    ## TODO this could be sped up by not calculating them in the first place maybe
    if requestedFeatures is not None:
      popFeatures=[]
      for rq in features.keys():
        tg, mtc, rid =rq.split('|')
        if mtc.lower() not in requestedFeatures.keys():
          #this apply to arma and fourier
          popFeatures.append(rq)
        elif mtc.lower()=='peak':
          gp, gpid, rrid =rid.split('_')
          if rrid.startswith('amp'):
            popFeatures.append(rq)
          elif rrid.startswith('ind'):
            popFeatures.append(rq)
          elif rrid.startswith('max'):
            popFeatures.append(rq)
          elif rrid.startswith('min'):
            popFeatures.append(rq)
        elif mtc.lower()=='arma':
          if rid.startswith('bin'):
            popFeatures.append(rq)
          elif rid.startswith('counts'):
            popFeatures.append(rq)
          elif rid.startswith('l'):
            popFeatures.append(rq)
      for p in popFeatures:
        del features[p]
    return features

  def readFundamentalFeatures(self, features):
    """
      Reads in the requested ARMA model properties from a feature dictionary
      @ In, features, dict, dictionary of fundamental features
      @ Out, readFundamentalFeatures, dict, more clear list of features for construction
    """
    # collect all the data
    fourier = collections.defaultdict(dict)
    arma = collections.defaultdict(dict)
    cdf = collections.defaultdict(dict)
    peak = collections.defaultdict(dict)
    for feature, val in features.items():
      target, metric, ID = feature.split('|')

      if metric == 'Fourier':
        if ID == 'fittingIntercept':
          fourier[target]['intercept'] = val
        else:
          period, wave = ID.split('_')
          period = float(period)
          if period not in fourier[target]:
            fourier[target][period] = {}
          fourier[target][period][wave] = val

      elif metric == 'arma':
        if ID == 'std':
          arma[target]['std'] = val
        if ID == 'len':
          arma[target]['len'] = val
        elif ID.startswith('AR_'):
          p = int(ID[3:])
          if 'AR' not in arma[target]:
            arma[target]['AR'] = {}
          arma[target]['AR'][p] = val
        elif ID.startswith('MA_'):
          p = int(ID[3:])
          if 'MA' not in arma[target]:
            arma[target]['MA'] = {}
          arma[target]['MA'][p] = val
        elif ID.startswith('bin_'):
          p = int(ID[4:])
          if 'bin' not in arma[target]:
            arma[target]['bin'] = {}
          arma[target]['bin'][p] = val
        elif ID.startswith('counts_'):
          p = int(ID[7:])
          if 'counts' not in arma[target]:
            arma[target]['counts'] = {}
          arma[target]['counts'][p] = val

      elif metric == 'cdf':
        if ID.startswith('counts_'):
          c = int(ID.split('_')[1])
          if 'counts' not in cdf[target]:
            cdf[target]['counts'] = {}
          cdf[target]['counts'][c] = val
        elif ID.startswith('edges_'):
          e = int(ID.split('_')[1])
          if 'edges' not in cdf[target]:
            cdf[target]['edges'] = {}
          cdf[target]['edges'][e] = val

      elif metric == 'peak':
        _, group, realID = ID.split('_')
        if group not in peak[target]:
          peak[target][group] = {}
        if realID.startswith('amp'):
          c = int(realID.split(' ')[1])
          if 'ampCounts' not in peak[target][group]:
            peak[target][group]['ampCounts'] = {}
          peak[target][group]['ampCounts'][c] = val
        elif realID.startswith('ind'):
          c = int(realID.split(' ')[1])
          if 'indCounts' not in peak[target][group]:
            peak[target][group]['indCounts'] = {}
          peak[target][group]['indCounts'][c] = val
        else:
          peak[target][group][realID]=val

      else:
        raise KeyError('Unrecognized metric: "{}"'.format(metric))

    return {'fourier': fourier,
            'arma': arma,
            'cdf': cdf,
            'peak': peak}

  def setFundamentalFeatures(self, features):
    """
      opposite of getFundamentalFeatures, expects results as from readFundamentalFeatures
      Constructs this ROM by setting fundamental features from "features"
      @ In, features, dict, dictionary of info as from readFundamentalFeatures
      @ Out, None
    """
    self._setFourierResults(features.get('fourier', {}))
    self._setArmaResults(features.get('arma', {}))
    self._setCDFResults(features.get('cdf', {}))
    self._setPeakResults(features.get('peak', {}))
    self.amITrained = True

  def _setFourierResults(self, paramDict):
    """
      Sets Fourier fundamental parameters
      @ In, paramDict, dictionary of parameters to set
      @ Out, None
    """
    for target, info in paramDict.items():
      predict = np.ones(len(self.pivotParameterValues)) * info['intercept']
      params = {'coeffs': {}}
      for period, waves in info.items():
        if period == 'intercept':
          params[period] = waves
        else:
          # either A, B or C, p
          if 'sineAmp' in waves:
            A = waves['sineAmp']
            B = waves['cosineAmp']
            C, p = mathUtils.convertSinCosToSinPhase(A, B)
          else:
            C = waves['amplitude']
            p = waves['phase']
          params['coeffs'][period] = {}
          params['coeffs'][period]['amplitude'] = C
          params['coeffs'][period]['phase'] = p
          predict += C * np.sin(2.*np.pi / period * self.pivotParameterValues + p)
      params['periods'] = list(params['coeffs'].keys())
      self.fourierResults[target] = {'regression': params,
                                     'predict': predict}

  def _setArmaResults(self, paramDict):
    """
      Sets ARMA fundamental parameters
      @ In, paramDict, dictionary of parameters to set
      @ Out, None
    """
    for target, info in paramDict.items():
      if 'AR' in info:
        AR_keys, AR_vals = zip(*list(info['AR'].items()))
        AR_keys, AR_vals = zip(*sorted(zip(AR_keys, AR_vals), key=lambda x:x[0]))
        AR_vals = np.concatenate(([1], -np.asarray(AR_vals)))  # convert the AR params into polynomial form
      else:
        AR_vals = np.array([1])  # must include a 1 for the zero lag term
      if 'MA' in info:
        MA_keys, MA_vals = zip(*list(info['MA'].items()))
        MA_keys, MA_vals = zip(*sorted(zip(MA_keys, MA_vals), key=lambda x:x[0]))
        MA_vals = np.concatenate(([1], np.asarray(MA_vals)))  # converts the MA params into polynomial form
      else:
        MA_vals = np.array([1])
      if 'bin' in info:
        bin_keys, bin_vals = zip(*list(info['bin'].items()))
        bin_keys, bin_vals = zip(*sorted(zip(bin_keys, bin_vals), key=lambda x:x[0]))
        bin_vals = np.asarray(bin_vals)
      # FIXME no else in here
      # else:
      #   bin_vals = np.array([])
      if 'counts' in info:
        counts_keys, counts_vals = zip(*list(info['counts'].items()))
        counts_keys, counts_vals = zip(*sorted(zip(counts_keys, counts_vals), key=lambda x:x[0]))
        counts_vals = np.asarray(counts_vals)
      # FIXME no else

      sigma = info['std']
      result = armaResultsProxy(AR_vals, MA_vals, sigma)
      self.armaResult[target] = result
      lengthOfData=info['len']
      nBins=len(counts_vals)
      cdf = np.cumsum(counts_vals)
      cdf = np.insert(cdf, 0, 0)
      counts_vals = np.array(counts_vals) * float(lengthOfData)
      params = {'bins': bin_vals,
              'counts':counts_vals,
              'pdf' : counts_vals * nBins,
              'cdf' : cdf,
              'lens' : lengthOfData}
      self.cdfParams[target] = params

  def _setCDFResults(self, paramDict):
    """
      Sets CDF preservation fundamental parameters
      @ In, paramDict, dictionary of parameters to set
      @ Out, None
    """
    for target, info in paramDict.items():
      # counts
      cs = list(info['counts'].items())
      c_idx, c_vals = zip(*sorted(cs, key=lambda x: x[0]))
      c_vals = np.asarray(c_vals)
      ## renormalize counts
      counts = c_vals / float(c_vals.sum())
      # edges
      es = list(info['edges'].items())
      e_idx, e_vals = zip(*sorted(es, key=lambda x: x[0]))
      histogram = (counts, e_vals)
      dist = stats.rv_histogram(histogram)
      self._trainingCDF[target] = (dist, histogram)

  def _setPeakResults(self, paramDict):
    """
      Sets Peaks fundamental parameters
      @ In, paramDict, dictionary of parameters to set
      @ Out, None
    """
    for target, info in paramDict.items():
      groupWin=[]
      for g, groupInfo in info.items():
        g = int(g)
        lenWin=min(len(self.peaks[target]['rangeWindow'][g]['bg']),len(self.peaks[target]['rangeWindow'][g]['end']))
        groupWin.append({})

        lsCs=list(groupInfo['ampCounts'].items())
        _, hisCs = zip(*sorted(lsCs, key=lambda x: x[0]))
        ampHisCs = np.asarray(hisCs)
        maxAmp=groupInfo['maxAmp']
        minAmp=groupInfo['minAmp']
        probExist=groupInfo['probExist']

        if maxAmp>minAmp:
          ampHisEg=np.linspace(minAmp, maxAmp, num=len(ampHisCs)+1)
          histogram = (ampHisCs, ampHisEg)
          dist = stats.rv_histogram(histogram)
          ampLocal=dist.rvs(size=int(round(probExist*lenWin))).tolist()
        else:
          histogram = None
          ampLocal = [maxAmp]*int(round(probExist*lenWin))

        lsIndCs = list(groupInfo['indCounts'].items())
        _, hisIndCs = zip(*sorted(lsIndCs, key=lambda x: x[0]))
        indHisCs = np.asarray(hisIndCs)
        histogramInd = (indHisCs, np.arange(len(indHisCs)+1)+1)

        distInd = stats.rv_histogram(histogramInd)
        indLocal=distInd.rvs(size=int(round(probExist*lenWin))).tolist()
        # If the probability of exist is 0 then the indLocal is an empty list, size = 0
        # if probability of exist is not 0 then the distInd will contain real number, so
        # the rvs will not generate nan
        for indexOfIndex,valueOfIndex in enumerate(indLocal):
          valueOfIndex=int(valueOfIndex)
          indLocal[indexOfIndex]=valueOfIndex

        groupWin[g]['Ind']=indLocal
        groupWin[g]['Amp']=ampLocal
      self.peaks[target]['groupWin']=groupWin

  def getGlobalRomSegmentSettings(self, trainingDict, divisions):
    """
      Allows the ROM to perform some analysis before segmenting.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL subsegment ROMs!
      @ In, trainingDict, dict, data for training, full and unsegmented
      @ In, divisions, tuple, (division slice indices, unclustered spaces)
      @ Out, settings, object, arbitrary information about ROM clustering settings
      @ Out, trainingDict, dict, adjusted training data (possibly unchanged)
    """
    trainingDict = copy.deepcopy(trainingDict) # otherwise we destructively tamper with the input data object
    settings = {}
    targets = self.target

    # set up for input CDF preservation on a global scale
    if self.preserveInputCDF:
      inputDists = {}
      for target in targets:
        if target == self.pivotParameterID:
          continue
        targetVals = trainingDict[target][0]
        nbins=max(self._minBins,int(np.sqrt(len(targetVals))))
        inputDists[target] = mathUtils.trainEmpiricalFunction(targetVals, bins=nbins)
      settings['input CDFs'] = inputDists
    # zero filtering
    if self.zeroFilterTarget:
      self._masks[self.zeroFilterTarget]['zeroFilterMask'] = self._trainZeroRemoval(trainingDict[self.zeroFilterTarget][0], tol=self.zeroFilterTol) # where zeros are not
      self._masks[self.zeroFilterTarget]['notZeroFilterMask'] = np.logical_not(self._masks[self.zeroFilterTarget]['zeroFilterMask']) # where zeroes are
      print('DEBUGG setting ZF masks!', self.zeroFilterTarget, self._masks[self.zeroFilterTarget]['zeroFilterMask'].sum(), self._masks[self.zeroFilterTarget]['notZeroFilterMask'].sum())
      # if the zero filter target is correlated, the same masks apply to the correlated vars
      if self.zeroFilterTarget in self.correlations:
        for cor in (c for c in self.correlations if c != self.zeroFilterTarget):
          print('DEBUGG setting ZF masks c!', cor)
          self._masks[cor]['zeroFilterMask'] = self._masks[self.zeroFilterTarget]['zeroFilterMask']
          self._masks[cor]['notZeroFilterMask'] = self._masks[self.zeroFilterTarget]['notZeroFilterMask']
    else:
      print('DEBUGG no ZF here!')

    # do global Fourier analysis on combined signal for all periods longer than the segment
    if self.fourierParams:
      # determine the Nyquist length for the clustered params
      slicers = divisions[0]
      pivotValues = trainingDict[self.pivotParameterID][0]
      # use the first segment as typical of all of them, NOTE might be bad assumption
      delta = pivotValues[slicers[0][-1]] - pivotValues[slicers[0][0]]
      # any Fourier longer than the delta should be trained a priori, leaving the reaminder
      #    to be specific to individual ROMs
      full = {}      # train these periods on the full series
      segment = {}   # train these periods on the segments individually
      for target in (t for t in targets if t != self.pivotParameterID):
        # only do separation for targets for whom there's a Fourier request
        if target in self.fourierParams:
          # NOTE: assuming training on only one history!
          targetVals = trainingDict[target][0]
          periods = np.asarray(self.fourierParams[target])
          full = periods[periods > (delta*self.nyquistScalar)]
          segment[target] = periods[np.logical_not(periods > (delta*self.nyquistScalar))]
          if len(full):
            # train Fourier on longer periods
            self.fourierResults[target] = self._trainFourier(pivotValues, full, targetVals, target=target)
            # remove longer signal from training data
            signal = self.fourierResults[target]['predict']
            targetVals = np.array(targetVals, dtype=np.float64)
            targetVals -= signal
            trainingDict[target][0] = targetVals
      # store the segment-based periods in the settings to return
      settings['segment Fourier periods'] = segment
      settings['long Fourier signal'] = self.fourierResults
    return settings, trainingDict

  def parametrizeGlobalRomFeatures(self, featureDict):
    """
      Parametrizes the GLOBAL features of the ROM (assumes this is the templateROM and segmentation is active)
      @ In, featureDict, dictionary of features to parametrize
      @ Out, params, dict, dictionary of collected parametrized features
    """
    t = 'GLOBAL_{target}|{metric}|{ID}'
    params = {}
    ## TODO FIXME duplicated code with getFundamentalFeatures! Extract for commonality!
    # CDF
    cdf = featureDict.get('input CDFs', None)
    if cdf:
      for target, (rvs, (counts, edges)) in cdf.items():
        for c, count in enumerate(counts):
          params[t.format(target=target, metric='cdf', ID='counts_{}'.format(c))] = count
        for e, edge in enumerate(edges):
          params[t.format(target=target, metric='cdf', ID='edges_{}'.format(e))] = edge
    # long Fourier
    fourier = featureDict.get('long Fourier signal', None)
    if fourier:
      for target, info in fourier.items():
        feature = t.format(target=target, metric='Fourier', ID='fittingIntercept')
        params[feature] = info['regression']['intercept']
        coeffMap = info['regression']['coeffs']
        for period, wave in coeffMap.items():
          amp = wave['amplitude']
          phase = wave['phase']
          sinAmp = amp * np.cos(phase)
          cosAmp = amp * np.sin(phase)
          ID = '{}_{}'.format(period, 'sineAmp')
          feature = t.format(target=target, metric='Fourier', ID=ID)
          params[feature] = sinAmp
          ID = '{}_{}'.format(period, 'cosineAmp')
          feature = t.format(target=target, metric='Fourier', ID=ID)
          params[feature] = cosAmp
    return params

  def setGlobalRomFeatures(self, params, pivotValues):
    """
      Sets global ROM properties for a templateROM when using segmenting
      Returns settings rather than "setting" them for use in ROMCollection classes
      @ In, params, dict, dictionary of parameters to set
      @ In, pivotValues, np.array, values of time parameter
      @ Out, results, dict, global ROM feature set
    """
    results = {}
    # TODO FIXME duplicate algorithm with readFundamentalFeatures!!
    cdf = collections.defaultdict(dict)
    fourier = collections.defaultdict(dict)
    for key, val in params.items():
      assert key.startswith('GLOBAL_')
      target, metric, ID = key[7:].split('|')

      if metric == 'cdf':
        if ID.startswith('counts_'):
          c = int(ID.split('_')[1])
          if 'counts' not in cdf[target]:
            cdf[target]['counts'] = {}
          cdf[target]['counts'][c] = val
        elif ID.startswith('edges_'):
          e = int(ID.split('_')[1])
          if 'edges' not in cdf[target]:
            cdf[target]['edges'] = {}
          cdf[target]['edges'][e] = val

      elif metric == 'Fourier':
        if ID == 'fittingIntercept':
          fourier[target]['intercept'] = val
        else:
          period, wave = ID.split('_')
          period = float(period)
          if period not in fourier[target]:
            fourier[target][period] = {}
          fourier[target][period][wave] = val

    # TODO FIXME duplicate algorithm with setFundamentalFeatures!
    # fourier
    if fourier:
      results['long Fourier signal'] = {}
    for target, info in fourier.items():
      predict = np.ones(len(pivotValues)) * info['intercept']
      fparams = {'coeffs': {}}
      for period, waves in info.items():
        if period == 'intercept':
          fparams[period] = waves
        else:
          # either A, B or C, p
          if 'sineAmp' in waves:
            A = waves['sineAmp']
            B = waves['cosineAmp']
            C, p = mathUtils.convertSinCosToSinPhase(A, B)
          else:
            C = waves['amplitude']
            p = waves['phase']
          fparams['coeffs'][period] = {}
          fparams['coeffs'][period]['amplitude'] = C
          fparams['coeffs'][period]['phase'] = p
          predict += C * np.sin(2.*np.pi / period * pivotValues + p)
      fparams['periods'] = list(fparams['coeffs'].keys())
      results['long Fourier signal'][target] = {'regression': fparams,
                                                'predict': predict}

    # cdf
    if cdf:
      results['input CDFs'] = {}
    for target, info in cdf.items():
      # counts
      cs = list(info['counts'].items())
      c_idx, c_vals = zip(*sorted(cs, key=lambda x: x[0]))
      c_vals = np.asarray(c_vals)
      ## renormalize counts
      counts = c_vals / float(c_vals.sum())
      # edges
      es = list(info['edges'].items())
      e_idx, e_vals = zip(*sorted(es, key=lambda x: x[0]))
      histogram = (counts, e_vals)
      dist = stats.rv_histogram(histogram)
      results['input CDFs'][target] = (dist, histogram)
    return results

  def adjustLocalRomSegment(self, settings, picker):
    """
      Adjusts this ROM to account for it being a segment as a part of a larger ROM collection.
      Call this before training the subspace segment ROMs
      Note this is called on the LOCAL subsegment ROMs, NOT on the GLOBAL templateROM from the ROMcollection!
      @ In, settings, object, arbitrary information about ROM clustering settings from getGlobalRomSegmentSettings
      @ In, picker, slice, slice object for selecting the desired segment
      @ Out, None
    """
    if self.zeroFilterTarget:
      print('DEBUGG adj local rom seg, zerofiltering!', self.zeroFilterTarget)
      print(' ... ZF:', self._masks[self.zeroFilterTarget]['zeroFilterMask'][picker].sum())
      # FIXME is self._masks really correct? Did that copy down from the templateROM?
      self._masks[self.zeroFilterTarget]['zeroFilterMask'] = self._masks[self.zeroFilterTarget]['zeroFilterMask'][picker]
      self._masks[self.zeroFilterTarget]['notZeroFilterMask'] = self._masks[self.zeroFilterTarget]['notZeroFilterMask'][picker]
      # also correlated targets
      if self.zeroFilterTarget in self.correlations:
        for cor in (c for c in self.correlations if c != self.zeroFilterTarget):
          self._masks[cor]['zeroFilterMask'] = self._masks[self.zeroFilterTarget]['zeroFilterMask']
          self._masks[cor]['notZeroFilterMask'] = self._masks[self.zeroFilterTarget]['notZeroFilterMask']
    # some Fourier periods have already been handled, so reset the ones that actually are needed
    newFourier = settings.get('segment Fourier periods', None)
    if newFourier is not None:
      for target in list(self.fourierParams.keys()):
        periods = newFourier.get(target, [])
        # if any sub-segment Fourier remaining, send it through
        if len(periods):
          self.fourierParams[target] = periods
        else:
          # otherwise, remove target from fourierParams so no Fourier is applied
          self.fourierParams.pop(target,None)
    # disable CDF preservation on subclusters
    ## Note that this might be a good candidate for a user option someday,
    ## but right now we can't imagine a use case that would turn it on
    self.preserveInputCDF = False
    if 'long Fourier signal' in settings:
      for target, peak in self.peaks.items():
        subMean = self._getMeanFromGlobal(settings, picker)
        subMean=subMean[0][target]
        th = self.peaks[target]['threshold']
        th = th - subMean
        self.peaks[target]['threshold'] = th

  def finalizeLocalRomSegmentEvaluation(self, settings, evaluation, globalPicker, localPicker=None):
    """
      Allows global settings in "settings" to affect a LOCAL evaluation of a LOCAL ROM
      Note this is called on the LOCAL subsegment ROM and not the GLOBAL templateROM.
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, evaluation, dict, preliminary evaluation from the local segment ROM as {target: [values]}
      @ In, globalPicker, slice, indexer for data range of this segment FROM GLOBAL SIGNAL
      @ In, localPicker, slice, optional, indexer for part of signal that should be adjusted IN LOCAL SIGNAL
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    # globalPicker always says where segment is within GLOBAL signal
    ## -> anyGlobalSignal[picker] is the portion of the global signal which represents this segment.
    # localPicker, if present, means that the evaluation is part of a larger history
    ## -> in this case, evaluation[localPicker] gives the location of this segment's values
    # Examples:
    ## - full evaluation: localPicker = globalPicker # NOTE: this is (default)
    ## - truncated evaluation: localPicker = slice(start, end, None)
    ## - ND clustered evaluation: localPicker = slice(None, None, None)
    if localPicker is None:
      # TODO assertion that signal and evaluation are same length?
      # This should only occur when performing a full, unclustered evaluation
      # TODO should this not be optional? Should we always take both?
      localPicker = globalPicker
    # add global Fourier to evaluated signals
    if 'long Fourier signal' in settings:
      for target, signal in settings['long Fourier signal'].items():
        # NOTE might need to put zero filter back into it
        # "sig" is variable for the sampled result
        sig = signal['predict'][globalPicker]
        # if multidimensional, need to scale by growth factor over cycles.
        if self.multicycle:
          scales = self._evaluateScales(self.growthFactors[target], np.arange(self.numCycles))
          # do multicycle signal (m.y.Sig) all at once
          mySig = np.tile(sig, (self.numCycles, 1))
          mySig = (mySig.T * scales).T
          # TODO can we do this all at once with a vector operation? -> you betcha
          evaluation[target][:, localPicker] += mySig
        else:
          # if last segment is shorter than other clusters, just keep the part of the evaluation
          #     that makes sense? I guess? What about the "truncated" case above? - talbpaul 2020-10
          evaluation[target][localPicker] += sig
    return evaluation

  def finalizeGlobalRomSegmentEvaluation(self, settings, evaluation, weights=None, slicer=None):
    """
      Allows any global settings to be applied to the signal collected by the ROMCollection instance.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL supspace segment ROMs!
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, evaluation, dict, {target: np.ndarray} evaluated full (global) signal from ROMCollection
      @ In, weights, np.array(float), optional, if included then gives weight to histories for CDF preservation
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    # backtransform signal to preserve CDF
    ## how nicely does this play with zerofiltering?
    evaluation = self._finalizeGlobalRSE_preserveCDF(settings, evaluation, weights)
    evaluation = self._finalizeGlobalRSE_zeroFilter(settings, evaluation, weights, slicer=slicer)
    return evaluation

  def _finalizeGlobalRSE_preserveCDF(self, settings, evaluation, weights):
    """
      Helper method for finalizeGlobalRomSegmentEvaluation,
      particularly for "full" or "truncated" representation.
      -> it turns out, this works for "clustered" too because of how element-wise numpy works.
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, evaluation, dict, {target: np.ndarray} evaluated full (global) signal from ROMCollection
      @ In, weights, np.array(float), optional, if included then gives weight to histories for CDF preservation
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    # TODO FIXME
    import scipy.stats as stats
    if self.preserveInputCDF:
      for target, dist in settings['input CDFs'].items():
        if self.multicycle: #TODO check this gets caught correctly by the templateROM.
          cycles = range(len(evaluation[target]))
          scaling = self._evaluateScales(self.growthFactors[target], cycles)
          # multicycle option
          for y in range(len(evaluation[target])):
            scale = scaling[y]
            if scale != 1:
              # apply it to the preserve CDF histogram BOUNDS (bin edges)
              objectDist = dist[0]
              histDist = tuple([dist[1][0], dist[1][1]*scale])
              newObject = stats.rv_histogram(histDist)
              newDist = tuple([newObject, histDist])
              evaluation[target][y] = self._transformThroughInputCDF(evaluation[target][y], newDist, weights)
            else:
              evaluation[target][y] = self._transformThroughInputCDF(evaluation[target][y], dist, weights)
        else:
          evaluation[target] = self._transformThroughInputCDF(evaluation[target], dist, weights)
    return evaluation

  def _finalizeGlobalRSE_zeroFilter(self, settings, evaluation, weights, slicer):
    """
      Helper method for finalizeGlobalRomSegmentEvaluation,
      particularly for zerofiltering
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, evaluation, dict, {target: np.ndarray} evaluated full (global) signal from ROMCollection
      @ In, weights, np.array(float), optional, if included then gives weight to histories for CDF preservation
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    if self.zeroFilterTarget:
      mask = self._masks[self.zeroFilterTarget]['zeroFilterMask']
      if slicer is not None:
        # truncated evaluation
        newMask = []
        for sl in slicer:
          m = mask[sl.start:sl.stop].tolist()
          newMask.append(np.asarray(m))
        newMask = np.asarray(newMask)
      else:
        newMask = mask
      if self.multicycle:
        evaluation[self.zeroFilterTarget][:, newMask] = 0
      else:
        evaluation[self.zeroFilterTarget][newMask] = 0
    return evaluation



  ### Peak Picker ###
  def _peakPicker(self,signal,low):
    """
      Peak picker, this method find the local maxima index inside the signal by comparing the
      neighboring values. Threshold of peaks is required to output the height of each peak.
      @ In, signal, np.array(float), signal to transform
      @ In, low, float, required height of peaks.
      @ Out, peaks, np.array, indices of peaks in x that satisfy all given conditions
      @ Out, heights, np.array, boolean mask where is the residual signal
    """
    peaks, properties = find_peaks(signal, height=low)
    heights = properties['peak_heights']
    return peaks, heights

  def rangeWindow(self,windowDict):
    """
      Collect the window index in to groups and store the information in dictionariy for each target
      @ In, windowDict, dict, dictionary for specefic target peaks
      @ Out, rangeWindow, list, list of dictionaries which store the window index for each target
    """
    rangeWindow = []
    windowType = len(windowDict['windows'])
    windows = windowDict['windows']
    period = windowDict['period']
    for i in range(windowType):
      windowRange={}
      bgP=(windows[i]['window'][0]-1)%period
      endP=(windows[i]['window'][1]+2)%period
      timeInd=np.arange(len(self.pivotParameterValues))
      bgPInd  = np.where(timeInd%period==bgP )[0].tolist()
      endPInd = np.where(timeInd%period==endP)[0].tolist()
      if bgPInd[0]>endPInd[0]:
        tail=endPInd[0]
        endPInd.pop(0)
        endPInd.append(tail)
      windowRange['bg']=bgPInd
      windowRange['end']=endPInd
      rangeWindow.append(windowRange)
    return rangeWindow

  def _peakGroupWindow(self,signal,windowDict):
    """
      Collect the peak information in to groups, define the residual signal.
      Including the index and amplitude of each peak found in the window.
      @ In, signal, np.array(float), signal to transform
      @ In, windowDict, dict, dictionary for specefic target peaks
      @ Out, groupWin, list, list of dictionaries which store the peak information
      @ Out, maskPeakRes, np.array, boolean mask where is the residual signal
    """
    groupWin = []
    maskPeakRes = np.ones(len(signal), dtype=bool)
    rangeWindow = self.rangeWindow(windowDict)
    low = windowDict['threshold']
    windows = windowDict['windows']
    period = windowDict['period']
    for i in range(len(windowDict['windows'])):
      bg  = rangeWindow[i]['bg']
      end = rangeWindow[i]['end']
      peakInfo   = {}
      indLocal   = []
      ampLocal   = []
      for j in range(min(len(bg), len(end))):
        ##FIXME this might ignore one window, because the amount of the
        # staring points and the ending points might be different here,
        # we choose the shorter one to make sure each window is complete.
        # Future developer can extend the head and tail of the signal to
        # include all the posible windows
        bgLocal = bg[j]
        endLocal = end[j]
        if bgLocal<endLocal:
          peak, height = self._peakPicker(signal[bgLocal:endLocal], low=low)
        else:
          peak, height = self._peakPicker(np.concatenate([signal[bgLocal:], signal[:endLocal]]), low=low)
        if len(peak) == 1:
          indLocal.append(int(peak))
          ampLocal.append(float(height))
          maskBg=int((int(peak)+bgLocal-int(np.floor(windows[i]['width']/2)))%len(self.pivotParameterValues))
          maskEnd=int((int(peak)+bgLocal+int(np.ceil(windows[i]['width']/2)))%len(self.pivotParameterValues))
          if maskBg>maskEnd:
            maskPeakRes[maskBg:] = False
            maskPeakRes[:maskEnd] = False
          else:
            maskPeakRes[maskBg:maskEnd] = False
        elif len(peak) > 1:
          indLocal.append(int(peak[np.argmax(height)]))
          ampLocal.append(float(height[np.argmax(height)]))
          maskBg=int((int(peak[np.argmax(height)])+bgLocal-int(np.floor(windows[i]['width']/2)))%len(self.pivotParameterValues))
          maskEnd=int((int(peak[np.argmax(height)])+bgLocal+int(np.ceil(windows[i]['width']/2)))%len(self.pivotParameterValues))
          if maskBg>maskEnd:
            maskPeakRes[maskBg:] = False
            maskPeakRes[:maskEnd] = False
          else:
            maskPeakRes[maskBg:maskEnd] = False

      peakInfo['Ind'] = indLocal
      peakInfo['Amp'] = ampLocal
      groupWin.append(peakInfo)
    return groupWin , maskPeakRes

  def _transformBackPeaks(self,signal,windowDict):
    """
      Transforms a signal by regenerate the peaks signal
      @ In, signal, np.array(float), signal to transform
      @ In, windowDict, dict, dictionary for specefic target peaks
      @ Out, signal, np.array(float), new signal after transformation
    """
    groupWin = windowDict['groupWin']
    windows  = windowDict['windows']
    # rangeWindow = self.rangeWindow(windowDict)
    rangeWindow = windowDict['rangeWindow']
    for i in range(len(windows)):
      prbExist = len(groupWin[i]['Ind'])/len(rangeWindow[i]['bg'])
      # (amount of peaks that collected in the windows)/(the amount of windows)
      # this is the probability to check if we should add a peak in each type of window
      histAmp = np.histogram(groupWin[i]['Amp'])
      # generate the distribution of the amplitude for this type of peak
      histInd = np.histogram(groupWin[i]['Ind'])
      # generate the distribution of the position( relative index) in the window
      for j in range(min(len(rangeWindow[i]['bg']),len(rangeWindow[i]['end']))):
        # the length of the starting points and ending points might be different
        bgLocal = rangeWindow[i]['bg'][j]
        # choose the starting index for specific window
        exist = np.random.choice(2, 1, p=[1-prbExist,prbExist])
        # generate 1 or 0 base on the prbExist
        if exist == 1:
          Amp = rv_histogram(histAmp).rvs()
          Ind = int(rv_histogram(histInd).rvs())
          # generate the amplitude and the relative position base on the distribution
          SigIndOrg = bgLocal+Ind
          #signalOrg can be longer than the segment length
          SigInd = int(SigIndOrg%len(self.pivotParameterValues))
          signal[SigInd] = Amp
          # replace the signal with peak in this window
          maskBg = SigInd-int(np.floor(windows[i]['width']/2))
          ## peaks begin index can be negative end index can be more than the length of the segments
          maskEnd = SigInd+int(np.ceil(windows[i]['width']/2))
          bgValue = signal[maskBg-1]
          endVaue = signal[int((maskEnd+1)%len(self.pivotParameterValues))]
          # valueBg=np.interp(range(maskBg,SigInd), [maskBg-1,SigInd], [bgValue,  Amp])
          # valueEnd=np.interp(range(SigInd+1,maskEnd+1), [SigInd,maskEnd+1],   [Amp,endVaue])
          valuePeak=np.interp(range(maskBg,maskEnd+1), [maskBg-1,SigInd,maskEnd+1],   [bgValue,Amp,endVaue])
          maskBg=int(maskBg%len(self.pivotParameterValues))
          maskEnd=int(maskEnd%len(self.pivotParameterValues))
          # maskbg and end now can be used as index in segment
          # replace the signal inside the width of this peak by interpolation
          if maskEnd > maskBg:
            signal[maskBg:maskEnd+1]=valuePeak
          else:
            localTailInd=list(range(maskBg, int(len(self.pivotParameterValues))))
            localHeadInd=list(range(0, maskEnd+1))
            actPeakInd=localTailInd+localHeadInd
            for idd, ind in enumerate(actPeakInd):
              signal[ind]=valuePeak[idd]
      return signal

  ### ESSENTIALLY UNUSED ###
  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure, since we do not desire normalization in this implementation.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def __confidenceLocal__(self,featureVals):
    """
      This method is currently not needed for ARMA
    """
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      Currently not implemented for ARMA
    """
    pass

  def __returnInitialParametersLocal__(self):
    """
      there are no possible default parameters to report
    """
    localInitParam = {}
    return localInitParam

  def __returnCurrentSettingLocal__(self):
    """
      override this method to pass the set of parameters of the ROM that can change during simulation
      Currently not implemented for ARMA
    """
    pass

  def setEngine(self,eng,seed=None,count=None):
    """
     Set up the random engine for arma
     @ In, eng, instance, random number generator
     @ In, seed, int, optional, the seed, if None then use the global seed from ARMA
     @ In, count, int, optional, advances the state of the generator, if None then use the current ARMA.randomEng count
     @ Out, None
    """
    if seed is None:
      seed = self.seed
    seed = abs(seed)
    randomUtils.randomSeed(seed, engine=eng)
    if count is not None:
      randomUtils.forwardSeed(count, engine=eng)
    self.randomEng = eng

#
#
#
#
# Dummy class for replacing a statsmodels ARMAResults with a surrogate.
class armaResultsProxy:
  """
    Class that can be used to artifically construct ARMA information
    from pre-determined values
  """
  def __init__(self, polyar, polyma, sigma):
    """
      Constructor.
      @ In, polyar, np.array(float), autoregressive coefficients
      @ In, polyma, np.array(float), moving average coefficients
      @ In, sigma, float, standard deviation of ARMA residual noise
      @ Out, None
    """
    self.polyar = polyar
    self.polyma = polyma
    self.sigma = sigma
