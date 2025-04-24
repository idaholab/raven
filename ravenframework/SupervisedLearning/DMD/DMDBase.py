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
  Created on July 21, 2024

  @author: Andrea Alfonsi
  Dynamic Mode Decomposition base class

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
from ...utils.importerUtils import importModuleLazy
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
np = importModuleLazy("numpy")
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..SupervisedLearning import SupervisedLearning
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------


approximationDefaults = {'GPR': {},'RBF': {}}
# RBF
approximationDefaults['RBF']['kernel'] = 'multiquadric'
approximationDefaults['RBF']['smooth'] = 0.
approximationDefaults['RBF']['neighbors'] = None
approximationDefaults['RBF']['epsilon'] = 1.
approximationDefaults['RBF']['degree'] = None
# GPR
approximationDefaults['GPR']['optimization_restart'] = 0
approximationDefaults['GPR']['normalize_y'] = True

class DMDBase(SupervisedLearning):
  """
    Base Class for DMD-based surrogate models
  """
  info = {'problemtype':None, 'normalize':None}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    # handling time series?
    self._dynamicHandling = True
    # initial settings for the ROM (coming from input)
    self.settings = {}
    # dmd-based model parameters (used in the initialization of the DMD models)
    self.dmdParams = {}
    # parametric model
    self.model = None #  ParametericDMD
    # local models
    ## POD
    self._dimReductionRom = None
    ## RBF
    self._interpolator = None
    ## base specific DMD estimator/model (Set by derived classes)
    self._dmdBase = None
    ## DMD fit arguments (overloaded by derived classes (if needed))
    self.fitArguments = {}
    # flag to indicate that the model has a single target (in addition to the pivot parameter)
    # This flag is needed because the DMD based model has an issue with single target (space dimension == 1) and
    # a counter mesurament (concatenation of snapshots) is required
    self.singleTarget = False
    # target indeces (positions in self.target list)
    self.targetIndices = None

  @property
  def featureImportances_(self):
    """
      This property is in charge of extracting from the estimators
      the importance of the features used in the training process
      @ In, None
      @ Out, importances, dict, {featName:float or array(nTargets)} importances of the features
    """
    # store importances
    importances = {}
    return importances

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
    specs.addSub(InputData.parameterInputFactory("light", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether this instance should be light or not. A light instance uses
                                                 less memory since it caches a smaller number of resources.""", default=False))
    specs.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType,
                                                 descr=r"""defines the pivot variable (e.g., time) that represents the
                                                 independent monotonic variable""", default="time"))
    specs.addSub(InputData.parameterInputFactory("reductionMethod", contentType=InputTypes.makeEnumType("reductionMethod", "reductionMethodType",
                                                                                                        ["svd", "correlation_matrix", "randomized_svd"]),
                                                 descr=r"""the type of method used for the dimensionality reduction.Available are:
                                                  \begin{itemize}
                                                    \item \textit{svd}, single value decomposition
                                                    \item \textit{svd}, randomized single value decomposition
                                                    \item \textit{correlation\_matrix}, correlation-based reduction.
                                                  \end{itemize}""", default="svd"))
    specs.addSub(InputData.parameterInputFactory("reductionRank", contentType=InputTypes.IntegerType,
                                                 descr=r"""defines the truncation rank to be used for the reduction method.
                                                 Available options are:
                                                 \begin{itemize}
                                                 \item \textit{-1}, no truncation is performed
                                                 \item \textit{0}, optimal rank is internally computed
                                                 \item \textit{$>1$}, this rank is going to be used for the truncation
                                                 \end{itemize}""", default=0))
    specs.addSub(InputData.parameterInputFactory("approximationMethod", contentType=InputTypes.makeEnumType("approximationMethod", "approximationMethodType",
                                                                                                        ["RBF", "GPR"]),
                                                 descr=r"""the type of method used for the interpolation of the parameter space.Available are:
                                                  \begin{itemize}
                                                    \item \textit{RBF}, Radial-basis functions
                                                    \item \textit{GPR}, Gaussian Process Regression
                                                  \end{itemize}""", default="RBF"))

    approximationSettings = InputData.parameterInputFactory("approximationSettings",
                                                 descr=r"""the settings available depending on the different type of method used for the interpolation of the parameter space""")
    #RBF
    approximationSettings.addSub(InputData.parameterInputFactory("kernel", contentType=InputTypes.makeEnumType("kernelRBF", "kernelRBFType",
                                                                                                        ["cubic", "quintic", "linear",
                                                                                                         "gaussian", "inverse", "multiquadric", "thin_plate_spline"]),
                                                 descr=r"""RBF kernel.
                                                 Available options are:
                                                 \begin{itemize}
                                                 \item \textit{thin\_plate\_spline}, thin-plate spline ($r**2 * log(r)$)
                                                 \item \textit{cubic}, cubic kernel ($r**3$)
                                                 \item \textit{quintic}, quintic kernel ($r**5$)
                                                 \item \textit{linear}, linear kernel ($r$)
                                                 \item \textit{gaussian}, gaussian kernel ($exp(-(r/self.epsilon)**2)$)
                                                 \item \textit{inverse}, inverse kernel ($1.0/sqrt((r/self.epsilon)**2 + 1)$)
                                                 \item \textit{multiquadric}, multiquadric kernel ($sqrt((r/self.epsilon)**2 + 1)$)
                                                 \end{itemize}""", default=approximationDefaults['RBF']['kernel']))
    approximationSettings.addSub(InputData.parameterInputFactory("smooth", contentType=InputTypes.FloatType,
                                                 descr=r"""RBF smooth factor. Values greater than zero increase the smoothness of the approximation.
                                                 0 is for interpolation (default), the function will always go through the nodal points in this case.
                                                 """, default=approximationDefaults['RBF']['smooth']))
    approximationSettings.addSub(InputData.parameterInputFactory("neighbors", contentType=InputTypes.IntegerType,
                                                 descr=r"""RBF number of neighbors. If specified, the value of the interpolant at each
                                                           evaluation point will be computed using only the nearest data points.
                                                           If None (default), all the data points are used by default.""",
                                                 default=approximationDefaults['RBF']['neighbors']))
    approximationSettings.addSub(InputData.parameterInputFactory("epsilon", contentType=InputTypes.FloatType,
                                                 descr=r"""RBF Shape parameter that scales the input to the RBF.
                                                           If kernel is ``linear'', ‘thin_plate_spline'', ``cubic'', or ``quintic'', this
                                                           defaults to 1 and can be ignored. Otherwise, this must be specified.""",
                                                 default=approximationDefaults['RBF']['epsilon']))
    approximationSettings.addSub(InputData.parameterInputFactory("degree", contentType=InputTypes.IntegerType,
                                                 descr=r"""RBF Degree of the added polynomial. The default value is
                                                           the minimum degree for kernel or 0 if there is no minimum degree.""",
                                                 default=approximationDefaults['RBF']['degree']))
    #GPR
    approximationSettings.addSub(InputData.parameterInputFactory("optimization_restart", contentType=InputTypes.IntegerType,
                                                 descr=r"""GPR restart parameter. The number of restarts of the optimizer for finding the
                                                 kernel parameters which maximize the log-marginal likelihood. The first run of the optimizer
                                                 is performed from the kernel’s initial parameters, the remaining ones (if any) from thetas
                                                 sampled log-uniform randomly from the space of allowed theta-values. If greater than 0,
                                                 all bounds must be finite. Note that $n\_restarts\_optimizer == 0$ implies that one run is performed.""",
                                                 default=approximationDefaults['GPR']['optimization_restart']))
    approximationSettings.addSub(InputData.parameterInputFactory("normalize_y", contentType=InputTypes.BoolType,
                                                 descr=r"""GPR normalization. Whether or not to normalize the target values y by removing the mean and scaling
                                                 to unit-variance. This is recommended for cases where zero-mean, unit-variance priors are used.
                                                 Note that, in this implementation, the normalisation is reversed before the GP predictions are reported.""",
                                                 default=approximationDefaults['GPR']['normalize_y']))

    specs.addSub(approximationSettings)
    return specs


  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['pivotParameter','light', 'reductionRank', 'reductionMethod', 'approximationMethod'])
    # notFound must be empty
    assert(not notFound)
    self.settings = {}
    self.pivotParameterID  = settings.get("pivotParameter")  # pivot parameter
    self.settings['light'] = settings.get('light')
    self.settings['reductionMethod'] = settings.get('reductionMethod')
    self.settings['reductionRank'] = settings.get('reductionRank')
    self.settings['approximationMethod'] = settings.get('approximationMethod')
    approximationSettings = paramInput.findFirst("approximationSettings")
    self.settings['approximationSettings'] = {}
    if self.settings['approximationMethod'] == 'RBF':
      if  approximationSettings is not None:
        RBFsettings, RBFnotFound = approximationSettings.findNodesAndExtractValues(['kernel','smooth', 'neighbors', 'epsilon', 'degree'])
        # RBFnotFound must be empty
        assert(not RBFnotFound)
      else:
        RBFsettings = approximationDefaults['RBF']
      self.settings['approximationSettings']['kernel'] = RBFsettings.get('kernel')
      self.settings['approximationSettings']['smooth'] = RBFsettings.get('smooth')
      self.settings['approximationSettings']['neighbors'] = RBFsettings.get('neighbors')
      self.settings['approximationSettings']['epsilon'] = RBFsettings.get('epsilon')
      self.settings['approximationSettings']['degree'] = RBFsettings.get('degree')
    elif self.settings['approximationMethod'] == 'GPR':
      if  approximationSettings is not None:
        GPRsettings, GPRnotFound = approximationSettings.findNodesAndExtractValues(['optimization_restart','normalize_y'])
        # GPRnotFound must be empty
        assert(not GPRnotFound)
      else:
        GPRsettings =  approximationDefaults['GPR']
      self.settings['approximationSettings']['optimization_restart'] = GPRsettings.get('optimization_restart')
      self.settings['approximationSettings']['normalize_y'] = GPRsettings.get('normalize_y')
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,f"The pivotParameter {self.pivotParameterID} must be part of the Target space!")
    if len(self.target) < 2:
      self.raiseAnError(IOError,f"At least one Target in addition to the pivotParameter {self.pivotParameterID} must be part of the Target space!")

  def initializeModel(self, dmdParams):
    """
      Method to initialize the surrogate model with a dmdParams dictionary
      @ In, dmdParams, dict, the dictionary containin the parameters/settings to instanciate the model
      @ Out, None
    """
    from pydmd import ParametricDMD
    from ...contrib.ezyrb import POD, RBF, GPR

    assert(self._dmdBase is not None)
    self.dmdParams = dmdParams
    print(self.dmdParams)

    # intialize dimensionality reduction
    self._dimReductionRom = POD(self.settings['reductionMethod'], rank=self.settings['reductionRank'])
    # initialize coefficient interpolator
    if self.settings['approximationMethod'] == 'RBF':
      self._interpolator = RBF(kernel=self.settings['approximationSettings']['kernel'], smooth=self.settings['approximationSettings']['smooth'],
                               neighbors=self.settings['approximationSettings']['neighbors'], epsilon=self.settings['approximationSettings']['epsilon'],
                               degree=self.settings['approximationSettings']['degree'])
    elif self.settings['approximationMethod'] == 'GPR':
      self._interpolator = GPR(optimization_restart=self.settings['approximationSettings']['optimization_restart'],
                               normalizer=self.settings['approximationSettings']['normalize_y'])
    # initialize the base model
    self._dmdBase = self._dmdBase(**self.dmdParams)
    self.model = ParametricDMD(self._dmdBase, self._dimReductionRom, self._interpolator, light=self.settings['light'], dmd_fit_kwargs=self.fitArguments)
    # set type of dmd class
    self.dmdType = self.__class__.__name__
    # check if single target
    self.singleTarget = len(self.target) == 2
    self.targetIndices = tuple([i for i,x in enumerate(self.target) if x != self.pivotID])

  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  #######
  def _getTimeScale(self):
    """
      Get the ts of the dmd (if dmd = True) or training (if dmd = False) reconstructed time scale.
      @ In, None
      @ Out, timeScale, numpy.array, the dmd or training reconstructed time scale
    """
    try:
      timeScaleInfo = self.model.dmd_time
      if isinstance(timeScaleInfo, dict):
        timeScale = np.arange(timeScaleInfo['t0'], (timeScaleInfo['tend']+1)*timeScaleInfo['dt'], timeScaleInfo['dt'])
      else:
        timeScale = timeScaleInfo
    except AttributeError:
      if 'time' in  dir(self.model._reference_dmd):
        timeScale = self.model._reference_dmd.time
      else:
        timeScale = self.pivotValues.flatten()
    return timeScale

  def _preFitModifications(self):
    """
      Method to modify parameters and populate fit argument before fitting
      @ In, None
      @ Out, None
    """
    pass

  def _train(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape=[n_samples, n_features], an array of input data # Not use for ARMA training
      @ In, targetVals, numpy.ndarray, shape = [n_samples, n_timeStep, n_targets], an array of time series data
    """

    #        - 0: Training parameters;
    #        - 1: Space;
    #        - 2: Training time instants.
    self.featureVals  = featureVals
    pivotParamIndex   = self.target.index(self.pivotParameterID)
    self.pivotValues  = targetVals[0,:,pivotParamIndex]

    snapshots = np.swapaxes(targetVals, 1, 2)
    if self.singleTarget:
      targetSnaps = snapshots[:, self.targetIndices, :].reshape((snapshots.shape[0], 1, snapshots.shape[-1]))
      targetSnaps = np.concatenate((targetSnaps, targetSnaps), axis=1)
    else:
      targetSnaps = snapshots[:, self.targetIndices, :]
    # populate fit arguments and allow for modifications (if needed)
    self._preFitModifications()
    # fit model
    self.model.fit(targetSnaps, training_parameters=featureVals)
    self.model.parameters = featureVals

  def __evaluateLocal__(self,featureVals):
    """
      This method is used to inquire the DMD to evaluate (after normalization that in
      this case is not performed)  a set of points contained in featureVals.
      a KDTree algorithm is used to construct a weighting function for the reconstructed space
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    returnEvaluation = dict.fromkeys(self.target)
    returnEvaluation[self.pivotID] = self.pivotValues
    self.model.parameters = featureVals
    data = self.model.reconstructed_data
    for didx, tidx in enumerate(self.targetIndices):
      target = self.target[tidx]
      returnEvaluation[target] = data[:, didx, :].flatten().real

    return returnEvaluation

  def writeXMLPreamble(self, writeTo, targets = None):
    """
      Specific local method for printing anything desired to xml file at the begin of the print.
      @ In, writeTo, xmlUtils.StaticXmlElement instance, element to write to
      @ In, targets, list, list of targets for whom information should be written.
      @ Out, None
    """
    # add description
    super().writeXMLPreamble(writeTo, targets)
    description  = ' This XML file contains the main information of the DMD-based ROM .'
    description += ' If "modes" (dynamic modes), "eigs" (eigenvalues), "amplitudes" (mode amplitudes)'
    description += ' and "dmdTimeScale" (internal dmd time scale) are dumped, the basic method'
    description += ' is explained in P.J. Schmid, Dynamic mode decomposition'
    description += ' of numerical and experimental data, Journal of Fluid Mechanics 656.1 (2010), 5-28'
    writeTo.addScalar('ROM',"description",description)

  def writeXML(self, writeTo, targets = None, skip = None):
    """
      Adds requested entries to XML node.
      @ In, writeTo, xmlTuils.StaticXmlElement, element to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    if skip is None:
      skip = []

    # check what

    what = ['features','timeScale','eigs','amplitudes','modes','dmdTimeScale'] + list(self.dmdParams.keys())
    if targets is None:
      readWhat = what
    else:
      readWhat = targets
    for s in skip:
      if s in readWhat:
        readWhat.remove(s)
    if not set(readWhat) <= set(what):
      self.raiseAnError(IOError, "The following variables specified in <what> node are not recognized: "+ ",".join(np.setdiff1d(readWhat, what).tolist()) )
    else:
      what = readWhat

    target = self.name
    toAdd = list(self.dmdParams.keys())

    for add in toAdd:
      if add in what :
        writeTo.addScalar(target,add,self.dmdParams[add])
    targNode = writeTo._findTarget(writeTo.getRoot(), target)
    if "features" in what:
      writeTo.addScalar(target,"features",' '.join(self.features))
    if "timeScale" in what:
      writeTo.addScalar(target,"timeScale",' '.join(['%.6e' % elm for elm in self.pivotValues.ravel()]))
    if "dmdTimeScale" in what:
      writeTo.addScalar(target,"dmdTimeScale",' '.join(['%.6e' % elm for elm in self._getTimeScale()]))
    if "eigs" in what:
      eigsReal = " ".join(['%.6e' % self.model._reference_dmd.eigs[indx].real for indx in
                       range(len(self.model._reference_dmd.eigs))])
      writeTo.addScalar("eigs","real", eigsReal, root=targNode)
      eigsImag = " ".join(['%.6e' % self.model._reference_dmd.eigs.imag[indx] for indx in
                               range(len(self.model._reference_dmd.eigs))])
      writeTo.addScalar("eigs","imaginary", eigsImag, root=targNode)
    if "amplitudes" in what and 'amplitudes' in dir(self.model._reference_dmd) and self.model._reference_dmd.amplitudes is not None:
      ampsReal = " ".join(['%.6e' % self.model._reference_dmd.amplitudes.real[indx] for indx in
                       range(len(self.model._reference_dmd.amplitudes))])
      writeTo.addScalar("amplitudes","real", ampsReal, root=targNode)
      ampsImag = " ".join(['%.6e' % self.model._reference_dmd.amplitudes.imag[indx] for indx in
                               range(len(self.model._reference_dmd.amplitudes))])
      writeTo.addScalar("amplitudes","imaginary", ampsImag, root=targNode)
    if "modes" in what:
      nSamples = self.featureVals.shape[0]
      delays = max(1, int(self.model._reference_dmd.modes.shape[0] / nSamples))
      loopCnt = 0
      noSampled = False
      if nSamples * delays !=  self.model._reference_dmd.modes.shape[0]:
        nSamples = self.model._reference_dmd.modes.shape[0]
        noSampled = True
      for smp in range(nSamples):
        valDict = {'real':'', 'imaginary': ''}
        for _ in range(delays):
          valDict['real'] += ' '.join([ '%.6e' % elm for elm in self.model._reference_dmd.modes[loopCnt,:].real]) + ' '
          valDict['imaginary'] += ' '.join([ '%.6e' % elm for elm in self.model._reference_dmd.modes[loopCnt,:].imag]) +' '
          loopCnt += 1
        if noSampled:
          attributeDict = {"index":f'{loopCnt}'}
        else:
          attributeDict = {self.features[index]:'%.6e' % self.featureVals[smp,index] for index in range(len(self.features))}
        if delays > 1:
          attributeDict['shape'] = f"({self.model._reference_dmd.modes.shape[1]},{delays})"
        writeTo.addVector("modes","realization" if not noSampled else "element",valDict, root=targNode, attrs=attributeDict)

  def __confidenceLocal__(self,featureVals):
    """
      The confidence associate with a set of requested evaluations
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, None
    """
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      @ In, featureVals, numpy.ndarray, shape= (n_samples, n_dimensions), an array of input data (training data)
      @ Out, None
    """
    self.amITrained   = False
    self.model = {}
    self.pivotValues  = None
    self.featureVals  = None

  def __returnInitialParametersLocal__(self):
    """
      This method returns the initial parameters of the SM
      @ In, None
      @ Out, params, dict, the dict of the SM settings
    """
    params = self.dmdParams
    params.update(self.settings)
    return params

  def __returnCurrentSettingLocal__(self):
    """
      This method is used to pass the set of parameters of the ROM that can change during simulation
      @ In, None
      @ Out, params, dict, the dict of the SM settings
    """
    return self.__returnInitialParametersLocal__()


#magic to allow DMDBase to be pickled
DMDBase.getInputSpecification()
