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
  Created on Jan 21, 2020

  @author: alfoa, wangc
  Support Vector Regression

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
from ...utils.importerUtils import importModuleLazy
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
np = importModuleLazy("numpy")
import inspect
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..SupervisedLearning import SupervisedLearning
from ...utils import utils
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class ScikitLearnBase(SupervisedLearning):
  """
    Base Class for Scikitlearn-based surrogate models (classifiers and regressors)
  """
  info = {'problemtype':None, 'normalize':None}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    import pydmd
    import ezyrb
    import pydmd.ParametricDMD
    from ezyrb import POD, RBF    
    super().__init__()
    self.settings = None # initial settings for the ROM
    # parametric model
    self.model = pydmd.ParametricDMD
    # local models
    self._DRrom = POD
    self._interpolator = RBF    
    self._dmdBase = None  # base specific DMD estimator/model (Set by derived classes)
    self.uniqueVals = None # flag to indicate targets only have a single unique value

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
                                                 descr=r"""TWhether this instance should be light or not. A light instance uses
                                                 less memory since it caches a smaller number of resources.""", default=False))
    specs.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType,
                                                 descr=r"""defines the pivot variable (e.g., time) that represents the
                                                 independent monotonic variable""", default="time"))
    return specs

  def __init__(self):
    """
      DMD constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.dmdParams = {}          # dmd settings container
    self._dynamicHandling = True # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self.pivotParameterID = None # pivot parameter
                                 # variables filled up in the training stages
    self._amplitudes = {}        # {'target1': vector of amplitudes,'target2':vector of amplitudes, etc.}
    self._eigs = {}              # {'target1': vector of eigenvalues,'target2':vector of eigenvalues, etc.}
    self._modes = {}             # {'target1': matrix of dynamic modes,'target2':matrix of dynamic modes, etc.}
    self.__Atilde = {}           # {'target1': matrix of lowrank operator from the SVD,'target2':matrix of lowrank operator from the SVD, etc.}
    self.pivotValues = None      # pivot values (e.g. time)
    self.timeScales = {}         # time-scales (training and dmd). {'training' and 'dmd':{t0:float,'dt':float,'intervals':int}}
    self.featureVals = None      # feature values

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['pivotParameter','light'])
    # notFound must be empty
    assert(not notFound)
    self.pivotParameterID  = settings.get("pivotParameter")  # pivot parameter
    self.dmdParams['light'] = settings.get('light')         
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,f"The pivotParameter {self.pivotParameterID} must be part of the Target space!")
    if len(self.target) < 2:
      self.raiseAnError(IOError,f"At least one Target in addition to the pivotParameter {self.pivotParameterID} must be part of the Target space!")

  def initializeModel(self, settings):
    """
      Method to initialize the surrogate model with a settings dictionary
      @ In, settings, dict, the dictionary containin the parameters/settings to instanciate the model
      @ Out, None
    """
    if self.settings is None:
      self.settings = settings
    if inspect.isclass(self.model):
      self.model = self.model(**settings)
      if self.multioutputWrapper:
        self.multioutput(self.info['problemtype'])
    else:
      setts = self.updateSettings(settings)
      self.model.set_params(**setts)
      
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
  def _getTimeScale(self,dmd=True):
    """
      Get the ts of the dmd (if dmd = True) or training (if dmd = False) reconstructed time scale.
      @ In, dmd, bool, optional, True if dmd time scale needs to be returned, othewise training one
      @ Out, timeScale, numpy.array, the dmd or training reconstructed time scale
    """
    timeScaleInfo = self.timeScales['dmd'] if dmd else self.timeScales['training']
    timeScale = np.arange(timeScaleInfo['t0'], (timeScaleInfo['intervals']+1)*timeScaleInfo['dt'], timeScaleInfo['dt'])
    return timeScale

  def __getTimeEvolution(self, target):
    """
      Get the time evolution of each mode
      @ In, target, str, the target for which mode evolution needs to be retrieved for
      @ Out, timeEvol, numpy.ndarray, the matrix that contains all the time evolution (by row)
    """
    omega = np.log(self._eigs[target]) / self.timeScales['training']['dt']
    van = np.exp(np.multiply(*np.meshgrid(omega, self._getTimeScale())))
    timeEvol = (van * self._amplitudes[target]).T
    return timeEvol

  def _reconstructData(self, target):
    """
      Retrieve the reconstructed data
      @ In, target, str, the target for which the data needs to be reconstructed
      @ Out, data, numpy.ndarray, the matrix (nsamples,n_time_steps) containing the reconstructed data
    """
    data = self._modes[target].dot(self.__getTimeEvolution(target))
    return data

  def _train(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, numpy.ndarray, shape = [n_timeStep, n_dimensions], an array of time series data
    """
    
    self.model.fit()
    
 

  def __evaluateLocal__(self,featureVals):
    """
      This method is used to inquire the DMD to evaluate (after normalization that in
      this case is not performed)  a set of points contained in featureVals.
      a KDTree algorithm is used to construct a weighting function for the reconstructed space
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
 

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
    description  = ' This XML file contains the main information of the DMD ROM.'
    description += ' If "modes" (dynamic modes), "eigs" (eigenvalues), "amplitudes" (mode amplitudes)'
    description += ' and "dmdTimeScale" (internal dmd time scale) are dumped, the method'
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
    what = ['exactModes','optimized','dmdType','features','timeScale','eigs','amplitudes','modes','dmdTimeScale']
    if self.dmdParams['rankTLSQ'] is not None:
      what.append('rankTLSQ')
    what.append('energyRankSVD' if self.dmdParams['energyRankSVD'] is not None else 'rankSVD')
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

    target = self.target[-1]
    toAdd = ['exactModes','optimized','dmdType']
    if self.dmdParams['rankTLSQ'] is not None:
      toAdd.append('rankTLSQ')
    toAdd.append('energyRankSVD' if self.dmdParams['energyRankSVD'] is not None else 'rankSVD')
    self.dmdParams['rankSVD'] = self.dmdParams['rankSVD'] if self.dmdParams['rankSVD'] is not None else -1

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
      eigsReal = " ".join(['%.6e' % self._eigs[target][indx].real for indx in
                       range(len(self._eigs[target]))])
      writeTo.addScalar("eigs","real", eigsReal, root=targNode)
      eigsImag = " ".join(['%.6e' % self._eigs[target][indx].imag for indx in
                               range(len(self._eigs[target]))])
      writeTo.addScalar("eigs","imaginary", eigsImag, root=targNode)
    if "amplitudes" in what:
      ampsReal = " ".join(['%.6e' % self._amplitudes[target][indx].real for indx in
                       range(len(self._amplitudes[target]))])
      writeTo.addScalar("amplitudes","real", ampsReal, root=targNode)
      ampsImag = " ".join(['%.6e' % self._amplitudes[target][indx].imag for indx in
                               range(len(self._amplitudes[target]))])
      writeTo.addScalar("amplitudes","imaginary", ampsImag, root=targNode)
    if "modes" in what:
      for smp in range(len(self._modes[target])):
        valDict = {'real': ' '.join([ '%.6e' % elm for elm in self._modes[target][smp,:].real]),
                   'imaginary':' '.join([ '%.6e' % elm for elm in self._modes[target][smp,:].imag])}
        attributeDict = {self.features[index]:'%.6e' % self.featureVals[smp,index] for index in range(len(self.features))}
        writeTo.addVector("modes","realization",valDict, root=targNode, attrs=attributeDict)

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
    self._amplitudes  = {}
    self._eigs        = {}
    self._modes       = {}
    self.__Atilde     = {}
    self.pivotValues  = None
    self.KDTreeFinder = None
    self.featureVals  = None

  def __returnInitialParametersLocal__(self):
    """
      This method returns the initial parameters of the SM
      @ In, None
      @ Out, self.dmdParams, dict, the dict of the SM settings
    """
    return self.dmdParams

  def __returnCurrentSettingLocal__(self):
    """
      This method is used to pass the set of parameters of the ROM that can change during simulation
      @ In, None
      @ Out, self.dmdParams, dict, the dict of the SM settings
    """
    return self.dmdParams