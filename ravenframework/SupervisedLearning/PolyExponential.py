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

  @author: alfoa, talbpaul, wangc
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Base subclass definition for PolyExponential ROM (transferred from alfoa in SupervisedLearning)
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..SupervisedLearning import SupervisedLearning
from ..SupervisedLearning.NDspline import NDspline
from ..utils import InputTypes, InputData
#Internal Modules End--------------------------------------------------------------------------------


class PolyExponential(SupervisedLearning):
  """
    This surrogate is aimed to construct a time-dep surrogate based on a polynomial sum of exponentials
    The surrogate will have the form:
    $SM(X,z) = sum_{i=1}^N P_{i}(X) exp ( - Q_{i}(X) z )$
    where:
      z is the independent  monotonic variable (e.g. time)
      X is the vector of the other independent (parametric) variables
      $P_{i}(X)$ is a polynomial of rank M function of the parametric space X
      $Q_{i}(X)$ is a polynomial of rank M function of the parametric space X
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    spec.description = r"""The \xmlNode{PolyExponential} contains a single ROM type, aimed to construct a
    time-dependent (or any other monotonic variable) surrogate model based on polynomial sum of exponential term.
    This surrogate have the form:
    \begin{equation}
      SM(X,z) = \sum_{i=1}^{N} P_{i}(X) \times \exp ( - Q_{i}(X) \times z )
    \end{equation}
    where:
    \begin{itemize}
      \item $\mathbf{z}$ is the independent  monotonic variable (e.g. time)
      \item $\mathbf{X}$  is the vector of the other independent (parametric) variables  (Features)
      \item $\mathbf{P_{i}}(X)$ is a polynomial of rank M function of the parametric space X
      \item  $\mathbf{Q_{i}}(X)$ is a polynomial of rank M function of the parametric space X
      \item  $\mathbf{N}$ is the number of requested exponential terms.
    \end{itemize}
    It is crucial to notice that this model is quite suitable for FOMs whose drivers are characterized by an exponential-like behavior.
    In addition, it is important to notice that the exponential terms' coefficients are computed running a genetic-algorithm optimization
    problem, which is quite slow in case of increasing number of ``numberExpTerms''.
    In order to use this Reduced Order Model, the \xmlNode{ROM} attribute
    \xmlAttr{subType} needs to be set equal to \xmlString{PolyExponential}.
    \\
    Once the ROM is trained (\textbf{Step} \xmlNode{RomTrainer}), its coefficients can be exported into an XML file
    via an \xmlNode{OutStream} of type \xmlAttr{Print}. The following variable/parameters can be exported (i.e. \xmlNode{what} node
    in \xmlNode{OutStream} of type \xmlAttr{Print}):
    \begin{itemize}
      \item \xmlNode{expTerms}, see XML input specifications above, inquired pre-pending the keyword ``output|'' (e.g. output| expTerms)
      \item \xmlNode{coeffRegressor}, see XML input specifications above
      \item \xmlNode{polyOrder}, see XML input specifications above
      \item \xmlNode{features}, see XML input specifications above
      \item \xmlNode{timeScale}, XML node containing the array of the training time steps values
      \item \xmlNode{coefficients}, XML node containing the exponential terms' coefficients for each realization
    \end{itemize}"""

    spec.addSub(InputData.parameterInputFactory('pivotParameter',contentType=InputTypes.StringType,
                                                descr=r"""defines the pivot variable (e.g., time) that represents the
                                                independent monotonic variable""", default='time'))
    spec.addSub(InputData.parameterInputFactory('numberExpTerms',contentType=InputTypes.IntegerType,
                                                descr=r"""the number of exponential terms to be used ($N$ above)""", default=3))
    coeffRegressorEnumType = InputTypes.makeEnumType("coeffRegressor","coeffRegressorType",["poly","spline","nearest"])
    spec.addSub(InputData.parameterInputFactory('coeffRegressor',contentType=coeffRegressorEnumType,
                                                descr=r"""defines which regressor to use for interpolating the
                                                exponential coefficient. Available are ``spline'',``poly'' and ``nearest''.""",
                                                default='spline'))
    spec.addSub(InputData.parameterInputFactory('polyOrder',contentType=InputTypes.IntegerType,
                                                descr=r"""the polynomial order to be used for interpolating the exponential
                                                coefficients. Only valid in case of  \xmlNode{coeffRegressor} set to ``poly''.""", default=3))
    spec.addSub(InputData.parameterInputFactory('tol',contentType=InputTypes.FloatType,
                                                descr=r"""relative tolerance of the optimization problem (differential evolution optimizer)""",
                                                default=1e-3))
    spec.addSub(InputData.parameterInputFactory('max_iter',contentType=InputTypes.IntegerType,
                                                descr=r"""maximum number of iterations (generations) for the
                                                optimization problem  (differential evolution optimizer)""", default=5000))
    return spec

  def __init__(self):
    """
      PolyExponential constructor
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
    """

    super().__init__()
    self.availCoeffRegressors               = ['nearest','poly','spline']                   # avail coeff regressors
    self.printTag                           = 'PolyExponential'                             # Print tag
    self._dynamicHandling                   = True                                          # This ROM is able to manage the time-series on its own
    self.polyExpParams                      = {}                                            # poly exponential options' container
    self.aij                                = None                                          # a_ij coefficients of the exponential terms {'target1':ndarray(nsamples, self.polyExpParams['expTerms']),'target2',ndarray,etc}
    self.bij                                = None                                          # b_ij coefficients of the exponent of the exponential terms {'target1':ndarray(nsamples, self.polyExpParams['expTerms']),'target2',ndarray,etc}
    self.model                              = None                                          # the surrogate model itself {'target1':model,'target2':model, etc.}

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['pivotParameter','numberExpTerms', 'coeffRegressor',
                                                               'polyOrder','tol','max_iter'])
    # notFound must be empty
    assert(not notFound)
    self.pivotParameterID                   = settings.get("pivotParameter")           # Pivot parameter ID
    self.polyExpParams['expTerms']          = settings.get('numberExpTerms')         # the number of exponential terms
    self.polyExpParams['coeffRegressor']    = settings.get('coeffRegressor') # which regressor to use for interpolating the coefficient
    self.polyExpParams['polyOrder']         = settings.get('polyOrder')                # the polynomial order
    self.polyExpParams['tol']               = settings.get('tol')                # optimization tolerance
    self.polyExpParams['maxNumberIter']     = settings.get('max_iter')         # maximum number of iterations in optimization
    # check if the pivotParameter is among the targetValues
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")

  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def __computeExpTerms(self, x, y, returnPredictDiff=True):
    """
      Method to compute the coefficients of "n" exponential terms that minimize the
      difference between the training data and the "predicted" data
      y(x) = sum_{i=1}**n a_i exp ( - bi x )
      @ In, x, numpy.ndarray, the x values
      @ In, y, numpy.ndarray, the target values
      @ In, storePredictDiff, bool, optional, True if the prediction differences need to be returned (default False)
      @ Out, (fi, taui**(-1), predictionErr (optional) ), tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray (optional)),
             ai and bi and predictionErr (if returnPredictDiff=True)
    """
    def _objective(s):
      """
        Objective function for the optimization
        @ In, s, numpy.ndarray, the array of coefficient
        @ Out, objective, float, the cumulative difference between the predicted and the real data
      """
      l = int(s.size/2)
      return np.sum((y - np.dot(s[l:], np.exp(-np.outer(1./s[:l], x))))**2.)
    # I import the differential_evolution here since it is available for scipy ver > 0.15 only and
    # we do not require it yet
    ##TODO: update library requirement
    try:
      from scipy.optimize import differential_evolution
    except ImportError:
      self.raiseAnError(ImportError, "Minimum scipy version to use this SM is 0.15")
    numberTerms   = self.polyExpParams['expTerms']
    predictionErr = None
    x, y          = np.array(x), np.array(y)
    bounds        = [[min(x), max(x)]]*numberTerms + [[min(y), max(y)]]*numberTerms
    result        = differential_evolution(_objective, bounds,
                                           maxiter=self.polyExpParams['maxNumberIter'],
                                           tol=self.polyExpParams['tol'],
                                           disp=False,
                                           seed=200286)
    taui, fi      = np.split(result['x'], 2)
    sortIndexes   = np.argsort(fi)
    fi, taui      = fi[sortIndexes], taui[sortIndexes]
    if returnPredictDiff:
      predictionErr = (y-self.__evaluateExpTerm(x, fi, 1./taui))/y
    return fi, 1./taui, predictionErr

  def __evaluateExpTerm(self,x, a, b):
    """
      Evaluate exponential term given x, a and b
      y(x) = sum_{i=1}**n ai exp ( - bi x )
      @ In, x, numpy.ndarray, the x values
      @ In, a, numpy.ndarray, the a values
      @ In, b, numpy.ndarray, the b values
      @ Out, y, numpy.ndarray, the outcome y(x)
    """
    return np.dot(a, np.exp(-np.outer(b, x)))

  def _train(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape= (n_samples, n_dimensions), an array of input data
      @ In, targetVals, numpy.ndarray, shape = (n_samples, n_timeStep), an array of time series data
    """
    import sklearn.preprocessing
    import sklearn.pipeline
    import sklearn.linear_model
    import sklearn.neighbors
    # check if the data are time-dependent, otherwise error out
    if (len(targetVals.shape) != 3) :
      self.raiseAnError(Exception, "This ROM is specifically usable for time-series data surrogating (i.e. HistorySet)!")
    targetIndexes     = {}
    self.aij          = {}
    self.bij          = {}
    self.predictError = {}
    self.model        = {}
    pivotParamIndex   = self.target.index(self.pivotParameterID)
    nsamples          = len(targetVals[:,:,pivotParamIndex])
    for index, target in enumerate(self.target):
      if target != self.pivotParameterID:
        targetIndexes[target]     = index
        self.aij[target]          = np.zeros( (nsamples, self.polyExpParams['expTerms']))
        self.bij[target]          = np.zeros((nsamples, self.polyExpParams['expTerms']))
        self.predictError[target] = np.zeros( (nsamples, len(targetVals[0,:,index]) ))
    #TODO: this can be parallelized
    for smp in range(nsamples):
      self.raiseADebug("Computing exponential terms for sample ID "+str(smp+1))
      for target in targetIndexes:
        resp = self.__computeExpTerms(targetVals[smp,:,pivotParamIndex],targetVals[smp,:,targetIndexes[target]])
        self.aij[target][smp,:], self.bij[target][smp,:], self.predictError[target][smp,:] = resp
    # store the pivot values
    self.pivotValues = targetVals[0,:,pivotParamIndex]
    if self.polyExpParams['coeffRegressor']== 'nearest':
      self.scaler = sklearn.preprocessing.StandardScaler().fit(featureVals)
    # construct poly
    for target in targetIndexes:
      # the targets are the coefficients
      expTermCoeff = np.concatenate( (self.aij[target],self.bij[target]), axis=1)
      if self.polyExpParams['coeffRegressor']== 'poly':
        # now that we have the coefficients, we can construct the polynomial expansion whose targets are the just computed coefficients
        self.model[target] = sklearn.pipeline.make_pipeline(sklearn.preprocessing.PolynomialFeatures(self.polyExpParams['polyOrder']), sklearn.linear_model.Ridge())
        self.model[target].fit(featureVals, expTermCoeff)
        # print the coefficient
        self.raiseADebug('poly coefficients for target "'+target+'":')
        coefficients = self.model[target].steps[1][1].coef_
        for l, coeff in enumerate(coefficients):
          coeff_str = "    a_"+str(l+1) if l < self.polyExpParams['expTerms'] else "    b_"+str((l-self.polyExpParams['expTerms'])+1)
          coeff_str+="(" + ",".join(self.features)+"):"
          self.raiseADebug(coeff_str)
          self.raiseADebug("      "+" ".join([str(elm) for elm in coeff]))
      elif self.polyExpParams['coeffRegressor']== 'nearest':
        # construct nearest
        self.model[target] = [None for _ in range(self.polyExpParams['expTerms']*2)]
        for cnt in range(len(self.model[target] )):
          self.model[target][cnt] = sklearn.neighbors.KNeighborsRegressor(n_neighbors=min(nsamples, 2**len(self.features)), weights='distance')
          self.model[target][cnt].fit(self.scaler.transform(featureVals),expTermCoeff[:,cnt])
      else:
        # construct spline
        numbTerms = self.polyExpParams['expTerms']
        targets   = ["a_"+str(cnt+1) if cnt < numbTerms else "b_"+str((cnt-numbTerms)+1) for cnt in range(numbTerms*2)]
        self.model[target] = NDspline()
        inputDict = {'Features':self.features, 'Target':targets}
        self.model[target].initializeFromDict(inputDict)
        self.model[target].__class__._train(self.model[target],featureVals,expTermCoeff)
    self.featureVals = featureVals

  def __evaluateLocal__(self,featureVals):
    """
      This method is used to inquire the PolyExponential to evaluate (after normalization that in
      this case is not performed)  a set of points contained in featureVals.
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    returnEvaluation = {self.pivotParameterID:self.pivotValues}
    for target in list(set(self.target) - set([self.pivotParameterID])):
      if isinstance(self.model[target], list):
        evaluation = np.zeros((len(featureVals),len(self.model[target])))
        for cnt,model in enumerate(self.model[target]):
          evaluation[:,cnt] = model.predict(self.scaler.transform(featureVals) )
      else:
        if 'predict' in dir(self.model[target]):
          evaluation = self.model[target].predict(featureVals)
        else:
          evaluation = np.zeros((len(featureVals),len(self.model[target].target)))
          evalDict = self.model[target].__class__.__evaluateLocal__(self.model[target],featureVals)
          for cnt,targ in enumerate(self.model[target].target):
            evaluation[:,cnt] = evalDict[targ][:]
      for point in range(len(evaluation)):
        l = int(evaluation[point].size/2)
        returnEvaluation[target] =  self.__evaluateExpTerm(self.pivotValues,
                                                                evaluation[point][:l],
                                                                evaluation[point][l:])
    return returnEvaluation

  def writeXMLPreamble(self, writeTo, targets = None):
    """
      Specific local method for printing anything desired to xml file at the begin of the print.
      Overwrite in inheriting classes.
      @ In, writeTo, xmlUtils.StaticXmlElement instance, element to write to
      @ In, targets, list, list of targets for whom information should be written.
      @ Out, None
    """
    # add description
    description  = r" This XML file contains the main information of the PolyExponential ROM."
    description += r" If ``coefficients'' are dumped for each realization, the evaluation function (for each realization ``j'') is as follows:"
    description += r" $SM_{j}(z) = \sum_{i=1}^{N}f_{i}*exp^{-tau_{i}*z}$, with ``z'' beeing the monotonic variable and ``N'' the"
    description += r" number of exponential terms (expTerms). If the Polynomial coefficients ``poly\_coefficients'' are"
    description += r" dumped, the SM evaluation function is as follows:"
    description += r" $SM(X,z) = \sum_{i=1}^{N} P_{i}(X)*exp^{-Q_{i}(X)*z}$, with ``P'' and ``Q'' the polynomial expressions of the exponential terms."
    writeTo.addScalar('ROM', "description", description)

  def writeXML(self, writeTo, targets = None, skip = None):
    """
      Adds requested entries to XML node.
      @ In, writeTo, xmlTuils.StaticXmlElement, element to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """
    ##TODO retrieve coefficients from spline interpolator
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    if skip is None:
      skip = []

    # check what
    what = ['expTerms','coeffRegressor','features','timeScale','coefficients']
    if self.polyExpParams['coeffRegressor'].strip() == 'poly':
      what.append('polyOrder')
    if targets is None:
      readWhat = what
    else:
      readWhat = targets
    for s in skip:
      if s in readWhat:
        readWhat.remove(s)
    if not set(readWhat) <= set(what):
      self.raiseAnError(IOError, "The following variables in <what> node are not recognized: "
                        + ",".join(np.setdiff1d(readWhat, what).tolist()) )
    else:
      what = readWhat

    # Target
    target = self.target[-1]
    toAdd = ['expTerms', 'coeffRegressor']
    if self.polyExpParams['coeffRegressor'].strip() == 'poly':
      toAdd.append('polyOrder')
    for add in toAdd:
      if add in what:
        writeTo.addScalar(target,add,self.polyExpParams[add])
    targNode = writeTo._findTarget(writeTo.getRoot(), target)
    if "features" in what:
      writeTo.addScalar(target,"features",' '.join(self.features))
    if "timeScale" in what:
      writeTo.addScalar(target,"timeScale",' '.join([str(elm) for elm in self.pivotValues]))
    if "coefficients" in what:
      for smp in range(len(self.aij[target])):
        valDict = {'fi': ' '.join([ '%.6e' % elm for elm in self.aij[target][smp,:]]),
                   'taui':' '.join([ '%.6e' % elm for elm in self.bij[target][smp,:]]),
                   'predictionRelDiff' :' '.join([ '%.6e' % elm for elm in self.predictError[target][smp,:]])}
        attributeDict = {self.features[index]:'%.6e' % self.featureVals[smp,index] for index in range(len(self.features))}
        writeTo.addVector("coefficients", "realization", valDict, root=targNode, attrs=attributeDict)

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
    self.aij          = None
    self.bij          = None
    self.model        = None
    self.pivotValues  = None
    self.predictError = None
    self.featureVals  = None

  def __returnInitialParametersLocal__(self):
    """
      This method returns the initial parameters of the SM
      @ In, None
      @ Out, self.polyExpParams, dict, the dict of the SM settings
    """
    return self.polyExpParams

  def __returnCurrentSettingLocal__(self):
    """
      This method is used to pass the set of parameters of the ROM that can change during simulation
      @ In, None
      @ Out, self.polyExpParams, dict, the dict of the SM settings
    """
    return self.polyExpParams
