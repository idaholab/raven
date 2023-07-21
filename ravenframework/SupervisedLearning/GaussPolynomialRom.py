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
  Class implementation for the GaussPolynomialRom
"""
from numpy import average
import sys

#External Modules------------------------------------------------------------------------------------
import numpy as np
from collections import OrderedDict
from scipy import spatial
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SupervisedLearning import SupervisedLearning
from ..utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------



class GaussPolynomialRom(SupervisedLearning):
  """
    Gauss Polynomial Rom Class
  """

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
    ## cross validation node 'CV' can not be used for GaussPolynomialRom
    specs.popSub('CV')
    specs.description = r"""The \xmlString{GaussPolynomialRom} is based on a
                        characteristic Gaussian polynomial fitting scheme: generalized polynomial chaos
                        expansion (gPC).
                        \\
                        In gPC, sets of polynomials orthogonal with respect to the distribution of uncertainty
                        are used to represent the original model.  The method converges moments of the original
                        model faster than Monte Carlo for small-dimension uncertainty spaces ($N<15$).
                        In order to use this ROM, the \xmlNode{ROM} attribute \xmlAttr{subType} needs to
                        be \xmlString{GaussPolynomialRom}.
                        \\
                        The GaussPolynomialRom is dependent on specific sampling; thus, this ROM cannot be trained unless a
                        SparseGridCollocation or similar Sampler specifies this ROM in its input and is sampled in a MultiRun step.
                        \begin{table}[htb]
                          \centering
                          \begin{tabular}{c | c c}
                            Unc. Distribution & Default Quadrature & Default Polynomials \\ \hline
                            Uniform & Legendre & Legendre \\
                            Normal & Hermite & Hermite \\ \hline
                            Gamma & Laguerre & Laguerre \\
                            Beta & Jacobi & Jacobi \\ \hline
                            Other & Legendre* & Legendre*
                          \end{tabular}
                          \caption{GaussPolynomialRom defaults}
                          \label{tab:gpcCompatible}
                        \end{table}
                        \nb This ROM type must be trained from a collocation quadrature set.
                        Thus, it can only be trained from the outcomes of a SparseGridCollocation sampler.
                        Also, this ROM must be referenced in the SparseGridCollocation sampler in order to
                        accurately produce the necessary sparse grid points to train this ROM.
                        \zNormalizationNotPerformed{GaussPolynomialRom}
                        \\
                        When Printing this ROM via a Print OutStream (see \ref{sec:printing}), the available metrics are:
                        \begin{itemize}
                          \item \xmlString{mean}, the mean value of the ROM output within the input space it was trained,
                          \item \xmlString{variance}, the variance of the ROM output within the input space it was trained,
                          \item \xmlString{samples}, the number of distinct model runs required to construct the ROM,
                          \item \xmlString{indices}, the Sobol sensitivity indices (in percent), Sobol total indices, and partial variances,
                          \item \xmlString{polyCoeffs}, the polynomial expansion coefficients (PCE moments) of the ROM.  These are
                            listed by each polynomial combination, with the polynomial order tags listed in the order of the variables
                            shown in the XML print.
                        \end{itemize}
                        """
    specs.addSub(InputData.parameterInputFactory("IndexSet", contentType=InputTypes.makeEnumType("IndexSet", "IndexSetType", ["TensorProduct", "TotalDegree", "HyperbolicCross", "Custom"]),
                                                 descr=r"""specifies the rules by which to construct multidimensional polynomials.  The options are
                                                 \xmlString{TensorProduct}, \xmlString{TotalDegree},\\
                                                 \xmlString{HyperbolicCross}, and \xmlString{Custom}.
                                                 Total degree is efficient for
                                                 uncertain inputs with a large degree of regularity, while hyperbolic cross is more efficient
                                                 for low-regularity input spaces.
                                                 If \xmlString{Custom} is chosen, the \xmlNode{IndexPoints} is required."""))
    specs.addSub(InputData.parameterInputFactory("PolynomialOrder", contentType=InputTypes.IntegerType,
                                                 descr=r"""indicates the maximum polynomial order in any one dimension to use in the
                                                 polynomial chaos expansion. \nb If non-equal importance weights are supplied in the optional
                                                 \xmlNode{Interpolation} node, the actual polynomial order in dimensions with high
                                                 importance might exceed this value; however, this value is still used to limit the
                                                 relative overall order."""))
    specs.addSub(InputData.parameterInputFactory("SparseGrid", contentType=InputTypes.makeEnumType("SparseGrid", "SparseGrid", ["smolyak", "tensor"]),
                                                 descr=r"""allows specification of the multidimensional
                                                   quadrature construction strategy.  Options are \xmlString{smolyak} and \xmlString{tensor}.""",
                                                   default="smolyak"))
    specs.addSub(InputData.parameterInputFactory("IndexPoints", contentType=InputTypes.IntegerTupleListType,
                                                 descr=r"""used to specify the index set points in a \xmlString{Custom} index set.  The tuples are
                                                 entered as comma-separated values between parenthesis, with each tuple separated by a comma.
                                                 Any amount of whitespace is acceptable.  For example, \xmlNode{IndexPoints}\verb'(0,1),(0,2),(1,1),(4,0)'\xmlNode{/IndexPoints}
                                                 \nb{Using custom index sets
                                                 does not guarantee accurate convergence.}""", default=None))
    interpolationParam = InputData.parameterInputFactory("Interpolation", contentType=InputTypes.StringType,
                                                 descr=r"""offers the option to specify quadrature, polynomials, and importance weights for the given
                                                 variable name.  The ROM accepts any number of \xmlNode{Interpolation} nodes up to the
                                                 dimensionality of the input space.""", default=None)
    interpolationParam.addParam("quad", InputTypes.StringType, required=False,
                  descr=r"""specifies the quadrature type to use for collocation in this dimension.  The default options
                  depend on the uncertainty distribution of the input dimension, as shown in Table
                  \ref{tab:gpcCompatible}. Additionally, Clenshaw Curtis quadrature can be used for any
                  distribution that doesn't include an infinite bound.
                  \default{see Table \ref{tab:gpcCompatible}.}
                  \nb For an uncertain distribution aside from the four listed on Table
                  \ref{tab:gpcCompatible}, this ROM
                  makes use of the uniform-like range of the distribution's CDF to apply quadrature that is
                  suited uniform uncertainty (Legendre).  It converges more slowly than the four listed, but are
                  viable choices.  Choosing polynomial type Legendre for any non-uniform distribution will
                  enable this formulation automatically.""")
    interpolationParam.addParam("poly", InputTypes.StringType, required=False,
                  descr=r"""specifies the interpolating polynomial family to use for the polynomial expansion in this
                  dimension.  The default options depend on the quadrature type chosen, as shown in Table
                  \ref{tab:gpcCompatible}.  Currently, no polynomials are available outside the
                  default. \default{see Table \ref{tab:gpcCompatible}.}""")
    interpolationParam.addParam("weight", InputTypes.FloatType, required=False,
                  descr=r"""delineates the importance weighting of this dimension.  A larger importance weight will
                  result in increased resolution for this dimension at the cost of resolution in lower-weighted
                  dimensions.  The algorithm normalizes weights at run-time.""", default=1)
    specs.addSub(interpolationParam)
    return specs

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    pass

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    pass

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    pass

  def __initLocal__(self):
    """
      Method used to add additional initialization features used by pickling
      @ In, None
      @ Out, None
    """
    pass

  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    super().__init__()
    self.initialized   = False #only True once self.initialize has been called
    self.interpolator  = None #FIXME what's this?
    self.printTag      = 'GAUSSgpcROM'
    self.indexSetType  = None #string of index set type, TensorProduct or TotalDegree or HyperbolicCross
    self.indexSetVals  = []   #list of tuples, custom index set to use if CustomSet is the index set type
    self.maxPolyOrder  = None #integer of relative maximum polynomial order to use in any one dimension
    self.itpDict       = {}   #dict of quad,poly,weight choices keyed on varName
    self.norm          = None #combined distribution normalization factors (product)
    self.sparseGrid    = None #Quadratures.SparseGrid object, has points and weights
    self.distDict      = None #dict{varName: Distribution object}, has point conversion methods based on quadrature
    self.quads         = None #dict{varName: Quadrature object}, has keys for distribution's point conversion methods
    self.polys         = None #dict{varName: OrthoPolynomial object}, has polynomials for evaluation
    self.indexSet      = None #array of tuples, polynomial order combinations
    self.polyCoeffDict = None #dict{index set point, float}, polynomial combination coefficients for each combination
    self.numRuns       = None #number of runs to generate ROM; default is len(self.sparseGrid)
    self.itpDict       = {}   #dict{varName: dict{attribName:value} }
    self.featv         = None  # list of feature variables
    self.targv         = None  # list of target variables
    self.mean          = None
    self.variance      = None
    self.sdx           = None
    self.partialVariances = None
    self.sparseGridType    = 'smolyak' #type of sparse quadrature to use,default smolyak

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['IndexSet','IndexPoints', 'PolynomialOrder',
                                                               'SparseGrid'])
    # notFound must be empty
    assert(not notFound)
    self.indexSetType = settings.get('IndexSet')
    self.indexSetVals = settings.get('IndexPoints')
    self.raiseADebug('points',self.indexSetVals)
    self.maxPolyOrder = settings.get('PolynomialOrder')
    self.sparseGridType = settings.get('SparseGrid')
    interpolationList = paramInput.findAll('Interpolation')
    for elem in interpolationList:
      var = elem.value
      poly = elem.parameterValues.get('poly', 'DEFAULT')
      quad = elem.parameterValues.get('quad', 'DEFAULT')
      weight = elem.parameterValues.get('weight', 1.0)
      self.itpDict[var]={'poly'  : poly,
                         'quad'  : quad,
                         'weight': weight}

    if self.indexSetType=='Custom':
      if len(self.indexSetVals)<1:
        self.raiseAnError(IOError,'If using CustomSet, must specify points in <IndexPoints> node!')
      else:
        for i in self.indexSetVals:
          if len(i)<len(self.features):
            self.raiseAnError(IOError,'CustomSet points',i,'is too small!')
    if self.maxPolyOrder < 1:
      self.raiseAnError(IOError,'Polynomial order cannot be less than 1 currently.')

  def initializeFromDict(self, kwargs):
    """
      Function which initializes the ROM given a the information contained in inputDict
      @ In, kwargs, dict, dictionary containing the values required to initialize the ROM
      @ Out, None
    """
    super().initializeFromDict(kwargs)
    for key,val in kwargs.items():
      if key=='IndexSet':
        self.indexSetType = val
      elif key=='IndexPoints':
        self.indexSetVals=[]
        strIndexPoints = val.strip()
        strIndexPoints = strIndexPoints.replace(' ','').replace('\n','').strip('()')
        strIndexPoints = strIndexPoints.split('),(')
        self.raiseADebug(strIndexPoints)
        for s in strIndexPoints:
          self.indexSetVals.append(tuple(int(i) for i in s.split(',')))
        self.raiseADebug('points',self.indexSetVals)
      elif key=='PolynomialOrder':
        self.maxPolyOrder = int(val)
      elif key=='Interpolation':
        for var,value in val.items():
          self.itpDict[var]={'poly'  :'DEFAULT',
                             'quad'  :'DEFAULT',
                             'weight':'1'}
          for atrName,atrVal in value.items():
            if atrName in ['poly','quad','weight']:
              self.itpDict[var][atrName]=atrVal
            else:
              self.raiseAnError(IOError,'Unrecognized option: '+atrName)
      elif key == 'SparseGrid':
        if val.lower() not in ['smolyak','tensor']:
          self.raiseAnError(IOError,'No such sparse quadrature implemented: %s.  Options are %s.' %(val,str(['smolyak','tensor'])))
        self.sparseGridType = val

    if not self.indexSetType:
      self.raiseAnError(IOError,'No IndexSet specified!')
    if self.indexSetType=='Custom':
      if len(self.indexSetVals)<1:
        self.raiseAnError(IOError,'If using CustomSet, must specify points in <IndexPoints> node!')
      else:
        for i in self.indexSetVals:
          if len(i)<len(self.features):
            self.raiseAnError(IOError,'CustomSet points',i,'is too small!')
    if not self.maxPolyOrder:
      self.raiseAnError(IOError,'No maxPolyOrder specified!')
    if self.maxPolyOrder < 1:
      self.raiseAnError(IOError,'Polynomial order cannot be less than 1 currently.')


  def writeXML(self, writeTo, requests = None, skip = None):
    """
      Adds requested entries to XML node.
      @ In, writeTo, xmlUtils.StaticXmlElement, StaticXmlElement to write to
      @ In, requests, list, optional, list of requests for whom to write
      @ In, skip, list, optional, list of targets to skip (often a pivot parameter)
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    if skip is None:
      skip = []
    #establish what we can handle, and how
    scalars = ['mean','expectedValue','variance','samples']
    vectors = ['polyCoeffs','partialVariance','sobolIndices','sobolTotalIndices']
    canDo = scalars + vectors
    #lowercase for convenience
    scalars = list(s.lower() for s in scalars)
    vectors = list(v.lower() for v in vectors)
    for target in self.target:
      if target in skip:
        continue
      if requests is None:
        requests = canDo
      # loop over the requested items
      for request in requests:
        request=request.strip()
        if request.lower() in scalars:
          if request.lower() in ['mean','expectedvalue']:
            val = self.__mean__(target)
          elif request.lower() == 'variance':
            val = self.__variance__(target)
          elif request.lower() == 'samples':
            if self.numRuns != None:
              val = self.numRuns
            else:
              val = len(self.sparseGrid)
          writeTo.addScalar(target,request,val)
        elif request.lower() in vectors:
          if request.lower() == 'polycoeffs':
            valueDict = OrderedDict()
            valueDict['inputVariables'] = ','.join(self.features)
            keys = list(self.polyCoeffDict[target].keys())
            keys.sort()
            for key in keys:
              valueDict['_'+'_'.join(str(k) for k in key)+'_'] = self.polyCoeffDict[target][key]
          elif request.lower() in ['partialvariance', 'sobolindices', 'soboltotalindices']:
            sobolIndices, partialVars = self.getSensitivities(target)
            sobolTotals = self.getTotalSensitivities(sobolIndices)
            #sort by value
            entries = []
            if request.lower() in ['partialvariance','sobolindices']:
              #these both will have same sort
              for key in sobolIndices.keys():
                entries.append( ('.'.join(key),partialVars[key],key) )
            elif request.lower() in ['soboltotalindices']:
              for key in sobolTotals.keys():
                entries.append( ('.'.join(key),sobolTotals[key],key) )
            entries.sort(key=lambda x: abs(x[1]),reverse=True)
            #add entries to results list
            valueDict=OrderedDict()
            for entry in entries:
              name,_,key = entry
              if request.lower() == 'partialvariance':
                valueDict[name] = partialVars[key]
              elif request.lower() == 'sobolindices':
                valueDict[name] = sobolIndices[key]
              elif request.lower() == 'soboltotalindices':
                valueDict[name] = sobolTotals[key]
          writeTo.addVector(target,request,valueDict)
        else:
          self.raiseAWarning('ROM does not know how to return "'+request+'".  Skipping...')

  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure.
      @ In, values, list(float), unused
      @ In, names, list(string), unused
      @ In, feat, string, feature to (not) normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def interpolationInfo(self):
    """
      Returns the interpolation information
      @ In, None
      @ Out, interpValues, dict, dictionary of interpolation information
    """
    interpValues = dict(self.itpDict)
    return interpValues

  def initialize(self,idict):
    """
      Initializes the instance.
      @ In, idict, dict, objects needed to initalize
      @ Out, None
    """
    self.sparseGrid     = idict.get('SG'        ,None)
    self.distDict       = idict.get('dists'     ,None)
    self.quads          = idict.get('quads'     ,None)
    self.polys          = idict.get('polys'     ,None)
    self.indexSet       = idict.get('iSet'      ,None)
    self.numRuns        = idict.get('numRuns'   ,None)
    #make sure requireds are not None
    if self.sparseGrid is None:
      self.raiseAnError(RuntimeError,'Tried to initialize without key object "SG"   ')
    if self.distDict   is None:
      self.raiseAnError(RuntimeError,'Tried to initialize without key object "dists"')
    if self.quads      is None:
      self.raiseAnError(RuntimeError,'Tried to initialize without key object "quads"')
    if self.polys      is None:
      self.raiseAnError(RuntimeError,'Tried to initialize without key object "polys"')
    if self.indexSet   is None:
      self.raiseAnError(RuntimeError,'Tried to initialize without key object "iSet" ')
    self.initialized = True

  def _multiDPolyBasisEval(self,orders,pts):
    """
      Evaluates each polynomial set at given orders and points, returns product.
      @ In orders, tuple(int), polynomial orders to evaluate
      @ In pts, tuple(float), values at which to evaluate polynomials
      @ Out, tot, float, product of polynomial evaluations
    """
    tot=1
    for i,(o,p) in enumerate(zip(orders,pts)):
      varName = self.sparseGrid.varNames[i]
      tot*=self.polys[varName](o,p)
    return tot

  def _train(self,featureVals,targetVals):
    """
      Trains ROM.
      @ In, featureVals, np.ndarray, feature values
      @ In, targetVals, np.ndarray, target values
    """
    #check to make sure ROM was initialized
    if not self.initialized:
      self.raiseAnError(RuntimeError,'ROM has not yet been initialized!  Has the Sampler associated with this ROM been used?')
    self.raiseADebug('training',self.features,'->',self.target)
    self.featv, self.targv = featureVals,targetVals
    self.polyCoeffDict = {key: dict({}) for key in self.target}
    #check equality of point space
    self.raiseADebug('...checking required points are available...')
    fvs = []
    tvs = {key: list({}) for key in self.target}
    sgs = list(self.sparseGrid.points())
    missing=[]
    kdTree = spatial.KDTree(featureVals)
    #TODO this is slowest loop in this algorithm, by quite a bit.
    for pt in sgs:
      #KDtree way
      distances,idx = kdTree.query(pt,k=1,distance_upper_bound=1e-9) #FIXME how to set the tolerance generically?
      #KDTree repots a "not found" as at infinite distance with index len(data)
      if idx >= len(featureVals):
        found = False
      else:
        found = True
        point = tuple(featureVals[idx])
      #end KDTree way
      if found:
        fvs.append(point)
        for cnt, target in enumerate(self.target):
          tvs[target].append(targetVals[idx,cnt])
      else:
        missing.append(pt)
    if len(missing)>0:
      msg='\n'
      msg+='DEBUG missing feature vals:\n' + '\n'.join(map(lambda x:'  '+str(x),missing))+ '\n'
      self.raiseADebug(msg)
      self.raiseADebug('sparse:',sgs)
      self.raiseADebug('solns :',fvs)
      self.raiseAnError(IOError,'input values do not match required values!')
    #make translation matrix between lists, also actual-to-standardized point map
    self.raiseADebug('...constructing translation matrices...')
    translate={}
    for i in range(len(fvs)):
      translate[tuple(fvs[i])]=sgs[i]
    standardPoints = {}
    for pt in fvs:
      stdPt = []
      for i,p in enumerate(pt):
        varName = self.sparseGrid.varNames[i]
        stdPt.append( self.distDict[varName].convertToQuad(self.quads[varName].type,p) )
      standardPoints[tuple(pt)] = stdPt[:]
    #make polynomials
    self.raiseADebug('...constructing polynomials...')
    self.norm = np.prod(list(self.distDict[v].measureNorm(self.quads[v].type) for v in self.distDict.keys()))
    for i,idx in enumerate(self.indexSet):
      idx=tuple(idx)
      for target in self.target:
        self.polyCoeffDict[target][idx]=0
        wtsum=0
        for pt,soln in zip(fvs,tvs[target]):
          tupPt = tuple(pt)
          stdPt = standardPoints[tupPt]
          wt = self.sparseGrid.weights(translate[tupPt])
          self.polyCoeffDict[target][idx]+=soln*self._multiDPolyBasisEval(idx,stdPt)*wt
        self.polyCoeffDict[target][idx]*=self.norm
    self.amITrained=True
    self.raiseADebug('...training complete!')

  def printPolyDict(self,printZeros=False):
    """
      Human-readable version of the polynomial chaos expansion.
      @ In, printZeros, bool, optional, optional flag for printing even zero coefficients
      @ Out, None
    """
    for target in self.target:
      data=[]
      for idx,val in self.polyCoeffDict[target].items():
        if abs(val) > 1e-12 or printZeros:
          data.append([idx,val])
      data.sort()
      self.raiseADebug('polyDict for ['+target+'] with inputs '+str(self.features)+':')
      for idx,val in data:
        self.raiseADebug('    '+str(idx)+' '+str(val))

  def checkForNonzeros(self,tol=1e-12):
    """
      Checks poly coefficient dictionary for nonzero entries.
      @ In, tol, float, optional, the tolerance under which is zero (default 1e-12)
      @ Out, data, dict, {'target1':list(tuple),'target2':list(tuple)}: the indices and values of the nonzero coefficients for each target
    """
    data = dict.fromkeys(self.target,[])
    for target in self.target:
      for idx,val in self.polyCoeffDict[target].items():
        if round(val,11) !=0:
          data[target].append([idx,val])
    return data

  def __mean__(self, targ=None):
    """
      Returns the mean of the ROM.
      @ In, None
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, __mean__, float, the mean
    """
    return self.__evaluateMoment__(1,targ)

  def __variance__(self, targ=None):
    """
      returns the variance of the ROM.
      @ In, None
      @ In, targ, str, optional, the target for which the __variance__ needs to be computed
      @ Out, __variance__, float, variance
    """
    mean = self.__evaluateMoment__(1,targ)
    return self.__evaluateMoment__(2,targ) - mean*mean

  def __evaluateMoment__(self,r, targ=None):
    """
      Use the ROM's built-in method to calculate moments.
      @ In, r, int, moment to calculate
      @ In, targ, str, optional, the target for which the moment needs to be computed
      @ Out, tot, float, evaluation of moment
    """
    target = self.target[0] if targ is None else targ
    #TODO is there a faster way still to do this?
    if r==1:
      return self.polyCoeffDict[target][tuple([0]*len(self.features))]
    elif r==2:
      return sum(s**2 for s in self.polyCoeffDict[target].values())
    tot=0
    for pt,wt in self.sparseGrid:
      tot+=self.__evaluateLocal__([pt])[target]**r*wt
    tot*=self.norm
    return tot

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, list, of values at which to evaluate the ROM
      @ Out, returnDict, dict, the evaluated point for each target
    """
    featureVals=featureVals[0]
    returnDict={}
    stdPt = np.zeros(len(featureVals))
    for p,pt in enumerate(featureVals):
      varName = self.sparseGrid.varNames[p]
      stdPt[p] = self.distDict[varName].convertToQuad(self.quads[varName].type,pt)
    for target in self.target:
      tot=0
      for idx,coeff in self.polyCoeffDict[target].items():
        tot+=coeff*self._multiDPolyBasisEval(idx,stdPt)
      returnDict[target] = tot
    return returnDict

  def _printPolynomial(self):
    """
      Prints each polynomial for each coefficient.
      @ In, None
      @ Out, None
    """
    for target in self.target:
      self.raiseADebug('Target:'+target+'.Coeff Idx:')
      for idx,coeff in self.polyCoeffDict[target].items():
        if abs(coeff)<1e-12:
          continue
        self.raiseADebug(str(idx))
        for i,ix in enumerate(idx):
          var = self.features[i]
          self.raiseADebug(self.polys[var][ix]*coeff,'|',var)

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = {}
    return params

  def getSensitivities(self,targ=None):
    """
      Calculates the Sobol indices (percent partial variances) of the terms in this expansion.
      @ In, targ, str, optional, the target for which the moment needs to be computed
      @ Out, getSensitivities, tuple(dict), Sobol indices and partial variances keyed by subset
    """
    target = self.target[0] if targ is None else targ
    totVar = self.__variance__(target)
    partials = {}
    #calculate partial variances
    self.raiseADebug('Calculating partial variances...')
    for poly,coeff in self.polyCoeffDict[target].items():
      #use poly to determine subset
      subset = self._polyToSubset(poly)
      # skip mean
      if len(subset) < 1:
        continue
      subset = tuple(subset)
      if subset not in partials.keys():
        partials[subset] = 0
      partials[subset] += coeff*coeff
    #calculate Sobol indices
    indices = {}
    for subset,partial in partials.items():
      indices[subset] = partial / totVar
    return (indices,partials)

  def getTotalSensitivities(self,indices):
    """
      Given the Sobol global sensitivity indices, calculates the total indices for each subset.
      @ In, indices, dict, tuple(subset):float(index)
      @ Out, totals, dict, tuple(subset):float(index)
    """
    #total index is the sum of all Sobol indices in which a subset belongs
    totals={}
    for subset in indices.keys():
      setSub = set(subset)
      totals[subset] = 0
      for checkSubset in indices.keys():
        setCheck = set(checkSubset)
        if setSub.issubset(setCheck):
          totals[subset] += indices[checkSubset]
    return totals

  def _polyToSubset(self,poly):
    """
      Given a tuple with polynomial orders, returns the subset it belongs exclusively to
      @ In, poly, tuple(int), polynomial index set entry
      @ Out, subset, tuple(str), subset
    """
    boolRep = tuple(False if poly[i]==0 else True for i in range(len(poly)))
    subset = []
    for i,p in enumerate(boolRep):
      if p:
        subset.append(self.features[i])
    return tuple(subset)
