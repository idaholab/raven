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

  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for HDMRRom
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils
from .GaussPolynomialRom import GaussPolynomialRom
#Internal Modules End--------------------------------------------------------------------------------

class HDMRRom(GaussPolynomialRom):
  """
    High-Dimention Model Reduction reduced order model.  Constructs model based on subsets of the input space.
  """
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

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    GaussPolynomialRom.__init__(self,messageHandler,**kwargs)
    self.initialized   = False #true only when self.initialize has been called
    self.printTag      = 'HDMR_ROM('+'-'.join(self.target)+')'
    self.sobolOrder    = None #depth of HDMR/Sobol expansion
    self.ROMs          = {}   #dict of GaussPolyROM objects keyed by combination of vars that make them up
    self.sdx           = None #dict of sobol sensitivity coeffs, keyed on order and tuple(varnames)
    self.mean          = None #mean, store to avoid recalculation
    self.variance      = None #variance, store to avoid recalculation
    self.anova         = None #converted true ANOVA terms, stores coefficients not polynomials
    self.partialVariances = None #partial variance contributions

    for key,val in kwargs.items():
      if key=='SobolOrder':
        self.sobolOrder = int(val)

  def _localPrintXML(self,outFile,pivotVal,options={}):
    """
      Adds requested entries to XML node.
      @ In, outFile, Files.File, either StaticXMLOutput or DynamicXMLOutput file
      @ In, pivotVal, float, value of pivot parameters to use in printing if dynamic
      @ In, options, dict, optional, dict of string-based options to use, including filename, things to print, etc
        May include:
        'what': comma-separated string list, the qualities to print out
        'pivotVal': float value of dynamic pivotParam value
      @ Out, None
    """
    #inherit from GaussPolynomialRom
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    self.mean=None
    canDo = ['mean','expectedValue','variance','samples','partialVariance','sobolIndices','sobolTotalIndices']
    if 'what' in options.keys():
      requests = list(o.strip() for o in options['what'].split(','))
      if 'all' in requests:
        requests = canDo
      #protect against things SCgPC can do that HDMR can't
      if 'polyCoeffs' in requests:
        self.raiseAWarning('HDMRRom cannot currently print polynomial coefficients.  Skipping...')
        requests.remove('polyCoeffs')
      options['what'] = ','.join(requests)
    else:
      self.raiseAWarning('No "what" options for XML printing are recognized!  Skipping...')
    GaussPolynomialRom._localPrintXML(self,outFile,pivotVal,options)

  def initialize(self,idict):
    """
      Initializes the instance.
      @ In, idict, dict, objects needed to initalize
      @ Out, None
    """
    for key,value in idict.items():
      if   key == 'ROMs':
        self.ROMs       = value
      elif key == 'dists':
        self.distDict   = value
      elif key == 'quads':
        self.quads      = value
      elif key == 'polys':
        self.polys      = value
      elif key == 'refs':
        self.references = value
      elif key == 'numRuns':
        self.numRuns    = value
    self.initialized = True

  def __trainLocal__(self,featureVals,targetVals):
    """
      Because HDMR rom is a collection of sub-roms, we call sub-rom "train" to do what we need it do.
      @ In, featureVals, np.array, training feature values
      @ In, targetVals, np.array, training target values
      @ Out, None
    """
    if not self.initialized:
      self.raiseAnError(RuntimeError,'ROM has not yet been initialized!  Has the Sampler associated with this ROM been used?')
    ft={}
    self.refSoln = {key:dict({}) for key in self.target}
    for i in range(len(featureVals)):
      ft[tuple(featureVals[i])]=targetVals[i,:]

    #get the reference case
    self.refpt = tuple(self.__fillPointWithRef((),[]))
    for cnt, target in enumerate(self.target):
      self.refSoln[target] = ft[self.refpt][cnt]
    for combo,rom in self.ROMs.items():
      subtdict = {key:list([]) for key in self.target}
      for c in combo:
        subtdict[c]=[]
      SG = rom.sparseGrid
      fvals=np.zeros([len(SG),len(combo)])
      tvals=np.zeros((len(SG),len(self.target)))
      for i in range(len(SG)):
        getpt=tuple(self.__fillPointWithRef(combo,SG[i][0]))
        #the 1e-10 is to be consistent with RAVEN's CSV print precision
        tvals[i,:] = ft[tuple(mathUtils.NDInArray(np.array(ft.keys()),getpt,tol=1e-10)[2])]
        for fp,fpt in enumerate(SG[i][0]):
          fvals[i][fp] = fpt
      for i,c in enumerate(combo):
        subtdict[c] = fvals[:,i]
      for cnt, target in enumerate(self.target):
        subtdict[target] = tvals[:,cnt]
      rom.train(subtdict)

    #make ordered list of combos for use later
    maxLevel = max(list(len(combo) for combo in self.ROMs.keys()))
    self.combos = []
    for i in range(maxLevel+1):
      self.combos.append([])
    for combo in self.ROMs.keys():
      self.combos[len(combo)].append(combo)

    #list of term objects
    self.terms = {():[]}  # each entry will look like 'x1,x2':('x1','x2'), missing the reference entry
    for l in range(1,maxLevel+1):
      for romName in self.combos[l]:
        self.terms[romName] = []
        # add subroms -> does this get referenece case, too?
        for key in self.terms.keys():
          if set(key).issubset(set(romName)) and key!=romName:
            self.terms[romName].append(key)
    #reduce terms
    self.reducedTerms = {}
    for term in self.terms.keys():
      self._collectTerms(term,self.reducedTerms)
    #remove zero entries
    self._removeZeroTerms(self.reducedTerms)

    self.amITrained = True

  def __fillPointWithRef(self,combo,pt):
    """
      Given a "combo" subset of the full input space and a partially-filled
      point within that space, fills the rest of space with the reference
      cut values.
      @ In, combo, tuple(str), names of subset dimensions
      @ In, pt, list(float), values of points in subset dimension
      @ Out, newpt, tuple(float), full point in input dimension space on cut-hypervolume
    """
    newpt=np.zeros(len(self.features))
    for v,var in enumerate(self.features):
      if var in combo:
        newpt[v] = pt[combo.index(var)]
      else:
        newpt[v] = self.references[var]
    return newpt

  def __fillIndexWithRef(self,combo,pt):
    """
       Given a "combo" subset of the full input space and a partially-filled
       polynomial order index within that space, fills the rest of index with zeros.
       @ In, combo, tuple of strings, names of subset dimensions
       @ In, pt, list of floats, values of points in subset dimension
       @ Out, newpt, tuple(int), full index in input dimension space on cut-hypervolume
    """
    newpt=np.zeros(len(self.features),dtype=int)
    for v,var in enumerate(self.features):
      if var in combo:
        newpt[v] = pt[combo.index(var)]
    return tuple(newpt)

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, list(float), list of values at which to evaluate the ROM
      @ Out, tot, float, the evaluated point
    """
    #am I trained?
    returnDict = dict.fromkeys(self.target,None)
    if not self.amITrained:
      self.raiseAnError(IOError,'Cannot evaluate, as ROM is not trained!')
    for target in self.target:
      tot = 0
      for term,mult in self.reducedTerms.items():
        if term == ():
          tot += self.refSoln[target]*mult
        else:
          cutVals = [list(featureVals[0][self.features.index(j)] for j in term)]
          tot += self.ROMs[term].__evaluateLocal__(cutVals)[target]*mult
      returnDict[target] = tot
    return returnDict

  def __mean__(self,targ=None):
    """
      The Cut-HDMR approximation can return its mean easily.
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, __mean__, float, the mean
    """
    if not self.amITrained:
      self.raiseAnError(IOError,'Cannot evaluate mean, as ROM is not trained!')
    return self._calcMean(self.reducedTerms,targ)

  def __variance__(self,targ=None):
    """
      The Cut-HDMR approximation can return its variance somewhat easily.
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, __variance__, float, the variance
    """
    if not self.amITrained:
      self.raiseAnError(IOError,'Cannot evaluate variance, as ROM is not trained!')
    target = self.target[0] if targ is None else targ
    self.getSensitivities(target)
    return sum(val for val in self.partialVariances[target].values())

  def _calcMean(self,fromDict,targ=None):
    """
      Given a subset, calculate mean from terms
      @ In, fromDict, dict{string:int}, ROM subsets and their multiplicity
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, tot, float, mean
    """
    tot = 0
    for term,mult in fromDict.items():
      tot += self._evaluateIntegral(term,targ)*mult
    return tot

  def _collectTerms(self,a,targetDict,sign=1,depth=0):
    """
      Adds main term multiplicity and subtracts sub term multiplicity for cross between terms
      @ In, targetDict, dict, dictionary to pace terms in
      @ In, a, string, main combo key from self.terms
      @ In, sign, int, optional, gives the signs of the terms (1 for positive, -1 for negative)
      @ In, depth, int, optional, recursion depth
      @ Out, None
    """
    if a not in targetDict.keys():
      targetDict[a] = sign
    else:
      targetDict[a] += sign
    for sub in self.terms[a]:
      self._collectTerms(sub,targetDict,sign*-1,depth+1)

  def _evaluateIntegral(self,term, targ=None):
    """
      Uses properties of orthonormal gPC to algebraically evaluate integrals gPC
      This does assume the integral is over all the constituent variables in the the term
      @ In, term, string, subset term to integrate
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, _evaluateIntegral, float, evaluation

    """
    if term in [(),'',None]:
      return self.refSoln[targ if targ is not None else self.target[0]]
    else:
      return self.ROMs[term].__evaluateMoment__(1,targ)

  def _removeZeroTerms(self,d):
    """
      Removes keys from d that have zero value
      @ In, d, dict, string:int
      @ Out, None
    """
    toRemove=[]
    for key,val in d.items():
      if abs(val) < 1e-15:
        toRemove.append(key)
    for rem in toRemove:
      del d[rem]

  def getSensitivities(self,targ=None):
    """
      Calculates the Sobol indices (percent partial variances) of the terms in this expansion.
      @ In, targ, str, optional, the target for which the moment needs to be computed
      @ Out, getSensitivities, tuple(dict), Sobol indices and partial variances keyed by subset
    """
    target = self.target[0] if targ is None else targ
    if self.sdx is not None and self.partialVariances is not None and target in self.sdx.keys():
      self.raiseADebug('Using previously-constructed ANOVA terms...')
      return self.sdx[target],self.partialVariances[target]
    self.raiseADebug('Constructing ANOVA terms...')
    #collect terms
    terms = {}
    allFalse = tuple(False for _ in self.features)
    for subset,mult in self.reducedTerms.items():
      #skip mean, since it will be subtracted off in the end
      if subset == ():
        continue
      for poly,coeff in self.ROMs[subset].polyCoeffDict[target].items():
        #skip mean terms
        if sum(poly) == 0:
          continue
        poly = self.__fillIndexWithRef(subset,poly)
        polySubset = self._polyToSubset(poly)
        if polySubset not in terms.keys():
          terms[polySubset] = {}
        if poly not in terms[polySubset].keys():
          terms[polySubset][poly] = 0
        terms[polySubset][poly] += coeff*mult
    #calculate partial variances
    self.partialVariances = {target: dict({})}
    self.sdx              = {target: dict({})}
    for subset in terms.keys():
      self.partialVariances[target][subset] = sum(v*v for v in terms[subset].values())
    #calculate indices
    totVar = sum(self.partialVariances[target].values())
    for subset,value in self.partialVariances[target].items():
      self.sdx[target][subset] = value / totVar
    return self.sdx[target],self.partialVariances[target]

