
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
  Base subclass definition for DynamicModeDecomposition ROM (transferred from alfoa in SupervisedLearning)
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import numpy as np
from scipy import spatial
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils
from SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------


class DynamicModeDecomposition(supervisedLearning):
  """
    This surrogate is aimed to construct a "time-dep" surrogate based on
    Dynamic Mode Decomposition method.
    Ref. Kutz, Brunton, Brunton, Proctor. Dynamic Mode Decomposition:
        Data-Driven Modeling of Complex Systems. SIAM Other Titles in
        Applied Mathematics, 2016
  """
  def __init__(self, **kwargs):
    """
      DMD constructor
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
    """
    supervisedLearning.__init__(self, **kwargs)
    self.availDmdAlgorithms          = ['dmd','hodmd']                      # available dmd types: basic dmd and high order dmd
    self.dmdParams                   = {}                                   # dmd settings container
    self.printTag                    = 'DMD'                                # print tag
    self.pivotParameterID            = kwargs.get("pivotParameter","time")  # pivot parameter
    self._dynamicHandling            = True                                 # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self.dmdParams['rankSVD'       ] = kwargs.get('rankSVD',None)           # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    self.dmdParams['energyRankSVD' ] = kwargs.get('energyRankSVD',None)     #  0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by "energyRankSVD"
    self.dmdParams['rankTLSQ'      ] = kwargs.get('rankTLSQ',None)          # truncation rank for total least square
    self.dmdParams['exactModes'    ] = kwargs.get('exactModes',True)        # True if the exact modes need to be computed (eigs and eigvs), otherwise the projected ones (using the left-singular matrix)
    self.dmdParams['optimized'     ] = kwargs.get('optimized',False)        # amplitudes computed minimizing the error between the mods and all the timesteps (True) or 1st timestep only (False)
    self.dmdParams['dmdType'       ] = kwargs.get('dmdType','dmd')          # the dmd type to be applied. Currently we support dmd and hdmd (high order dmd)
    # variables filled up in the training stages
    self._amplitudes                 = {}                                   # {'target1': vector of amplitudes,'target2':vector of amplitudes, etc.}
    self._eigs                       = {}                                   # {'target1': vector of eigenvalues,'target2':vector of eigenvalues, etc.}
    self._modes                      = {}                                   # {'target1': matrix of dynamic modes,'target2':matrix of dynamic modes, etc.}
    self.__Atilde                    = {}                                   # {'target1': matrix of lowrank operator from the SVD,'target2':matrix of lowrank operator from the SVD, etc.}
    self.pivotValues                 = None                                 # pivot values (e.g. time)
    self.KDTreeFinder                = None                                 # kdtree weighting model
    self.timeScales                  = {}                                   # time-scales (training and dmd). {'training' and 'dmd':{t0:float,'dt':float,'intervals':int}}

    # some checks
    if self.dmdParams['rankSVD'] is not None and self.dmdParams['energyRankSVD'] is not None:
      self.raiseAWarning('Both "rankSVD" and "energyRankSVD" have been inputted. "energyRankSVD" is predominant and will be used!')
    if self.dmdParams['dmdType'] not in self.availDmdAlgorithms:
      self.raiseAnError(IOError,'dmdType(s) available are "'+', '.join(self.availDmdAlgorithms)+'"!')
    # check if the pivotParameter is among the targetValues
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")

  def __setstate__(self,state):
    """
      Initializes the DMD with the data contained in state
      @ In, state, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(state)
    self.KDTreeFinder = spatial.KDTree(self.featureVals)

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
  def __getTimeScale(self,dmd=True):
    """
      Get the ts of the dmd (if dmd = True) or training (if dmd = False) reconstructed time scale.
      @ In, dmd, bool, optional, True if dmd time scale needs to be returned, othewise training one
      @ Out, timeScale, numpy.array, the dmd or training reconstructed time scale
    """
    timeScaleInfo = self.timeScales['dmd'] if dmd else self.timeScales['training']
    timeScale = np.arange(timeScaleInfo['t0'], timeScaleInfo['intervals'] + timeScaleInfo['dt'], timeScaleInfo['dt'])
    return timeScale

  def __getTimeEvolution(self, target):
    """
      Get the time evolution of each mode
      @ In, target, str, the target for which mode evolution needs to be retrieved for
      @ Out, timeEvol, numpy.ndarray, the matrix that contains all the time evolution (by row)
    """
    omega = np.log(self._eigs[target]) / self.timeScales['training']['dt']
    van = np.exp(np.multiply(*np.meshgrid(omega, self.__getTimeScale())))
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

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, numpy.ndarray, shape = [n_timeStep, n_dimensions], an array of time series data
    """
    self.featureVals  = featureVals
    self.KDTreeFinder = spatial.KDTree(featureVals)
    pivotParamIndex   = self.target.index(self.pivotParameterID)
    self.pivotValues  = targetVals[0,:,pivotParamIndex]
    ts                = len(self.pivotValues)
    for target in list(set(self.target) - set([self.pivotParameterID])):
      targetParamIndex  = self.target.index(target)
      snaps = targetVals[:,:,targetParamIndex]
      # if number of features (i.e. samples) > number of snapshots, we apply the high order DMD or HODMD has been requested
      imposedHODMD = False
      if self.dmdParams['dmdType'] == 'hodmd' or snaps.shape[0] < snaps.shape[1]:
        v = max(snaps.shape[1] - snaps.shape[0],2)
        imposedHODMD = True
        snaps = np.concatenate([snaps[:, i:snaps.shape[1] - v  + i + 1] for i in range(v) ], axis=0)
      # overlap snaps
      X, Y = snaps[:, :-1], snaps[:, 1:]
      if self.dmdParams['rankTLSQ'] is not None:
        X, Y = mathUtils.computeTruncatedTotalLeastSquare(X, Y, self.dmdParams['rankTLSQ'])
      rank = self.dmdParams['energyRankSVD'] if self.dmdParams['energyRankSVD'] is not None else (self.dmdParams['rankSVD'] if self.dmdParams['rankSVD'] is not None else -1)
      U, s, V = mathUtils.computeTruncatedSingularValueDecomposition(X, rank)
      # lowrank operator from the SVD of matrices X and Y
      self.__Atilde[target] = U.T.conj().dot(Y).dot(V) * np.reciprocal(s)
      self._eigs[target], self._modes[target] = mathUtils.computeEigenvaluesAndVectorsFromLowRankOperator(self.__Atilde[target],
                                                                                                          Y, U, s, V,
                                                                                                          self.dmdParams['exactModes'])
      if imposedHODMD:
        self._modes[target] = self._modes[target][:targetVals[:,:,targetParamIndex].shape[0],:]
      self._amplitudes[target] = mathUtils.computeAmplitudeCoefficients(self._modes[target],
                                                                        targetVals[:,:,targetParamIndex],
                                                                        self._eigs[target],
                                                                        self.dmdParams['optimized'])
    # Default timesteps (even if the time history is not equally spaced in time, we "trick" the dmd to think it).
    self.timeScales = dict.fromkeys( ['training','dmd'],{'t0': 0, 'intervals': ts - 1, 'dt': 1})

  def __evaluateLocal__(self,featureVals):
    """
      This method is used to inquire the DMD to evaluate (after normalization that in
      this case is not performed)  a set of points contained in featureVals.
      a KDTree algorithm is used to construct a weighting function for the reconstructed space
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    returnEvaluation = {self.pivotParameterID:self.pivotValues}
    for target in list(set(self.target) - set([self.pivotParameterID])):
      reconstructData = self._reconstructData(target).real
      # find the nearest data and compute weights
      if len(reconstructData) > 1:
        weights, indexes = self.KDTreeFinder.query(featureVals, k=min(2**len(self.features),len(reconstructData)))
        # if 0 (perfect match), assign minimum possible distance
        weights[weights == 0] = sys.float_info.min
        weights =1./weights
        # normalize to 1
        weights = weights/weights.sum()
        for point in range(len(weights)):
          returnEvaluation[target] =  np.sum ((weights[point,:]*reconstructData[indexes[point,:]].T) , axis=1)
      else:
        returnEvaluation[target] = reconstructData[0]

    return returnEvaluation

  def writeXMLPreamble(self, writeTo, targets = None):
    """
      Specific local method for printing anything desired to xml file at the begin of the print.
      @ In, writeTo, xmlUtils.StaticXmlElement instance, element to write to
      @ In, targets, list, list of targets for whom information should be written.
      @ Out, None
    """
    # add description
    supervisedLearning.writeXMLPreamble(self, writeTo, targets)
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
      writeTo.addScalar(target,"dmdTimeScale",' '.join(['%.6e' % elm for elm in self.__getTimeScale()]))
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

