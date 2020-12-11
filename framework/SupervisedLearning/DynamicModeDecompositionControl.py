
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
  Created on Oct 01, 2020

  @author: Haoyu Wang, Andrea Alfonsi

  Dynamic Mode Decomposition with Control (The class is based on the DynamicModeDecomposition class)
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import numpy as np
import scipy
from sklearn import neighbors
from scipy import spatial
import matplotlib.pyplot as plt
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils
from SupervisedLearning import supervisedLearning
from .DynamicModeDecomposition import DynamicModeDecomposition
#Internal Modules End--------------------------------------------------------------------------------

class DynamicModeDecompositionControl(DynamicModeDecomposition):
  """
    This surrogate is aimed to construct a "time-dep" surrogate based on
    Dynamic Mode Decomposition with control.
    Ref. Proctor, Joshua L., Steven L. Brunton, and J. Nathan Kutz.
        "Dynamic mode decomposition with control."
        SIAM Journal on Applied Dynamical Systems 15, no. 1 (2016): 142-161.

    Updated on July 1st, 2020

    @author: Haoyu Wang, Argonne National Laboratory
             Andrea Alfonsi, Idaho National Laboratory
  """
  def __init__(self, messageHandler, **kwargs):
    """
      DMD constructor
      @ In, messageHandler, MessageHandler.MessageUser, a MessageHandler object in charge of raising errors,
                           and printing messages
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
    """
    DynamicModeDecomposition.__init__(self,messageHandler,**kwargs)
    self.printTag = 'DMDC'
    self._dynamicFeatures = True
    ### Extract the Actuator Variable Names (u)
    self.actuatorsID = kwargs.get("Actuators", None)
    ### Extract the State Variable Names (x)
    self.stateID = kwargs.get("StateVariables", None)
    ### Extract the Output Names (Output, Y)
    self.outputID = set(self.target) - set([self.pivotParameterID])
    cUXY =  kwargs.get('SubtractNormUXY',False)
    self.dmdParams['centerUXY'] = cUXY # whether to subtract the nominal(initial) value from U, X and Y signal for calculation
    # variables filled up in the training stages
    self.__Btilde = {} # B matrix
    self.__Ctilde = {} # C matrix
    self.actuatorVals = None # Actuator values (e.g. U), the variable names are in self.ActuatorID
    self.stateVals = None # state values (e.g. X)
    self.outputVals = None # output values (e.g. Y)
    self.parameterValues = None #  parameter values
    # some checks
    if not self.actuatorsID:
      self.raiseAnError(IOError,'Actuators XML node must be present for constructing DMDc !')
    if not self.stateID:
      self.raiseAnError(IOError,'StateVariables XML node must be present for constructing DMDc !')
    # check if there are parameters
    self.parametersIDs = set(self.features) - set(self.actuatorsID) - set(self.stateID)

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape=[n_samples,n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, numpy.ndarray, shape = [n_samples,n_timeStep, n_dimensions], an array of time series data
    """
    ### Extract the Pivot Values (Actuator, U) ###
    self.neigh = None
    if len(self.parametersIDs):
      self.parameterValues =  np.asarray([featureVals[:, :, self.features.index(par)] for par in self.parametersIDs]).T[0, :, :]
      self.neigh = neighbors.KNeighborsRegressor(n_neighbors=1)
      y = np.asarray (range(featureVals.shape[0]))
      self.neigh.fit(self.parameterValues, y)   
    # self.ActuatorVals is Num_Entries*2 array, the snapshots of [u1, u2]. Shape is [n_samples, n_timesteps, n_actuators]
    self.actuatorVals = np.asarray([featureVals[:, :, self.features.index(act)] for act in self.actuatorsID]).T
    ### Extract the time marks "self.pivotValues" (discrete, in time step marks) ###
    self.pivotValues = targetVals[:, :, self.target.index(self.pivotParameterID)]
    # self.outputVals is Num_Entries*2 array, the snapshots of [y1, y2]. Shape is [n_samples, n_timesteps, n_targets]
    self.outputVals =  np.asarray([targetVals[:, :,self.target.index(out)] for out in self.outputID]).T
    ### Extract the State Values (State, X) ###
    # self.outputVals is Num_Entries*2 array, the snapshots of [y1, y2]. Shape is [n_samples, n_timesteps, n_state_variables]
    self.stateVals =  np.asarray([featureVals[:, :, self.features.index(st)] for st in self.stateID]).T
    # create matrices
    self.__Atilde = np.zeros((featureVals.shape[0], len(self.stateID), len(self.stateID)))
    self.__Btilde = np.zeros((featureVals.shape[0], len(self.stateID), len(self.actuatorsID)))
    self.__Ctilde = np.zeros((featureVals.shape[0], len(self.outputID), len(self.stateID)))
    for smp in range(featureVals.shape[0]):
      X1 = (self.stateVals[:-1,smp, :] - self.stateVals[0,smp,:]).T if self.dmdParams['centerUXY'] else self.stateVals[:-1,smp, :].T
      X2 = (self.stateVals[1:,smp, :] - self.stateVals[0,smp,:]).T  if self.dmdParams['centerUXY'] else self.stateVals[1:,smp, :].T
      U =  (self.actuatorVals[:-1,smp, :] - self.actuatorVals[0,smp, :]).T  if self.dmdParams['centerUXY'] else self.actuatorVals[:-1,smp, :].T
      Y1 = (self.outputVals[:-1,smp, :] - self.outputVals[0, smp, :]).T if self.dmdParams['centerUXY'] else self.outputVals[:-1,smp, :].T
      # compute A,B,C matrices
      self.__Atilde[smp, :, :] , self.__Btilde[smp, :, :], self.__Ctilde[smp, :, :] = self._evaluateMatrices(X1, X2, U, Y1, self.dmdParams['rankSVD'])
    # Default timesteps (even if the time history is not equally spaced in time, we "trick" the dmd to think it).
    self.timeScales = dict.fromkeys( ['training','dmd'],{'t0': self.pivotValues[0, 0], 'intervals': len(self.pivotValues[0, : ]) - 1, 'dt': self.pivotValues[0, 1]-self.pivotValues[0, 0]})

  #######
  def __evaluateLocal__(self,featureVals):
    """
      This method is used to inquire the DMD to evaluate (after normalization that in
      this case is not performed)  a set of points contained in featureVals.
      a KDTree algorithm is used to construct a weighting function for the reconstructed space
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    ### Initialize the final return value ###
    returnEvaluation = {}
    ### Extract the Actuator signal U ###
    Eval_U = []
    for VarID in self.actuatorsID:
      VarIndex = self.features.index(VarID)
      Eval_U.append(featureVals[:, VarIndex])
      returnEvaluation.update({VarID: featureVals[:, VarIndex]})
    Eval_U = np.asarray(Eval_U)
    _,ts_Eval = np.shape(Eval_U) # ts_Eval = 100


    ### Extract the initial state vector ###
    Eval_X = [[]]
    for VarID in self.stateID:
      VarIndex = self.features.index(VarID)
      Eval_X[0].append(featureVals[0, VarIndex])
    Eval_X = np.asarray(Eval_X).T

    ### perform the self-propagation of X, X[k+1] = A*X[k] + B*U[k] ###
    for i in range(0,ts_Eval-1):
      X_pred = np.reshape(self.__Atilde.dot(Eval_X[:,i]) + self.__Btilde.dot(Eval_U[:,i]),(-1,1))
      Eval_X = np.hstack((Eval_X,X_pred))

    ### Store the results to the dictionary "returnEvaluation"
    for VarID in self.stateID:
      VarIndex = self.stateID.index(VarID)
      returnEvaluation.update({VarID: Eval_X[VarIndex,:]})

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
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    description  = ' This XML file contains the main information of the DMDC ROM.'
    description += ' The method is explained in:'
    description += ' Proctor, Joshua L., Steven L. Brunton, and J. Nathan Kutz. '
    description += ' "Dynamic mode decomposition with control." '
    description += ' SIAM Journal on Applied Dynamical Systems 15, no. 1 (2016): 142-161.'
    writeTo.addScalar('ROM',"description",description)

  def writeXML(self, writeTo, targets = None, skip = None):
    """
      Adds requested entries to XML node.
      @ In, writeTo, xmlTuils.StaticXmlElement, element to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """

    if skip is None: # skip =  None
      skip = []

    what = ['dmdType','rankSVD','acturators','acturatorsCount',
            'states','statesCount','outputs','outputsCount',
            'Atilde','Btilde','Ctilde','UNorm','XNorm','YNorm',
            'XLast','dmdTimeScale']

    if targets is None: # targets = None
      readWhat = what
    else:
      readWhat = targets

    for s in skip: # skip is empty, so nothing to skip
      if s in readWhat:
        readWhat.remove(s) # "readWhat" remains unchanged.

    if not set(readWhat) <= set(what): # when "readWhat" contains something outside of "what", raise error
      self.raiseAnError(IOError, "The following variables specified in <what> node are not recognized: "+ ",".join(np.setdiff1d(readWhat, what).tolist()) )
    else:
      what = readWhat

    target = 'DMDcModel'

    toAdd = ['dmdType','rankSVD']
    self.dmdParams['rankSVD'] = self.dmdParams['rankSVD'] if self.dmdParams['rankSVD'] is not None else -1

    for add in toAdd: # toAdd = ['dmdType','rankSVD']
      if add in what :
        writeTo.addScalar(target,add,self.dmdParams[add])

    targNode = writeTo._findTarget(writeTo.getRoot(), target)
    if "acturators" in what:
      writeTo.addScalar(target, "acturators", ' '.join(self.actuatorsID))
    if "acturatorsCount" in what:
      writeTo.addScalar(target, "acturatorsCount", len(self.actuatorsID))
    if "states" in what:
      writeTo.addScalar(target, "states", ' '.join(self.stateID))
    if "statesCount" in what:
      writeTo.addScalar(target, "statesCount", len(self.stateID))
    if "outputs" in what:
      writeTo.addScalar(target, "outputs", ' '.join(self.outputID))
    if "outputsCount" in what:
      writeTo.addScalar(target, "outputsCount", len(self.outputID))
    if "dmdTimeScale" in what:
      writeTo.addScalar(target,"dmdTimeScale",' '.join(['%.3d' % elm for elm in self._getTimeScale()]))

    for smp in range(self.stateVals.shape[1]):
      attributeDict = {}
      if len(self.parametersIDs):
        attributeDict = {self.parametersIDs[index]:'%.6e' % self.parameterValues[smp,index] for index in range(len(self.parametersIDs))}
      attributeDict["sample"] = str(smp)

      if "UNorm" in what and self.dmdParams['centerUXY']:
        valCont = " ".join(['%.8e' % elm for elm in self.actuatorVals[:, smp, :].T.flatten().tolist()])
        writeTo.addVector("UNorm","realization",valCont, root=targNode, attrs=attributeDict)

      if "XNorm" in what and self.dmdParams['centerUXY']:
        valCont = " ".join(['%.8e' % elm for elm in self.stateVals[:, smp, :].T.flatten().tolist()])
        writeTo.addVector("XNorm","realization",valCont, root=targNode, attrs=attributeDict)
        # writeTo.addScalar(target, "XNorm",  valCont)

      if "XLast" in what:
        valCont = " ".join(['%.8e' % elm for elm in self.stateVals[-1, smp, :].T.flatten().tolist()])
        writeTo.addVector("XLast","realization",valCont, root=targNode, attrs=attributeDict)
        # writeTo.addScalar(target, "XLast", valCont)

      if "YNorm" in what and self.dmdParams['centerUXY']:
        valCont = " ".join(['%.8e' % elm for elm in self.outputVals[:, smp, :].T.flatten().tolist()])
        writeTo.addVector("YNorm","realization",valCont, root=targNode, attrs=attributeDict)
        # writeTo.addScalar(target, "YNorm", valCont)

      if "Atilde" in what:
        valDict = {'real': " ".join(['%.8e' % elm for elm in self.__Atilde[smp, :, :].T.real.flatten().tolist()]),
                   'imaginary':" ".join(['%.8e' % elm for elm in self.__Atilde[smp, :, :].T.imag.flatten().tolist()]),
                   "matrixShape":",".join(str(x) for x in np.shape(self.__Atilde[smp, :, :]))}
        writeTo.addVector("Atilde","realization",valDict, root=targNode, attrs=attributeDict)

        valDict = {'real': " ".join(['%.8e' % elm for elm in self.__Btilde[smp, :, :].T.real.flatten().tolist()]),
                   'imaginary':" ".join(['%.8e' % elm for elm in self.__Btilde[smp, :, :].T.imag.flatten().tolist()]),
                   "matrixShape":",".join(str(x) for x in np.shape(self.__Btilde[smp, :, :]))}
        writeTo.addVector("Btilde","realization",valDict, root=targNode, attrs=attributeDict)

      if "Ctilde" in what and len(self.outputID) > 0:
        valDict = {'real': " ".join(['%.8e' % elm for elm in self.__Ctilde[smp, :, :].T.real.flatten().tolist()]),
                   'imaginary':" ".join(['%.8e' % elm for elm in self.__Ctilde[smp, :, :].T.imag.flatten().tolist()]),
                   "matrixShape":",".join(str(x) for x in np.shape(self.__Ctilde[smp, :, :]))}
        writeTo.addVector("Ctilde","realization",valDict, root=targNode, attrs=attributeDict)

  def _evaluateMatrices(self, X1, X2, U, Y1, rankSVD):
    """
      Evaluate the the matrices (A and B tilde)
      @ In, X1, np.ndarray, n dimensional state vectors (n*L)
      @ In, X2, np.ndarray, n dimensional state vectors (n*L)
      @ In, U, np.ndarray, m-dimension control vector by L (m*L)
      @ In, Y1, np.ndarray, m-dimension output vector by L (y*L)
      @ In, rankSVD, int, rank of the SVD
      @ Out, A, np.ndarray, the A matrix
      @ Out, B, np.ndarray, the B matrix
      @ Out, C, np.ndarray, the C matrix
    """
    n = len(X2)
    # Omega Matrix, stack X1 and U
    omega = np.concatenate((X1, U), axis=0)
    # SVD
    Ut, St, Vt = mathUtils.computeTruncatedSingularValueDecomposition(omega, rankSVD, True, False)
    # QR decomp. St=Q*R, Q unitary, R upper triangular
    Q, R = np.linalg.qr(St)
    # if R is singular matrix, raise an error
    if np.linalg.det(R) == 0:
      self.raiseAnError(RuntimeError, "The R matrix is singlular, Please check the singularity of [X1;U]!")
    beta = X2.dot(Vt).dot(np.linalg.inv(R)).dot(Q.T)
    A = beta.dot(Ut[0:n, :].T)
    B = beta.dot(Ut[n:, :].T)
    C = Y1.dot(scipy.linalg.pinv2(X1))

    return A, B, C


