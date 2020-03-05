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
Created on September 2017

@author: wangc
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import sklearn
import ast
from utils import utils
from sklearn import model_selection as cross_validation
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .CrossValidation import CrossValidation
#Internal Modules End--------------------------------------------------------------------------------

class SciKitLearn(CrossValidation):
  """
    SciKitLearn inteface for Cross validation methods
  """
  # Minimum requirement for SciKit-Learn is 0.18
  # dictionary of available cross validation methods {mainClass:(classPointer, output Type (float))}
  availImpl = {}
  availImpl['KFold'                  ] = (cross_validation.KFold,                  'float')
  availImpl['StratifiedKFold'        ] = (cross_validation.StratifiedKFold,        'float')
  availImpl['LeaveOneOut'            ] = (cross_validation.LeaveOneOut,            'float')
  availImpl['LeavePOut'              ] = (cross_validation.LeavePOut,              'float')
  availImpl['ShuffleSplit'           ] = (cross_validation.ShuffleSplit,           'float')
  availImpl['StratifiedShuffleSplit' ] = (cross_validation.StratifiedShuffleSplit, 'float')
  availImpl['LabelKFold'             ] = (cross_validation.GroupKFold,             'float')
  availImpl['LabelShuffleSplit'      ] = (cross_validation.GroupShuffleSplit,      'float')
  availImpl['LeaveOneLabelOut'       ] = (cross_validation.LeaveOneGroupOut,       'float')
  availImpl['LeavePLabelOut'         ] = (cross_validation.LeavePGroupsOut,         'float')
  # Method may needed
  #availImpl['PredefinedSplit'         ] = (cross_validation.PredefinedSplit,        'float')
  #availImpl['TimeSeriesSplit'         ] = (cross_validation.TimeSeriesSplit,        'float')
  # Methods available for SciKit-Learn version >= 0.19
  #availImpl['RepeatedKFold'           ] = (cross_validation.RepeatedKFold,          'float')
  #availImpl['RepeatedStratifiedKFold' ] = (cross_validation.RepeatedStratifiedKFold,'float')

  def __init__(self, messageHandler, **kwargs):
    """
      Constructor for SciKitLearn class
      @ In, messageHandler, MessageHandler, Message handler object
      @ In, kwargs, dict, arguments for the SciKitLearn
      @ Out, None
    """
    CrossValidation.__init__(self, messageHandler, **kwargs)
    self.printTag = 'SKLearn Cross Validation'

    if 'SKLtype' not in self.initOptionDict.keys():
      self.raiseAnError(IOError, 'Missing XML node: Cross validation from SciKitLearn requires keyword "SKLtype"')
    self.SKLType = self.initOptionDict['SKLtype']
    self.initOptionDict.pop('SKLtype')

    if self.SKLType not in self.__class__.availImpl.keys():
      self.raiseAnError(IOError, 'Unknow SKLtype ', self.SKLType, ' from cross validation ', self.name)

    self.__class__.returnType = self.__class__.availImpl[self.SKLType][1]
    self.__CVInstance = self.__class__.availImpl[self.SKLType][0](**self.initOptionDict)
    self.outputDict = {}

  def reset(self):
    """
      Used to reset private variables
      @ In, None
      @ Out, None
    """
    self.__CVInstance = None

  def getCrossValidationType(self):
    """
      This method is used to return the type of cross validation to be employed
      @ In, None
      @ Out, None
    """
    return self.SKLType

  def generateTrainTestIndices(self, X, y=None, groups=None):
    """
      generate train/test indices
      @ In, None
      @ In, X, array_like, shape (n_samples,n_features), trainning data
      @ In, y, array_like, shape (n_samples,), the target variable
      @ In, groups, array_like, shape (n_samples,), group labels for samples used while splitting the dataset into
        train/test set.
      @ Out, Object, instance of cross validation
    """
    return self.__CVInstance.split(X, y, groups)

