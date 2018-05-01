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
import warnings
warnings.simplefilter('default', DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import sklearn
import ast
from utils import utils
# Modules of cross_validation will be deprecated in SciKit Learn version 0.18 and will be removed in 0.20
# Currently, I think this is fine, but in the future we will use sklearn.model_selection instead.
# from sklearn import model_selection as cross_validation
from sklearn import cross_validation
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .CrossValidation import CrossValidation

#Internal Modules End--------------------------------------------------------------------------------


class SciKitLearn(CrossValidation):
  """
    SciKitLearn inteface for Cross validation methods
  """
  # The following methods will be deprecated since 0.18 of SciKitLearn, and will be removed in 0.20 of SciKitLearn
  # FIXME: Swith to sklearn.model_selection instead.
  # dictionary of available cross validation methods {mainClass:(classPointer, output Type (float))}
  availImpl = {}
  availImpl['KFold'] = (cross_validation.KFold, 'float')
  availImpl['StratifiedKFold'] = (cross_validation.StratifiedKFold, 'float')
  availImpl['LeaveOneOut'] = (cross_validation.LeaveOneOut, 'float')
  availImpl['LeavePOut'] = (cross_validation.LeavePOut, 'float')
  availImpl['ShuffleSplit'] = (cross_validation.ShuffleSplit, 'float')
  availImpl['StratifiedShuffleSplit'] = (cross_validation.StratifiedShuffleSplit, 'float')
  availImpl['LabelKFold'] = (cross_validation.LabelKFold, 'float')
  availImpl['LabelShuffleSplit'] = (cross_validation.LabelShuffleSplit, 'float')
  availImpl['LeaveOneLabelOut'] = (cross_validation.LeaveOneLabelOut, 'float')
  availImpl['LeavePLabelOut'] = (cross_validation.LeavePLabelOut, 'float')

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
      self.raiseAnError(
          IOError, 'Missing XML node: Cross validation from SciKitLearn requires keyword "SKLtype"')
    self.SKLType = self.initOptionDict['SKLtype']
    self.initOptionDict.pop('SKLtype')

    if self.SKLType not in self.__class__.availImpl.keys():
      self.raiseAnError(IOError, 'Unknow SKLtype ', self.SKLType, ' from cross validation ',
                        self.name)

    self.__class__.returnType = self.__class__.availImpl[self.SKLType][1]

    for key, value in self.initOptionDict.items():
      try:
        newValue = ast.literal_eval(value)
        if type(newValue) == list:
          newValue = np.asarray(newValue)
        if key == 'n_splits':
          self.initOptionDict['n_folds'] = newValue
        else:
          self.initOptionDict[key] = newValue
      except:
        if key == 'n_splits':
          self.initOptionDict['n_folds'] = value
        else:
          self.initOptionDict[key] = value

    if 'n_splits' in self.initOptionDict.keys():
      self.initOptionDict.pop('n_splits')

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

  def generateTrainTestIndices(self):
    """
      generate train/test indices
      @ In, None
      @ Out, Object, instance of cross validation
    """
    return self.__CVInstance
