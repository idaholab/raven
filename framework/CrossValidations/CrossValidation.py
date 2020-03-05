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
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import MessageHandler
from utils import utils
#Internal Modules End--------------------------------------------------------------------------------

class CrossValidation(utils.metaclass_insert(abc.ABCMeta), MessageHandler.MessageUser):
  """
    Cross validation methods used to validate models
  """

  def __init__(self, messageHandler, **kwargs):
    """
      This is the basic method initialize the cross validation object
      @ In, messageHandler, object, Message handler object
      @ In, kwargs, dict, arguments for the Pairwise metric
      @ Out, none
    """
    self.messageHandler = messageHandler
    self.printTag = 'Cross Validation'
    if kwargs != None:
      self.initOptionDict = kwargs
    else:
      self.initOptionDict = {}

  def reset(self):
    """
      Used to reset private variables
      @ In, None
      @ Out, None
    """
    pass

  def getCrossValidationType(self):
    """
      This method is used to return the type of cross validation to be employed
      @ In, None
      @ Out, None
    """
    pass

  def generateTrainTestIndices(self):
    """
      This method is used to generate train/test indices to split data in train test sets
      @ In, None
      @ Out, None
    """
    pass


