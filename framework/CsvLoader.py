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
Created on Feb 7, 2013
@author: alfoa
This python module performs the loading of
data from csv files
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class CsvLoader(MessageHandler.MessageUser):
  """
    Class aimed to load the CSV files
  """
  def __init__(self,messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, the message handler
      @ Out, None
    """
    self.allOutParam      = False # all output parameters?
    self.allFieldNames    = []
    self.type               = 'CsvLoader'
    self.printTag           = self.type
    self.messageHandler     = messageHandler

  def loadCsvFile(self,myFile):
    """
      Function to load a csv file into a numpy array (2D)
      It also retrieves the headers
      The format of the csv must be:
      STRING,STRING,STRING,STRING
      FLOAT ,FLOAT ,FLOAT ,FLOAT
      ...
      FLOAT ,FLOAT ,FLOAT ,FLOAT
      @ In, fileIn, string, Input file name (absolute path)
      @ Out, data, numpy.ndarray, the loaded data
    """
    # open file
    myFile.open(mode='rb')
    # read the field names
    head = myFile.readline().decode()
    self.allFieldNames = head.split(',')
    for index in range(len(self.allFieldNames)):
      self.allFieldNames[index] = self.allFieldNames[index].strip()
    # load the table data (from the csv file) into a numpy nd array
    data = np.loadtxt(myFile,dtype='float',delimiter=',',ndmin=2,skiprows=1)
    # close file
    myFile.close()
    return data

  def getAllFieldNames(self):
    """
      Function to get all field names found in the csv file
      @ In, None
      @ Out, allFieldNames, list, list of field names (headers)
    """
    return self.allFieldNames
