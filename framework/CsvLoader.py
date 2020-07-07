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

#External
#Modules------------------------------------------------------------------------------------
import pandas as pd
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class CsvLoader(MessageHandler.MessageUser):
  """
    Class aimed to load the CSV files
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, the message handler
      @ Out, None
    """
    self.type = 'CsvLoader'               # naming type for this class
    self.printTag = self.type             # message handling representation
    self.allOutParam = False              # all output parameters?
    self.allFieldNames = []               # "header" of the CSV file
    self.messageHandler = messageHandler  # message handling utility

  def loadCsvFile(self, myFile, nullOK=None):
    """
      Function to load a csv file into realization format
      It also retrieves the headers
      The format of the csv must be comma-separated (pandas readable)
      @ In, myFile, string, Input file name (absolute path)
      @ In, nullOK, bool, indicates if null values are acceptable
      @ Out, df, pandas.DataFrame, the loaded data
    """
    # first try reading the file
    try:
      df = pd.read_csv(myFile)
    except pd.errors.EmptyDataError:
      # no data in file
      self.raiseAWarning(f'Tried to read data from "{myFile}", but the file is empty!')
      return
    else:
      self.raiseADebug(f'Reading data from "{myFile}"')
    # check for NaN contents -> this isn't allowed in RAVEN currently, although we might need to change this for ND
    if (not nullOK) and (pd.isnull(df).values.sum() != 0):
      bad = pd.isnull(df).any(1).nonzero()[0][0]
      self.raiseAnError(IOError, f'Invalid data in input file: row "{bad+1}" in "{myFile}"')
    self.allFieldNames = list(df.columns)
    return df

  def getAllFieldNames(self):
    """
      Function to get all field names found in the csv file
      @ In, None
      @ Out, allFieldNames, list, list of field names (headers)
    """
    return self.allFieldNames
