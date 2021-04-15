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
Created on Nov 1, 2019

@author: mandd
"""
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import pandas as pd
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.PostProcessorPluginBase import PostProcessorPluginBase
from utils import InputData, InputTypes
#Internal Modules End-----------------------------------------------------------

class MCSImporter(PostProcessorPluginBase):
  """
    This is the base class of the PostProcessor that imports Minimal Cut Sets (MCSs) into RAVEN as a PointSet
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag  = 'POSTPROCESSOR MCS IMPORTER'
    self.expand    = None  # option that controls the structure of the ET. If True, the tree is expanded so that
                           # all possible sequences are generated. Sequence label is maintained according to the
                           # original tree
    self.validDataType = ['PointSet'] # The list of accepted types of DataObject
    ## Currently, we have used both DataObject.addRealization and DataObject.load to
    ## collect the PostProcessor returned outputs. DataObject.addRealization is used to
    ## collect single realization, while DataObject.load is used to collect multiple realizations
    ## However, the DataObject.load can not be directly used to collect single realization
    self.outputMultipleRealizations = True

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for the class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory("expand",       contentType=InputTypes.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("BElistColumn", contentType=InputTypes.StringType))
    return inputSpecification

  def _handleInput(self, paramInput):
    """
      Method that handles PostProcessor parameter input block.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    expand = paramInput.findFirst('expand')
    self.expand = expand.value

    if self.expand:
      beListColumn = paramInput.findFirst('BElistColumn')
      self.beListColumn = beListColumn.value

  def run(self, inputs):
    """
      This method executes the PostProcessor action.
      @ In,  inputs, list, list of file objects
      @ Out, None
    """

    mcsFileFound = False
    beFileFound  = False

    for file in inputs:
      if file.getType()=="MCSlist":
        if mcsFileFound:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Multiple files with type=MCSlist have been found')
        else:
          mcsListFile = file
          mcsFileFound = True
      if file.getType()=="BElist":
        if self.expand==False:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', A file with type=BElist has been found but expand is set to False')
        if beFileFound:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Multiple files with type=BElist have been found')
        else:
          BElistFile = file
          beFileFound  = True

    if beFileFound==False and self.expand==True:
      self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Expand is set to False but no file with type=BElist has been found')

    self.mcsIDs, self.probability, self.mcsList, self.beList = mcsReader(mcsListFile)

    if self.expand:
      beData = pd.read_csv(BElistFile.getFilename())
      self.beList = beData[self.beListColumn].values.tolist()

    mcsPointSet = {}

    # MCS Input variables
    mcsPointSet['probability'] = self.probability
    mcsPointSet['MCS_ID']      = self.mcsIDs
    mcsPointSet['out']         = np.ones((self.probability.size))

    # MCS Output variables
    for be in self.beList:
      mcsPointSet[be]= np.zeros(self.probability.size)
    counter=0
    for mcs in self.mcsList:
      for be in mcs:
        mcsPointSet[be][counter] = 1.0
      counter = counter+1
    mcsPointSet = {'data': mcsPointSet, 'dims': {}}
    return mcsPointSet

def mcsReader(mcsListFile):
  """
    Function designed to read a file containing the set of MCSs and to save it as list of list
    @ In, mcsListFile, string, name of the file containing the set of MCSs
    @ Out, mcsIDs, np array, array containing the ID associated to each MCS
    @ Out, probability, np array, array containing the probability associated to each MCS
    @ Out, mcsList, list, list of MCS, each element is also a list containing the basic events of each MCS
    @ Out, beList, list, list of all basic events contained in the MCSs
  """
  mcsList=[]
  beList=set()
  probability = np.zeros((0))
  mcsIDs = np.zeros((0))

  # construct the list of MCSs and the list of BE
  with open(mcsListFile.getFilename(), 'r') as file:
    next(file) # skip header
    lines = file.read().splitlines()
    for l in lines:
      elementsList = l.split(',')

      mcsIDs = np.append(mcsIDs,elementsList[0])
      elementsList.pop(0)

      probability=np.append(probability,elementsList[0])
      elementsList.pop(0)

      mcsList.append(list(element.rstrip('\n') for element in elementsList))

      beList.update(elementsList)

  return mcsIDs, probability, mcsList, beList
