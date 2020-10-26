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
Created on March 25, 2020

@author: mandd
"""

#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData, InputTypes
from sympy.logic import SOPform, POSform
from itertools import product
import Runners
#Internal Modules End-----------------------------------------------------------

class FTgenerator(PostProcessor):
  """
    This postprocessor is designed to create a fault tree in OpenPSA format from a set of simulation data
    The postprocessor acts only on PointSet and return a reduced PointSet and print of file the FT in OpenPSA format 
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.inputVars = None   # variable associated with the lower limit of the value dimension
    self.topEventID = None   # variable associated with the upper limit of the cost dimension

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(FTgenerator, cls).getInputSpecification()
    
    inputSpecification.addSub(InputData.parameterInputFactory("topEventID", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("inputVars" , contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("fileName"  , contentType=InputTypes.StringType))
    
    typeAllowedFormats = InputTypes.makeEnumType("calculationFormat", "calculationFormatType", ["sop","pos"])
    inputSpecification.addSub(InputData.parameterInputFactory("type" , contentType=typeAllowedFormats))

    return inputSpecification

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    self.topEventID  = paramInput.findFirst('topEventID')

    inputVars = paramInput.findFirst('valueID')
    self.inputVars = inputVars.split(",")
    for var in self.inputVars:
      var = var.strip()
    
    self.fileName  = paramInput.findFirst('fileName')
    
    typeID = paramInput.findFirst('type')
    self.type = typeID.lower()

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      In this case, we only want data objects!
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject.HistorySet, input data
    """
    if len(currentInp) > 1:
      self.raiseAnError(IOError, 'FTgenerator postprocessor {} expects one input DataObject, but received {} inputs!".'
                                  .format(self.name,len(currentInp)))
    currentInp = currentInp[0]
    if currentInp.type not in ['PointSet']:
      self.raiseAnError(IOError, 'FTgenerator postprocessor "{}" requires a DataObject input! Got "{}".'
                                 .format(self.name, currentInp.type))
    return currentInp

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, DataObject, point set that contains the data to be processed
      @ Out, reducedDataset, dict, dictionary containing the Pareto Frontier information
    """
    inData = self.inputToInternal(inputIn)
    dataset = inData.asDataset()
    
    for var in dataset.keys():
      dataset[var] = np.where(dataset[var].values > 0.0, 1, 0)
    
    nVars = len(self.inputVars)
    combinations = np.array([i for i in product(range(2), repeat=nVars)])  
    self.reducedDataset = np.zeros([2**nVars,nVars+1])
    self.reducedDataset[:,:nVars] = combinations
    
    counter = 0
    for combination in combinations:
      indexes = np.where((dataset.values==combination).all(axis=1))
      if len(indexes)==0:
        self.raiseAnError(RuntimeError,'FTgenerator: combination ' + str(combination) + ' of variables ' + str(self.inputVars) +' has not been found in the dataset.')
      avg = np.average(np.absolute(dataset[self.topEventID][indexes]))
      self.reducedDataset[counter] = avg  
      if avg not in [0,1]:
        self.raiseAWarning('FTgenerator: combination ' + str(combination) + ' of variables ' + str(self.inputVars) +' has generated an average value not in {0,1}.')
      counter = counter + 1

    return self.reducedDataset

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, DataObject.DataObject, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()

    outputDict ={}
    outputDict['data'] = evaluation[1]

    if output.type in ['PointSet']:
      outputDict['dims'] = {}
      for key in outputDict.keys():
        outputDict['dims'][key] = []
      output.load(outputDict['data'], style='dict', dims=outputDict['dims'])
    else:
        self.raiseAnError(RuntimeError, 'ParetoFrontier failed: Output type ' + str(output.type) + ' is not supported.')

  def generateFT(self):
    indexes = np.where(self.reducedDataset[:,-1] > 0)
    minterms = self.reducedDataset(indexes)
    
    if self.type=='sop':
      formula = SOPform(self.inputVars,minterms)
      FTgeneratorSOP(formula)
    else:  # self.type=='pos'
      formula = POSform(self.inputVars,minterms)
      FTgeneratorPOS(formula)

def FTgeneratorSOP(formula):
  # Example: c | (a & b)
  terms = str(formula).split("|")  # ['c ', ' (a & b)']
  for term in terms:
    term = term.split("&")         # ['c ', [' (a', ' b)']]
    for element in term:
      element.replace('(', '')
      element.replace(')', '')
      element.strip()              # ['c', ['a',' b']]
        
def FTgeneratorPOS(formula):
  # Example: (a | c) & (b | c)
      
  
  
  