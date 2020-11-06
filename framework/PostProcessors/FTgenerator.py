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
import xml.etree.ElementTree as ET
import math
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils, xmlUtils
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
    self.inputVars = None    # variable associated with the lower limit of the value dimension
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
    inputSpecification.addSub(InputData.parameterInputFactory("simOnly"   , contentType=InputTypes.BoolType))
    
    typeAllowedFormats = InputTypes.makeEnumType("calculationFormat", "calculationFormatType", ["sop","pos"])
    inputSpecification.addSub(InputData.parameterInputFactory("type" , contentType=typeAllowedFormats))

    return inputSpecification

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    topEventID  = paramInput.findFirst('topEventID')
    self.topEventID = topEventID.value.strip()

    inputVars = paramInput.findFirst('inputVars')
    self.inputVars = inputVars.value.split(",")
    for var in self.inputVars:
      var = var.strip()
    
    fileName  = paramInput.findFirst('fileName')
    self.fileName = fileName.value
    
    simOnly  = paramInput.findFirst('simOnly')
    self.simOnly = simOnly.value
    
    typeConv = paramInput.findFirst('type')
    self.type = typeConv.value.lower()

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
      @ Out, reducedDataset, dict, dictionary containing the Karnaugh map information
    """
    inData = self.inputToInternal(inputIn)
    dataset = inData.asDataset()
    
    self.dataObjectName = inputIn
    
    for var in dataset.keys():
      dataset[var].values = np.where(dataset[var].values > 0.0, 1, 0)
      
    dsSel = dataset[self.inputVars]
    data = dsSel.to_array().data.transpose()
    keys = list(dsSel.keys())
    
    nVars = len(self.inputVars)
    combinations = np.array([i for i in product(range(2), repeat=nVars)])  
    self.reducedDataset = np.zeros([2**nVars,nVars+1])
    self.reducedDataset[:,:nVars] = combinations
 
    for counter,combination in enumerate(combinations):
      indexes = np.where(np.all(data==combination,axis=1))[0]
      if len(indexes)==0 and not self.simOnly:
        self.raiseAnError(RuntimeError,'FTgenerator: combination ' + str(combination) + ' of variables ' + str(self.inputVars) +' has not been found in the dataset.')
      avg=0

      for locIndex in indexes:
        avg = avg + inData.realization(index=locIndex)[self.topEventID]
      if len(indexes)==0 and self.simOnly:
        self.reducedDataset[counter,nVars] = -1
      else:
        avg=avg/np.size(indexes)
        self.reducedDataset[counter,nVars] = avg  
      if avg not in [0,1]:
        self.reducedDataset[counter,nVars] = 1
        self.raiseAWarning('FTgenerator: combination ' + str(combination) + ' of variables ' + str(self.inputVars) +' has generated an average value not in {0,1}.')
    
    self.reducedDatasetDict = {}
    for index,key in enumerate(keys):
      self.reducedDatasetDict[key] = self.reducedDataset[:,index]
    self.reducedDatasetDict[self.topEventID] = self.reducedDataset[:,-1]
    
    self.generateFT()
    
    return self.reducedDatasetDict

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
        self.raiseAnError(RuntimeError, 'FTgenerator failed: Output type ' + str(output.type) + ' is not supported.')

  def generateFT(self):
    """
      Class method designed to generate a fault tree from the obtained Karnaugh map
      @ In,  None
      @ Out, None
    """
    indexes = np.where(self.reducedDataset[:,-1] > 0)
    minterms = self.reducedDataset[indexes,:-1][0]
    mintermsConverted = minterms.tolist()
    
    indexDontCares = np.where(self.reducedDataset[:,-1] < 0)
    if len(indexes)>0:  
      dontCares = self.reducedDataset[indexDontCares,:-1][0]
      dontCaresConverted = dontCares.tolist()
    else:
      dontCaresConverted = None
  
    if self.type=='sop':
      formula = SOPform(self.inputVars, minterms=mintermsConverted, dontcares=dontCaresConverted)
    else:
      formula = POSform(self.inputVars, minterms=mintermsConverted, dontcares=dontCaresConverted)

    formulaText = structureFormula(formula,self.type)

    printFT(formulaText,self.fileName,'name', self.type)
      
def structureFormula(formula, typeSolver):
  """
    Method designed to generate the Boolean expression in terms of list of lists
    @ In,  formula, string, formula generated bu SymPy
    @ In,  typeSolver, string, type of solver that has been used to generate the Boolean expression
    @ Out, formulaMod, list, list containing the terms of the generated  Boolean expression
    """
  # Example: c | (a & b)
  if typeSolver=='sop':
    terms = str(formula).split("|")  # ['c ', ' (a & b)']
  else:
    terms = str(formula).split("&") 

  formulaMod=[]
  for term in terms:
    if typeSolver=='sop':
      term1 = term.split("&")         # ['c ', [' (a', ' b)']]
    else:
      term1 = term.split("|")

    elemMod=[]
    for element in term1:
      elemMod.append(element.replace('(', '').replace(')', '').strip())  # ['c', ['a',' b']]
    formulaMod.append(elemMod)  

  return formulaMod  

def printFT(formula, filename, dataObjectName, typeSolver):
  """
    Method designed to print on file the generated fault tree in Open PSA format
    @ In,  formula, list, list containing the terms of the generated  Boolean expression
    @ In,  filename, string, name of the xml file that will contain the fault tree in Open PSA format
    @ In,  dataObjectName, string, name of the original PointSet
    @ In,  typeSolver, string, type of solver that has been used to generate the Boolean expression
    @ Out, None, 
    """
  # 1) Create FT structure
  root = ET.Element('opsa-mef')
  
  FT = ET.SubElement(root, 'FT_' + dataObjectName)
  
  mainOR = ET.SubElement(FT, "define-gate", name="TOP")
  if typeSolver=='sop':
    orGate = ET.SubElement(mainOR, 'or')
  else:
    orGate = ET.SubElement(mainOR, 'and')
  
  for index,elem in enumerate(formula):
    if len(elem)==1:
      ET.SubElement(orGate, 'basic-event', name=str(elem[0]))
      mainBE = ET.SubElement(FT, "define-basic-event", name=str(elem[0]))
      ET.SubElement(mainBE, "float", value='1.0')
    else:
      gateID = "G"+str(index)
      ET.SubElement(orGate, "gate", name=gateID)
      elementGate = ET.SubElement(FT, "define-gate", name=gateID)
      
      if typeSolver=='sop':
        elementAndGate = ET.SubElement(elementGate, 'and')
      else:
        elementAndGate = ET.SubElement(elementGate, 'or')
      
      for term in elem:
        if "~" in term:
          notGate = ET.SubElement(elementAndGate, 'not')
          ET.SubElement(notGate, 'basic-event', name=str(term))     
        else:
          ET.SubElement(elementAndGate, 'basic-event', name=str(term))
          
        mainBE = ET.SubElement(FT, "define-basic-event", name=str(term))
        ET.SubElement(mainBE, "float", value='1.0')
  
  # 2) Print FT on file
  filename = filename + ".xml"
  open(filename, "w").writelines(xmlUtils.prettify(root))
        
      
    
      
  
  
  