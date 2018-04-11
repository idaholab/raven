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
Created on Feb 16, 2013

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import itertools
import numpy as np
import os
import copy
from scipy import spatial
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils.cached_ndarray import c1darray
from .Data import Data, NotConsistentData, ConstructError
import Files
from utils import utils,mathUtils
#Internal Modules End--------------------------------------------------------------------------------

class PointSet(Data):
  """
    PointSet is an object that stores multiple sets of inputs and outputs for a particular point in time!
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Data.__init__(self)

  def addSpecializedReadingSettings(self):
    """
      This function adds in the _dataParameters dict the options needed for reading and constructing this class
      @ In, None
      @ Out, None
    """
    # if hierarchical fashion has been requested, we set the type of the reading to a Point,
    #  since a PointSet in hierarchical fashion would be a tree of Points
    if self._dataParameters['hierarchical']:
      self._dataParameters['type'] = 'Point'
    # store the type into the _dataParameters dictionary
    else:
      self._dataParameters['type'] = self.type
    if hasattr(self._toLoadFromList[-1],'type'):
      sourceType = self._toLoadFromList[-1].type
    else:
      sourceType = None
    if('HDF5' == sourceType):
      self._dataParameters['type']       = self.type
      self._dataParameters['HistorySet'] = self._toLoadFromList[-1].getEndingGroupNames()
      self._dataParameters['filter'   ]  = 'whole'

  def checkConsistency(self):
    """
      Here we perform the consistency check for the structured data PointSet
      @ In, None
      @ Out, None
    """
    #The lenMustHave is a counter of the HistorySet contained in the
    #toLoadFromList list. Since this list can contain either CSVfiles
    #and HDF5, we can not use "len(_toLoadFromList)" anymore. For
    #example, if that list contains 10 csvs and 1 HDF5 (with 20
    #HistorySet), len(toLoadFromList) = 11 but the number of HistorySet
    #is actually 30.
    lenMustHave = self.numAdditionalLoadPoints
    sourceType = self._toLoadFromList[-1].type
    # here we assume that the outputs are all read....so we need to compute the total number of time point sets
    if'HDF5' == self._toLoadFromList[-1].type:
      for key in self._dataContainer['inputs'].keys():
        if (self._dataContainer['inputs'][key].size) != lenMustHave:
          self.raiseAnError(NotConsistentData,'The input parameter value, for key ' + key + ' has not a consistent shape for PointSet ' + self.name + '!! It should be an array of size ' + str(lenMustHave) + '.Actual size is ' + str(self._dataContainer['inputs'][key].size))
      for key in self._dataContainer['outputs'].keys():
        if (self._dataContainer['outputs'][key].size) != lenMustHave:
          self.raiseAnError(NotConsistentData,'The output parameter value, for key ' + key + ' has not a consistent shape for PointSet ' + self.name + '!! It should be an array of size ' + str(lenMustHave) + '.Actual size is ' + str(self._dataContainer['outputs'][key].size))
    else:
      if self._dataParameters['hierarchical']:
        for key in self._dataContainer['inputs'].keys():
          if (self._dataContainer['inputs'][key].size) != 1:
            self.raiseAnError(NotConsistentData,'The input parameter value, for key ' + key + ' has not a consistent shape for PointSet ' + self.name + '!! It should be a single value since we are in hierarchical mode.' + '.Actual size is ' + str(self._dataContainer['inputs'][key].size))
        for key in self._dataContainer['outputs'].keys():
          if (self._dataContainer['outputs'][key].size) != 1:
            self.raiseAnError(NotConsistentData,'The output parameter value, for key ' + key + ' has not a consistent shape for PointSet ' + self.name + '!! It should be a single value since we are in hierarchical mode.' + '.Actual size is ' + str(self._dataContainer['outputs'][key].size))
      else:
        for key in self._dataContainer['inputs'].keys():
          if (self._dataContainer['inputs'][key].size) != lenMustHave:
            self.raiseAnError(NotConsistentData,'The input parameter value, for key ' + key + ' has not a consistent shape for PointSet ' + self.name + '!! It should be an array of size ' + str(lenMustHave) + '.Actual size is ' + str(self._dataContainer['inputs'][key].size))
        for key in self._dataContainer['outputs'].keys():
          if (self._dataContainer['outputs'][key].size) != lenMustHave:
            self.raiseAnError(NotConsistentData,'The output parameter value, for key ' + key + ' has not a consistent shape for PointSet ' + self.name + '!! It should be an array of size ' + str(lenMustHave) + '.Actual size is ' + str(self._dataContainer['outputs'][key].size))

  def _updateSpecializedInputValue(self,name,value,options=None):
    """
      This function performs the updating of the values (input space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (single value)
      @ Out, None
    """

    ## So, you are trying to update a single data point, but you passed in
    ## more information, this means we need to reduce it down using one of our
    ## recipes, but only if it is not an unstructured input. Please refer
    ## questions about unstructured input to alfoa. -- DPM 8/28/17
    reducedValue = value = np.atleast_1d(value).flatten()
    if len(value) > 1:
      row = -1
      if self._dataParameters is not None:
        row = self._dataParameters.get('inputRow', -1)
      reducedValue = value[row]
      ## We don't have access to the pivot parameter's information at this
      ## point, so I will forego this implementation for now -- DPM 5/3/2017
      #else:
      #  value = interp1d(data[:,pivotIndex], value, kind='linear')(outputPivotVal)

    # if this flag is true, we accept realizations in the input space that are not only scalar but can be 1-D arrays!
    #acceptArrayRealizations = False if options == None else options.get('acceptArrayRealizations',False)
    unstructuredInput = False
    if isinstance(value,(np.ndarray,c1darray)):
      if np.asarray(value).ndim > 1 and max(np.asarray(value).shape) != np.asarray(value).size:
        self.raiseAnError(NotConsistentData,'PointSet Data accepts only a 1 Dimensional numpy array or a single value for method <_updateSpecializedInputValue>. Array shape is ' + str(value.shape))
      #if value.size != 1 and not acceptArrayRealizations: self.raiseAnError(NotConsistentData,'PointSet Data accepts only a numpy array of dim 1 or a single value for method <_updateSpecializedInputValue>. Size is ' + str(value.size))
      unstructuredInput = True if value.size > 1 else False
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'Point' has been stored
      parentID = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parentID' in options['metadata'].keys():
          parentID = options['metadata']['parentID']
      else:
        prefix    = options['prefix']
        if 'parentID' in options.keys():
          parentID = options['parentID']
      if parentID:
        tsnode = self.retrieveNodeInTreeMode(prefix,parentID)
      else:
        tsnode = self.retrieveNodeInTreeMode(prefix)
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'unstructuredInputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if name in self._dataContainer['inputs'].keys():
        self._dataContainer['inputs'].pop(name)
      if name in self._dataContainer['unstructuredInputs'].keys():
        self._dataContainer['unstructuredInputs'].pop(name)
      if name not in self._dataParameters['inParam']:
        self._dataParameters['inParam'].append(name)
      if not unstructuredInput:
        self._dataContainer['inputs'][name]             = c1darray(values=np.atleast_1d(np.ravel(reducedValue)[-1]))
      else:
        self._dataContainer['unstructuredInputs'][name] = [c1darray(values=np.atleast_1d(np.ravel(value)))]
      #self._dataContainer['inputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)[-1])) if not acceptArrayRealizations else c1darray(values=np.atleast_1d(np.atleast_1d(value)))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in itertools.chain(self._dataContainer['inputs'].keys(),self._dataContainer['unstructuredInputs'].keys()):
        #popped = self._dataContainer['inputs'].pop(name)
        if not unstructuredInput:
          self._dataContainer['inputs'][name].append(np.atleast_1d(np.ravel(reducedValue)[-1]))
        else:
          self._dataContainer['unstructuredInputs'][name].append(np.atleast_1d(np.ravel(value)))
        #self._dataContainer['inputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)[-1]))                     copy.copy(np.concatenate((np.atleast_1d(np.array(popped)), np.atleast_1d(np.atleast_1d(value)[-1]))))
      else:
        if name not in self._dataParameters['inParam']:
          self._dataParameters['inParam'].append(name)
        #if name not in self._dataParameters['inParam']: self.raiseAnError(NotConsistentData,'The input variable '+name+'is not among the input space of the DataObject '+self.name)
        #self._dataContainer['inputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)[-1])) if not acceptArrayRealizations else c1darray(values=np.atleast_1d(np.atleast_1d(value)))
        if not unstructuredInput:
          self._dataContainer['inputs'][name]             = c1darray(values=np.atleast_1d(np.ravel(reducedValue)[-1]))
        else:
          self._dataContainer['unstructuredInputs'][name] = [c1darray(values=np.atleast_1d(np.ravel(value)))]
        #self._dataContainer['inputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)[-1])) if not acceptArrayRealizations else c1darray(values=np.atleast_1d(np.atleast_1d(value)))

  def _updateSpecializedMetadata(self,name,value,valueType,options=None):
    """
      This function performs the updating of the values (metadata) into this Data
      @ In, name, string, parameter name (ex. probability)
      @ In, value, object, newer value
      @ In, valueType, dtype, the value type
      @ Out, None
      NB. This method, if the metadata name is already present, replaces it with the new value. No appending here, since the metadata are dishomogenius and a common updating strategy is not feasable.
    """
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'Point' has been stored
      parentID = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parentID' in options['metadata'].keys():
          parentID = options['metadata']['parentID']
      else:
        prefix    = options['prefix']
        if 'parentID' in options.keys():
          parentID = options['parentID']
      if parentID:
        tsnode = self.retrieveNodeInTreeMode(prefix,parentID)
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'metadata':{}})
        self._dataContainer = tsnode.get('dataContainer')
      else:
        if 'metadata' not in self._dataContainer.keys():
          self._dataContainer['metadata'] ={}
      if name in self._dataContainer['metadata'].keys():
        self._dataContainer['metadata'][name].append(np.atleast_1d(value)) # = np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value)))
      else:
        valueToAdd = np.array(value,dtype=valueType) if valueType is not None else np.array(value)
        self._dataContainer['metadata'][name] = c1darray(values=np.atleast_1d(valueToAdd))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['metadata'].keys():
        self._dataContainer['metadata'][name].append(np.atleast_1d(value)) # = np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value)))
      else:
        valueToAdd = np.array(value,dtype=valueType) if valueType is not None else np.array(value)
        self._dataContainer['metadata'][name] = c1darray(values=np.atleast_1d(valueToAdd))

  def _updateSpecializedOutputValue(self,name,value,options=None):
    """
      This function performs the updating of the values (output space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (single value)
      @ Out, None
    """
    ## So, you are trying to update a single data point, but you passed in
    ## more information, this means we need to reduce it down using one of our
    ## recipes.
    value = np.atleast_1d(value).flatten()
    if len(value) > 1:

      row = -1
      if self._dataParameters is not None:
        row = self._dataParameters.get('outputRow', -1)
        operator = self._dataParameters.get('operator', None)
      if operator == 'max':
        value = np.max(value)
      elif operator == 'min':
        value = np.min(value)
      elif operator == 'average':
        value = np.average(value)
      else: #elif outputRow is not None:
        value = value[row]
      ## We don't have access to the pivot parameter's information at this
      ## point, so I will forego this implementation for now -- DPM 5/3/2017
      #else:
      #  value = interp1d(data[:,pivotIndex], value, kind='linear')(outputPivotVal)

    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'Point' has been stored
      parentID = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parentID' in options['metadata'].keys():
          parentID = options['metadata']['parentID']
      else:
        prefix    = options['prefix']
        if 'parentID' in options.keys():
          parentID = options['parentID']
      if parentID:
        tsnode = self.retrieveNodeInTreeMode(prefix,parentID)

      #if 'parentID' in options.keys(): tsnode = self.retrieveNodeInTreeMode(options['prefix'], options['parentID'])
      #else:                             tsnode = self.retrieveNodeInTreeMode(options['prefix'])
      # we store the pointer to the container in the self._dataContainer because checkConsistency acts on this
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if name in self._dataContainer['outputs'].keys():
        self._dataContainer['outputs'].pop(name)
      if name not in self._dataParameters['outParam']:
        self._dataParameters['outParam'].append(name)
      self._dataContainer['outputs'][name] = c1darray(values=np.atleast_1d(value)) #np.atleast_1d(np.atleast_1d(value)[-1])
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['outputs'].keys():
        #popped = self._dataContainer['outputs'].pop(name)
        self._dataContainer['outputs'][name].append(np.atleast_1d(value)[-1])   #= copy.copy(np.concatenate((np.array(popped), np.atleast_1d(np.atleast_1d(value)[-1]))))
      else:
        if name not in self._dataParameters['outParam']:
          self._dataParameters['outParam'].append(name)
        self._dataContainer['outputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)[-1])) # np.atleast_1d(np.atleast_1d(value)[-1])

  def specializedPrintCSV(self,filenameLocal,options):
    """
      This function prints a CSV file with the content of this class (Input and Output space)
      @ In,  filenameLocal, string, filename root (for example, 'homo_homini_lupus' -> the final file name is gonna be called 'homo_homini_lupus.csv')
      @ In,  options, dict, dictionary of printing options
      @ Out, None (a csv is gonna be printed)
    """
    inpKeys               = []
    inpValues             = []
    unstructuredInpKeys   = []
    unstructuredInpValues = []
    outKeys               = []
    outValues             = []
    #Print input values
    if self._dataParameters['hierarchical']:
      # retrieve a serialized of DataObjects from the tree
      O_o = self.getHierParam('inout','*',serialize=True)
      for key in O_o.keys():
        inpKeys.append([])
        inpValues.append([])
        outKeys.append([])
        outValues.append([])
        if 'what' in options.keys():
          for var in options['what']:
            splitted = var.split('|')
            variableName = "|".join(splitted[1:])
            varType = splitted[0]
            if varType == 'input':
              inpKeys[-1].append(variableName)
              axa = np.zeros(len(O_o[key]))
              for index in range(len(O_o[key])):
                axa[index] = O_o[key][index]['inputs'][variableName][0]
              inpValues[-1].append(axa)
            if varType == 'output':
              outKeys[-1].append(variableName)
              axa = np.zeros(len(O_o[key]))
              for index in range(len(O_o[key])):
                axa[index] = O_o[key][index]['outputs'][variableName][0]
              outValues[-1].append(axa)
            if varType == 'metadata':
              inpKeys[-1].append(variableName)
              if type(O_o[key][index]['metadata'][splitted[1]]) not in self.metatype:
                self.raiseAnError(NotConsistentData,'metadata '+variableName +' not compatible with CSV output. Its type needs to be one of '+str(np.ndarray))
                axa = np.zeros(len(O_o[key]))
                for index in range(len(O_o[key])):
                  axa[index] = np.atleast_1d(np.float(O_o[key][index]['metadata'][variableName]))[0]
                inpValues[-1].append(axa)
        else:
          inpKeys[-1] = O_o[key][0]['inputs'].keys()
          for var in inpKeys[-1]:
            axa = np.zeros(len(O_o[key]))
            for index in range(len(O_o[key])):
              axa[index] = O_o[key][index]['inputs'][var][0]
            inpValues[-1].append(axa)
          outKeys[-1] = O_o[key][0]['outputs'].keys()
          for var in outKeys[-1]:
            axa = np.zeros(len(O_o[key]))
            for index in range(len(O_o[key])):
              axa[index] = O_o[key][index]['outputs'][var][0]
            outValues[-1].append(axa)
      if len(inpKeys[-1]) > 0 or len(outKeys[-1]) > 0:
        myFile = open(filenameLocal + '.csv', 'w')
      else:
        return
      O_o_keys = list(O_o.keys())
      for index in range(len(O_o.keys())):
        myFile.write('Ending branch,'+O_o_keys[index]+'\n')
        myFile.write('branch #')
        for item in inpKeys[index]:
          myFile.write(',' + item)
        for item in outKeys[index]:
          myFile.write(',' + item)
        myFile.write('\n')
        # maljdan: Generalized except caught
        try:
          sizeLoop = outValues[index][0].size
        except:
          sizeLoop = inpValues[index][0].size
        for j in range(sizeLoop):
          myFile.write(str(j+1))
          for i in range(len(inpKeys[index])):
            myFile.write(',' + str(inpValues[index][i][j]))
          for i in range(len(outKeys[index])):
            myFile.write(',' + str(outValues[index][i][j]))
          myFile.write('\n')
      myFile.close()
    else:
      #If not hierarchical
      #For Pointset it will create an XML file and one CSV file.
      #The CSV file will have a header with the input names and output
      #names, and multiple lines of data with the input and output
      #numeric values, one line for each input.
      if 'what' in options.keys():
        for var in options['what']:
          splitted = var.split('|')
          variableName = "|".join(splitted[1:])
          varType = splitted[0]
          if varType == 'input':
            if variableName not in self.getParaKeys('input'):
              self.raiseAnError(Exception,"variable named "+variableName+" is not among the "+varType+"s!")
            if variableName in self._dataContainer['inputs'].keys():
              inpKeys.append(variableName)
              inpValues.append(self._dataContainer['inputs'][variableName])
            else:
              unstructuredInpKeys.append(variableName)
              unstructuredInpValues.append(self._dataContainer['unstructuredInputs'][variableName])
          if varType == 'output':
            if variableName not in self.getParaKeys('output'):
              self.raiseAnError(Exception,"variable named "+variableName+" is not among the "+varType+"s!")
            outKeys.append(variableName)
            outValues.append(self._dataContainer['outputs'][variableName])
          if varType == 'metadata':
            inpKeys.append(variableName)
            inpValues.append(self._dataContainer['metadata'][variableName])
      else:
        unstructuredInpKeys   = sorted(self._dataContainer['unstructuredInputs'].keys())
        unstructuredInpValues = [self._dataContainer['unstructuredInputs'][var] for var in unstructuredInpKeys]
        inpKeys   = self._dataContainer['inputs'].keys()
        inpValues = self._dataContainer['inputs'].values()
        outKeys   = self._dataContainer['outputs'].keys()
        outValues = self._dataContainer['outputs'].values()
      if len(inpKeys) > 0 or len(outKeys) > 0:
        myFile = open(filenameLocal + '.csv', 'w')
      else:
        return
      if len(unstructuredInpKeys) > 0:
        filteredUnstructuredInpKeys   = [unstructuredInpKeys]*len(unstructuredInpValues[0])
        filteredUnstructuredInpValues = [[unstructuredInpValues[cnt][histNum] for cnt in range(len(unstructuredInpValues))] for histNum in range(len(unstructuredInpValues[0])) ]
      #Print header
      myFile.write(','.join([str(item) for item in itertools.chain(inpKeys,outKeys)]))
      myFile.write('\n')
      #Print values
      for j in range(len(next(iter(itertools.chain(inpValues,outValues))))):
        #myFile.write(','.join(['{:.17f}'.format(item[j]) for item in itertools.chain(inpValues,outValues)]))
        #str(item) can truncate the accuracy of the value.  However, we've lost that truncation before this point...
        #  ...as shown by the line above.
        myFile.write(','.join([str(item[j]) for item in itertools.chain(inpValues,outValues)]))
        myFile.write('\n')
      myFile.close()
      self._createXMLFile(filenameLocal,'Pointset',inpKeys,outKeys)
      if len(unstructuredInpKeys) > 0:
        # write unstructuredData
        self._writeUnstructuredInputInXML(filenameLocal +'_unstructured_inputs',filteredUnstructuredInpKeys,filteredUnstructuredInpValues)

  def _specializedLoadXMLandCSV(self, filenameRoot, options):
    """
      Function to load the xml additional file of the csv for data
      (it contains metadata, etc). It must be implemented by the specialized classes
      @ In, filenameRoot, string, file name root
      @ In, options, dict, dictionary -> options for loading
      @ Out, None
    """
    #For Pointset it will create an XML file and one CSV file.
    #The CSV file will have a header with the input names and output
    #names, and multiple lines of data with the input and output
    #numeric values, one line for each input.
    if options is not None and 'fileToLoad' in options.keys():
      name = os.path.join(options['fileToLoad'].getPath(),options['fileToLoad'].getBase())
    else:
      name = self.name

    filenameLocal = os.path.join(filenameRoot,name)

    if os.path.isfile(filenameLocal+'.xml'):
      xmlData = self._loadXMLFile(filenameLocal)
      assert(xmlData["fileType"] == "Pointset")
      if "metadata" in xmlData:
        self._dataContainer['metadata'] = xmlData["metadata"]

      mainCSV = os.path.join(filenameRoot,xmlData["filenameCSV"])
    else:
      mainCSV = os.path.join(filenameRoot,name+'.csv')

    myFile = open(mainCSV,"rU")
    header = myFile.readline().rstrip()
    inoutKeys = header.split(",")
    inoutValues = [[] for _ in range(len(inoutKeys))]

    for lineNumber,line in enumerate(myFile.readlines(),2):
      lineList = line.rstrip().split(",")
      for i in range(len(inoutKeys)):
        datum = utils.partialEval(lineList[i])
        if datum == '':
          self.raiseAnError(IOError, 'Invalid data in input file: {} at line {}: "{}"'.format(filenameLocal, lineNumber, line.rstrip()))
        inoutValues[i].append(utils.partialEval(lineList[i]))

    #extend the expected size of this point set
    self.numAdditionalLoadPoints += len(inoutValues[0]) #used in checkConsistency

    ## Do not reset these containers because it will wipe whatever information
    ## already exists in this PointSet. This is not one of the use cases for our
    ## data objects. We claim in the manual to construct or update information.
    ## These should be non-destructive operations. -- DPM 6/26/17
    # self._dataContainer['inputs'] = {}
    # self._dataContainer['outputs'] = {}
    inoutDict = {}
    for key,value in zip(inoutKeys,inoutValues):
      inoutDict[key] = value

    for key in self.getParaKeys('inputs'):
      ## Again, in order to be non-destructive we should only initialize on the
      ## first go-around, subsequent loads should append to the existing list.
      ## -- DPM 6/26/17
      if key not in self._dataContainer["inputs"]:
        self._dataContainer["inputs"][key] = c1darray(values=np.array(inoutDict[key]))
      else:
        self._dataContainer["inputs"][key].append(c1darray(values=np.array(inoutDict[key])))

    for key in self.getParaKeys('outputs'):
      ## Again, in order to be non-destructive we should only initialize on the
      ## first go-around, subsequent loads should append to the existing list.
      ## -- DPM 6/26/17
      if key not in self._dataContainer["outputs"]:
        self._dataContainer["outputs"][key] = c1darray(values=np.array(inoutDict[key]))
      else:
        self._dataContainer["outputs"][key].append(c1darray(values=np.array(inoutDict[key])))

  def _constructKDTree(self,requested):
    """
      Constructs a KD tree consisting of the variable values in "requested"
      @ In, requested, list, requested variable names
      @ Out, None
    """
    #set up data scaling, so that relative distances are used
    # scaling is so that scaled = (actual - mean)/scale
    inpVals = self.getParametersValues('inputs')
    floatVars = list(r for r in requested if r in inpVals.keys())
    for v in floatVars:
      mean,scale = mathUtils.normalizationFactors(inpVals[v])
      self.treeScalingFactors[v] = (mean,scale)
    #convert data into a matrix in the order of requested
    data = np.dstack(tuple((np.array(inpVals[v])-self.treeScalingFactors[v][0])/self.treeScalingFactors[v][1] for v in floatVars))[0]
    self.inputKDTree = spatial.KDTree(data)
