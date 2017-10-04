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

Please note the methods in this file have been alphabetized and clumped according
to their scope (public,protected,private), please do your best to preserve this
order when adding new methods
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import ast
import copy
import numpy as np
import itertools
from scipy import spatial
import xml.etree.ElementTree as ET
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from utils.cached_ndarray import c1darray
from CsvLoader import CsvLoader as ld
import Files
import utils.TreeStructure as TS
from utils import utils
from utils import mathUtils
from utils import InputData
#Internal Modules End--------------------------------------------------------------------------------

# Custom exceptions
class NotConsistentData(Exception):
  """
    Custom exception class for NotConsistentData
  """
  pass
class ConstructError(Exception):
  """
    Custom exception class for ConstructError
  """
  pass

class Data(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  """
    The Data object is the base class for constructing derived data object classes.
    It provides the common interfaces to access and to add data values into the RAVEN internal object format.
    This object is "understood" by all the "active" modules (e.g. postprocessors, models, etc) and represents the way
    RAVEN shares the information among the framework
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Data, cls).getInputSpecification()
    inputSpecification.addParam("hierarchical", InputData.StringType)
    inputSpecification.addParam("inputTs", InputData.StringType)
    inputSpecification.addParam("historyName", InputData.StringType)

    inputInput = InputData.parameterInputFactory("Input", contentType=InputData.StringListType)

    inputSpecification.addSub(inputInput)

    outputInput = InputData.parameterInputFactory("Output", contentType=InputData.StringListType)

    inputSpecification.addSub(outputInput)

    optionsInput = InputData.parameterInputFactory("options")
    for option in ['inputRow','inputPivotValue','outputRow','outputPivotValue','operator','pivotParameter']:
      optionSubInput = InputData.parameterInputFactory(option, contentType=InputData.StringType)
      optionsInput.addSub(optionSubInput)
    inputSpecification.addSub(optionsInput)

    return inputSpecification

  ## Special Overloaded Methods

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)
    self.numAdditionalLoadPoints         = 0                          # if points are loaded into csv, this will help keep tally
    self._dataParameters                 = {}                         # in here we store all the data parameters (inputs params, output params,etc)
    self._dataParameters['inParam'     ] = []                         # inParam list
    self._dataParameters['outParam'    ] = []                         # outParam list
    self._dataParameters['hierarchical'] = False                      # the structure of this data is hierarchical?
    self._dataParameters['typeMetadata'] = {}                         # store the type of the metadata in order to accellerate the collecting
    self._toLoadFromList                 = []                         # loading source
    self._dataContainer                  = {'inputs':{},'unstructuredInputs':{}, 'outputs':{}} # Dict that contains the actual data: self._dataContainer['inputs'] contains the input space in scalar form. self._dataContainer['output'] the output space
    #self._unstructuredInputContainer     = {}                         # Dict that contains the input space in unstrctured form (e.g. 1-D arrays)
    self._dataContainer['metadata'     ] = {}                         # In this dictionary we store metadata (For example, probability,input file names, etc)
    self.metaAdditionalInOrOut           = ['PointProbability','ProbabilityWeight']            # list of metadata keys that will be printed in the CSV one
    self.inputKDTree                     = None                       # KDTree for speedy querying of input space
    self.treeScalingFactors = {}                                      # dictionary of means, scaling factors for KDTree searches
    self.notAllowedInputs  = []                                       # this is a list of keyword that are not allowed as Inputs
    self.notAllowedOutputs = []                                       # this is a list of keyword that are not allowed as Outputs
    # This is a list of metadata types that are CSV-compatible...we build the list this way to catch when a python implementation doesn't
    #   have some type or another (ie. Windows doesn't have np.float128, but does have np.float96)
    self.metatype = []
    for typeString in ["float","bool","int","np.ndarray","np.float16","np.float32","np.float64","np.float96","np.float128",
                       "np.int16","np.int32","np.int64","np.bool8","c1darray"]:
      try:
        self.metatype.append(eval(typeString))  # eval turns the string into the internal type
      except AttributeError:
        # Catches the type not being defined somewhere
        pass
    self.type = self.__class__.__name__
    self.printTag  = 'DataObjects'

  def __len__(self):
    """
      Overriding of the __len__ method for data.
      len(dataobject) is going to return the size of the first output element found in the self._dataParameters['outParams']
      @ In, None
      @ Out, __len__, integer, size of first output element
    """
    if len(self._dataParameters['outParam']) == 0:
      return 0
    else:
      return self.sizeData()

  ## Public Methods

  def addOutput(self,toLoadFrom,options=None):
    """
      Function to construct a data from a source
      @ In, toLoadFrom, string, loading source, it can be an HDF5 database, a csv file and in the future a xml file
      @ In, options, dict, optional, it is a dictionary of options. For example useful for metadata storing or,
                     in case an hierarchical fashion has been requested, it must contain the parentID and the name of the actual 'branch'
      @ Out, None
    """
    self._toLoadFromList.append(toLoadFrom)
    self.addSpecializedReadingSettings()
    self._dataParameters['SampledVars'] = copy.deepcopy(options['metadata']['SampledVars']) if options != None and 'metadata' in options.keys() and 'SampledVars' in options['metadata'].keys() else None
    if options is not None and 'alias' in options.keys():
      self._dataParameters['alias'] = options['alias']
    self.raiseAMessage('Object type ' + self._toLoadFromList[-1].type + ' named "' + self._toLoadFromList[-1].name+'"')
    if(self._toLoadFromList[-1].type == 'HDF5'):
      tupleVar = self._toLoadFromList[-1].retrieveData(self._dataParameters)
      if options:
        parentID = options['metadata']['parentID'] if 'metadata' in options.keys() and 'parentID' in options['metadata'].keys() else (options['parentID'] if 'parentID' in options.keys() else None)
        if parentID and self._dataParameters['hierarchical']:
          self.raiseAWarning('-> Data storing in hierarchical fashion from HDF5 not yet implemented!')
          self._dataParameters['hierarchical'] = False
      self.numAdditionalLoadPoints += len(self._toLoadFromList[-1].getEndingGroupNames())
    elif (isinstance(self._toLoadFromList[-1],Files.File)):
      tupleVar = ld(self.messageHandler).csvLoadData([toLoadFrom],self._dataParameters)
      self.numAdditionalLoadPoints += 1
    else:
      self.raiseAnError(ValueError, "Type "+self._toLoadFromList[-1].type+ "from which the DataObject "+ self.name +" should be constructed is unknown!!!")

    for hist in tupleVar[0].keys():
      if type(tupleVar[0][hist]) == dict:
        for key in tupleVar[0][hist].keys():
          self.updateInputValue(key, tupleVar[0][hist][key], options)
      else:
        if self.type in ['PointSet']:
          for index in range(len(tupleVar[0][hist])):
            if hist in self.getParaKeys('input'):
              self.updateInputValue(hist, tupleVar[0][hist][index], options)
        else:
          self.updateInputValue(hist, tupleVar[0][hist], options)
    for hist in tupleVar[1].keys():
      if type(tupleVar[1][hist]) == dict:
        for key in tupleVar[1][hist].keys():
          self.updateOutputValue(key, tupleVar[1][hist][key], options)
      else:
        if self.type in ['PointSet']:
          for index in range(np.asarray(tupleVar[1][hist]).size):
            if hist in self.getParaKeys('output'):
              self.updateOutputValue(hist, tupleVar[1][hist][index], options)
        else:
          self.updateOutputValue(hist, tupleVar[1][hist], options)
    if len(tupleVar) > 2:
      #metadata
      for hist in tupleVar[2].keys():
        if type(tupleVar[2][hist]) == list:
          for element in tupleVar[2][hist]:
            if type(element) == dict:
              for key,value in element.items():
                self.updateMetadata(key, value, options)
        elif type(tupleVar[2][hist]) == dict:
          for key,value in tupleVar[2][hist].items():
            if value:
              for elem in value:
                if type(elem) == dict:
                  for ke ,val  in elem.items():
                    self.updateMetadata(ke, val, options)
                else:
                  self.raiseAnError(IOError,'unknown type for metadata adding process. Relevant type = '+ str(elem))

        else:
          if tupleVar[2][hist]:
            self.raiseAnError(IOError,'unknown type for metadata adding process. Relevant type = '+ str(type(tupleVar[2][hist])))
    self.checkConsistency()

  @abc.abstractmethod
  def addSpecializedReadingSettings(self):
    """
      This function is used to add specialized attributes to the data in order to retrieve the data properly.
      Every specialized data needs to overwrite it!!!!!!!!
      @ In, None
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def checkConsistency(self):
    """
      This function checks the consistency of the data structure... every specialized data needs to overwrite it!!!!!
      @ In, None
      @ Out, None
    """
    pass

  def getAllMetadata(self,nodeId=None,serialize=False):
    """
      Function to get all the metadata
      @ In, nodeId, string, optional, id of the node if hierarchical
      @ In, serialize, bool, optional, serialize the tree if in hierarchical mode
      @ Out, dictionary, dict, return the metadata dictionary
    """
    if self._dataParameters['hierarchical']:
      return self.getHierParam('metadata',nodeId,None,serialize)
    else:
      return self._dataContainer['metadata']

  def getHierParam(self,typeVar,nodeId,keyword=None,serialize=False):
    """
      This function get a parameter when we are in hierarchical mode
      @ In,  typeVar,  string, it's the variable type... input,unstructuredInput,output, or inout
      @ In,  nodeId,   string, it's the node name... if == None or *, a dictionary of of data is returned, otherwise the actual node data is returned in a dict as well (see serialize attribute)
      @ In, keyword,   string, it's a parameter name (for example, cladTemperature), if None, the whole dict is returned, otherwise the parameter value is got (see serialize attribute)
      @ In, serialize, bool  , if true a sequence of PointSet is generated (a dictionary where the keys are the 'ending' branches and the values are a sorted list of _dataContainers (from first branch to the ending ones)
                               if false see explanation for nodeId
      @ Out, nodesDict, dict, a dictionary of data (see above)
    """
    if type(keyword).__name__ in ['str','unicode','bytes']:
      if keyword == 'none':
        keyword = None
    nodesDict = {}
    if not self.TSData:
      return nodesDict
    if not nodeId or nodeId=='*':
      # we want all the nodes
      if serialize:
        # we want all the nodes and serialize them
        for TSData in self.TSData.values():
          for node in TSData.iterEnding():
            nodesDict[node.name] = []
            for se in list(TSData.iterWholeBackTrace(node)):
              if typeVar in 'inout' and not keyword:
                nodesDict[node.name].append(se.get('dataContainer'))
              elif typeVar in ['inputs','input'] and not keyword:
                nodesDict[node.name].append(se.get('dataContainer')['inputs'])
              elif typeVar in ['unstructuredInput','unstructuredInputs'] and not keyword:
                nodesDict[node.name].append(se.get('dataContainer')['unstructuredInputs'])
              elif typeVar in ['output','outputs'] and not keyword:
                nodesDict[node.name].append(se.get('dataContainer')['outputs'])
              elif typeVar in 'metadata' and not keyword:
                nodesDict[node.name].append(se.get('dataContainer')['metadata'])
              elif typeVar in ['inputs','input'] and keyword:
                nodesDict[node.name].append(np.asarray(se.get('dataContainer')['inputs'  ][keyword]))
              elif typeVar in ['unstructuredInput','unstructuredInputs'] and keyword:
                nodesDict[node.name].append(np.asarray(se.get('dataContainer')['unstructuredInputs'][keyword]))
              elif typeVar in ['output','outputs'] and keyword:
                nodesDict[node.name].append(np.asarray(se.get('dataContainer')['outputs' ][keyword]))
              elif typeVar in 'metadata' and keyword:
                nodesDict[node.name].append(np.asarray(se.get('dataContainer')['metadata'][keyword]))
      else:
        for TSData in self.TSData.values():
          for node in TSData.iter():
            if typeVar in 'inout' and not keyword:
              nodesDict[node.name] = node.get('dataContainer')
            elif typeVar in ['inputs','input'] and not keyword:
              nodesDict[node.name] = node.get('dataContainer')['inputs']
            elif typeVar in ['unstructuredInput','unstructuredInputs'] and not keyword:
              nodesDict[node.name] = node.get('dataContainer')['unstructuredInputs']
            elif typeVar in ['output','outputs'] and not keyword:
              nodesDict[node.name] = node.get('dataContainer')['outputs']
            elif typeVar in 'metadata' and not keyword:
              nodesDict[node.name] = node.get('dataContainer')['metadata']
            elif typeVar in ['inputs','input'] and keyword:
              nodesDict[node.name] = np.asarray(node.get('dataContainer')['inputs'][keyword])
            elif typeVar in ['unstructuredInput','unstructuredInputs'] and keyword:
              nodesDict[node.name] = np.asarray(node.get('dataContainer')['unstructuredInputs'][keyword])
            elif typeVar in ['output','outputs'] and keyword:
              nodesDict[node.name] = np.asarray(node.get('dataContainer')['outputs'][keyword])
            elif typeVar in 'metadata' and keyword:
              nodesDict[node.name] = np.asarray(node.get('dataContainer')['metadata'][keyword])
    elif nodeId == 'ending':
      for TSDat in self.TSData.values():
        for ending in TSDat.iterEnding():
          if typeVar in 'inout' and not keyword:
            nodesDict[ending.name] = ending.get('dataContainer')
          elif typeVar in ['inputs','input']   and not keyword:
            nodesDict[ending.name] = ending.get('dataContainer')['inputs']
          elif typeVar in ['unstructuredInput','unstructuredInputs'] and not keyword:
            nodesDict[ending.name] = ending.get('dataContainer')['unstructuredInputs']
          elif typeVar in ['output','outputs'] and not keyword:
            nodesDict[ending.name] = ending.get('dataContainer')['outputs']
          elif typeVar in 'metadata' and not keyword:
            nodesDict[ending.name] = ending.get('dataContainer')['metadata']
          elif typeVar in ['inputs','input'] and keyword:
            nodesDict[ending.name] = np.asarray(ending.get('dataContainer')['inputs'][keyword])
          elif typeVar in ['unstructuredInput','unstructuredInputs'] and     keyword:
            nodesDict[ending.name] = np.asarray(ending.get('dataContainer')['unstructuredInputs'][keyword])
          elif typeVar in ['output','outputs'] and keyword:
            nodesDict[ending.name] = np.asarray(ending.get('dataContainer')['outputs'][keyword])
          elif typeVar in 'metadata' and keyword:
            nodesDict[ending.name] = np.asarray(ending.get('dataContainer')['metadata'][keyword])
    elif nodeId == 'RecontructEnding':
      # if history, reconstruct the history... if Point set take the last one (see below)
      backTrace = {}
      for TSData in self.TSData.values():
        for node in TSData.iterEnding():
          if self.type == 'HistorySet':
            backTrace[node.name] = []
            for se in list(TSData.iterWholeBackTrace(node)):
              if typeVar in 'inout' and not keyword:
                backTrace[node.name].append(se.get('dataContainer'))
              elif typeVar in ['inputs','input'] and not keyword:
                backTrace[node.name].append(se.get('dataContainer')['inputs'])
              elif typeVar in ['unstructuredInput','unstructuredInputs'] and not keyword:
                backTrace[node.name].append(se.get('dataContainer')['unstructuredInputs'])
              elif typeVar in ['output','outputs'] and not keyword:
                backTrace[node.name].append(se.get('dataContainer')['outputs'])
              elif typeVar in 'metadata' and not keyword:
                backTrace[node.name].append(se.get('dataContainer')['metadata'])
              elif typeVar in ['inputs','input'] and keyword:
                backTrace[node.name].append(np.asarray(se.get('dataContainer')['inputs'][keyword]))
              elif typeVar in ['unstructuredInput','unstructuredInputs'] and keyword:
                backTrace[node.name].append(np.asarray(se.get('dataContainer')['unstructuredInputs'][keyword]))
              elif typeVar in ['output','outputs'] and keyword:
                backTrace[node.name].append(np.asarray(se.get('dataContainer')['outputs' ][keyword]))
              elif typeVar in 'metadata' and keyword:
                backTrace[node.name].append(np.asarray(se.get('dataContainer')['metadata'][keyword]))
            #reconstruct history
            nodesDict[node.name] = None
            for element in backTrace[node.name]:
              if type(element) == dict:
                if not nodesDict[node.name]:
                  nodesDict[node.name] = {}
                for innerkey in element.keys():
                  if type(element[innerkey]) == dict:
                    #inputs outputs metadata
                    if innerkey not in nodesDict[node.name].keys():
                      nodesDict[node.name][innerkey] = {}
                    for ininnerkey in element[innerkey].keys():
                      if ininnerkey not in nodesDict[node.name][innerkey].keys():
                        nodesDict[node.name][innerkey][ininnerkey] = element[innerkey][ininnerkey]
                      else:
                        nodesDict[node.name][innerkey][ininnerkey] = np.concatenate((nodesDict[node.name][innerkey][ininnerkey],element[innerkey][ininnerkey]))
                  else:
                    if innerkey not in nodesDict[node.name].keys():
                      nodesDict[node.name][innerkey] = np.atleast_1d(element[innerkey])
                    else:
                      nodesDict[node.name][innerkey] = np.concatenate((nodesDict[node.name][innerkey],element[innerkey]))
              else:
                # it is a value
                if not nodesDict[node.name]:
                  nodesDict[node.name] = element
                else:
                  nodesDict[node.name] = np.concatenate((nodesDict[node.name],element))
          else:
            #Pointset
            if typeVar in 'inout' and not keyword:
              backTrace[node.name] = node.get('dataContainer')
            elif typeVar in ['inputs','input'] and not keyword:
              backTrace[node.name] = node.get('dataContainer')['inputs']
            elif typeVar in ['unstructuredInput','unstructuredInputs']   and not keyword:
              backTrace[node.name] = node.get('dataContainer')['unstructuredInputs']
            elif typeVar in ['output','outputs'] and not keyword:
              backTrace[node.name] = node.get('dataContainer')['outputs']
            elif typeVar in 'metadata' and not keyword:
              backTrace[node.name] = node.get('dataContainer')['metadata']
            elif typeVar in ['inputs','input'] and keyword:
              backTrace[node.name] = np.asarray(node.get('dataContainer')['inputs'][keyword])
            elif typeVar in ['unstructuredInput','unstructuredInputs'] and     keyword:
              backTrace[node.name] = np.asarray(node.get('dataContainer')['unstructuredInputs'][keyword])
            elif typeVar in ['output','outputs'] and keyword:
              backTrace[node.name] = np.asarray(node.get('dataContainer')['outputs'][keyword])
            elif typeVar in 'metadata' and keyword:
              backTrace[node.name] = np.asarray(node.get('dataContainer')['metadata'][keyword])
            if type(backTrace[node.name]) == dict:
              for innerkey in backTrace[node.name].keys():
                if type(backTrace[node.name][innerkey]) == dict:
                  #inputs outputs metadata
                  if innerkey not in backTrace[node.name][innerkey].keys():
                    nodesDict[innerkey] = {}
                  for ininnerkey in backTrace[node.name][innerkey].keys():
                    if ininnerkey not in nodesDict[innerkey].keys():
                      nodesDict[innerkey][ininnerkey] = backTrace[node.name][innerkey][ininnerkey]
                    else:
                      nodesDict[innerkey][ininnerkey] = np.concatenate((nodesDict[innerkey][ininnerkey],backTrace[node.name][innerkey][ininnerkey]))
                else:
                  if innerkey not in nodesDict.keys():
                    nodesDict[innerkey] = np.atleast_1d(backTrace[node.name][innerkey])
                  else:
                    nodesDict[innerkey] = np.concatenate((nodesDict[innerkey],backTrace[node.name][innerkey]))
            else:
              #it is a value
              if type(nodesDict) == dict:
                nodesDict = np.empty(0)
              nodesDict = np.concatenate((nodesDict,backTrace[node.name]))
    else:
      # we want a particular node
      found = False
      for TSDat in self.TSData.values():
        #a = TSDat.iter(nodeId)
        #b = TSDat.iterWholeBackTrace(a)
        nodelist = []
        for node in TSDat.iter(nodeId):
          if serialize:
            for se in list(TSDat.iterWholeBackTrace(node)):
              nodelist.append(se)
          else:
            nodelist.append(node)
          break
        #nodelist = list(TSDat.iterWholeBackTrace(TSDat.iter(nodeId)[0]))
        if len(nodelist) > 0:
          found = True
          break
      if not found:
        self.raiseAnError(RuntimeError,'Starting node called '+ nodeId+ ' not found!')
      if serialize:
        # we want a particular node and serialize it
        nodesDict[nodeId] = []
        for se in nodelist:
          if typeVar in 'inout' and not keyword:
            nodesDict[node.name].append(se.get('dataContainer'))
          elif typeVar in ['inputs','input'] and not keyword:
            nodesDict[node.name].append(se.get('dataContainer')['inputs'])
          elif typeVar in ['unstructuredInput','unstructuredInputs'] and not keyword:
            nodesDict[node.name].append(se.get('dataContainer')['unstructuredInputs'])
          elif typeVar in ['output','outputs'] and not keyword:
            nodesDict[node.name].append(se.get('dataContainer')['outputs'])
          elif typeVar in 'metadata' and not keyword:
            nodesDict[node.name].append(se.get('dataContainer')['metadata'])
          elif typeVar in ['inputs','input'] and keyword:
            nodesDict[node.name].append(np.asarray(se.get('dataContainer')['inputs'][keyword]))
          elif typeVar in ['unstructuredInput','unstructuredInputs'] and keyword:
            nodesDict[node.name].append(np.asarray(se.get('dataContainer')['unstructuredInputs'][keyword]))
          elif typeVar in ['output','outputs'] and keyword:
            nodesDict[node.name].append(np.asarray(se.get('dataContainer')['outputs' ][keyword]))
          elif typeVar in 'metadata' and keyword:
            nodesDict[node.name].append(np.asarray(se.get('dataContainer')['metadata'][keyword]))
      else:
        if typeVar in 'inout' and not keyword:
          nodesDict[nodeId] = nodelist[-1].get('dataContainer')
        elif typeVar in ['inputs','input'] and not keyword:
          nodesDict[nodeId] = nodelist[-1].get('dataContainer')['inputs']
        elif typeVar in ['unstructuredInput','unstructuredInputs'] and not keyword:
          nodesDict[nodeId] = nodelist[-1].get('dataContainer')['unstructuredInputs']
        elif typeVar in ['output','outputs'] and not keyword:
          nodesDict[nodeId] = nodelist[-1].get('dataContainer')['outputs']
        elif typeVar in 'metadata' and not keyword:
          nodesDict[nodeId] = nodelist[-1].get('dataContainer')['metadata']
        elif typeVar in ['inputs','input'] and keyword:
          nodesDict[nodeId] = np.asarray(nodelist[-1].get('dataContainer')['inputs'][keyword])
        elif typeVar in ['unstructuredInput','unstructuredInputs'] and     keyword:
          nodesDict[nodeId] = np.asarray(nodelist[-1].get('dataContainer')['unstructuredInputs'][keyword])
        elif typeVar in ['output','outputs'] and keyword:
          nodesDict[nodeId] = np.asarray(nodelist[-1].get('dataContainer')['outputs'][keyword])
        elif typeVar in 'metadata' and keyword:
          nodesDict[nodeId] = np.asarray(nodelist[-1].get('dataContainer')['metadata'][keyword])
    return nodesDict

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for i,inParam in enumerate(self._dataParameters['inParam' ]):
      paramDict['Input_'+str(i)]  = inParam

    for i,outParam in enumerate(self._dataParameters['outParam']):
      paramDict['Output_'+str(i)] = outParam

    for key,value in self._dataParameters.items():
      paramDict[key] = value

    return paramDict

  def getInpParametersValues(self,nodeId=None,serialize=False,unstructuredInputs=False):
    """
      Function to get a reference to the input parameter dictionary
      @, In, nodeId, string, optional, in hierarchical mode, if nodeId is provided, the data for that node is returned,
                                  otherwise check explanation for getHierParam
      @ In, serialize, bool, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                  PLEASE check explanation for getHierParam
      @ In, unstructuredInputs, bool, optional, True if the unstructured input space needs to be returned
      @, Out, dictionary, dict, Reference to self._dataContainer['inputs'] or something else in hierarchical
    """
    if self._dataParameters['hierarchical']:
      return self.getHierParam('inputs' if not unstructuredInputs else 'unstructuredInput',nodeId,serialize=serialize)
    else:
      return self._dataContainer['inputs'] if not unstructuredInputs else self._dataContainer['unstructuredInputs']

  def getMatchingRealization(self,requested,tol=1e-15):
    """
      Finds first appropriate match within tolerance and returns it.
      @ In, requested, dict, {inName:inValue, inName:inValue}
      @ In, tol, float, relative tolerance with which to match
      @ Out, realization, dict, {'inputs':{inpName:value, inpName:value},'outputs':{outName:value, outName:value}} or None if not found
    """
    #if there's no entries, just return
    if len(self) < 1:
      return
    #check input spaces match
    #FIXME I don't like this fix.  Because of the Transformed space, our internal space includes latent variables, so we check subset.
    #  This is potentially flawed, though, in case you're taking points from a higher-dimension space!
    if not set(requested.keys()).issubset(set(self.getParaKeys('inputs'))):
      self.raiseADebug('Requested Space :',requested.keys())
      self.raiseADebug('DataObject Space:',self.getParaKeys('inputs'))
      self.raiseADebug('Requested realization input space does not match DataObject input space!  Assuming not found...')
      return
    inpVals = self.getParametersValues('inputs')
    #in benchmarking, using KDTree to query was shown to be consistently and on-average faster
    #  for each tensor case of dimensions=[2,5,10], number of realizations=[10,100,1000]
    #  when compared with brute force search through tuples
    #  This speedup was realized both in formatting, as well as creating the tree/querying the tree
    #if inputs have changed or this if first query, build the tree
    if self.inputKDTree is None:
      #set up data scaling, so that relative distances are used
      # scaling is so that scaled = (actual - mean)/scale
      for v in requested.keys():
        mean,scale = mathUtils.normalizationFactors(inpVals[v])
        self.treeScalingFactors[v] = (mean,scale)
      #convert data into a matrix in the order of requested
      data = np.dstack(tuple((np.array(inpVals[v])-self.treeScalingFactors[v][0])/self.treeScalingFactors[v][1] for v in requested.keys()))[0] #[0] is for the way dstack constructs the stack
      self.inputKDTree = spatial.KDTree(data)
    #query the tree
    distances,indices = self.inputKDTree.query(tuple((v-self.treeScalingFactors[k][0])/self.treeScalingFactors[k][1] for k,v in requested.items()),\
                  distance_upper_bound=tol, #acceptable distance
                  k=1, #number of points to find
                  p=2) #use Euclidean distance
    #if multiple entries were within tolerance, accept the minimum one
    if hasattr(distances,'__len__'):
      index = indices[distances.index(min(distances))]
    else:
      index = indices
    #KDTree reports a "not found" as at infinite distance, at len(data) index
    if index >= len(self):
      return None
    else:
      realization = self.getRealization(index)
    return realization

    #brute force approach, for comparison
    #prepare list of tuples to search from
    #have = []
    #for i in range(len(inpVals.values()[0])):
    #  have.append(tuple(inpVals[var][i] for var in requested.keys()))
    #have = np.array(have)
    #found,idx,match = mathUtils.NDInArray(have,tuple(val for val in requested.values()),tol=tol)
    #if not found:
    #  return
    #realization = self.getRealization(idx)
    #return realization

  def getMetadata(self,keyword,nodeId=None,serialize=False):
    """
      Function to get a value from the dictionary metadata
      @ In, keyword, string, parameter name
      @ In, nodeId, string, optional, id of the node if hierarchical
      @ In, serialize, bool, optional, serialize the tree if in hierarchical mode
      @ Out, dictionary, dict, return the metadata dictionary
    """
    if self._dataParameters['hierarchical']:
      if type(keyword) == int:
        return list(self.getHierParam('metadata',nodeId,None,serialize).values())[keyword-1]
      else:
        return self.getHierParam('metadata',nodeId,keyword,serialize)
    else:
      if keyword in self._dataContainer['metadata'].keys():
        return self._dataContainer ['metadata'][keyword]
      else:
        self.raiseAnError(RuntimeError,'parameter ' + str(keyword) + ' not found in metadata dictionary. Available keys are '+str(self._dataContainer['metadata'].keys())+'.Function: Data.getMetadata')

  def getOutParametersValues(self,nodeId=None,serialize=False):
    """
      Function to get a reference to the output parameter dictionary
      @, In, nodeId, string, optional, in hierarchical mode, if nodeId is provided, the data for that node is returned,
                                  otherwise check explanation for getHierParam
      @ In, serialize, bool, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                  PLEASE check explanation for getHierParam
      @, Out, dictionary, dict, Reference to self._dataContainer['outputs'] or something else in hierarchical
    """
    if self._dataParameters['hierarchical']:
      return self.getHierParam('outputs',nodeId,serialize=serialize)
    else:
      return self._dataContainer['outputs']

  def getParaKeys(self,typePara):
    """
      Functions to get the parameter keys
      @ In, typePara, string, variable type (input, output or metadata)
      @ Out, keys, list, list of requested keys
    """
    if typePara.lower() not in ['input','inputs','output','outputs','metadata']:
      self.raiseAnError(RuntimeError,'type ' + typePara + ' is not a valid type. Function: Data.getParaKeys')
    keys = self._dataParameters['inParam' ] if typePara.lower() in 'inputs' else (self._dataParameters['outParam'] if typePara.lower() in 'outputs' else self._dataContainer['metadata'].keys())
    return keys

  def getParam(self,typeVar,keyword,nodeId=None,serialize=False):
    """
      Function to get a reference to an output or input parameter
      @ In, typeVar, string, input, unstructuredInput or output
      @ In, keyword, string, keyword
      @, In, nodeId, string, optional, in hierarchical mode, if nodeId is provided, the data for that node is returned,
                                  otherwise check explanation for getHierParam
      @ In, serialize, bool, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                  PLEASE check explanation for getHierParam
      @ Out, value, float, Reference to the parameter
    """
    if self.type == 'HistorySet':
      acceptedType = ['str','unicode','bytes','int']
      convertArr = lambda x: x
      #convertArr = lambda x: np.asarray(x)
    else                       :
      acceptedType = ['str','unicode','bytes']
      convertArr = lambda x: np.asarray(x)

    if type(typeVar).__name__ not in ['str','unicode','bytes']:
      self.raiseAnError(RuntimeError,'type of parameter typeVar needs to be a string. Function: Data.getParam')
    if type(keyword).__name__ not in acceptedType        :
      self.raiseAnError(RuntimeError,'type of parameter keyword needs to be '+str(acceptedType)+' . Function: Data.getParam')
    if nodeId:
      if type(nodeId).__name__ not in ['str','unicode','bytes']:
        self.raiseAnError(RuntimeError,'type of parameter nodeId needs to be a string. Function: Data.getParam')
    if typeVar.lower() not in ['input','inout','inputs','unstructuredInput','output','outputs']:
      self.raiseAnError(RuntimeError,'type ' + typeVar + ' is not a valid type. Function: Data.getParam')
    if self._dataParameters['hierarchical']:
      if type(keyword) == int:
        return list(self.getHierParam(typeVar.lower(),nodeId,None,serialize).values())[keyword-1]
      else:
        return self.getHierParam(typeVar.lower(),nodeId,keyword,serialize)
    else:
      if typeVar.lower() in ['input','inputs','unstructuredInput']:
        returnDict = {}
        if keyword in itertools.chain(self._dataContainer['inputs'].keys(),self._dataContainer['unstructuredInputs'].keys()):
          returnDict[keyword] = {}
          if self.type == 'HistorySet':
            for key in self._dataContainer['inputs'][keyword].keys():
              returnDict[keyword][key] = np.resize(self._dataContainer['inputs'][keyword][key],len(self._dataContainer['outputs'][keyword].values()[0]))
            if len(self._dataContainer['unstructuredInputs'].keys()) > 0:
              for key in self._dataContainer['unstructuredInputs'][keyword].keys():
                returnDict[keyword][key] = self._dataContainer['unstructuredInputs'][keyword][key]
            return convertArr(returnDict[keyword])
          else:
            return convertArr(self._dataContainer['inputs'][keyword] if keyword in self._dataContainer['inputs'].keys() else self._dataContainer['unstructuredInputs'][keyword])
        else:
          self.raiseAnError(RuntimeError,self.name+' : parameter ' + str(keyword) + ' not found in inpParametersValues dictionary. Available keys are '+str(self._dataContainer['inputs'].keys())+'.Function: Data.getParam')
      elif typeVar.lower() in ['output','outputs']:
        if keyword in self._dataContainer['outputs'].keys():
          return convertArr(self._dataContainer['outputs'][keyword])
        else:
          self.raiseAnError(RuntimeError,self.name+' : parameter ' + str(keyword) + ' not found in outParametersValues dictionary. Available keys are '+str(self._dataContainer['outputs'].keys())+'.Function: Data.getParam')

  def getParametersValues(self,typeVar,nodeId=None, serialize=False):
    """
      Functions to get the parameter values
      @ In, typeVar, string, variable type (input, unstructuredInputs or output)
      @ In, nodeId, string, optional, id of the node if hierarchical
      @ In, serialize, bool, optional, serialize the tree if in hierarchical mode
      @ Out, dictionary, dict, dictionary of parameter values
    """
    if typeVar.lower() in 'inputs':
      return self.getInpParametersValues(nodeId,serialize)
    elif typeVar.lower() in 'unstructuredinputs':
      return self.getInpParametersValues(nodeId,serialize,True)
    elif typeVar.lower() in 'outputs':
      return self.getOutParametersValues(nodeId,serialize)
    else:
      self.raiseAnError(RuntimeError,'type ' + typeVar + ' is not a valid type. Function: Data.getParametersValues')

  def getRealization(self,index):
    """
      Returns the indexed entry of inputs and outputs
      @ In, index, int, index of realization to return
      @ Out, realization, dict, {'inputs':{inName:value}, 'outputs':{outName:value}}
    """
    if index >= len(self):
      self.raiseAnError(IndexError,'Requested entry %i but only entries 0 through %i exist!' %(index,len(self)-1))
    realization = {}
    inps       = self.getParaKeys('inputs')
    outs       = self.getParaKeys('outputs')
    allInVals  = self.getParametersValues('inputs', nodeId = 'RecontructEnding')
    allOutVals = self.getParametersValues('outputs', nodeId = 'RecontructEnding')
    inVals     = list(allInVals[var][index] for var in inps)   if self.type == 'PointSet' else list(allInVals[index+1][var] for var in inps)
    outVals    = list(allOutVals[var][index] for var in outs) if self.type == 'PointSet' else list(allOutVals[index+1][var] for var in outs)
    realization['inputs'] = dict((k,v) for (k,v) in zip(inps,inVals))
    realization['outputs'] = dict((k,v) for (k,v) in zip(outs,outVals))
    return realization

  def isItEmpty(self):
    """
      Function to check if the data is empty
      @ In, None
      @ Out, empty, bool, True if this instance is empty
    """
    empty = True if len(self.getInpParametersValues().keys()) == 0 and len(self.getOutParametersValues()) == 0 and len(self.getInpParametersValues(unstructuredInputs=True).keys()) == 0 else False
    return empty

  def loadXMLandCSV(self,filenameRoot,options=None):
    """
      Function to load the xml additional file of the csv for data
      (it contains metadata, etc)
      @ In, filenameRoot, string, file name root
      @ In, options, dict, optional, dictionary -> options for loading
      @ Out, None
    """
    self.inputKDTree = None
    self.treeScalingFactors = {}
    self._specializedLoadXMLandCSV(filenameRoot,options)

  def printCSV(self,options=None):
    """
      Function used to dump the data into a csv file
      Every class must implement the specializedPrintCSV method
      that is going to be called from here
      @ In, options, dict, optional, dictionary of options... it can contain the filename to be used, the parameters need to be printed.
      @ Out, None
    """
    optionsInt = {}
    # print content of data in a .csv format
    self.raiseADebug(' '*len(self.printTag)+':=============================')
    self.raiseADebug(' '*len(self.printTag)+':DataObjects: print on file(s)')
    self.raiseADebug(' '*len(self.printTag)+':=============================')
    variablesToPrint = []
    if options:
      if ('filenameroot' in options.keys()):
        filenameLocal = options['filenameroot']
      else:
        filenameLocal = self.name + '_dump'
      if 'what' in options.keys():
        for var in options['what'].split(','):
          lvar = var.lower()
          if lvar.startswith('input'):
            variablesToPrint.extend(self.__getVariablesToPrint(var,'input'))
          elif lvar.startswith('output'):
            variablesToPrint.extend(self.__getVariablesToPrint(var,'output'))
          else:
            self.raiseAnError(RuntimeError,'variable ' + var + ' is unknown in Data ' + self.name + '. When specifying \'what\' remember to prepend parameter names with \'Input|\' or \'Output|\'')
        optionsInt['what'] = variablesToPrint
    else:
      filenameLocal = self.name + '_dump'
    # this not needed since the variables are taken from inside
    #if 'what' not in optionsInt.keys():
    #  inputKeys, outputKeys = sorted(self.getParaKeys('inputs')), sorted(self.getParaKeys('outputs'))
    #  for inKey in inputKeys  : variablesToPrint.append('input|'+inKey)
    #  for outKey in outputKeys: variablesToPrint.append('output|'+outKey)
    #  optionsInt['what'] = variablesToPrint
    self.specializedPrintCSV(filenameLocal,optionsInt)

  def _writeUnstructuredInputInXML(self,fileRoot,historyVariableKeys,historyVariableValues):
    """
      Method to write the unstructured inputs into an XML file
      @ In, fileRoot, string, filename root (the generated file will be <fileRoot>.xml
      @ In, historyVariableKeys, list, list of lists containing the variable keys ([[varName1,varName2]i  i=1,n histories])
      @ In, historyVariableValues, list, list of lists containing the variable values ([[varValue1,varValue2]i  i=1,n histories])
      @ Out, None
    """
    unstructuredDataFile = open(fileRoot+".xml","w")
    unstructuredDataFile.write("<unstructuredInputSpace>\n")
    for histNum in range(len(historyVariableKeys)):
      unstructuredDataFile.write(" "*3+"<unstructuredInputData id='"+str(histNum)+"'>\n")
      for cnt,var in enumerate(historyVariableKeys[histNum]):
        unstructuredDataFile.write(" "*5+"<"+var.strip()+">\n")
        unstructuredDataFile.write(" "*7+np.array_str(np.asarray(historyVariableValues[histNum][cnt])).replace("[","").replace("]","")+"\n")
        unstructuredDataFile.write(" "*5+"</"+var.strip()+">\n")
      unstructuredDataFile.write(" "*3+"</unstructuredInputData>\n")
    unstructuredDataFile.write("</unstructuredInputSpace>\n")
    unstructuredDataFile.close()


  def removeInputValue(self,name):
    """
      Function to remove a value from the dictionary inpParametersValues
      @ In, name, string, parameter name
      @ Out, None
    """
    if self._dataParameters['hierarchical']:
      for TSData in self.TSData.values():
        for node in list(TSData.iter('*')):
          if name in node.get('dataContainer')['inputs'].keys():
            node.get('dataContainer')['inputs'].pop(name)
          elif name in node.get('unstructuredInputContainer').keys():
            node.get('dataContainer')['unstructuredInputs'].pop(name)
    else:
      if name in self._dataContainer['inputs'].keys():
        self._dataContainer['inputs'].pop(name)
      elif name in self._dataContainer['unstructuredInputs'].keys():
        self._dataContainer['unstructuredInputs'].pop(name)
    self.inputKDTree = None
    self.treeScalingFactors = {}

  def removeOutputValue(self,name):
    """
      Function to remove a value from the dictionary outParametersValues
      @ In, name, string, parameter name
      @ Out, None
    """
    if self._dataParameters['hierarchical']:
      for TSData in self.TSData.values():
        for node in list(TSData.iter('*')):
          if name in node.get('dataContainer')['outputs'].keys():
            node.get('dataContainer')['outputs'].pop(name)
    else:
      if name in self._dataContainer['outputs'].keys():
        self._dataContainer['outputs'].pop(name)

  def resetData(self):
    """
      Function to remove all the data in this dataobject
      @ In, None
      @ Out, None
    """
    if self._dataParameters['hierarchical']:
      self.TSData, self.rootToBranch = None, {}
    else:
      # we get the type of the metadata
      typeMetadata = self._dataParameters.get("typeMetadata",{})
      self._dataContainer                  = {'inputs':{},'unstructuredInputs':{},'outputs':{},'typeMetadata':typeMetadata}
      self._dataContainer['metadata'     ] = {}
    self.inputKDTree = None
    self.treeScalingFactors = {}

  def retrieveNodeInTreeMode(self,nodeName,parentName=None):
    """
      This Method is used to retrieve a node (a list...) when the hierarchical mode is requested
      If the node has not been found, Create a new one
      @ In, nodeName, string, is the node we want to retrieve
      @ In, parentName, string, optional, is the parent name... It's possible that multiple nodes have the same name.
                                          With the parentName, it's possible to perform a double check
      @ Out, foundNodes, TreeStructure.Node, the found nodes
    """
    if not self.TSData:
      # there is no tree yet
      self.TSData = {nodeName:TS.HierarchicalTree(self.messageHandler,TS.HierarchicalNode(self.messageHandler,nodeName))}
      return self.TSData[nodeName].getrootnode()
    else:
      if nodeName in self.TSData.keys():
        return self.TSData[nodeName].getrootnode()
      elif parentName == 'root':
        self.TSData[nodeName] = TS.HierarchicalTree(self.messageHandler,TS.HierarchicalNode(self.messageHandler,nodeName))
        return self.TSData[nodeName].getrootnode()
      else:
        for TSDat in self.TSData.values():
          foundNodes = list(TSDat.iter(nodeName))
          if len(foundNodes) > 0:
            break
        if len(foundNodes) == 0:
          return TS.HierarchicalNode(self.messageHandler,nodeName)
        else:
          if parentName:
            for node in foundNodes:
              if node.getParentName() == parentName:
                return node
            self.raiseAnError(RuntimeError,'the node ' + nodeName + 'has been found but no one has a parent named '+ parentName)
          else:
            return(foundNodes[0])

  def sizeData(self):
    """
      Method to get the size of the Data. (Number of realizations)
      @ In, None
      @ Out, outcome, int, number of realizations
    """
    outcome   = 0
    if self.isItEmpty():
      return outcome
    if self.type == 'PointSet':
      outcome = len(self.getParam('input',self.getParaKeys('input')[-1],nodeId = 'RecontructEnding'))
    else:
      outcome = len(self.getParametersValues('input', nodeId='RecontructEnding').keys())
    return outcome

  def updateInputValue(self,name,value,options=None):
    """
      Function to update a value from the input dictionary
      @ In, name, string, parameter name
      @ In, value, float, the new value
      @ In, options, dict, optional, the dictionary of options to update the value (e.g. parentId, etc.)
      @ Out, None
    """
    self.inputKDTree = None
    self.treeScalingFactors = {}
    self._updateSpecializedInputValue(name,value,options)

  def updateOutputValue(self,name,value,options=None):
    """
      Function to update a value from the output dictionary
      @ In, name, string, parameter name
      @ In, value, float, the new value
      @ In, options, dict, optional,  the dictionary of options to update the value (e.g. parentId, etc.)
      @ Out, None
    """
    self._updateSpecializedOutputValue(name,value,options)

  def updateMetadata(self,name,value,options=None):
    """
      Function to update a value from the dictionary metadata
      @ In, name, string, parameter name
      @ In, value, float, the new value
      @ In, options, dict, optional, dictionary of options
      @ Out, None
    """
    dtype = self.__getMetadataType(name,value)
    self._updateSpecializedMetadata(name,value,dtype,options)

  def addNodeInTreeMode(self,tsnode,options):
    """
      This Method is used to add a node into the tree when the hierarchical mode is requested
      If the node has not been found, Create a new one
      @ In, tsnode, TreeStructure.Node, the node
      @ In, options, dict, dictionary of options. parentID must be present if newer node
    """
    if not tsnode.getParentName():
      parentID = None
      if 'metadata' in options.keys():
        if 'parentID' in options['metadata'].keys():
          parentID = options['metadata']['parentID']
      else:
        if 'parentID' in options.keys():
          parentID = options['parentID']
      if not parentID:
        self.raiseAnError(ConstructError,'the parentID must be provided if a new node needs to be appended')
      self.retrieveNodeInTreeMode(parentID).appendBranch(tsnode)

  ##Protected Methods

  def _createXMLFile(self,filenameLocal,fileType,inpKeys,outKeys):
    """
      Creates an XML file to contain the input and output data list
      @ In, filenameLocal, string, file name
      @ In, fileType, string, file type (csv, xml)
      @ In, inpKeys, list, input keys
      @ In, outKeys, list, output keys
      @ Out, None
    """
    myXMLFile = open(filenameLocal + '.xml', 'w')
    root = ET.Element('data',{'name':filenameLocal,'type':fileType})
    inputNode = ET.SubElement(root,'input')
    inputNode.text = ','.join(inpKeys)
    outputNode = ET.SubElement(root,'output')
    outputNode.text = ','.join(outKeys)
    filenameNode = ET.SubElement(root,'inputFilename')
    filenameNode.text = filenameLocal + '.csv'
    if len(self._dataContainer['metadata']) > 0:
      #write metadata as well_known_implementations
      metadataNode = ET.SubElement(root,'metadata')
      submetadataNodes = []
      for key,value in self._dataContainer['metadata'].items():
        submetadataNodes.append(ET.SubElement(metadataNode,key))
        submetadataNodes[-1].text = utils.toString(str(value))
    myXMLFile.write(utils.toString(ET.tostring(root)))
    myXMLFile.write('\n')
    myXMLFile.close()

  def _loadXMLFile(self, filenameLocal):
    """
      Function to load the xml additional file of the csv for data
      (it contains metadata, etc). It must be implemented by the specialized classes
      @ In, filenameLocal, string, file name
      @ Out, retDict, dict, dictionary of keys, fileType etc.
    """
    myXMLFile = open(filenameLocal + '.xml', 'r')
    root = ET.fromstring(myXMLFile.read())
    myXMLFile.close()
    assert(root.tag == 'data')
    retDict = {}
    retDict["fileType"] = root.attrib['type']
    inputNode = root.find("input")
    outputNode = root.find("output")
    filenameNode = root.find("inputFilename")
    if inputNode is None:
      self.raiseAnError(RuntimeError,'input XML node not found in file ' + filenameLocal + '.xml')
    if outputNode is None:
      self.raiseAnError(RuntimeError,'output XML node not found in file ' + filenameLocal + '.xml')
    if filenameNode is None:
      self.raiseAnError(RuntimeError,'inputFilename XML node not found in file ' + filenameLocal + '.xml')
    retDict["inpKeys"] = inputNode.text.split(",")
    retDict["outKeys"] = outputNode.text.split(",")
    retDict["filenameCSV"] = filenameNode.text
    metadataNode = root.find("metadata")
    if metadataNode is not None:
      metadataDict = {}
      for child in metadataNode:
        key = child.tag
        value = child.text
        value.replace('\n','')
        # ast.literal_eval can't handle numpy arrays, so we'll handle that.
        if value.startswith('array('):
          isArray=True
          value=value.split('dtype')[0].lstrip('ary(').rstrip('),\n ')
        else:
          isArray = False
        try:
          value = ast.literal_eval(value)
        except ValueError as e:
          # these aren't real fails, they just don't actually need converting
          self.raiseAWarning('ast.literal_eval failed on "',value,'"')
          self.raiseAWarning('ValueError was "',e,'", but continuing on...')
        except SyntaxError as e:
          # these aren't real fails, they just don't actually need converting
          self.raiseAWarning('ast.literal_eval failed on "',value,'"')
          self.raiseAWarning('SyntaxError was "',e,'", but continuing on...')
        if isArray:
          # convert back
          value = np.array(value)
          value = c1darray(values=value)
        metadataDict[key] = value
      retDict["metadata"] = metadataDict
    return retDict

  def _readMoreXML(self,xmlNode):
    """
      Function to read the xml input block.
      @ In, xmlNode, xml.etree.ElementTree.Element, xml node
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    # retrieve input/outputs parameters' keywords
    inputFind = paramInput.findFirst('Input' )
    outputFind =  paramInput.findFirst('Output' )
    if inputFind is None and outputFind is None:
      self.raiseAnError(IOError,"At least one of the Input or Output XML block needs to be inputted!")
    # we allow to avoid to have an <Input> block if not needed (InputPlaceHolder) or a <Output> block if not needed (OutputPlaceHolder)
    self._dataParameters['inParam']  = list(inputFind.value) if inputFind is not None else ['InputPlaceHolder']
    self._dataParameters['outParam'] = list(outputFind.value) if outputFind is not None else ['OutputPlaceHolder']
    if '' in self._dataParameters['inParam']:
      self.raiseAnError(IOError, 'In DataObject  ' +self.name+' there is a trailing comma in the "Input" XML block!')
    if '' in self._dataParameters['outParam']:
      self.raiseAnError(IOError, 'In DataObject  ' +self.name+' there is a trailing comma in the "Output" XML block!')
    #test for keywords not allowed
    if len(set(self._dataParameters['inParam'])&set(self.notAllowedInputs))!=0:
      self.raiseAnError(IOError,'the keyword '+str(set(self._dataParameters['inParam'])&set(self.notAllowedInputs))+' is not allowed among inputs')
    if len(set(self._dataParameters['outParam'])&set(self.notAllowedOutputs))!=0:
      self.raiseAnError(IOError,'the keyword '+str(set(self._dataParameters['outParam'])&set(self.notAllowedOutputs))+' is not allowed among inputs')
    # test if some parameters are repeated
    for inp in self._dataParameters['inParam']:
      if self._dataParameters['inParam'].count(inp) > 1:
        self.raiseAnError(IOError,'the keyword '+inp+' is listed, in <Input> block, more then once!')
    for out in self._dataParameters['outParam']:
      if self._dataParameters['outParam'].count(out) > 1:
        self.raiseAnError(IOError,'the keyword '+out+' is listed, in <Output> block, more then once!')
    #test for same input/output variables name
    if len(set(self._dataParameters['inParam'])&set(self._dataParameters['outParam']))!=0:
      self.raiseAnError(IOError,'It is not allowed to have the same name of input/output variables in the data '+self.name+' of type '+self.type)
    optionsData = paramInput.findFirst('options')
    if optionsData != None:
      for child in optionsData.subparts:
        self._dataParameters[child.getName()] = child.value
    if set(self._dataParameters.keys()).issubset(['inputRow','inputPivotValue']):
      self.raiseAnError(IOError,'It is not allowed to simultaneously specify the nodes: inputRow and inputPivotValue!')
    if set(self._dataParameters.keys()).issubset(['outputRow','outputPivotValue','operator']):
      self.raiseAnError(IOError,'It is not allowed to simultaneously specify the nodes: outputRow, outputPivotValue and operator!')
    self._specializedInputCheckParam(paramInput)
    if 'hierarchical' in paramInput.parameterValues:
      if paramInput.parameterValues['hierarchical'].lower() in utils.stringsThatMeanTrue():
        self._dataParameters['hierarchical'] = True
      else:
        self._dataParameters['hierarchical'] = False
      if self._dataParameters['hierarchical']:
        self.TSData, self.rootToBranch = None, {}
    else:
      self._dataParameters['hierarchical'] = False

  def _specializedInputCheckParam(self,paramInput):
    """
      Function to check the input parameters that have been read for each DataObject subtype
      @ In, paramInput, ParameterInput, the input
      @ Out, None
    """
    pass

  def _specializedLoadXMLandCSV(self,filenameRoot,options):
    """
      Function to load the xml additional file of the csv for data
      (it contains metadata, etc). It must be implemented by the specialized classes
      @ In, filenameRoot, string, file name root
      @ In, options, dict, dictionary -> options for loading
      @ Out, None
    """
    self.raiseAnError(RuntimeError,"specializedloadXMLandCSV not implemented "+str(self))

  ##Private Methods

#   COMMENTED SINCE NO USED. NEEDS TO BE REMOVED IN THE FUTURE
#   @abc.abstractmethod
#   def __extractValueLocal__(self,inOutType,varTyp,varName,varID=None,stepID=None,nodeId='root'):
#     """
#       This method has to be override to implement the specialization of extractValue for each data class
#       @ In, inOutType, string, the type of data to extract (input or output)
#       @ In, varTyp, string, is the requested type of the variable to be returned (bool, int, float, numpy.ndarray, etc)
#       @ In, varName, string, is the name of the variable that should be recovered
#       @ In, varID, tuple or int, optional,  is the ID of the value that should be retrieved within a set
#         if varID.type!=tuple only one point along sampling of that variable is retrieved
#           else:
#             if varID=(int,int) the slicing is [varID[0]:varID[1]]
#             if varID=(int,None) the slicing is [varID[0]:]
#       @ In, stepID, tuple or int, optional, it  determines the slicing of an history.
#           if stepID.type!=tuple only one point along the history is retrieved
#           else:
#             if stepID=(int,int) the slicing is [stepID[0]:stepID[1]]
#             if stepID=(int,None) the slicing is [stepID[0]:]
#       @ In, nodeId, string, optional, in hierarchical mode, is the node from which the value needs to be extracted... by default is the root
#       @ Out, value, the requested value
#     """
#     pass

  def __getVariablesToPrint(self,var,inOrOut):
    """
      Returns a list of variables to print.
      Takes the variable and either 'input' or 'output'
      In addition, if the variable belong to the metadata and metaAdditionalInOrOut, it will also return to print
      @ In, var, string, variable name
      @ In, inOrOut, string, type of variable (input or output)
      @ Out, variablesToPrint, list, list of variables to print
    """
    variablesToPrint = []
    lvar = var.lower()
    inOrOuts = inOrOut + 's'
    if lvar == inOrOut:
      if type(list(self._dataContainer[inOrOuts].values())[0]) == dict:
        varKeys = list(self._dataContainer[inOrOuts].values())[0].keys()
      else:
        varKeys = self._dataContainer[inOrOuts].keys()
      for invar in varKeys:
        variablesToPrint.append(inOrOut+'|'+str(invar))
    elif '|' in var and lvar.startswith(inOrOut+'|'):
      varName = var.split('|')[1]
      # get the variables from the metadata if the variables are in the list metaAdditionalInOrOut
      if varName in self.metaAdditionalInOrOut:
        varKeys = self._dataContainer['metadata'].keys()
        if varName not in varKeys:
          self.raiseAnError(RuntimeError,'variable ' + varName + ' is not present among the ' +inOrOuts+' of Data ' + self.name)
        if type(self._dataContainer['metadata'][varName]) not in self.metatype:
          self.raiseAnError(NotConsistentData,inOrOut + var.split('|')[1]+' not compatible with CSV output. Its type needs to be one of '+str(self.metatype))
        else:
          variablesToPrint.append('metadata'+'|'+str(varName))
      else:
        if type(list(self._dataContainer[inOrOuts].values())[0]) == dict:
          varKeys = list(self._dataContainer[inOrOuts].values())[0].keys()
        else:
          varKeys = self._dataContainer[inOrOuts].keys()
        if varName not in varKeys:
          self.raiseAnError(RuntimeError,'variable %s is not present among the %s of Data %s (available: %s)' % (varName,inOrOuts,self.name,varKeys))
        else:
          variablesToPrint.append(inOrOut+'|'+str(varName))
    else:
      self.raiseAnError(RuntimeError,'unexpected variable '+ var)
    return variablesToPrint

  def __getMetadataType(self,name,value):
    """
      Utility method to get the metadata type. If the type is not stored in
      the self._dataParameters['typeMetadata'] this method will add it
      @ In, name, str, the metadata name
      @ In, value, object, the metadata value to analyze
      @ Out, valueType, type, the metadata type
    """
    try:
      valueType = self._dataParameters['typeMetadata'][name]
    except KeyError:
      valueType = None if utils.checkTypeRecursively(value) not in ['str','unicode','bytes'] else object
      self._dataParameters['typeMetadata'][name] = valueType
    return valueType
