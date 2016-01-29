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
import os
import copy
import itertools
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from cached_ndarray import c1darray
from .Data import Data, NotConsistentData, ConstructError
import utils
#Internal Modules End--------------------------------------------------------------------------------

class Point(Data):
  """
  Point is an object that stores a set of inputs and outputs for a particular point in time!
  """

  def _specializedInputCheck(self,xmlNode):
    """
     Here we check if the parameters read by the global reader are compatible with this type of Data
     @ In, ElementTree object, xmlNode
     @ Out, None
    """
    if "historyName" in xmlNode.attrib.keys(): self._dataParameters['history'] = xmlNode.attrib['historyName']
    else                                     : self._dataParameters['history'] = None

  def addSpecializedReadingSettings(self):
    """
      This function adds in the dataParameters dict the options needed for reading and constructing this class
      @ In, None
      @ Out, None
    """
    self._dataParameters['type'] = self.type # store the type into the _dataParameters dictionary
    #The source is the last item we added, so use [-1]
    if hasattr(self._toLoadFromList[-1], 'type'):
      sourceType = self._toLoadFromList[-1].type
    else:
      sourceType = None
    if('HDF5' == sourceType):
      if(not self._dataParameters['history']): self.raiseAnError(IOError,'In order to create a Point data, history name must be provided')
      self._dataParameters['filter'] = 'whole'

  def checkConsistency(self):
    """
      Here we perform the consistency check for the structured data Point
      @ In, None
      @ Out, None
    """
    for key in self._dataContainer['inputs'].keys():
      if (self._dataContainer['inputs'][key].size) != 1:
        self.raiseAnError(NotConsistentData,'The input parameter value, for key ' + key + ' has not a consistent shape for Point ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(self._dataContainer['inputs'][key].size))
    for key in self._dataContainer['outputs'].keys():
      if (self._dataContainer['outputs'][key].size) != 1:
        self.raiseAnError(NotConsistentData,'The output parameter value, for key ' + key + ' has not a consistent shape for Point ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(self._dataContainer['outputs'][key].size))

  def _updateSpecializedInputValue(self,name,value,options=None):
    """
      This function performs the updating of the values (input space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (1-D array)
      @ Out, None
    """
    if name in self._dataContainer['inputs'].keys():
      self._dataContainer['inputs'].pop(name)
    if name not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(name)
    self._dataContainer['inputs'][name] = c1darray(values=np.atleast_1d(value))

  def _updateSpecializedMetadata(self,name,value,options=None):
    """
      This function performs the updating of the values (metadata) into this Data
      @ In,  name, string, parameter name (ex. probability)
      @ In,  value, whatever type, newer value
      @ Out, None
    """
    self._dataContainer['metadata'][name] = copy.copy(value)

  def _updateSpecializedOutputValue(self,name,value,options=None):
    """
      This function performs the updating of the values (output space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (1-D array)
      @ Out, None
    """
    if name in self._dataContainer['inputs'].keys():
      self._dataContainer['outputs'].pop(name)
    if name not in self._dataParameters['outParam']: self._dataParameters['outParam'].append(name)
    self._dataContainer['outputs'][name] = c1darray(values=np.atleast_1d(value))

  def specializedPrintCSV(self,filenameLocal,options):
    """
      This function prints a CSV file with the content of this class (Input and Output space)
      @ In,  filenameLocal, string, filename root (for example, 'homo_homini_lupus' -> the final file name is gonna be called 'homo_homini_lupus.csv')
      @ In,  options, dictionary, dictionary of printing options
      @ Out, None (a csv is gonna be printed)
    """
    #For Point it creates an XML file and one csv file.  The
    #CSV file will have a header with the input names and output
    #names, and one line of data with the input and output numeric
    #values.
    inpKeys   = []
    inpValues = []
    outKeys   = []
    outValues = []
    #Print input values
    if 'what' in options.keys():
      for var in options['what']:
        splitted = var.split('|')
        variableName = "|".join(splitted[1:])
        varType = splitted[0]
        if varType == 'input':
          inpKeys.append(variableName)
          inpValues.append(self._dataContainer['inputs'][variableName])
        if varType == 'output':
          outKeys.append(variableName)
          outValues.append(self._dataContainer['outputs'][variableName])
        if varType == 'metadata':
          inpKeys.append(variableName)
          inpValues.append(self._dataContainer['metadata'][variableName])
    else:
      inpKeys   = self._dataContainer['inputs'].keys()
      inpValues = self._dataContainer['inputs'].values()
      outKeys   = self._dataContainer['outputs'].keys()
      outValues = self._dataContainer['outputs'].values()
    if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '.csv', 'w')
    else: return

    #Print header
    myFile.write(','.join([item for item in itertools.chain(inpKeys,outKeys)]))
    myFile.write('\n')
    #Print values
    myFile.write(','.join([str(item[0]) for item in  itertools.chain(inpValues,outValues)]))
    myFile.write('\n')
    myFile.close()
    self._createXMLFile(filenameLocal,'Point',inpKeys,outKeys)

  def _specializedLoadXMLandCSV(self, filenameRoot, options):
    #For Point it creates an XML file and one csv file.  The
    #CSV file will have a header with the input names and output
    #names, and one line of data with the input and output numeric
    #values.
    if options is not None and 'fileToLoad' in options.keys():
      name = os.path.join(options['fileToLoad'].getPath(),options['fileToLoad'].getBase())
    else: name = self.name
    filenameLocal = os.path.join(filenameRoot,name)
    xmlData = self._loadXMLFile(filenameLocal)
    assert(xmlData["fileType"] == "Point")
    if "metadata" in xmlData:
      self._dataContainer['metadata'] = xmlData["metadata"]
    mainCSV = os.path.join(filenameRoot,xmlData["filenameCSV"])
    myFile = open(mainCSV,"rU")
    header = myFile.readline().rstrip()
    firstLine = myFile.readline().rstrip()
    myFile.close()
    inoutKeys = header.split(",")
    inoutValues = [utils.partialEval(a) for a in firstLine.split(",")]
    inoutDict = {}
    for key,value in zip(inoutKeys,inoutValues):
      inoutDict[key] = value
    self._dataContainer['inputs'] = {}
    self._dataContainer['outputs'] = {}
    #NOTE it's critical to cast these as c1darray!
    for key in xmlData["inpKeys"]:
      self._dataContainer["inputs"][key] = c1darray(values=np.array([inoutDict[key]]))
    for key in xmlData["outKeys"]:
      self._dataContainer["outputs"][key] = c1darray(values=np.array([inoutDict[key]]))

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    """override of the method in the base class DataObjects"""
    if varID!=None or stepID!=None: self.raiseAnError(RuntimeError,'seeking to extract a slice from a Point type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':exec ('return '+varTyp+'(self.getParam(inOutType,varName)[0])')
    else: return self.getParam(inOutType,varName)
