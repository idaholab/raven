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
import copy
import itertools
import numpy as np
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from cached_ndarray import c1darray
from DataObjects.Data import Data, NotConsistentData, ConstructError
import utils
#Internal Modules End--------------------------------------------------------------------------------

class History(Data):
  """
  History is an object that stores a set of inputs and associated history for output parameters.
  """
  def _specializedInputCheck(self,xmlNode):
    if "historyName" in xmlNode.attrib.keys(): self._dataParameters['history'] = xmlNode.attrib['historyName']
    else                                     : self._dataParameters['history'] = None
    if set(self._dataParameters.keys()).issubset(['operator','outputRow']): self.raiseAnError(IOError,"Inputted operator or outputRow attributes are available for Point and PointSet only!")

  def addSpecializedReadingSettings(self):
    """
      This function adds in the _dataParameters dict the options needed for reading and constructing this class
    """
    self._dataParameters['type'] = self.type # store the type into the _dataParameters dictionary
    try: sourceType = self._toLoadFromList[-1].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      if(not self._dataParameters['history']): self.raiseAnError(IOError,'In order to create a History data, history name must be provided')
      self._dataParameters['filter'] = 'whole'

  def checkConsistency(self):
    """
      Here we perform the consistency check for the structured data History
      @ In, None
      @ Out, None
    """
    for key in self._dataContainer['inputs'].keys():
      if (self._dataContainer['inputs'][key].size) != 1:
        self.raiseAnError(NotConsistentData,'The input parameter value, for key ' + key + ' has not a consistent shape for History ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(len(self._dataContainer['inputs'][key])))
    for key in self._dataContainer['outputs'].keys():
      if (self._dataContainer['outputs'][key].ndim) != 1:
        self.raiseAnError(NotConsistentData,'The output parameter value, for key ' + key + ' has not a consistent shape for History ' + self.name + '!! It should be an 1D array.' + '.Actual dimension is ' + str(self._dataContainer['outputs'][key].ndim))

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
      NB. This method, if the metadata name is already present, replaces it with the new value. No appending here, since the metadata are dishomogenius and a common updating strategy is not feasable.
    """
    self._dataContainer['metadata'][name] = copy.copy(value)

  def _updateSpecializedOutputValue(self,name,value,options=None):
    """
      This function performs the updating of the values (output space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (1-D array)
      @ Out, None
    """
    if name in self._dataContainer['outputs'].keys():
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
    #For history, create an XML file and two CSV files.  The
    #first CSV file has a header with the input names, and a column
    #for the filename.  The second CSV file is named the same as the
    #filename, and has the output names for a header, a column for
    #time, and the rest of the file is data for different times.
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

    #Create Input file
    #Print header
    myFile.write(','.join([item for item in  itertools.chain(inpKeys,['filename'])]))
    myFile.write('\n')
    #Print data
    myFile.write(','.join([str(item[0]) for item in itertools.chain(inpValues,[[filenameLocal + '_0' + '.csv']])]))
    myFile.write('\n')
    myFile.close()
    #Create Output file
    myDataFile = open(filenameLocal + '_0' + '.csv', 'w')
    #Print headers
    #Print time + output values
    myDataFile.write(','.join([item for item in outKeys]))
    myDataFile.write('\n')
    #Print data
    for j in range(next(iter(outValues)).size):
      myDataFile.write(','.join([str(item[j]) for item in outValues]))
      myDataFile.write('\n')
    myDataFile.close()
    self._createXMLFile(filenameLocal,'history',inpKeys,outKeys)

  def _specializedLoadXMLandCSV(self, filenameRoot, options):
    #For history, create an XML file and two CSV files.  The
    #first CSV file has a header with the input names, and a column
    #for the filename.  The second CSV file is named the same as the
    #filename, and has the output names for a header, a column for
    #time, and the rest of the file is data for different times.

    if options is not None and 'fileToLoad' in options.keys():
      name = os.path.join(options['fileToLoad'].getPath(),options['fileToLoad'].getBase())
    else: name = self.name
    filenameLocal = os.path.join(filenameRoot,name)
    xmlData = self._loadXMLFile(filenameLocal)
    assert(xmlData["fileType"] == "history")
    if "metadata" in xmlData:
      self._dataContainer['metadata'] = xmlData["metadata"]
    mainCSV = os.path.join(filenameRoot,xmlData["filenameCSV"])
    myFile = open(mainCSV,"rU")
    header = myFile.readline().rstrip()
    firstLine = myFile.readline().rstrip()
    myFile.close()
    inpKeys = header.split(",")[:-1]
    subCSVFilename = os.path.join(filenameRoot,firstLine.split(",")[-1])
    inpValues = [utils.partialEval(a) for a in firstLine.split(",")[:-1]]
    myDataFile = open(subCSVFilename, "rU")
    header = myDataFile.readline().rstrip()
    outKeys = header.split(",")
    outValues = [[] for a in range(len(outKeys))]
    for line in myDataFile.readlines():
      lineList = line.rstrip().split(",")
      for i in range(len(outKeys)):
        outValues[i].append(utils.partialEval(lineList[i]))
    self._dataContainer['inputs'] = {}
    self._dataContainer['outputs'] = {}
    for key,value in zip(inpKeys,inpValues):
      self._dataContainer['inputs'][key] = c1darray(values=np.array([value]*len(outValues[0])))
    for key,value in zip(outKeys,outValues):
      self._dataContainer['outputs'][key] = c1darray(values=np.array(value))

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    """override of the method in the base class DataObjects"""
    if varID!=None: self.raiseAnError(RuntimeError,'seeking to extract a slice over number of parameters an History type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':
      if varName in self._dataParameters['inParam']: exec ('return varTyp(self.getParam('+inOutType+','+varName+')[0])')
      else:
        if stepID!=None and type(stepID)!=tuple: exec ('return self.getParam('+inOutType+','+varName+')['+str(stepID)+']')
        else: self.raiseAnError(RuntimeError,'To extract a scalar from an history a step id is needed. Variable: '+varName+', Data: '+self.name)
    else:
      if stepID==None : return self.getParam(inOutType,varName)
      elif stepID!=None and type(stepID)==tuple: return self.getParam(inOutType,varName)[stepID[0]:stepID[1]]
      else: self.raiseAnError(RuntimeError,'trying to extract variable '+varName+' from '+self.name+' the id coordinate seems to be incoherent: stepID='+str(stepID))
