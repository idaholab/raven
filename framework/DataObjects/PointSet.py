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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from cached_ndarray import c1darray
from DataObjects.Data import Data, NotConsistentData, ConstructError
import Files
import utils
#Internal Modules End--------------------------------------------------------------------------------

class PointSet(Data):
  """
  PointSet is an object that stores multiple sets of inputs and outputs for a particular point in time!
  """
  def __init__(self):
    Data.__init__(self)
    self.acceptHierarchy = True

  def addSpecializedReadingSettings(self):
    """
      This function adds in the _dataParameters dict the options needed for reading and constructing this class
    """
    # if hierarchical fashion has been requested, we set the type of the reading to a Point,
    #  since a PointSet in hierarchical fashion would be a tree of Points
    if self._dataParameters['hierarchical']: self._dataParameters['type'] = 'Point'
    # store the type into the _dataParameters dictionary
    else:                                   self._dataParameters['type'] = self.type
    try: sourceType = self._toLoadFromList[-1].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      self._dataParameters['type']       = self.type
      self._dataParameters['HistorySet'] = self._toLoadFromList[-1].getEndingGroupNames()
      self._dataParameters['filter'   ]  = 'whole'

  def checkConsistency(self):
    """
      Here we perform the consistency check for the structured data PointSet
    """
    #The lenMustHave is a counter of the HistorySet contained in the
    #toLoadFromList list. Since this list can contain either CSVfiles
    #and HDF5, we can not use "len(_toLoadFromList)" anymore. For
    #example, if that list contains 10 csvs and 1 HDF5 (with 20
    #HistorySet), len(toLoadFromList) = 11 but the number of HistorySet
    #is actually 30.
    lenMustHave = 0
    sourceType = self._toLoadFromList[-1].type
    # here we assume that the outputs are all read....so we need to compute the total number of time point sets
    for sourceLoad in self._toLoadFromList:
      if'HDF5' == sourceLoad.type:  lenMustHave = lenMustHave + len(sourceLoad.getEndingGroupNames())
      elif isinstance(sourceLoad,Files.File): lenMustHave += 1
      else: self.raiseAnError(Exception,'The type ' + sourceLoad.type + ' is unknown!')

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
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'Point' has been stored
      parentID = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
      else:
        prefix    = options['prefix']
        if 'parentID' in options.keys(): parentID = options['parentID']
      if parentID: tsnode = self.retrieveNodeInTreeMode(prefix,parentID)
      else:                             tsnode = self.retrieveNodeInTreeMode(prefix)
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if name in self._dataContainer['inputs'].keys():
        self._dataContainer['inputs'].pop(name)
      if name not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(name)
      self._dataContainer['inputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['inputs'].keys():
        #popped = self._dataContainer['inputs'].pop(name)
        self._dataContainer['inputs'][name].append(np.atleast_1d(np.atleast_1d(value)[-1]))
        #self._dataContainer['inputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)[-1]))                     copy.copy(np.concatenate((np.atleast_1d(np.array(popped)), np.atleast_1d(np.atleast_1d(value)[-1]))))
      else:
        if name not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(name)
        self._dataContainer['inputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)[-1]))

  def _updateSpecializedMetadata(self,name,value,options=None):
    """
      This function performs the updating of the values (metadata) into this Data
      @ In,  name, string, parameter name (ex. probability)
      @ In,  value, whatever type, newer value
      @ Out, None
      NB. This method, if the metadata name is already present, replaces it with the new value. No appending here, since the metadata are dishomogenius and a common updating strategy is not feasable.
    """
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'Point' has been stored
      parentID = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
      else:
        prefix    = options['prefix']
        if 'parentID' in options.keys(): parentID = options['parentID']
      if parentID: tsnode = self.retrieveNodeInTreeMode(prefix,parentID)

      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'metadata':{}})
        self._dataContainer = tsnode.get('dataContainer')
      else:
        if 'metadata' not in self._dataContainer.keys(): self._dataContainer['metadata'] ={}
      if name in self._dataContainer['metadata'].keys(): self._dataContainer['metadata'][name].append(np.atleast_1d(value)) # = np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value)))
      else                                             : self._dataContainer['metadata'][name] = c1darray(values=np.atleast_1d(value),dtype=type(value))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['metadata'].keys(): self._dataContainer['metadata'][name].append(np.atleast_1d(value)) # = np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value)))
      else                                             : self._dataContainer['metadata'][name] = c1darray(values=np.atleast_1d(value),dtype=type(value))

  def _updateSpecializedOutputValue(self,name,value,options=None):
    """
      This function performs the updating of the values (output space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (single value)
      @ Out, None
    """
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'Point' has been stored
      parentID = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
      else:
        prefix    = options['prefix']
        if 'parentID' in options.keys(): parentID = options['parentID']
      if parentID: tsnode = self.retrieveNodeInTreeMode(prefix,parentID)

      #if 'parentID' in options.keys(): tsnode = self.retrieveNodeInTreeMode(options['prefix'], options['parentID'])
      #else:                             tsnode = self.retrieveNodeInTreeMode(options['prefix'])
      # we store the pointer to the container in the self._dataContainer because checkConsistency acts on this
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if name in self._dataContainer['outputs'].keys():
        self._dataContainer['outputs'].pop(name)
      if name not in self._dataParameters['outParam']: self._dataParameters['outParam'].append(name)
      self._dataContainer['outputs'][name] = c1darray(values=np.atleast_1d(value)) #np.atleast_1d(np.atleast_1d(value)[-1])
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['outputs'].keys():
        #popped = self._dataContainer['outputs'].pop(name)
        self._dataContainer['outputs'][name].append(np.atleast_1d(value)[-1])   #= copy.copy(np.concatenate((np.array(popped), np.atleast_1d(np.atleast_1d(value)[-1]))))
      else:
        if name not in self._dataParameters['outParam']: self._dataParameters['outParam'].append(name)
        self._dataContainer['outputs'][name] = c1darray(values=np.atleast_1d(np.atleast_1d(value)[-1])) # np.atleast_1d(np.atleast_1d(value)[-1])

  def specializedPrintCSV(self,filenameLocal,options):
    """
      This function prints a CSV file with the content of this class (Input and Output space)
      @ In,  filenameLocal, string, filename root (for example, 'homo_homini_lupus' -> the final file name is gonna be called 'homo_homini_lupus.csv')
      @ In,  options, dictionary, dictionary of printing options
      @ Out, None (a csv is gonna be printed)
    """
    inpKeys   = []
    inpValues = []
    outKeys   = []
    outValues = []
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
              for index in range(len(O_o[key])): axa[index] = O_o[key][index]['inputs'][variableName][0]
              inpValues[-1].append(axa)
            if varType == 'output':
              outKeys[-1].append(variableName)
              axa = np.zeros(len(O_o[key]))
              for index in range(len(O_o[key])): axa[index] = O_o[key][index]['outputs'][variableName][0]
              outValues[-1].append(axa)
            if varType == 'metadata':
              inpKeys[-1].append(variableName)
              if type(O_o[key][index]['metadata'][splitted[1]]) not in self.metatype:
                self.raiseAnError(NotConsistentData,'metadata '+variableName +' not compatible with CSV output. Its type needs to be one of '+str(np.ndarray))
                axa = np.zeros(len(O_o[key]))
                for index in range(len(O_o[key])): axa[index] = np.atleast_1d(np.float(O_o[key][index]['metadata'][variableName]))[0]
                inpValues[-1].append(axa)
        else:
          inpKeys[-1] = O_o[key][0]['inputs'].keys()
          for var in inpKeys[-1]:
            axa = np.zeros(len(O_o[key]))
            for index in range(len(O_o[key])): axa[index] = O_o[key][index]['inputs'][var][0]
            inpValues[-1].append(axa)
          outKeys[-1] = O_o[key][0]['outputs'].keys()
          for var in outKeys[-1]:
            axa = np.zeros(len(O_o[key]))
            for index in range(len(O_o[key])): axa[index] = O_o[key][index]['outputs'][var][0]
            outValues[-1].append(axa)
      if len(inpKeys[-1]) > 0 or len(outKeys[-1]) > 0: myFile = open(filenameLocal + '.csv', 'w')
      else: return
      O_o_keys = list(O_o.keys())
      for index in range(len(O_o.keys())):
        myFile.write('Ending branch,'+O_o_keys[index]+'\n')
        myFile.write('branch #')
        for item in inpKeys[index]:
          myFile.write(',' + item)
        for item in outKeys[index]:
          myFile.write(',' + item)
        myFile.write('\n')
        try   : sizeLoop = outValues[index][0].size
        except: sizeLoop = inpValues[index][0].size
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
      myFile.write(','.join([str(item) for item in itertools.chain(inpKeys,outKeys)]))
      myFile.write('\n')
      #Print values
      for j in range(len(next(iter(itertools.chain(inpValues,outValues))))):
        myFile.write(','.join([str(item[j]) for item in itertools.chain(inpValues,outValues)]))
        myFile.write('\n')
      myFile.close()
      self._createXMLFile(filenameLocal,'Pointset',inpKeys,outKeys)

  def _specializedLoadXMLandCSV(self, filenameRoot, options):
    """
    Loads a CSV-XML file pair into a PointSet.
    @ In, filenameRoot, path to files
    @ In, options, can optionally contain the following:
        - nameToLoad, filename base (no extension) of CSV-XML pair
    @Out, None
    """
    #For Pointset it will create an XML file and one CSV file.
    #The CSV file will have a header with the input names and output
    #names, and multiple lines of data with the input and output
    #numeric values, one line for each input.
    if options is not None and 'fileToLoad' in options.keys():
      name = os.path.join(options['fileToLoad'].getPath(),options['fileToLoad'].getBase())
    else: name = self.name
    filenameLocal = os.path.join(filenameRoot,name)
    xmlData = self._loadXMLFile(filenameLocal)
    assert(xmlData["fileType"] == "Pointset")
    if "metadata" in xmlData:
      self._dataContainer['metadata'] = xmlData["metadata"]
    mainCSV = os.path.join(filenameRoot,xmlData["filenameCSV"])
    myFile = open(mainCSV,"rU")
    header = myFile.readline().rstrip()
    inoutKeys = header.split(",")
    inoutValues = [[] for _ in range(len(inoutKeys))]
    for line in myFile.readlines():
      lineList = line.rstrip().split(",")
      for i in range(len(inoutKeys)):
        inoutValues[i].append(utils.partialEval(lineList[i]))
    self._dataContainer['inputs'] = {}
    self._dataContainer['outputs'] = {}
    inoutDict = {}
    for key,value in zip(inoutKeys,inoutValues):
      inoutDict[key] = value
    for key in xmlData["inpKeys"]:
      self._dataContainer["inputs"][key] = c1darray(values=np.array(inoutDict[key]))
    for key in xmlData["outKeys"]:
      self._dataContainer["outputs"][key] = c1darray(values=np.array(inoutDict[key]))

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    """override of the method in the base class DataObjects"""
    if stepID!=None: self.raiseAnError(RuntimeError,'seeking to extract a history slice over an PointSet type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':
      if varID!=None:
        if self._dataParameters['hierarchical']: exec('extractedValue ='+varTyp +'(self.getHierParam(inOutType,nodeid,varName,serialize=False)[nodeid])')
        else: exec('extractedValue ='+varTyp +'(self.getParam(inOutType,varName)[varID])')
        return extractedValue
      #if varID!=None: exec ('return varTyp(self.getParam('+inOutType+','+varName+')[varID])')
      else: self.raiseAnError(RuntimeError,'trying to extract a scalar value from a time point set without an index')
    else:
      if self._dataParameters['hierarchical']:
        paramss = self.getHierParam(inOutType,nodeid,varName,serialize=True)
        extractedValue = np.zeros(len(paramss[nodeid]))
        for index in range(len(paramss[nodeid])): extractedValue[index] = paramss[nodeid][index]
        return extractedValue
      else: return self.getParam(inOutType,varName)
