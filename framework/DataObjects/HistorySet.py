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
import Files
#Internal Modules End--------------------------------------------------------------------------------

class HistorySet(Data):
  """
  HistorySet is an object that stores multiple sets of inputs and associated history for output parameters.
  """
  def __init__(self):
    Data.__init__(self)
    self.acceptHierarchy = True

  def _specializedInputCheck(self,xmlNode):
    """
     Here we check if the parameters read by the global reader are compatible with this type of Data
     @ In, ElementTree object, xmlNode
     @ Out, None
    """
    if set(self._dataParameters.keys()).issubset(['operator','outputRow']): self.raiseAnError(IOError,"Inputted operator or outputRow attributes are available for Point and PointSet only!")

  def addSpecializedReadingSettings(self):
    """
      This function adds in the _dataParameters dict the options needed for reading and constructing this class
      @ In,  None
      @ Out, None
    """
    if self._dataParameters['hierarchical']: self._dataParameters['type'] = 'History'
    else: self._dataParameters['type'] = self.type # store the type into the _dataParameters dictionary
    try: sourceType = self._toLoadFromList[-1].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      self._dataParameters['type']      =  self.type
      self._dataParameters['filter'   ] = 'whole'

  def checkConsistency(self):
    """
      Here we perform the consistency check for the structured data HistorySet
      @ In,  None
      @ Out, None
    """
    lenMustHave = 0
    sourceType = self._toLoadFromList[-1].type
    # here we assume that the outputs are all read....so we need to compute the total number of time point sets
    for sourceLoad in self._toLoadFromList:
      if'HDF5' == sourceLoad.type:  lenMustHave = lenMustHave + len(sourceLoad.getEndingGroupNames())
      elif isinstance(sourceLoad,Files.File): lenMustHave += 1
      else:
        self.raiseAnError(Exception,'The type ' + sourceLoad.type + ' is unknown!')

    if self._dataParameters['hierarchical']:
      for key in self._dataContainer['inputs'].keys():
        if (self._dataContainer['inputs'][key].size) != 1:
          self.raiseAnError(NotConsistentData,'The input parameter value, for key ' + key + ' has not a consistent shape for History in HistorySet ' + self.name + '!! It should be a single value since we are in hierarchical mode.' + '.Actual size is ' + str(len(self._dataContainer['inputs'][key])))
      for key in self._dataContainer['outputs'].keys():
        if (self._dataContainer['outputs'][key].ndim) != 1:
          self.raiseAnError(NotConsistentData,'The output parameter value, for key ' + key + ' has not a consistent shape for History in HistorySet ' + self.name + '!! It should be an 1D array since we are in hierarchical mode.' + '.Actual dimension is ' + str(self._dataContainer['outputs'][key].ndim))
    else:
      if('HDF5' == sourceType):
        #eg = self._toLoadFromList[-1].getEndingGroupNames()
        if(lenMustHave != len(self._dataContainer['inputs'].keys())):
          self.raiseAnError(NotConsistentData,'Number of HistorySet contained in HistorySet data ' + self.name + ' != number of loading sources!!! ' + str(lenMustHave) + ' !=' + str(len(self._dataContainer['inputs'].keys())))
      else:
        if(len(self._toLoadFromList) != len(self._dataContainer['inputs'].keys())):
          self.raiseAnError(NotConsistentData,'Number of HistorySet contained in HistorySet data ' + self.name + ' != number of loading sources!!! ' + str(len(self._toLoadFromList)) + ' !=' + str(len(self._dataContainer['inputs'].keys())))
      for key in self._dataContainer['inputs'].keys():
        for key2 in self._dataContainer['inputs'][key].keys():
          if (self._dataContainer['inputs'][key][key2].size) != 1:
            self.raiseAnError(NotConsistentData,'The input parameter value, for key ' + key2 + ' has not a consistent shape for History ' + key + ' contained in HistorySet ' +self.name+ '!! It should be a single value.' + '.Actual size is ' + str(len(self._dataContainer['inputs'][key][key2])))
      for key in self._dataContainer['outputs'].keys():
        for key2 in self._dataContainer['outputs'][key].keys():
          if (self._dataContainer['outputs'][key][key2].ndim) != 1:
            self.raiseAnError(NotConsistentData,'The output parameter value, for key ' + key2 + ' has not a consistent shape for History ' + key + ' contained in HistorySet ' +self.name+ '!! It should be an 1D array.' + '.Actual dimension is ' + str(self._dataContainer['outputs'][key][key2].ndim))

  def _updateSpecializedInputValue(self,name,value,options=None):
    """
      This function performs the updating of the values (input space) into this Data
      @ In,  name, either 1) list (size = 2), name[0] == history number(ex. 1 or 2 etc) - name[1], parameter name (ex. cladTemperature)
                       or 2) string, parameter name (ex. cladTemperature) -> in this second case,the parameter is added in the last history (if not present),
                                                                             otherwise a new history is created and the new value is inserted in it
      @ In, value, newer value
      @ Out, None
    """
    if (not isinstance(value,(float,int,bool,np.ndarray))):
      self.raiseAnError(NotConsistentData,'HistorySet Data accepts only a numpy array (dim 1) or a single value for method <_updateSpecializedInputValue>. Got type ' + str(type(value)))
    if isinstance(value,np.ndarray):
      if value.size != 1: self.raiseAnError(NotConsistentData,'HistorySet Data accepts only a numpy array of dim 1 or a single value for method <_updateSpecializedInputValue>. Size is ' + str(value.size))

    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'History' has been stored
      parentID = None
      if type(name) == list:
        namep = name[1]
        if type(name[0]) == str: nodeid = name[0]
        else:
          if 'metadata' in options.keys():
            nodeid = options['metadata']['prefix']
            if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
          else:
            nodeid = options['prefix']
            if 'parentID' in options.keys(): parentID = options['parentID']
      else:
        if 'metadata' in options.keys():
          nodeid = options['metadata']['prefix']
          if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
        else:
          nodeid = options['prefix']
          if 'parentID' in options.keys(): parentID = options['parentID']
        namep = name
      if parentID: tsnode = self.retrieveNodeInTreeMode(nodeid, parentID)
      else:         tsnode = self.retrieveNodeInTreeMode(nodeid)
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if namep in self._dataContainer['inputs'].keys():
        self._dataContainer['inputs'].pop(name)
      if namep not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(namep)
      self._dataContainer['inputs'][namep] = c1darray(values=np.atleast_1d(value)) # np.atleast_1d(np.array(value))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if type(name) == list:
        # there are info regarding the history number
        if name[0] in self._dataContainer['inputs'].keys():
          gethistory = self._dataContainer['inputs'].pop(name[0])
          popped = gethistory[name[1]]
          if name[1] in popped.keys():
            gethistory[name[1]] = c1darray(values=np.atleast_1d(np.array(value,dtype=float))) #np.atleast_1d(np.array(value))
            self._dataContainer['inputs'][name[0]] = gethistory
        else:
          self._dataContainer['inputs'][name[0]] = {name[1]:c1darray(values=np.atleast_1d(np.array(value,dtype=float)))}
      else:
        # no info regarding the history number => use internal counter
        if len(self._dataContainer['inputs'].keys()) == 0: self._dataContainer['inputs'][1] = {name:c1darray(values=np.atleast_1d(np.array(value,dtype=float)))}
        else:
          hisn = max(self._dataContainer['inputs'].keys())
          if name in list(self._dataContainer['inputs'].values())[-1]:
            hisn += 1
            self._dataContainer['inputs'][hisn] = {}
          self._dataContainer['inputs'][hisn][name] = c1darray(values=np.atleast_1d(np.array(value,dtype=float))) # np.atleast_1d(np.array(value))

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
      if type(name) == list:
        if type(name[0]) == str: nodeid = name[0]
        else:
          if 'metadata' in options.keys():
            nodeid = options['metadata']['prefix']
            if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
          else:
            nodeid = options['prefix']
            if 'parentID' in options.keys(): parentID = options['parentID']
      else:
        if 'metadata' in options.keys():
          nodeid = options['metadata']['prefix']
          if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
        else:
          nodeid = options['prefix']
          if 'parentID' in options.keys(): parentID = options['parentID']
      if parentID: tsnode = self.retrieveNodeInTreeMode(nodeid, parentID)
      #if 'parentID' in options.keys(): tsnode = self.retrieveNodeInTreeMode(options['prefix'], options['parentID'])
      #else:                             tsnode = self.retrieveNodeInTreeMode(options['prefix'])
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'metadata':{}})
        self._dataContainer = tsnode.get('dataContainer')
      else:
        if 'metadata' not in self._dataContainer.keys(): self._dataContainer['metadata'] ={}
      if name in self._dataContainer['metadata'].keys(): self._dataContainer['metadata'][name].append(np.atleast_1d(np.array(value))) #= copy.copy(np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value))))
      else                                             : self._dataContainer['metadata'][name] = copy.copy(c1darray(values=np.atleast_1d(np.array(value)),dtype=type(value)))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['metadata'].keys():
        self._dataContainer['metadata'][name].append(np.atleast_1d(value)) # = copy.copy(np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value))))
      else                                             : self._dataContainer['metadata'][name] = copy.copy(c1darray(values=np.atleast_1d(np.array(value)),dtype=type(value)))

  def _updateSpecializedOutputValue(self,name,value,options=None):
    """
      This function performs the updating of the values (output space) into this Data
      @ In,  name, either 1) list (size = 2), name[0] == history number(ex. 1 or 2 etc) - name[1], parameter name (ex. cladTemperature)
                       or 2) string, parameter name (ex. cladTemperature) -> in this second case,the parameter is added in the last history (if not present),
                                                                             otherwise a new history is created and the new value is inserted in it
      @ Out, None
    """
    if not isinstance(value,np.ndarray):
        self.raiseAnError(NotConsistentData,'HistorySet Data accepts only numpy array as type for method <_updateSpecializedOutputValue>. Got ' + str(type(value)))

    if options and self._dataParameters['hierarchical']:
      parentID = None
      if type(name) == list:
        namep = name[1]
        if type(name[0]) == str: nodeid = name[0]
        else:
          if 'metadata' in options.keys():
            nodeid = options['metadata']['prefix']
            if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
          else:
            nodeid = options['prefix']
            if 'parentID' in options.keys(): parentID = options['parentID']
      else:
        if 'metadata' in options.keys():
          nodeid = options['metadata']['prefix']
          if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
        else:
          nodeid = options['prefix']
          if 'parentID' in options.keys(): parentID = options['parentID']
        namep = name
      if parentID: tsnode = self.retrieveNodeInTreeMode(nodeid, parentID)

      # we store the pointer to the container in the self._dataContainer because checkConsistency acts on this
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if namep in self._dataContainer['outputs'].keys(): self._dataContainer['outputs'].pop(namep)
      if namep not in self._dataParameters['inParam']: self._dataParameters['outParam'].append(namep)
      self._dataContainer['outputs'][namep] = c1darray(values=np.atleast_1d(np.array(value,dtype=float))) #np.atleast_1d(np.array(value))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if type(name) == list:
        # there are info regarding the history number
        if name[0] in self._dataContainer['outputs'].keys():
          gethistory = self._dataContainer['outputs'].pop(name[0])
          popped = gethistory[name[1]]
          if name[1] in popped.keys():
            gethistory[name[1]] = np.atleast_1d(np.array(value))
            self._dataContainer['outputs'][name[0]] =gethistory
        else:
          self._dataContainer['outputs'][name[0]] = {name[1]:c1darray(values=np.atleast_1d(np.array(value,dtype=float)))} #np.atleast_1d(np.array(value))}
      else:
        # no info regarding the history number => use internal counter
        if len(self._dataContainer['outputs'].keys()) == 0: self._dataContainer['outputs'][1] = {name:c1darray(values=np.atleast_1d(np.array(value,dtype=float)))} #np.atleast_1d(np.array(value))}
        else:
          hisn = max(self._dataContainer['outputs'].keys())
          if name in list(self._dataContainer['outputs'].values())[-1]:
            hisn += 1
            self._dataContainer['outputs'][hisn] = {}
          self._dataContainer['outputs'][hisn][name] = copy.copy(c1darray(values=np.atleast_1d(np.array(value,dtype=float)))) #np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal,options):
    """
      This function prints a CSV file with the content of this class (Input and Output space)
      @ In,  filenameLocal, string, filename root (for example, 'homo_homini_lupus' -> the final file name is gonna be called 'homo_homini_lupus.csv')
      @ In,  options, dictionary, dictionary of printing options
      @ Out, None (a csv is gonna be printed)
    """

    if self._dataParameters['hierarchical']:
      outKeys   = []
      inpKeys   = []
      inpValues = []
      outValues = []
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
              axa = O_o[key][0]['outputs'][variableName]
              for index in range(len(O_o[key])-1): axa = np.concatenate((axa,O_o[key][index+1]['outputs'][variableName]))
              outValues[-1].append(axa)
        else:
          inpKeys[-1] = O_o[key][0]['inputs'].keys()
          outKeys[-1] = O_o[key][0]['outputs'].keys()
          for var in O_o[key][0]['inputs'].keys():
            axa = np.zeros(len(O_o[key]))
            for index in range(len(O_o[key])): axa[index] = O_o[key][index]['inputs'][var][0]
            inpValues[-1].append(axa)
          for var in O_o[key][0]['outputs'].keys():
            axa = O_o[key][0]['outputs'][var]
            for index in range(len(O_o[key])-1):
              axa = np.concatenate((axa,O_o[key][index+1]['outputs'][var]))
            outValues[-1].append(axa)

        if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '_' + key + '.csv', 'w')
        else: return
        myFile.write('Ending branch,'+key+'\n')
        myFile.write('branch #')
        for item in inpKeys[-1]:
          myFile.write(',' + item)
        myFile.write('\n')
        # write the input paramters' values for each branch
        for i in range(inpValues[-1][0].size):
          myFile.write(str(i+1))
          for index in range(len(inpValues[-1])):
            myFile.write(',' + str(inpValues[-1][index][i]))
          myFile.write('\n')
        # write out keys
        myFile.write('\n')
        myFile.write('TimeStep #')
        for item in outKeys[-1]:
          myFile.write(',' + item)
        myFile.write('\n')
        for i in range(outValues[-1][0].size):
          myFile.write(str(i+1))
          for index in range(len(outValues[-1])):
            myFile.write(',' + str(outValues[-1][index][i]))
          myFile.write('\n')
        myFile.close()
    else:
      #if not hierarchical
      #For HistorySet, create an XML file, and multiple CSV
      #files.  The first CSV file has a header with the input names,
      #and a column for the filenames.  There is one CSV file for each
      #data line in the first CSV and they are named with the
      #filename.  They have the output names for a header, a column
      #for time, and the rest of the file is data for different times.
      inpValues = list(self._dataContainer['inputs'].values())
      outKeys   = self._dataContainer['outputs'].keys()
      outValues = list(self._dataContainer['outputs'].values())
      #Create Input file
      myFile = open(filenameLocal + '.csv','w')
      for n in range(len(outKeys)):
        inpKeys_h   = []
        inpValues_h = []
        outKeys_h   = []
        outValues_h = []
        if 'what' in options.keys():
          for var in options['what']:
            splitted = var.split('|')
            variableName = "|".join(splitted[1:])
            varType = splitted[0]
            if varType == 'input':
              inpKeys_h.append(variableName)
              inpValues_h.append(inpValues[n][variableName])
            if varType == 'output':
              outKeys_h.append(var.split('|')[1])
              outValues_h.append(outValues[n][variableName])
        else:
          inpKeys_h   = list(inpValues[n].keys())
          inpValues_h = list(inpValues[n].values())
          outKeys_h   = list(outValues[n].keys())
          outValues_h = list(outValues[n].values())

        dataFilename = filenameLocal + '_'+ str(n) + '.csv'
        if len(inpKeys_h) > 0 or len(outKeys_h) > 0: myDataFile = open(dataFilename, 'w')
        else: return #XXX should this just skip this iteration?
        #Write header for main file
        if n == 0:
          myFile.write(','.join([item for item in
                                  itertools.chain(inpKeys_h,['filename'])]))
          myFile.write('\n')
          self._createXMLFile(filenameLocal,'HistorySet',inpKeys_h,outKeys_h)
        myFile.write(','.join([str(item[0]) for item in
                                itertools.chain(inpValues_h,[[dataFilename]])]))
        myFile.write('\n')
        #Data file
        #Print time + output values
        myDataFile.write(','.join([item for item in outKeys_h]))
        if len(outKeys_h) > 0:
          myDataFile.write('\n')
          for j in range(outValues_h[0].size):
            myDataFile.write(','.join([str(item[j]) for item in
                                    outValues_h]))
            myDataFile.write('\n')
        myDataFile.close()
      myFile.close()

  def _specializedLoadXMLandCSV(self, filenameRoot, options):
    #For HistorySet, create an XML file, and multiple CSV
    #files.  The first CSV file has a header with the input names,
    #and a column for the filenames.  There is one CSV file for each
    #data line in the first CSV and they are named with the
    #filename.  They have the output names for a header, a column
    #for time, and the rest of the file is data for different times.
    if options is not None and 'fileToLoad' in options.keys():
      name = os.path.join(options['fileToLoad'].getPath(),options['fileToLoad'].getBase())
    else: name = self.name
    filenameLocal = os.path.join(filenameRoot,name)
    xmlData = self._loadXMLFile(filenameLocal)
    assert(xmlData["fileType"] == "HistorySet")
    if "metadata" in xmlData:
      self._dataContainer['metadata'] = xmlData["metadata"]
    mainCSV = os.path.join(filenameRoot,xmlData["filenameCSV"])
    myFile = open(mainCSV,"rU")
    header = myFile.readline().rstrip()
    inpKeys = header.split(",")[:-1]
    inpValues = []
    outKeys = []
    outValues = []
    for mainLine in myFile.readlines():
      mainLineList = mainLine.rstrip().split(",")
      inpValues_h = [utils.partialEval(a) for a in mainLineList[:-1]]
      inpValues.append(inpValues_h)
      dataFilename = mainLineList[-1]
      subCSVFilename = os.path.join(filenameRoot,dataFilename)
      myDataFile = open(subCSVFilename, "rU")
      header = myDataFile.readline().rstrip()
      outKeys_h = header.split(",")
      outValues_h = [[] for a in range(len(outKeys_h))]
      for line in myDataFile.readlines():
        lineList = line.rstrip().split(",")
        for i in range(len(outKeys_h)):
          outValues_h[i].append(utils.partialEval(lineList[i]))
      myDataFile.close()
      outKeys.append(outKeys_h)
      outValues.append(outValues_h)
    self._dataContainer['inputs'] = {} #XXX these are indexed by 1,2,...
    self._dataContainer['outputs'] = {} #XXX these are indexed by 1,2,...
    for i in range(len(inpValues)):
      mainKey = i + 1
      subInput = {}
      subOutput = {}
      for key,value in zip(inpKeys,inpValues[i]):
        subInput[key] = c1darray(values=np.array([value]*len(outValues[0][0])))
      for key,value in zip(outKeys[i],outValues[i]):
        subOutput[key] = c1darray(values=np.array(value))
      self._dataContainer['inputs'][mainKey] = subInput
      self._dataContainer['outputs'][mainKey] = subOutput

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    """
      override of the method in the base class DataObjects
      @ In,  myType, string, unused
      @ In,  inOutType
      IMPLEMENT COMMENT HERE
    """
    if varTyp!='numpy.ndarray':
      if varName in self._dataParameters['inParam']:
        if varID!=None: exec ('return varTyp(self.getParam('+inOutType+','+str(varID)+')[varName]')
        else: self.raiseAnError(RuntimeError,'to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID to identify the history (varID missed)')
      else:
        if varID!=None:
          if stepID!=None and type(stepID)!=tuple: exec ('return varTyp(self.getParam('+inOutType+','+str(varID)+')[varName][stepID]')
          else: self.raiseAnError(RuntimeError,'to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID of the input set used and a time coordinate (time or timeID missed or tuple)')
        else: self.raiseAnError(RuntimeError,'to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID of the input set used (varID missed)')
    else:
      if varName in self._dataParameters['inParam']:
        myOut=np.zeros(len(self.getInpParametersValues().keys()))
        for key in self.getInpParametersValues().keys():
          myOut[int(key)]=self.getParam(inOutType,key)[varName][0]
        return myOut
      else:
        if varID!=None:
          if stepID==None:
            return self.getParam(inOutType,varID)[varName]
          elif type(stepID)==tuple:
            if stepID[1]==None: return self.getParam(inOutType,varID)[varName][stepID[0]:]
            else: return self.getParam(inOutType,varID)[varName][stepID[0]:stepID[1]]
          else: return self.getParam(inOutType,varID)[varName][stepID]
        else:
          if stepID==None: self.raiseAnError(RuntimeError,'more info needed trying to extract '+varName+' from data '+self.name)
          elif type(stepID)==tuple:
            if stepID[1]!=None:
              myOut=np.zeros((len(self.getOutParametersValues().keys()),stepID[1]-stepID[0]))
              for key in self.getOutParametersValues().keys():
                myOut[int(key),:]=self.getParam(inOutType,key)[varName][stepID[0]:stepID[1]]
            else: self.raiseAnError(RuntimeError,'more info needed trying to extract '+varName+' from data '+self.name)
          else:
            myOut=np.zeros(len(self.getOutParametersValues().keys()))
            for key in self.getOutParametersValues().keys():
              myOut[int(key)]=self.getParam(inOutType,key)[varName][stepID]
            return myOut
