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
import abc
import ast
import copy
import numpy as np
import xml.etree.ElementTree as ET
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from cached_ndarray import c1darray
from Csv_loader import CsvLoader as ld
import Files
import TreeStructure as TS
import utils
#Internal Modules End--------------------------------------------------------------------------------

# Custom exceptions
class NotConsistentData(Exception): pass
class ConstructError(Exception)   : pass

class Data(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  """
  The Data object is the base class for constructing derived data object classes.
  It provides the common interfaces to access and to add data values into the RAVEN internal object format.
  This object is "understood" by all the "active" modules (e.g. postprocessors, models, etc) and represents the way
  RAVEN shares the information among the framework
  """
  def __init__(self):
    BaseType.__init__(self)
    self._dataParameters                 = {}                         # in here we store all the data parameters (inputs params, output params,etc)
    self._dataParameters['inParam'     ] = []                         # inParam list
    self._dataParameters['outParam'    ] = []                         # outParam list
    self._dataParameters['hierarchical'] = False                      # the structure of this data is hierarchical?
    self._toLoadFromList                 = []                         # loading source
    self._dataContainer                  = {'inputs':{},'outputs':{}} # Dict that contains the actual data. self._dataContainer['inputs'] contains the input space, self._dataContainer['output'] the output space
    self._dataContainer['metadata'     ] = {}                         # In this dictionary we store metadata (For example, probability,input file names, etc)
    self.metaAdditionalInOrOut           = ['PointProbability','ProbabilityWeight']            # list of metadata keys that will be printed in the CSV one
    self.acceptHierarchy                 = False                      # flag to tell if a sub-type accepts hierarchy
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

  def _readMoreXML(self,xmlNode):
    """
    Function to read the xml input block.
    @ In, xmlNode, xml node
    """
    # retrieve input/outputs parameters' keywords
    self._dataParameters['inParam']  = list(inp.strip() for inp in xmlNode.find('Input' ).text.strip().split(','))
    self._dataParameters['outParam'] = list(out.strip() for out in xmlNode.find('Output').text.strip().split(','))
    #test for keywords not allowed
    if len(set(self._dataParameters['inParam'])&set(self.notAllowedInputs))!=0  : self.raiseAnError(IOError,'the keyword '+str(set(self._dataParameters['inParam'])&set(self.notAllowedInputs))+' is not allowed among inputs')
    if len(set(self._dataParameters['outParam'])&set(self.notAllowedOutputs))!=0: self.raiseAnError(IOError,'the keyword '+str(set(self._dataParameters['outParam'])&set(self.notAllowedOutputs))+' is not allowed among inputs')
    #test for same input/output variables name
    if len(set(self._dataParameters['inParam'])&set(self._dataParameters['outParam']))!=0: self.raiseAnError(IOError,'It is not allowed to have the same name of input/output variables in the data '+self.name+' of type '+self.type)
    optionsData = xmlNode.find('options')
    if optionsData != None:
      for child in optionsData: self._dataParameters[child.tag] = child.text
    if set(self._dataParameters.keys()).issubset(['inputRow','inputPivotValue'])             : self.raiseAnError(IOError,'It is not allowed to simultaneously specify the nodes: inputRow and inputPivotValue!')
    if set(self._dataParameters.keys()).issubset(['outputRow','outputPivotValue','operator']): self.raiseAnError(IOError,'It is not allowed to simultaneously specify the nodes: outputRow, outputPivotValue and operator!')
    self._specializedInputCheck(xmlNode)
    if 'hierarchical' in xmlNode.attrib.keys():
      if xmlNode.attrib['hierarchical'].lower() in utils.stringsThatMeanTrue(): self._dataParameters['hierarchical'] = True
      else                                                                    : self._dataParameters['hierarchical'] = False
      if self._dataParameters['hierarchical'] and not self.acceptHierarchical():
        self.raiseAWarning('hierarchical fashion is not available (No Sense) for Data named '+ self.name + 'of type ' + self.type + '!!!')
        self._dataParameters['hierarchical'] = False
      else: self.TSData, self.rootToBranch = None, {}
    else: self._dataParameters['hierarchical'] = False

  def _specializedInputCheck(self,xmlNode):
    """
    Function to check the input parameters that have been read for each DataObject subtype
    @ In, None
    @ Out, None
    """
    pass

  def addInitParams(self,tempDict):
    """
    Function to get the input params that belong to this class
    @ In, tempDict, temporary dictionary
    """
    for i in range(len(self._dataParameters['inParam' ])):  tempDict['Input_'+str(i)]  = self._dataParameters['inParam' ][i]
    for i in range(len(self._dataParameters['outParam'])):  tempDict['Output_'+str(i)] = self._dataParameters['outParam'][i]
    for key,value in self._dataParameters.items(): tempDict[key] = value
    return tempDict

  def removeInputValue(self,name):
    """
    Function to remove a value from the dictionary inpParametersValues
    @ In, name, parameter name
    """
    if self._dataParameters['hierarchical']:
      for TSData in self.TSData.values():
        for node in list(TSData.iter('*')):
          if name in node.get('dataContainer')['inputs'].keys(): node.get('dataContainer')['inputs'].pop(name)
    else:
      if name in self._dataContainer['inputs'].keys(): self._dataContainer['inputs'].pop(name)

  def removeOutputValue(self,name):
    """
    Function to remove a value from the dictionary outParametersValues
    @ In, name, parameter name
    """
    if self._dataParameters['hierarchical']:
      for TSData in self.TSData.values():
        for node in list(TSData.iter('*')):
          if name in node.get('dataContainer')['outputs'].keys(): node.get('dataContainer')['outputs'].pop(name)
    else:
      if name in self._dataContainer['outputs'].keys(): self._dataContainer['outputs'].pop(name)

  def updateInputValue(self,name,value,options=None):
    """
    Function to update a value from the input dictionary
    @ In, name, parameter name
    @ In, value, the new value
    @ In, parentID, optional, parent identifier in case Hierarchical fashion has been requested
    """
    self._updateSpecializedInputValue(name,value,options)

  def updateOutputValue(self,name,value,options=None):
    """
    Function to update a value from the output dictionary
    @ In, name, parameter name
    @ In, value, the new value
    @ In, parentID, optional, parent identifier in case Hierarchical fashion has been requested
    """
    self._updateSpecializedOutputValue(name,value,options)

  def updateMetadata(self,name,value,options=None):
    """
    Function to update a value from the dictionary metadata
    @ In, name, parameter name
    @ In, value, the new value
    @ In, parentID, optional, parent identifier in case Hierarchical fashion has been requested
    """
    self._updateSpecializedMetadata(name,value,options)

  def getMetadata(self,keyword,nodeid=None,serialize=False):
    """
    Function to get a value from the dictionary metadata
    @ In, keyword, parameter name
    @ In, nodeid, optional, id of the node if hierarchical
    @ In, serialize, optional, serialize the tree if in hierarchical mode
    @ Out, return the metadata
    """
    if self._dataParameters['hierarchical']:
      if type(keyword) == int: return list(self.getHierParam('metadata',nodeid,None,serialize).values())[keyword-1]
      else: return self.getHierParam('metadata',nodeid,keyword,serialize)
    else:
      if keyword in self._dataContainer['metadata'].keys(): return self._dataContainer ['metadata'][keyword]
      else: self.raiseAnError(RuntimeError,'parameter ' + str(keyword) + ' not found in metadata dictionary. Available keys are '+str(self._dataContainer['metadata'].keys())+'.Function: Data.getMetadata')

  def getAllMetadata(self,nodeid=None,serialize=False):
    """
    Function to get all the metadata
    @ In, nodeid, optional, id of the node if hierarchical
    @ In, serialize, optional, serialize the tree if in hierarchical mode
    @ Out, return the metadata (s)
    """
    if self._dataParameters['hierarchical']: return self.getHierParam('metadata',nodeid,None,serialize)
    else                                   : return self._dataContainer['metadata']

  @abc.abstractmethod
  def addSpecializedReadingSettings(self):
    """
      This function is used to add specialized attributes to the data in order to retrieve the data properly.
      Every specialized data needs to overwrite it!!!!!!!!
    """
    pass

  @abc.abstractmethod
  def checkConsistency(self):
    """
      This function checks the consistency of the data structure... every specialized data needs to overwrite it!!!!!
    """
    pass

  def acceptHierarchical(self):
    """
      This function returns a boolean. True if the specialized Data accepts the hierarchical structure
    """
    return self.acceptHierarchy

  def __getVariablesToPrint(self,var,inOrOut):
    """
    Returns a list of variables to print.
    Takes the variable and either 'input' or 'output'
    In addition, if the variable belong to the metadata and metaAdditionalInOrOut, it will also return to print
    """
    variablesToPrint = []
    lvar = var.lower()
    inOrOuts = inOrOut + 's'
    if lvar == inOrOut:
      if type(list(self._dataContainer[inOrOuts].values())[0]) == dict: varKeys = list(self._dataContainer[inOrOuts].values())[0].keys()
      else: varKeys = self._dataContainer[inOrOuts].keys()
      for invar in varKeys: variablesToPrint.append(inOrOut+'|'+str(invar))
    elif '|' in var and lvar.startswith(inOrOut+'|'):
      varName = var.split('|')[1]
      # get the variables from the metadata if the variables are in the list metaAdditionalInOrOut
      if varName in self.metaAdditionalInOrOut:
        varKeys = self._dataContainer['metadata'].keys()
        if varName not in varKeys: self.raiseAnError(RuntimeError,'variable ' + varName + ' is not present among the ' +inOrOuts+' of Data ' + self.name)
        if type(self._dataContainer['metadata'][varName]) not in self.metatype:
          self.raiseAnError(NotConsistentData,inOrOut + var.split('|')[1]+' not compatible with CSV output. Its type needs to be one of '+str(self.metatype))
        else: variablesToPrint.append('metadata'+'|'+str(varName))
      else:
        if type(list(self._dataContainer[inOrOuts].values())[0]) == dict: varKeys = list(self._dataContainer[inOrOuts].values())[0].keys()
        else: varKeys = self._dataContainer[inOrOuts].keys()
        if varName not in varKeys: self.raiseAnError(RuntimeError,'variable ' + varName + ' is not present among the '+inOrOuts+' of Data ' + self.name)
        else: variablesToPrint.append(inOrOut+'|'+str(varName))
    else: self.raiseAnError(RuntimeError,'unexpected variable '+ var)
    return variablesToPrint

  def printCSV(self,options=None):
    """
    Function used to dump the data into a csv file
    Every class must implement the specializedPrintCSV method
    that is going to be called from here
    @ In, OPTIONAL, options, dictionary of options... it can contain the filename to be used, the parameters need to be printed....
    """
    optionsInt = {}
    # print content of data in a .csv format
    self.raiseADebug(' '*len(self.printTag)+':=============================')
    self.raiseADebug(' '*len(self.printTag)+':DataObjects: print on file(s)')
    self.raiseADebug(' '*len(self.printTag)+':=============================')
    if options:
      if ('filenameroot' in options.keys()): filenameLocal = options['filenameroot']
      else: filenameLocal = self.name + '_dump'
      if 'what' in options.keys():
        variablesToPrint = []
        for var in options['what'].split(','):
          lvar = var.lower()
          if lvar.startswith('input'):
            variablesToPrint.extend(self.__getVariablesToPrint(var,'input'))
          elif lvar.startswith('output'):
            variablesToPrint.extend(self.__getVariablesToPrint(var,'output'))
          else: self.raiseAnError(RuntimeError,'variable ' + var + ' is unknown in Data ' + self.name + '. You need to specify an input or a output')
        optionsInt['what'] = variablesToPrint
    else:   filenameLocal = self.name + '_dump'

    self.specializedPrintCSV(filenameLocal,optionsInt)

  def loadXMLandCSV(self,filenameRoot,options=None):
    """
    Function to load the xml additional file of the csv for data
    (it contains metadata, etc)
    @ In, filenameRoot, file name
    @ In, options, optional, dictionary -> options for loading
    """
    self._specializedLoadXMLandCSV(filenameRoot,options)

  def _specializedLoadXMLandCSV(self,filenameRoot,options):
    """
    Function to load the xml additional file of the csv for data
    (it contains metadata, etc). It must be implemented by the specialized classes
    @ In, filenameRoot, file name
    @ In, options, optional, dictionary -> options for loading
    """
    self.raiseAnError(RuntimeError,"specializedloadXMLandCSV not implemented "+str(self))

  def _createXMLFile(self,filenameLocal,fileType,inpKeys,outKeys):
    """
    Creates an XML file to contain the input and output data list
    @ In, filenameLocal, file name
    @ In, fileType, file type (csv, xml)
    @ In, inpKeys, list, input keys
    @ In, outKeys, list, output keys
    """
    myXMLFile = open(filenameLocal + '.xml', 'w')
    root = ET.Element('data',{'name':filenameLocal,'type':fileType})
    inputNode = ET.SubElement(root,'input')
    inputNode.text = ','.join(inpKeys)
    outputNode = ET.SubElement(root,'output')
    outputNode.text = ','.join(outKeys)
    filenameNode = ET.SubElement(root,'inputFilename')
    filenameNode.text = filenameLocal + '.csv'
    if len(self._dataContainer['metadata'].keys()) > 0:
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
    @ In, filenameRoot, file name
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
    if inputNode    ==  None: self.raiseAnError(RuntimeError,'input XML node not found in file ' + filenameLocal + '.xml')
    if outputNode   ==  None: self.raiseAnError(RuntimeError,'output XML node not found in file ' + filenameLocal + '.xml')
    if filenameNode ==  None: self.raiseAnError(RuntimeError,'inputFilename XML node not found in file ' + filenameLocal + '.xml')
    retDict["inpKeys"] = inputNode.text.split(",")
    retDict["outKeys"] = outputNode.text.split(",")
    retDict["filenameCSV"] = filenameNode.text
    metadataNode = root.find("metadata")
    if metadataNode:
      metadataDict = {}
      for child in metadataNode:
        key = child.tag
        value = child.text
        value.replace('\n','')
        # ast.literal_eval can't handle numpy arrays, so we'll handle that.
        if value.startswith('array('):
          isArray=True
          value=value.split('dtype')[0].lstrip('ary(').rstrip('),\n ')
        else: isArray = False
        try: value = ast.literal_eval(value)
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

  def addOutput(self,toLoadFrom,options=None):
    """
      Function to construct a data from a source
      @ In, toLoadFrom, loading source, it can be an HDF5 database, a csv file and in the future a xml file
      @ In, options, it's a dictionary of options. For example useful for metadata storing or,
                     in case an hierarchical fashion has been requested, it must contain the parentID and the name of the actual 'branch'
    """
    self._toLoadFromList.append(toLoadFrom)
    self.addSpecializedReadingSettings()
    self._dataParameters['SampledVars'] = copy.deepcopy(options['metadata']['SampledVars']) if options != None and 'metadata' in options.keys() and 'SampledVars' in options['metadata'].keys() else None
    self.raiseAMessage('Object type ' + self._toLoadFromList[-1].type + ' named "' + self._toLoadFromList[-1].name+'"')
    if(self._toLoadFromList[-1].type == 'HDF5'):
      tupleVar = self._toLoadFromList[-1].retrieveData(self._dataParameters)
      if options:
        parentID = options['metadata']['parentID'] if 'metadata' in options.keys() and 'parentID' in options['metadata'].keys() else (options['parentID'] if 'parentID' in options.keys() else None)
        if parentID and self._dataParameters['hierarchical']:
          self.raiseAWarning('-> Data storing in hierarchical fashion from HDF5 not yet implemented!')
          self._dataParameters['hierarchical'] = False
    elif (isinstance(self._toLoadFromList[-1],Files.File)): tupleVar = ld(self.messageHandler).csvLoadData([toLoadFrom],self._dataParameters)
    else: self.raiseAnError(ValueError, "Type "+self._toLoadFromList[-1].type+ "from which the DataObject "+ self.name +" should be constructed is unknown!!!")

    for hist in tupleVar[0].keys():
      if type(tupleVar[0][hist]) == dict:
        for key in tupleVar[0][hist].keys(): self.updateInputValue(key, tupleVar[0][hist][key], options)
      else:
        if self.type in ['Point','PointSet']:
          for index in range(tupleVar[0][hist].size): self.updateInputValue(hist, tupleVar[0][hist][index], options)
        else: self.updateInputValue(hist, tupleVar[0][hist], options)
    for hist in tupleVar[1].keys():
      if type(tupleVar[1][hist]) == dict:
        for key in tupleVar[1][hist].keys(): self.updateOutputValue(key, tupleVar[1][hist][key], options)
      else:
        if self.type in ['Point','PointSet']:
          for index in range(tupleVar[1][hist].size): self.updateOutputValue(hist, tupleVar[1][hist][index], options)
        else: self.updateOutputValue(hist, tupleVar[1][hist], options)
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
                else: self.raiseAnError(IOError,'unknown type for metadata adding process. Relevant type = '+ str(elem))

        else:
          if tupleVar[2][hist]: self.raiseAnError(IOError,'unknown type for metadata adding process. Relevant type = '+ str(type(tupleVar[2][hist])))
    self.checkConsistency()
    return

  def getParametersValues(self,typeVar,nodeid=None, serialize=False):
    """
    Functions to get the parameter values
    @ In, variable type (input or output)
    """
    if    typeVar.lower() in 'inputs' : return self.getInpParametersValues(nodeid,serialize)
    elif  typeVar.lower() in 'outputs': return self.getOutParametersValues(nodeid,serialize)
    else: self.raiseAnError(RuntimeError,'type ' + typeVar + ' is not a valid type. Function: Data.getParametersValues')

  def getParaKeys(self,typePara):
    """
    Functions to get the parameter keys
    @ In, typePara, variable type (input or output)
    """
    if   typePara.lower() in 'inputs' : return self._dataParameters['inParam' ]
    elif typePara.lower() in 'outputs': return self._dataParameters['outParam']
    else: self.raiseAnError(RuntimeError,'type ' + typePara + ' is not a valid type. Function: Data.getParaKeys')

  def isItEmpty(self):
    """
    Function to check if the data is empty
    @ In, None
    """
    if len(self.getInpParametersValues().keys()) == 0 and len(self.getOutParametersValues()) == 0: return True
    else:                                                                                          return False

  def __len__(self):
    """
    Overriding of the __len__ method for data.
    len(dataobject) is going to return the size of the first output element found in the self._dataParameters['outParams']
    @ In, None
    @ Out, integer, size of first output element
    """
    if len(self._dataParameters['outParam']) == 0:
      return 0
    else:
      if self.type != "HistorySet":
        return self.sizeData('output',keyword=self._dataParameters['outParam'][0])[self._dataParameters['outParam'][0]]
      else:
        return self.sizeData('output',keyword=1)[1]

  def sizeData(self,typeVar,keyword=None,nodeid=None,serialize=False):
    """
    Function to get the size of the Data.
    @ In, typeVar, string, required, variable type (input/inputs, output/outputs, metadata)
    @ In, keyword, string, optional, variable keyword. If None, the sizes of each variables are returned
    @ In, nodeid, string, optional, id of the node if hierarchical
    @ In, serialize, string, optional, serialize the tree if in hierarchical mode
    @ Out, dictionary, keyword:size
    """
    outcome   = {}
    emptyData = False
    if self.isItEmpty(): emptyData = True
    if typeVar.lower() in ['input','inputs','output','outputs']:
      if keyword != None:
        if not emptyData: outcome[keyword] = len(self.getParam(typeVar,keyword,nodeid,serialize))
        else            : outcome[keyword] = 0
      else:
        for key in self.getParaKeys(typeVar):
          if not emptyData: outcome[key] = len(self.getParam(typeVar,key,nodeid,serialize))
          else            : outcome[key] = 0
    elif typeVar.lower() == 'metadata':
      if keyword != None:
        if not emptyData: outcome[keyword] = len(self.getMetadata(keyword,nodeid,serialize))
        else            : outcome[keyword] = 0
      else:
        for key,value in self.getAllMetadata(nodeid,serialize):
          if not emptyData: outcome[key] = len(value)
          else            : outcome[key] = 0
    else: self.raiseAnError(RuntimeError,'type ' + typeVar + ' is not a valid type. Function: Data.sizeData')
    return outcome

  def getInpParametersValues(self,nodeid=None,serialize=False):
    """
    Function to get a reference to the input parameter dictionary
    @, In, nodeid, optional, in hierarchical mode, if nodeid is provided, the data for that node is returned,
                             otherwise check explanation for getHierParam
    @, In, serialize, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                PLEASE check explanation for getHierParam
    @, Out, Reference to self._dataContainer['inputs'] or something else in hierarchical
    """
    if self._dataParameters['hierarchical']: return self.getHierParam('inputs',nodeid,serialize=serialize)
    else:                                    return self._dataContainer['inputs']

  def getOutParametersValues(self,nodeid=None,serialize=False):
    """
    Function to get a reference to the output parameter dictionary
    @, In, nodeid, optional, in hierarchical mode, if nodeid is provided, the data for that node is returned,
                             otherwise check explanation for getHierParam
    @, In, serialize, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                PLEASE check explanation for getHierParam
    @, Out, Reference to self._dataContainer['outputs'] or something else in hierarchical
    """
    if self._dataParameters['hierarchical']: return self.getHierParam('outputs',nodeid,serialize=serialize)
    else:                                    return self._dataContainer['outputs']

  def getParam(self,typeVar,keyword,nodeid=None,serialize=False):
    """
    Function to get a reference to an output or input parameter
    @ In, typeVar, input or output
    @ In, keyword, keyword
    @ Out, Reference to the parameter
    """
    if self.type == 'HistorySet':
      acceptedType = ['str','unicode','bytes','int']
      convertArr = lambda x: x
      #convertArr = lambda x: np.asarray(x)
    else                       :
      acceptedType = ['str','unicode','bytes']
      convertArr = lambda x: np.asarray(x)

    if type(typeVar).__name__ not in ['str','unicode','bytes'] : self.raiseAnError(RuntimeError,'type of parameter typeVar needs to be a string. Function: Data.getParam')
    if type(keyword).__name__ not in acceptedType        :
      self.raiseAnError(RuntimeError,'type of parameter keyword needs to be '+str(acceptedType)+' . Function: Data.getParam')
    if nodeid:
      if type(nodeid).__name__ not in ['str','unicode','bytes']  : self.raiseAnError(RuntimeError,'type of parameter nodeid needs to be a string. Function: Data.getParam')
    if typeVar.lower() not in ['input','inout','inputs','output','outputs']: self.raiseAnError(RuntimeError,'type ' + typeVar + ' is not a valid type. Function: Data.getParam')
    if self._dataParameters['hierarchical']:
      if type(keyword) == int:
        return list(self.getHierParam(typeVar.lower(),nodeid,None,serialize).values())[keyword-1]
      else: return self.getHierParam(typeVar.lower(),nodeid,keyword,serialize)
    else:
      if typeVar.lower() in ['input','inputs']:
        returnDict = {}
        if keyword in self._dataContainer['inputs'].keys():
            returnDict[keyword] = {}
            if self.type == 'HistorySet':
                for key in self._dataContainer['inputs'][keyword].keys(): returnDict[keyword][key] = np.resize(self._dataContainer['inputs'][keyword][key],len(self._dataContainer['outputs'][keyword].values()[0]))
                return convertArr(returnDict[keyword])
            elif self.type == 'History':
                returnDict[keyword] = np.resize(self._dataContainer['inputs'][keyword],len(self._dataContainer['outputs'].values()[0]))
                return convertArr(returnDict[keyword])
            else:
                return convertArr(self._dataContainer['inputs'][keyword])
        else: self.raiseAnError(RuntimeError,self.name+' : parameter ' + str(keyword) + ' not found in inpParametersValues dictionary. Available keys are '+str(self._dataContainer['inputs'].keys())+'.Function: Data.getParam')
      elif typeVar.lower() in ['output','outputs']:
        if keyword in self._dataContainer['outputs'].keys(): return convertArr(self._dataContainer['outputs'][keyword])
        else: self.raiseAnError(RuntimeError,self.name+' : parameter ' + str(keyword) + ' not found in outParametersValues dictionary. Available keys are '+str(self._dataContainer['outputs'].keys())+'.Function: Data.getParam')

  def extractValue(self,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    """
    this a method that is used to extract a value (both array or scalar) attempting an implicit conversion for scalars
    the value is returned without link to the original
    @in varType is the requested type of the variable to be returned (bool, int, float, numpy.ndarray, etc)
    @in varName is the name of the variable that should be recovered
    @in varID is the ID of the value that should be retrieved within a set
      if varID.type!=tuple only one point along sampling of that variable is retrieved
        else:
          if varID=(int,int) the slicing is [varID[0]:varID[1]]
          if varID=(int,None) the slicing is [varID[0]:]
    @in stepID determine the slicing of an history.
        if stepID.type!=tuple only one point along the history is retrieved
        else:
          if stepID=(int,int) the slicing is [stepID[0]:stepID[1]]
          if stepID=(int,None) the slicing is [stepID[0]:]
    @in nodeid , in hierarchical mode, is the node from which the value needs to be extracted... by default is the root
    """

    myType=self.type
    if   varName in self._dataParameters['inParam' ]: inOutType = 'input'
    elif varName in self._dataParameters['outParam']: inOutType = 'output'
    else: self.raiseAnError(RuntimeError,'the variable named '+varName+' was not found in the data: '+self.name)
    return self.__extractValueLocal__(myType,inOutType,varTyp,varName,varID,stepID,nodeid)

  @abc.abstractmethod
  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    """
      this method has to be override to implement the specialization of extractValue for each data class
    """
    pass

  def getHierParam(self,typeVar,nodeid,keyword=None,serialize=False):
    """
      This function get a parameter when we are in hierarchical mode
      @ In,  typeVar,  string, it's the variable type... input,output, or inout
      @ In,  nodeid,   string, it's the node name... if == None or *, a dictionary of of data is returned, otherwise the actual node data is returned in a dict as well (see serialize attribute)
      @ In, keyword,   string, it's a parameter name (for example, cladTemperature), if None, the whole dict is returned, otherwise the parameter value is got (see serialize attribute)
      @ In, serialize, bool  , if true a sequence of PointSet is generated (a dictionary where the keys are the 'ending' branches and the values are a sorted list of _dataContainers (from first branch to the ending ones)
                               if false see explanation for nodeid
      @ Out, a dictionary of data (see above)
    """
    if type(keyword).__name__ in ['str','unicode','bytes']:
      if keyword == 'none': keyword = None
    nodesDict = {}
    if not self.TSData: return nodesDict
    if not nodeid or nodeid=='*':
      # we want all the nodes
      if serialize:
        # we want all the nodes and serialize them
        for TSData in self.TSData.values():
          for node in TSData.iterEnding():
            nodesDict[node.name] = []
            for se in list(TSData.iterWholeBackTrace(node)):
              if typeVar   in 'inout'              and not keyword: nodesDict[node.name].append( se.get('dataContainer'))
              elif typeVar in ['inputs','input']   and not keyword: nodesDict[node.name].append( se.get('dataContainer')['inputs'  ])
              elif typeVar in ['output','outputs'] and not keyword: nodesDict[node.name].append( se.get('dataContainer')['outputs' ])
              elif typeVar in 'metadata'           and not keyword: nodesDict[node.name].append( se.get('dataContainer')['metadata'])
              elif typeVar in ['inputs','input']   and     keyword: nodesDict[node.name].append( np.asarray(se.get('dataContainer')['inputs'  ][keyword]))
              elif typeVar in ['output','outputs'] and     keyword: nodesDict[node.name].append( np.asarray(se.get('dataContainer')['outputs' ][keyword]))
              elif typeVar in 'metadata'           and     keyword: nodesDict[node.name].append( np.asarray(se.get('dataContainer')['metadata'][keyword]))
      else:
        for TSData in self.TSData.values():
          for node in TSData.iter():
            if typeVar   in 'inout'              and not keyword: nodesDict[node.name] = node.get('dataContainer')
            elif typeVar in ['inputs','input']   and not keyword: nodesDict[node.name] = node.get('dataContainer')['inputs'  ]
            elif typeVar in ['output','outputs'] and not keyword: nodesDict[node.name] = node.get('dataContainer')['outputs' ]
            elif typeVar in 'metadata'           and not keyword: nodesDict[node.name] = node.get('dataContainer')['metadata']
            elif typeVar in ['inputs','input']   and     keyword: nodesDict[node.name] = np.asarray(node.get('dataContainer')['inputs'  ][keyword])
            elif typeVar in ['output','outputs'] and     keyword: nodesDict[node.name] = np.asarray(node.get('dataContainer')['outputs' ][keyword])
            elif typeVar in 'metadata'           and     keyword: nodesDict[node.name] = np.asarray(node.get('dataContainer')['metadata'][keyword])
    elif nodeid == 'ending':
      for TSDat in self.TSData.values():
        for ending in TSDat.iterEnding():
          if typeVar   in 'inout'              and not keyword: nodesDict[ending.name] = ending.get('dataContainer')
          elif typeVar in ['inputs','input']   and not keyword: nodesDict[ending.name] = ending.get('dataContainer')['inputs'  ]
          elif typeVar in ['output','outputs'] and not keyword: nodesDict[ending.name] = ending.get('dataContainer')['outputs' ]
          elif typeVar in 'metadata'           and not keyword: nodesDict[ending.name] = ending.get('dataContainer')['metadata']
          elif typeVar in ['inputs','input']   and     keyword: nodesDict[ending.name] = np.asarray(ending.get('dataContainer')['inputs'  ][keyword])
          elif typeVar in ['output','outputs'] and     keyword: nodesDict[ending.name] = np.asarray(ending.get('dataContainer')['outputs' ][keyword])
          elif typeVar in 'metadata'           and     keyword: nodesDict[ending.name] = np.asarray(ending.get('dataContainer')['metadata'][keyword])
    elif nodeid == 'RecontructEnding':
      # if history, reconstruct the history... if Point set take the last one (see below)
      backTrace = {}
      for TSData in self.TSData.values():
        for node in TSData.iterEnding():
          if self.type == 'HistorySet':
            backTrace[node.name] = []
            for se in list(TSData.iterWholeBackTrace(node)):
              if typeVar   in 'inout'              and not keyword: backTrace[node.name].append( se.get('dataContainer'))
              elif typeVar in ['inputs','input']   and not keyword: backTrace[node.name].append( se.get('dataContainer')['inputs'  ])
              elif typeVar in ['output','outputs'] and not keyword: backTrace[node.name].append( se.get('dataContainer')['outputs' ])
              elif typeVar in 'metadata'           and not keyword: backTrace[node.name].append( se.get('dataContainer')['metadata'])
              elif typeVar in ['inputs','input']   and     keyword: backTrace[node.name].append( np.asarray(se.get('dataContainer')['inputs'  ][keyword]))
              elif typeVar in ['output','outputs'] and     keyword: backTrace[node.name].append( np.asarray(se.get('dataContainer')['outputs' ][keyword]))
              elif typeVar in 'metadata'           and     keyword: backTrace[node.name].append( np.asarray(se.get('dataContainer')['metadata'][keyword]))
            #reconstruct history
            nodesDict[node.name] = None
            for element in backTrace[node.name]:
              if type(element) == dict:
                if not nodesDict[node.name]: nodesDict[node.name] = {}
                for innerkey in element.keys():
                  if type(element[innerkey]) == dict:
                    #inputs outputs metadata
                    if innerkey not in nodesDict[node.name].keys(): nodesDict[node.name][innerkey] = {}
                    for ininnerkey in element[innerkey].keys():
                      if ininnerkey not in nodesDict[node.name][innerkey].keys(): nodesDict[node.name][innerkey][ininnerkey] = element[innerkey][ininnerkey]
                      else: nodesDict[node.name][innerkey][ininnerkey] = np.concatenate((nodesDict[node.name][innerkey][ininnerkey],element[innerkey][ininnerkey]))
                  else:
                    if innerkey not in nodesDict[node.name].keys(): nodesDict[node.name][innerkey] = np.atleast_1d(element[innerkey])
                    else: nodesDict[node.name][innerkey] = np.concatenate((nodesDict[node.name][innerkey],element[innerkey]))
              else:
                # it is a value
                if not nodesDict[node.name]: nodesDict[node.name] = element
                else: nodesDict[node.name] = np.concatenate((nodesDict[node.name],element))
          else:
            #Pointset
            if typeVar   in 'inout'              and not keyword: backTrace[node.name] = node.get('dataContainer')
            elif typeVar in ['inputs','input']   and not keyword: backTrace[node.name] = node.get('dataContainer')['inputs'  ]
            elif typeVar in ['output','outputs'] and not keyword: backTrace[node.name] = node.get('dataContainer')['outputs' ]
            elif typeVar in 'metadata'           and not keyword: backTrace[node.name] = node.get('dataContainer')['metadata']
            elif typeVar in ['inputs','input']   and     keyword: backTrace[node.name] = np.asarray(node.get('dataContainer')['inputs'  ][keyword])
            elif typeVar in ['output','outputs'] and     keyword: backTrace[node.name] = np.asarray(node.get('dataContainer')['outputs' ][keyword])
            elif typeVar in 'metadata'           and     keyword: backTrace[node.name] = np.asarray(node.get('dataContainer')['metadata'][keyword])
            if type(backTrace[node.name]) == dict:
              for innerkey in backTrace[node.name].keys():
                if type(backTrace[node.name][innerkey]) == dict:
                  #inputs outputs metadata
                  if innerkey not in backTrace[node.name][innerkey].keys(): nodesDict[innerkey] = {}
                  for ininnerkey in backTrace[node.name][innerkey].keys():
                    if ininnerkey not in nodesDict[innerkey].keys(): nodesDict[innerkey][ininnerkey] = backTrace[node.name][innerkey][ininnerkey]
                    else: nodesDict[innerkey][ininnerkey] = np.concatenate((nodesDict[innerkey][ininnerkey],backTrace[node.name][innerkey][ininnerkey]))
                else:
                  if innerkey not in nodesDict.keys(): nodesDict[innerkey] = np.atleast_1d(backTrace[node.name][innerkey])
                  else: nodesDict[innerkey] = np.concatenate((nodesDict[innerkey],backTrace[node.name][innerkey]))
            else:
              #it is a value
              if type(nodesDict) == dict: nodesDict = np.empty(0)
              nodesDict = np.concatenate((nodesDict,backTrace[node.name]))
    else:
      # we want a particular node
      found = False
      for TSDat in self.TSData.values():
        #a = TSDat.iter(nodeid)
        #b = TSDat.iterWholeBackTrace(a)
        nodelist = []
        for node in TSDat.iter(nodeid):
          if serialize:
            for se in list(TSDat.iterWholeBackTrace(node)): nodelist.append(se)
          else: nodelist.append(node)
          break
        #nodelist = list(TSDat.iterWholeBackTrace(TSDat.iter(nodeid)[0]))
        if len(nodelist) > 0:
          found = True
          break
      if not found: self.raiseAnError(RuntimeError,'Starting node called '+ nodeid+ ' not found!')
      if serialize:
        # we want a particular node and serialize it
        nodesDict[nodeid] = []
        for se in nodelist:
          if typeVar   in 'inout' and not keyword             : nodesDict[node.name].append( se.get('dataContainer'))
          elif typeVar in ['inputs','input'] and not keyword  : nodesDict[node.name].append( se.get('dataContainer')['inputs'  ])
          elif typeVar in ['output','outputs'] and not keyword: nodesDict[node.name].append( se.get('dataContainer')['outputs' ])
          elif typeVar in 'metadata' and not keyword          : nodesDict[node.name].append( se.get('dataContainer')['metadata'])
          elif typeVar in ['inputs','input'] and keyword      : nodesDict[node.name].append( np.asarray(se.get('dataContainer')['inputs'  ][keyword]))
          elif typeVar in ['output','outputs'] and keyword    : nodesDict[node.name].append( np.asarray(se.get('dataContainer')['outputs' ][keyword]))
          elif typeVar in 'metadata' and keyword              : nodesDict[node.name].append( np.asarray(se.get('dataContainer')['metadata'][keyword]))
      else:
        if typeVar   in 'inout'              and not keyword: nodesDict[nodeid] = nodelist[-1].get('dataContainer')
        elif typeVar in ['inputs','input']   and not keyword: nodesDict[nodeid] = nodelist[-1].get('dataContainer')['inputs'  ]
        elif typeVar in ['output','outputs'] and not keyword: nodesDict[nodeid] = nodelist[-1].get('dataContainer')['outputs' ]
        elif typeVar in 'metadata'           and not keyword: nodesDict[nodeid] = nodelist[-1].get('dataContainer')['metadata']
        elif typeVar in ['inputs','input']   and     keyword: nodesDict[nodeid] = np.asarray(nodelist[-1].get('dataContainer')['inputs'  ][keyword])
        elif typeVar in ['output','outputs'] and     keyword: nodesDict[nodeid] = np.asarray(nodelist[-1].get('dataContainer')['outputs' ][keyword])
        elif typeVar in 'metadata'           and     keyword: nodesDict[nodeid] = np.asarray(nodelist[-1].get('dataContainer')['metadata'][keyword])
    return nodesDict

  def retrieveNodeInTreeMode(self,nodeName,parentName=None):
    """
      This Method is used to retrieve a node (a list...) when the hierarchical mode is requested
      If the node has not been found, Create a new one
      @ In, nodeName, string, is the node we want to retrieve
      @ In, parentName, string, optional, is the parent name... It's possible that multiple nodes have the same name.
                                          With the parentName, it's possible to perform a double check
    """
    if not self.TSData: # there is no tree yet
      self.TSData = {nodeName:TS.NodeTree(TS.Node(nodeName))}
      return self.TSData[nodeName].getrootnode()
    else:
      if nodeName in self.TSData.keys(): return self.TSData[nodeName].getrootnode()
      elif parentName == 'root':
        self.TSData[nodeName] = TS.NodeTree(TS.Node(nodeName))
        return self.TSData[nodeName].getrootnode()
      else:
        for TSDat in self.TSData.values():
          foundNodes = list(TSDat.iter(nodeName))
          if len(foundNodes) > 0: break
        if len(foundNodes) == 0: return TS.Node(nodeName)
        else:
          if parentName:
            for node in foundNodes:
              if node.getParentName() == parentName: return node
            self.raiseAnError(RuntimeError,'the node ' + nodeName + 'has been found but no one has a parent named '+ parentName)
          else: return(foundNodes[0])

  def addNodeInTreeMode(self,tsnode,options):
    """
      This Method is used to add a node into the tree when the hierarchical mode is requested
      If the node has not been found, Create a new one
      @ In, tsnode, the node
      @ In, options, dict, parentID must be present if newer node
    """
    if not tsnode.getParentName():
      parentID = None
      if 'metadata' in options.keys():
        if 'parentID' in options['metadata'].keys(): parentID = options['metadata']['parentID']
      else:
        if 'parentID' in options.keys(): parentID = options['parentID']
      if not parentID: self.raiseAnError(ConstructError,'the parentID must be provided if a new node needs to be appended')
      self.retrieveNodeInTreeMode(parentID).appendBranch(tsnode)
