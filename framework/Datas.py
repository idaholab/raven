'''
Created on Feb 16, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
from BaseType import BaseType
from Csv_loader import CsvLoader as ld
import copy
import itertools
import abc
import numpy as np
import utils
import TreeStructure as TS
import xml.etree.ElementTree as ET

# Custom exceptions
class NotConsistentData(Exception): pass
class ConstructError(Exception)   : pass

class Data(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  def __init__(self,inParamValues = None, outParamValues = None):
    BaseType.__init__(self)
    self._dataParameters                 = {}                         # in here we store all the data parameters (inputs params, output params,etc)
    self._dataParameters['inParam'     ] = []                         # inParam list
    self._dataParameters['outParam'    ] = []                         # outParam list
    self._dataParameters['hierarchical'] = False                      # the structure of this data is hierarchical?
    self._toLoadFromList                 = []                         # loading source
    self._dataContainer                  = {'inputs':{},'outputs':{}} # Dict that contains the actual data. self._dataContainer['inputs'] contains the input space, self._dataContainer['output'] the output space
    self._dataContainer['metadata'     ] = {}                         # In this dictionary we store metadata (For example, probability,input file names, etc)
    self.metaExclXml                     = ['probability']            # list of metadata keys that are excluded from xml outputter, and included in the CSV one
    if inParamValues:
      if type(inParamValues) != dict: raise ConstructError('DATAS'.ljust(25)+': ' +utils.returnPrintPostTag('ERROR') + '-> in __init__  in Datas of type ' + self.type + ' . inParamValues is not a dictionary')
      self._dataContainer['inputs'] = inParamValues
    if outParamValues:
      if type(outParamValues) != dict: raise ConstructError('DATAS'.ljust(25)+': ' +utils.returnPrintPostTag('ERROR') + '-> in __init__  in Datas of type ' + self.type + ' . outParamValues is not a dictionary')
      self._dataContainer['outputs'] = outParamValues
    self.notAllowedInputs  = []#this is a list of keyword that are not allowed as inputs
    self.notAllowedOutputs = []#this is a list of keyword that are not allowed as Outputs
    self.metatype  = [float,bool,int,np.ndarray,np.float16,np.float32,np.float64,np.float128,np.int16,np.int32,np.int64,np.bool8]
    self.printTag = utils.returnPrintTag('DATAS')

  def _readMoreXML(self,xmlNode):
    '''
    Function to read the xml input block.
    @ In, xmlNode, xml node
    '''
    # retrieve input/outputs parameters' keywords
    self._dataParameters['inParam']  = xmlNode.find('Input' ).text.strip().split(',')
    self._dataParameters['outParam'] = xmlNode.find('Output').text.strip().split(',')
    #test for keywords not allowed
    if len(set(xmlNode.find('Input' ).text.strip().split(','))&set(self.notAllowedInputs))!=0:
      raise IOError('the keyword '+str(set(xmlNode.find('Input' ).text.strip().split(','))&set(self.notAllowedInputs))+' is not allowed among inputs')
    if len(set(xmlNode.find('Output' ).text.strip().split(','))&set(self.notAllowedOutputs))!=0:
      raise IOError('the keyword '+str(set(xmlNode.find('Output' ).text.strip().split(','))&set(self.notAllowedOutputs))+' is not allowed among inputs')
    #test for same input/output variables name
    if len(set(xmlNode.find('Input' ).text.strip().split(','))&set(xmlNode.find('Output' ).text.strip().split(',')))!=0:
      raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->It is not allowed to have the same name of input/output variables in the data '+self.name+' of type '+self.type)
    #
    # retrieve history name if present
    try:   self._dataParameters['history'] = xmlNode.find('Input' ).attrib['name']
    except KeyError:self._dataParameters['history'] = None

    if 'time' in xmlNode.attrib.keys():
      # check if time information are present... in case, store it
      if not (self._dataParameters['time'] == 'end' or self._dataParameters['time'] == 'all'):
        try:   self._dataParameters['time'] = float(self._dataParameters['time'])
        except ValueError: self._dataParameters['time'] = float(self._dataParameters['time'].split(','))
    else:self._dataParameters['time'] = None

    if 'operator' in xmlNode.attrib.keys():
      # check if time information are present... in case, store it
      self._dataParameters['operator'] = xmlNode.attrib['operator'].lower()
      if self._dataParameters['operator'] not in ['min','max','average']: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->Only operation available are '+str(['min','max','average'])+' .Data named '+ self.name + 'of type ' + self.type  )

    # check if inputTs is provided => the time step that the inputs refer to
    try: self._dataParameters['inputTs'] = int(xmlNode.attrib['inputTs'])
    except KeyError:self._dataParameters['inputTs'] = None
    # check if this data needs to be in hierarchical fashion
    if 'hierarchical' in xmlNode.attrib.keys():
      if xmlNode.attrib['hierarchical'].lower() in ['true','t','1','yes','y']: self._dataParameters['hierarchical'] = True
      else: self._dataParameters['hierarchical'] = False
      if self._dataParameters['hierarchical'] and not self.acceptHierarchical():
        print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '-> hierarchical fashion is not available (No Sense) for Data named '+ self.name + 'of type ' + self.type + '!!!')
        self._dataParameters['hierarchical'] = False
      else:
        self.TSData = None
        self.rootToBranch = {}
    else: self._dataParameters['hierarchical'] = False

  def addInitParams(self,tempDict):
    '''
    Function to get the input params that belong to this class
    @ In, tempDict, temporary dictionary
    '''
    for i in range(len(self._dataParameters['inParam' ])):  tempDict['Input_'+str(i)]  = self._dataParameters['inParam' ][i]
    for i in range(len(self._dataParameters['outParam'])):  tempDict['Output_'+str(i)] = self._dataParameters['outParam'][i]
    tempDict['Time'                       ] = self._dataParameters['time']
    tempDict['Hierarchical mode'          ] = self._dataParameters['hierarchical']
    tempDict['TimeStep of the input space'] = self._dataParameters['inputTs']
    return tempDict

  def removeInputValue(self,name):
    '''
    Function to remove a value from the dictionary inpParametersValues
    @ In, name, parameter name
    '''
    if self._dataParameters['hierarchical']:
      for TSData in self.TSData.values():
        for node in list(TSData.iter('*')):
          if name in node.get('dataContainer')['inputs'].keys(): node.get('dataContainer')['inputs'].pop(name)
    else:
      if name in self._dataContainer['inputs'].keys(): self._dataContainer['inputs'].pop(name)

  def removeOutputValue(self,name):
    '''
    Function to remove a value from the dictionary outParametersValues
    @ In, name, parameter name
    '''
    if self._dataParameters['hierarchical']:
      for TSData in self.TSData.values():
        for node in list(TSData.iter('*')):
          if name in node.get('dataContainer')['outputs'].keys(): node.get('dataContainer')['outputs'].pop(name)
    else:
      if name in self._dataContainer['outputs'].keys(): self._dataContainer['outputs'].pop(name)

  def updateInputValue(self,name,value,options=None):
    '''
    Function to update a value from the input dictionary
    @ In, name, parameter name
    @ In, value, the new value
    @ In, parent_id, optional, parent identifier in case Hierarchical fashion has been requested
    '''
    self._updateSpecializedInputValue(name,value,options)

  def updateOutputValue(self,name,value,options=None):
    '''
    Function to update a value from the output dictionary
    @ In, name, parameter name
    @ In, value, the new value
    @ In, parent_id, optional, parent identifier in case Hierarchical fashion has been requested
    '''
    self._updateSpecializedOutputValue(name,value,options)

  def updateMetadata(self,name,value,options=None):
    '''
    Function to update a value from the dictionary metadata
    @ In, name, parameter name
    @ In, value, the new value
    @ In, parent_id, optional, parent identifier in case Hierarchical fashion has been requested
    '''
    self._updateSpecializedMetadata(name,value,options)

  def getMetadata(self,keyword,nodeid=None,serialize=False):
    if self._dataParameters['hierarchical']:
      if type(keyword) == int: return list(self.getHierParam('metadata',nodeid,None,serialize).values())[keyword-1]
      else: return self.getHierParam('metadata',nodeid,keyword,serialize)
    else:
      if keyword in self._dataContainer['metadata'].keys(): return self._dataContainer ['metadata'][keyword]
      else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> parameter ' + str(keyword) + ' not found in metadata dictionary. Available keys are '+str(self._dataContainer['metadata'].keys())+'.Function: Data.getMetadata')

  def getAllMetadata(self,nodeid=None,serialize=False):
    if self._dataParameters['hierarchical']: return self.getHierParam('metadata',nodeid,None,serialize)
    else                                   : return self._dataContainer['metadata']


  @abc.abstractmethod
  def addSpecializedReadingSettings(self):
    '''
      This function is used to add specialized attributes to the data in order to retrieve the data properly.
      Every specialized data needs to overwrite it!!!!!!!!
    '''
    pass

  @abc.abstractmethod
  def checkConsistency(self):
    '''
      This function checks the consistency of the data structure... every specialized data needs to overwrite it!!!!!
    '''
    pass

  @abc.abstractmethod
  def acceptHierarchical(self):
    '''
      This function returns a boolean. True if the specialized Data accepts the hierarchical structure
    '''
    pass

  def __getVariablesToPrint(self,var,inOrOut):
    ''' Returns a list of variables to print.
    Takes the variable and either 'input' or 'output'
    '''
    variables_to_print = []
    lvar = var.lower()
    if type(list(self._dataContainer[inOrOut+'s'].values())[0]) == dict: varKeys = list(self._dataContainer[inOrOut+'s'].values())[0].keys()
    else: varKeys = self._dataContainer[inOrOut+'s'].keys()
    if lvar == inOrOut:
      for invar in varKeys: variables_to_print.append(inOrOut+'|'+str(invar))
    elif '|' in var and lvar.startswith(inOrOut+'|'):
      varName = var.split('|')[1]
      if varName not in varKeys: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> variable ' + varName + ' is not present among the '+inOrOut+'s of Data ' + self.name)
      else: variables_to_print.append(inOrOut+'|'+str(varName))
    else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> unexpected variable '+ var)
    return variables_to_print

  def printCSV(self,options=None):
    '''
    Function used to dump the data into a csv file
    Every class must implement the specializedPrintCSV method
    that is going to be called from here
    @ In, OPTIONAL, options, dictionary of options... it can contain the filename to be used, the parameters need to be printed....
    '''
    options_int = {}
    # print content of data in a .csv format
    if self.debug:
      print(' '*len(self.printTag)+':=======================')
      print(' '*len(self.printTag)+':DATAS: print on file(s)')
      print(' '*len(self.printTag)+':=======================')
    if options:
      if ('filenameroot' in options.keys()): filenameLocal = options['filenameroot']
      else: filenameLocal = self.name + '_dump'
      if 'variables' in options.keys():
        variables_to_print = []
        for var in options['variables'].split(','):
          lvar = var.lower()
          if lvar.startswith('input'):
            variables_to_print.extend(self.__getVariablesToPrint(var,'input'))
          elif lvar.startswith('output'):
            variables_to_print.extend(self.__getVariablesToPrint(var,'output'))
          else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->variable ' + var + ' is unknown in Data ' + self.name + '. You need to specify an input or a output')
        options_int['variables'] = variables_to_print
    else:   filenameLocal = self.name + '_dump'

    self.specializedPrintCSV(filenameLocal,options_int)

  def _createXMLFile(self,filenameLocal,fileType,inpKeys,outKeys):
    '''Creates an XML file to contain the input and output data list
    and the type.
    '''
    myXMLFile = open(filenameLocal + '.xml', 'w')
    root = ET.Element('data',{'name':filenameLocal,'type':fileType})
    inputNode = ET.SubElement(root,'input')
    inputNode.text = ','.join(inpKeys)
    outputNode = ET.SubElement(root,'output')
    outputNode.text = ','.join(outKeys)
    filenameNode = ET.SubElement(root,'input_filename')
    filenameNode.text = filenameLocal + '.csv'
    if len(self._dataContainer['metadata'].keys()) > 0:
      #write metadata as well_known_implementations
      metadataNode = ET.SubElement(root,'metadata')
      submetadataNodes = []
      for key,value in self._dataContainer['metadata'].items():
        if key not in self.metaExclXml:
          submetadataNodes.append(ET.SubElement(metadataNode,key))
          submetadataNodes[-1].text = utils.toString(str(value)).replace('[','').replace(']','').replace('{','').replace('}','')
    myXMLFile.write(utils.toString(ET.tostring(root)))
    myXMLFile.write('\n')
    myXMLFile.close()

  def addOutput(self,toLoadFrom,options=None):
    '''
      Function to construct a data from a source
      @ In, toLoadFrom, loading source, it can be an HDF5 database, a csv file and in the future a xml file
      @ In, options, it's a dictionary of options. For example useful for metadata storing or,
                     in case an hierarchical fashion has been requested, it must contain the parent_id and the name of the actual 'branch'
    '''
    self._toLoadFromList.append(toLoadFrom)
    self.addSpecializedReadingSettings()

    sourceType = None
    print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> Constructing data type ' +self.type +' named '+ self.name + ' from:')
    try:
      sourceType =  self._toLoadFromList[-1].type
      print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> Object type ' + self._toLoadFromList[-1].type + ' named "' + self._toLoadFromList[-1].name+'"')
    except AttributeError:
      print(self.printTag+': ' +utils.returnPrintPostTag('Message') + '-> Object type' +' CSV named "' + toLoadFrom+'"')

    if(sourceType == 'HDF5'):
      tupleVar = self._toLoadFromList[-1].retrieveData(self._dataParameters)
      if options:
        parent_id = None
        if 'metadata' in options.keys():
          if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
        else:
          if 'parent_id' in options.keys(): parent_id = options['parent_id']
        if parent_id and self._dataParameters['hierarchical']:
          print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '-> Data storing in hierarchical fashion from HDF5 not yet implemented!')
          self._dataParameters['hierarchical'] = False
    else: tupleVar = ld().csvLoadData([toLoadFrom],self._dataParameters)

    for hist in tupleVar[0].keys():
      if type(tupleVar[0][hist]) == dict:
        for key in tupleVar[0][hist].keys(): self.updateInputValue(key, tupleVar[0][hist][key], options)
      else:
        if self.type in ['TimePoint','TimePointSet']:
          for index in range(tupleVar[0][hist].size): self.updateInputValue(hist, tupleVar[0][hist][index], options)
        else: self.updateInputValue(hist, tupleVar[0][hist], options)
    for hist in tupleVar[1].keys():
      if type(tupleVar[1][hist]) == dict:
        for key in tupleVar[1][hist].keys(): self.updateOutputValue(key, tupleVar[1][hist][key], options)
      else:
        if self.type in ['TimePoint','TimePointSet']:
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
                else: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> unknown type for metadata adding process. Relevant type = '+ str(elem))

        else:
          print('FIXME: This if statement is for back Compatibility... Remove it whitin end of July')
          if tupleVar[2][hist]: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> unknown type for metadata adding process. Relevant type = '+ str(type(tupleVar[2][hist])))
    self.checkConsistency()
    return

  def getParametersValues(self,typeVar,nodeid=None, serialize=False):
    '''
    Functions to get the parameter values
    @ In, variable type (input or output)
    '''
    if    typeVar.lower() in 'inputs' : return self.getInpParametersValues(nodeid,serialize)
    elif  typeVar.lower() in 'outputs': return self.getOutParametersValues(nodeid,serialize)
    else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> type ' + typeVar + ' is not a valid type. Function: Data.getParametersValues')

  def getParaKeys(self,typePara):
    '''
    Functions to get the parameter keys
    @ In, typePara, variable type (input or output)
    '''
    if   typePara.lower() in 'inputs' : return self._dataParameters['inParam' ]
    elif typePara.lower() in 'outputs': return self._dataParameters['outParam']
    else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> type ' + typePara + ' is not a valid type. Function: Data.getParaKeys')

  def isItEmpty(self):
    '''
    Functions to check if the data is empty
    @ In, None
    '''
    if len(self.getInpParametersValues().keys()) == 0 and len(self.getOutParametersValues()) == 0: return True
    else:                                                                                          return False

  def getInpParametersValues(self,nodeid=None,serialize=False):
    '''
    Function to get a reference to the input parameter dictionary
    @, In, nodeid, optional, in hierarchical mode, if nodeid is provided, the data for that node is returned,
                             otherwise check explaination for getHierParam
    @, In, serialize, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                PLEASE check explaination for getHierParam
    @, Out, Reference to self._dataContainer['inputs'] or something else in hierarchical
    '''
    if self._dataParameters['hierarchical']: return self.getHierParam('inputs',nodeid,serialize=serialize)
    else:                                   return self._dataContainer['inputs']

  def getOutParametersValues(self,nodeid=None,serialize=False):
    '''
    Function to get a reference to the output parameter dictionary
    @, In, nodeid, optional, in hierarchical mode, if nodeid is provided, the data for that node is returned,
                             otherwise check explaination for getHierParam
    @, In, serialize, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                PLEASE check explaination for getHierParam
    @, Out, Reference to self._dataContainer['outputs'] or something else in hierarchical
    '''
    if self._dataParameters['hierarchical']: return self.getHierParam('outputs',nodeid,serialize=serialize)
    else:                                    return self._dataContainer['outputs']

  def getParam(self,typeVar,keyword,nodeid=None,serialize=False):
    '''
    Function to get a reference to an output or input parameter
    @ In, typeVar, input or output
    @ In, keyword, keyword
    @ Out, Reference to the parameter
    '''
    if self.type == 'Histories': acceptedType = [str,unicode,bytes,int]
    else                       : acceptedType = [str,unicode,bytes]
    if type(typeVar) not in [str,unicode,bytes] : raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> type of parameter typeVar needs to be a string. Function: Data.getParam')
    if type(keyword) not in acceptedType        : raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> type of parameter keyword needs to be '+str(acceptedType)+' . Function: Data.getParam')
    if nodeid:
      if type(nodeid) not in [str,unicode,bytes]  : raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> type of parameter nodeid needs to be a string. Function: Data.getParam')
    if typeVar.lower() not in ['input','inputs','output','outputs']: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> type ' + typeVar + ' is not a valid type. Function: Data.getParam')
    if self._dataParameters['hierarchical']:
      if type(keyword) == int:
        return list(self.getHierParam(typeVar.lower(),nodeid,None,serialize).values())[keyword-1]
      else: return self.getHierParam(typeVar.lower(),nodeid,keyword,serialize)
    else:
      if typeVar.lower() in ['input','inputs']:
        if keyword in self._dataContainer['inputs'].keys(): return self._dataContainer['inputs'][keyword]
        else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> parameter ' + str(keyword) + ' not found in inpParametersValues dictionary. Available keys are '+str(self._dataContainer['inputs'].keys())+'.Function: Data.getParam')
      elif typeVar.lower() in ['output','outputs']:
        if keyword in self._dataContainer['outputs'].keys(): return self._dataContainer['outputs'][keyword]
        else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> parameter ' + str(keyword) + ' not found in outParametersValues dictionary. Available keys are '+str(self._dataContainer['outputs'].keys())+'.Function: Data.getParam')

  def extractValue(self,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    '''
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
    '''

    myType=self.type
    if   varName in self._dataParameters['inParam' ]: inOutType = 'input'
    elif varName in self._dataParameters['outParam']: inOutType = 'output'
    else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->the variable named '+varName+' was not found in the data: '+self.name)
    return self.__extractValueLocal__(myType,inOutType,varTyp,varName,varID,stepID,nodeid)

  @abc.abstractmethod
  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    '''
      this method has to be override to implement the specialization of extractValue for each data class
    '''
    pass

  def getHierParam(self,typeVar,nodeid,keyword=None,serialize=False):
    '''
      This function get a parameter when we are in hierarchical mode
      @ In,  typeVar,  string, it's the variable type... input,output, or inout
      @ In,  nodeid,   string, it's the node name... if == None or *, a dictionary of of data is returned, otherwise the actual node data is returned in a dict as well (see serialize attribute)
      @ In, keyword,   string, it's a parameter name (for example, cladTemperature), if None, the whole dict is returned, otherwise the parameter value is got (see serialize attribute)
      @ In, serialize, bool  , if true a sequence of TimePointSet is generated (a dictionary where the keys are the 'ending' branches and the values are a sorted list of _dataContainers (from first branch to the ending ones)
                               if false see explaination for nodeid
      @ Out, a dictionary of data (see above)
    '''
    nodesDict = {}
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
              elif typeVar in ['inputs','input']   and     keyword: nodesDict[node.name].append( se.get('dataContainer')['inputs'  ][keyword])
              elif typeVar in ['output','outputs'] and     keyword: nodesDict[node.name].append( se.get('dataContainer')['outputs' ][keyword])
              elif typeVar in 'metadata'           and     keyword: nodesDict[node.name].append( se.get('dataContainer')['metadata'][keyword])
      else:
        for TSData in self.TSData.values():
          for node in TSData.iter():
            if typeVar   in 'inout'              and not keyword: nodesDict[node.name] = node.get('dataContainer')
            elif typeVar in ['inputs','input']   and not keyword: nodesDict[node.name] = node.get('dataContainer')['inputs'  ]
            elif typeVar in ['output','outputs'] and not keyword: nodesDict[node.name] = node.get('dataContainer')['outputs' ]
            elif typeVar in 'metadata'           and not keyword: nodesDict[node.name] = node.get('dataContainer')['metadata']
            elif typeVar in ['inputs','input']   and     keyword: nodesDict[node.name] = node.get('dataContainer')['inputs'  ][keyword]
            elif typeVar in ['output','outputs'] and     keyword: nodesDict[node.name] = node.get('dataContainer')['outputs' ][keyword]
            elif typeVar in 'metadata'           and     keyword: nodesDict[node.name] = node.get('dataContainer')['metadata'][keyword]
    elif nodeid == 'ending':
      for TSDat in self.TSData.values():
        for ending in TSDat.iterEnding():
          if typeVar   in 'inout'              and not keyword: nodesDict[ending.name] = ending.get('dataContainer')
          elif typeVar in ['inputs','input']   and not keyword: nodesDict[ending.name] = ending.get('dataContainer')['inputs'  ]
          elif typeVar in ['output','outputs'] and not keyword: nodesDict[ending.name] = ending.get('dataContainer')['outputs' ]
          elif typeVar in 'metadata'           and not keyword: nodesDict[ending.name] = ending.get('dataContainer')['metadata']
          elif typeVar in ['inputs','input']   and     keyword: nodesDict[ending.name] = ending.get('dataContainer')['inputs'  ][keyword]
          elif typeVar in ['output','outputs'] and     keyword: nodesDict[ending.name] = ending.get('dataContainer')['outputs' ][keyword]
          elif typeVar in 'metadata'           and     keyword: nodesDict[ending.name] = ending.get('dataContainer')['metadata'][keyword]
    elif nodeid == 'RecontructEnding':
      # if history, reconstruct the history... if timepoint set take the last one (see below)
      backTrace = {}
      for TSData in self.TSData.values():
        for node in TSData.iterEnding():
          if self.type == 'Histories':
            backTrace[node.name] = []
            for se in list(TSData.iterWholeBackTrace(node)):
              if typeVar   in 'inout'              and not keyword: backTrace[node.name].append( se.get('dataContainer'))
              elif typeVar in ['inputs','input']   and not keyword: backTrace[node.name].append( se.get('dataContainer')['inputs'  ])
              elif typeVar in ['output','outputs'] and not keyword: backTrace[node.name].append( se.get('dataContainer')['outputs' ])
              elif typeVar in 'metadata'           and not keyword: backTrace[node.name].append( se.get('dataContainer')['metadata'])
              elif typeVar in ['inputs','input']   and     keyword: backTrace[node.name].append( se.get('dataContainer')['inputs'  ][keyword])
              elif typeVar in ['output','outputs'] and     keyword: backTrace[node.name].append( se.get('dataContainer')['outputs' ][keyword])
              elif typeVar in 'metadata'           and     keyword: backTrace[node.name].append( se.get('dataContainer')['metadata'][keyword])
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
            #timepointset
            if typeVar   in 'inout'              and not keyword: backTrace[node.name] = node.get('dataContainer')
            elif typeVar in ['inputs','input']   and not keyword: backTrace[node.name] = node.get('dataContainer')['inputs'  ]
            elif typeVar in ['output','outputs'] and not keyword: backTrace[node.name] = node.get('dataContainer')['outputs' ]
            elif typeVar in 'metadata'           and not keyword: backTrace[node.name] = node.get('dataContainer')['metadata']
            elif typeVar in ['inputs','input']   and     keyword: backTrace[node.name] = node.get('dataContainer')['inputs'  ][keyword]
            elif typeVar in ['output','outputs'] and     keyword: backTrace[node.name] = node.get('dataContainer')['outputs' ][keyword]
            elif typeVar in 'metadata'           and     keyword: backTrace[node.name] = node.get('dataContainer')['metadata'][keyword]
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
        nodelist = list(TSDat.iterWholeBackTrace(TSDat.iter(nodeid)[0]))
        if len(nodelist) > 0:
          found = True
          TSData = TSDat
          break
      if not found: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> Starting node called '+ nodeid+ ' not found!')
      if serialize:
        # we want a particular node and serialize it
        nodesDict[nodeid] = []
        for se in nodelist:
          if typeVar   in 'inout' and not keyword             : nodesDict[node.name].append( se.get('dataContainer'))
          elif typeVar in ['inputs','input'] and not keyword  : nodesDict[node.name].append( se.get('dataContainer')['inputs'  ])
          elif typeVar in ['output','outputs'] and not keyword: nodesDict[node.name].append( se.get('dataContainer')['outputs' ])
          elif typeVar in 'metadata' and not keyword          : nodesDict[node.name].append( se.get('dataContainer')['metadata'])
          elif typeVar in ['inputs','input'] and keyword      : nodesDict[node.name].append( se.get('dataContainer')['inputs'  ][keyword])
          elif typeVar in ['output','outputs'] and keyword    : nodesDict[node.name].append( se.get('dataContainer')['outputs' ][keyword])
          elif typeVar in 'metadata' and keyword              : nodesDict[node.name].append( se.get('dataContainer')['metadata'][keyword])
      else:
        if typeVar   in 'inout'              and not keyword: nodesDict[nodeid] = TSData.iter(nodeid)[0].get('dataContainer')
        elif typeVar in ['inputs','input']   and not keyword: nodesDict[nodeid] = TSData.iter(nodeid)[0].get('dataContainer')['inputs'  ]
        elif typeVar in ['output','outputs'] and not keyword: nodesDict[nodeid] = TSData.iter(nodeid)[0].get('dataContainer')['outputs' ]
        elif typeVar in 'metadata'           and not keyword: nodesDict[nodeid] = TSData.iter(nodeid)[0].get('dataContainer')['metadata']
        elif typeVar in ['inputs','input']   and     keyword: nodesDict[nodeid] = TSData.iter(nodeid)[0].get('dataContainer')['inputs'  ][keyword]
        elif typeVar in ['output','outputs'] and     keyword: nodesDict[nodeid] = TSData.iter(nodeid)[0].get('dataContainer')['outputs' ][keyword]
        elif typeVar in 'metadata'           and     keyword: nodesDict[nodeid] = TSData.iter(nodeid)[0].get('dataContainer')['metadata'][keyword]
    return nodesDict

  def retrieveNodeInTreeMode(self,nodeName,parentName=None):
    '''
      This Method is used to retrieve a node (a list...) when the hierarchical mode is requested
      If the node has not been found, Create a new one
      @ In, nodeName, string, is the node we want to retrieve
      @ In, parentName, string, optional, is the parent name... It's possible that multiple nodes have the same name.
                                          With the parentName, it's possible to perform a double check
    '''
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
            raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> the node ' + nodeName + 'has been found but no one has a parent named '+ parentName)
          else: return(foundNodes[0])

  def addNodeInTreeMode(self,tsnode,options):
    '''
      This Method is used to add a node into the tree when the hierarchical mode is requested
      If the node has not been found, Create a new one
      @ In, tsnode, the node
      @ In, options, dict, parent_id must be present if newer node
    '''
    if not tsnode.getParentName():
      parent_id = None
      if 'metadata' in options.keys():
        if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
      else:
        if 'parent_id' in options.keys(): parent_id = options['parent_id']
      if not parent_id: raise ConstructError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> the parent_id must be provided if a new node needs to be appended')
      self.retrieveNodeInTreeMode(parent_id).appendBranch(tsnode)

class TimePoint(Data):
  def acceptHierarchical(self):
    ''' Overwritten from baseclass'''
    return False

  def addSpecializedReadingSettings(self):
    '''
      This function adds in the dataParameters dict the options needed for reading and constructing this class
      @ In, None
      @ Out, None
    '''
    self._dataParameters['type'] = self.type # store the type into the _dataParameters dictionary
    try: sourceType = self._toLoadFromList[0].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      if(not self._dataParameters['history']): raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> DATAS     : ERROR: In order to create a TimePoint data, history name must be provided')
      self._dataParameters['filter'] = 'whole'

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data TimePoint
      @ In, None
      @ Out, None
    '''
    for key in self._dataContainer['inputs'].keys():
      if (self._dataContainer['inputs'][key].size) != 1:
        raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The input parameter value, for key ' + key + ' has not a consistent shape for TimePoint ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(self._dataContainer['inputs'][key].size))
    for key in self._dataContainer['outputs'].keys():
      if (self._dataContainer['outputs'][key].size) != 1:
        raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The output parameter value, for key ' + key + ' has not a consistent shape for TimePoint ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(self._dataContainer['outputs'][key].size))

  def _updateSpecializedInputValue(self,name,value,options=None):
    '''
      This function performs the updating of the values (input space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (1-D array)
      @ Out, None
    '''
    if name in self._dataContainer['inputs'].keys():
      self._dataContainer['inputs'].pop(name)
    if name not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(name)
    self._dataContainer['inputs'][name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def _updateSpecializedMetadata(self,name,value,options=None):
    '''
      This function performs the updating of the values (metadata) into this Data
      @ In,  name, string, parameter name (ex. probability)
      @ In,  value, whatever type, newer value
      @ Out, None
    '''
    self._dataContainer['metadata'][name] = copy.deepcopy(value)

  def _updateSpecializedOutputValue(self,name,value,options=None):
    '''
      This function performs the updating of the values (output space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (1-D array)
      @ Out, None
    '''
    if name in self._dataContainer['inputs'].keys():
      self._dataContainer['outputs'].pop(name)
    if name not in self._dataParameters['outParam']: self._dataParameters['outParam'].append(name)
    self._dataContainer['outputs'][name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal,options):
    '''
      This function prints a CSV file with the content of this class (Input and Output space)
      @ In,  filenameLocal, string, filename root (for example, 'homo_homini_lupus' -> the final file name is gonna be called 'homo_homini_lupus.csv')
      @ In,  options, dictionary, dictionary of printing options
      @ Out, None (a csv is gonna be printed)
    '''

    #For timepoint it creates an XML file and one csv file.  The
    #CSV file will have a header with the input names and output
    #names, and one line of data with the input and output numeric
    #values.

    inpKeys   = []
    inpValues = []
    outKeys   = []
    outValues = []
    #Print input values
    if 'variables' in options.keys():
      for var in options['variables']:
        if var.split('|')[0] == 'input':
          inpKeys.append(var.split('|')[1])
          inpValues.append(self._dataContainer['inputs'][var.split('|')[1]])
        if var.split('|')[0] == 'output':
          outKeys.append(var.split('|')[1])
          outValues.append(self._dataContainer['outputs'][var.split('|')[1]])
        if var.split('|')[0] == 'metadata':
          if var.split('|')[1] in self.metaExclXml:
            if type(self._dataContainer['metadata'][var.split('|')[1]]) not in self.metatype:
              raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> metadata '+var.split('|')[1]+' not compatible with CSV output. Its type needs to be one of '+str(self.metatype))
            inpKeys.append(var.split('|')[1])
            inpValues.append(np.atleast_1d(np.float(self._dataContainer['metadata'][var.split('|')[1]])))
          else: print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '-> metadata '+var.split('|')[1]+' not compatible with CSV output.It is going to be outputted into Xml out')
    else:
      inpKeys   = self._dataContainer['inputs'].keys()
      inpValues = self._dataContainer['inputs'].values()
      outKeys   = self._dataContainer['outputs'].keys()
      outValues = self._dataContainer['outputs'].values()
      if len(self._dataContainer['metadata'].keys()) > 0:
        #write metadata as well_known_implementations
        for key,value in self._dataContainer['metadata'].items():
          if key in self.metaExclXml:
            if type(value) not in self.metatype:
              raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> metadata '+key+' not compatible with CSV output. Its type needs to be one of '+str(self.metatype))
            inpKeys.append(key)
            inpValues.append(np.atleast_1d(np.float(value)))
    if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '.csv', 'wb')
    else: return

    #Print header
    myFile.write(b','.join([utils.toBytes(item) for item in itertools.chain(inpKeys,outKeys)]))
    myFile.write(b'\n')
    #Print values
    myFile.write(b','.join([utils.toBytes(str(item[0])) for item in  itertools.chain(inpValues,outValues)]))
    myFile.write(b'\n')
    myFile.close()
    self._createXMLFile(filenameLocal,'timepoint',inpKeys,outKeys)


  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    '''override of the method in the base class Datas'''
    if varID!=None or stepID!=None: raise Exception('seeking to extract a slice from a TimePoint type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':exec ('return '+varTyp+'(self.getParam(inOutType,varName)[0])')
    else: return self.getParam(inOutType,varName)

class TimePointSet(Data):
  def acceptHierarchical(self):
    ''' Overwritten from baseclass'''
    return True

  def addSpecializedReadingSettings(self):
    '''
      This function adds in the _dataParameters dict the options needed for reading and constructing this class
    '''
    # if hierarchical fashion has been requested, we set the type of the reading to a TimePoint,
    #  since a TimePointSet in hierarchical fashion would be a tree of TimePoints
    if self._dataParameters['hierarchical']: self._dataParameters['type'] = 'TimePoint'
    # store the type into the _dataParameters dictionary
    else:                                   self._dataParameters['type'] = self.type
    try: sourceType = self._toLoadFromList[0].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      self._dataParameters['histories'] = self._toLoadFromList[0].getEndingGroupNames()
      self._dataParameters['filter'   ] = 'whole'

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data TimePointSet
    '''
    lenMustHave = 0
    try:   sourceType = self._toLoadFromList[-1].type
    except AttributeError:sourceType = None
    # here we assume that the outputs are all read....so we need to compute the total number of time point sets
    for sourceLoad in self._toLoadFromList:
      if not isinstance(sourceLoad, basestring):
        if('HDF5' == sourceLoad.type):  lenMustHave = lenMustHave + len(sourceLoad.getEndingGroupNames())
      else: lenMustHave += 1

    if('HDF5' == sourceType):
      eg = self._toLoadFromList[0].getEndingGroupNames()
      for key in self._dataContainer['inputs'].keys():
        if (self._dataContainer['inputs'][key].size) != len(eg):
          raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(eg)) + '.Actual size is ' + str(self._dataContainer['inputs'][key].size))
      for key in self._dataContainer['outputs'].keys():
        if (self._dataContainer['outputs'][key].size) != len(eg):
          raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(eg)) + '.Actual size is ' + str(self._dataContainer['outputs'][key].size))
    else:
      if self._dataParameters['hierarchical']:
        for key in self._dataContainer['inputs'].keys():
          if (self._dataContainer['inputs'][key].size) != 1:
            raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be a single value since we are in hierarchical mode.' + '.Actual size is ' + str(self._dataContainer['inputs'][key].size))
        for key in self._dataContainer['outputs'].keys():
          if (self._dataContainer['outputs'][key].size) != 1:
            raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be a single value since we are in hierarchical mode.' + '.Actual size is ' + str(self._dataContainer['outputs'][key].size))
      else:
        for key in self._dataContainer['inputs'].keys():
          if (self._dataContainer['inputs'][key].size) != lenMustHave:
            raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(lenMustHave) + '.Actual size is ' + str(self._dataContainer['inputs'][key].size))
        for key in self._dataContainer['outputs'].keys():
          if (self._dataContainer['outputs'][key].size) != lenMustHave:
            raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(lenMustHave) + '.Actual size is ' + str(self._dataContainer['outputs'][key].size))


  def _updateSpecializedInputValue(self,name,value,options=None):
    '''
      This function performs the updating of the values (input space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (single value)
      @ Out, None
    '''
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'TimePoint' has been stored
      parent_id = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
      else:
        prefix    = options['prefix']
        if 'parent_id' in options.keys(): parent_id = options['parent_id']
      if parent_id: tsnode = self.retrieveNodeInTreeMode(prefix,parent_id)
      else:                             tsnode = self.retrieveNodeInTreeMode(prefix)
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if name in self._dataContainer['inputs'].keys():
        self._dataContainer['inputs'].pop(name)
      if name not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(name)
      self._dataContainer['inputs'][name] = copy.deepcopy(np.atleast_1d(np.atleast_1d(value)[-1]))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['inputs'].keys():
        popped = self._dataContainer['inputs'].pop(name)
        self._dataContainer['inputs'][name] = copy.copy(np.concatenate((np.atleast_1d(np.array(popped)), np.atleast_1d(np.atleast_1d(value)[-1]))))
      else:
        if name not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(name)
        self._dataContainer['inputs'][name] = copy.deepcopy(np.atleast_1d(np.atleast_1d(value)[-1]))

  def _updateSpecializedMetadata(self,name,value,options=None):
    '''
      This function performs the updating of the values (metadata) into this Data
      @ In,  name, string, parameter name (ex. probability)
      @ In,  value, whatever type, newer value
      @ Out, None
      NB. This method, if the metadata name is already present, replaces it with the new value. No appending here, since the metadata are dishomogenius and a common updating strategy is not feasable.
    '''
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'TimePoint' has been stored
      parent_id = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
      else:
        prefix    = options['prefix']
        if 'parent_id' in options.keys(): parent_id = options['parent_id']
      if parent_id: tsnode = self.retrieveNodeInTreeMode(prefix,parent_id)

      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'metadata':{}})
        self._dataContainer = tsnode.get('dataContainer')
      else:
        if 'metadata' not in self._dataContainer.keys(): self._dataContainer['metadata'] ={}
      if name in self._dataContainer['metadata'].keys(): self._dataContainer['metadata'][name] = copy.deepcopy(np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value))))
      else                                             : self._dataContainer['metadata'][name] = copy.deepcopy(np.atleast_1d(value))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['metadata'].keys(): self._dataContainer['metadata'][name] = copy.copy(np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value))))
      else                                             : self._dataContainer['metadata'][name] = copy.deepcopy(np.atleast_1d(value))

  def _updateSpecializedOutputValue(self,name,value,options=None):
    '''
      This function performs the updating of the values (output space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (single value)
      @ Out, None
    '''
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'TimePoint' has been stored
      parent_id = None
      if 'metadata' in options.keys():
        prefix    = options['metadata']['prefix']
        if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
      else:
        prefix    = options['prefix']
        if 'parent_id' in options.keys(): parent_id = options['parent_id']
      if parent_id: tsnode = self.retrieveNodeInTreeMode(prefix,parent_id)

      #if 'parent_id' in options.keys(): tsnode = self.retrieveNodeInTreeMode(options['prefix'], options['parent_id'])
      #else:                             tsnode = self.retrieveNodeInTreeMode(options['prefix'])
      # we store the pointer to the container in the self._dataContainer because checkConsistency acts on this
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if name in self._dataContainer['outputs'].keys():
        self._dataContainer['outputs'].pop(name)
      if name not in self._dataParameters['inParam']: self._dataParameters['outParam'].append(name)
      self._dataContainer['outputs'][name] = copy.deepcopy(np.atleast_1d(np.atleast_1d(value)[-1]))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['outputs'].keys():
        popped = self._dataContainer['outputs'].pop(name)
        self._dataContainer['outputs'][name] = copy.copy(np.concatenate((np.array(popped), np.atleast_1d(np.atleast_1d(value)[-1]))))
      else:
        if name not in self._dataParameters['outParam']: self._dataParameters['outParam'].append(name)
        self._dataContainer['outputs'][name] = copy.deepcopy(np.atleast_1d(np.atleast_1d(value)[-1]))

  def specializedPrintCSV(self,filenameLocal,options):
    '''
      This function prints a CSV file with the content of this class (Input and Output space)
      @ In,  filenameLocal, string, filename root (for example, 'homo_homini_lupus' -> the final file name is gonna be called 'homo_homini_lupus.csv')
      @ In,  options, dictionary, dictionary of printing options
      @ Out, None (a csv is gonna be printed)
    '''
    inpKeys   = []
    inpValues = []
    outKeys   = []
    outValues = []
    #Print input values
    if self._dataParameters['hierarchical']:
      # retrieve a serialized of datas from the tree
      O_o = self.getHierParam('inout','*',serialize=True)
      for key in O_o.keys():
        inpKeys.append([])
        inpValues.append([])
        outKeys.append([])
        outValues.append([])
        if 'variables' in options.keys():
          for var in options['variables']:
            if var.split('|')[0] == 'input':
              inpKeys[-1].append(var.split('|')[1])
              axa = np.zeros(len(O_o[key]))
              for index in range(len(O_o[key])): axa[index] = O_o[key][index]['inputs'][var.split('|')[1]][0]
              inpValues[-1].append(axa)
            if var.split('|')[0] == 'output':
              outKeys[-1].append(var.split('|')[1])
              axa = np.zeros(len(O_o[key]))
              for index in range(len(O_o[key])): axa[index] = O_o[key][index]['outputs'][var.split('|')[1]][0]
              outValues[-1].append(axa)
            if var.split('|')[0] == 'metadata':
              if var.split('|')[1] in self.metaExclXml:
                if type(O_o[key][index]['metadata'][var.split('|')[1]]) not in self.metatype:
                  raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> metadata '+var.split('|')[1] +' not compatible with CSV output. Its type needs to be one of '+str(np.ndarray))
                inpKeys[-1].append(var.split('|')[1])
                axa = np.zeros(len(O_o[key]))
                for index in range(len(O_o[key])): axa[index] = np.atleast_1d(np.float(O_o[key][index]['metadata'][var.split('|')[1]]))[0]
                inpValues[-1].append(copy.deepcopy(axa))
        else:
          inpKeys[-1] = O_o[key][0]['inputs'].keys()
          for var in inpKeys[-1]:
            axa = np.zeros(len(O_o[key]))
            for index in range(len(O_o[key])): axa[index] = O_o[key][index]['inputs'][var][0]
            inpValues[-1].append(copy.deepcopy(axa))
          outKeys[-1] = O_o[key][0]['outputs'].keys()
          for var in outKeys[-1]:
            axa = np.zeros(len(O_o[key]))
            for index in range(len(O_o[key])): axa[index] = O_o[key][index]['outputs'][var][0]
            outValues[-1].append(copy.deepcopy(axa))
          if len(O_o[key][0]['metadata'].keys()) > 0:
            #write metadata as well_known_implementations
            for metaname,value in O_o[key][0]['metadata'].items():
              if metaname in self.metaExclXml:
                if type(value) not in self.metatype:
                  raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> metadata '+metaname+' not compatible with CSV output. Its type needs to be one of '+str(np.ndarray))
                inpKeys[-1].append(metaname)
                axa = np.zeros(len(O_o[key]))
                for index in range(len(O_o[key])): axa[index] = np.atleast_1d(np.float(O_o[key][index]['metadata'][metaname]))[0]
                inpValues[-1].append(copy.deepcopy(axa))
      if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '.csv', 'wb')
      else: return
      O_o_keys = list(O_o.keys())
      for index in range(len(O_o.keys())):
        myFile.write(b'Ending branch,'+utils.toBytes(O_o_keys[index])+b'\n')
        myFile.write(b'branch #')
        for item in inpKeys[index]:
            myFile.write(b',' + utils.toBytes(item))
        for item in outKeys[index]:
            myFile.write(b',' + utils.toBytes(item))
        myFile.write(b'\n')
        for j in range(outValues[index][0].size):
          myFile.write(utils.toBytes(str(j+1)))
          for i in range(len(inpKeys[index])):
            myFile.write(b',' + utils.toBytes(str(inpValues[index][i][j])))
          for i in range(len(outKeys[index])):
            myFile.write(b',' + utils.toBytes(str(outValues[index][i][j])))
          myFile.write(b'\n')
      myFile.close()
    else:
      #If not hierarchical

      #For timepointset it will create an XML file and one CSV file.
      #The CSV file will have a header with the input names and output
      #names, and multiple lines of data with the input and output
      #numeric values, one line for each input.
      if 'variables' in options.keys():
        for var in options['variables']:
          if var.split('|')[0] == 'input':
            inpKeys.append(var.split('|')[1])
            inpValues.append(self._dataContainer['inputs'][var.split('|')[1]])
          if var.split('|')[0] == 'output':
            outKeys.append(var.split('|')[1])
            outValues.append(self._dataContainer['outputs'][var.split('|')[1]])
          if var.split('|')[0] == 'metadata':
            inpKeys.append(var.split('|')[1])
            inpValues.append(self._dataContainer['metadata'][var.split('|')[1]])
          if var.split('|')[0] == 'metadata':
            if var.split('|')[1] in self.metaExclXml:
              if type(self._dataContainer['metadata'][var.split('|')[1]]) not in self.metatype:
                raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> metadata '+var.split('|')[1]+' not compatible with CSV output. Its type needs to be one of '+str(self.metatype))
              inpKeys.append(var.split('|')[1])
              if type(value) != np.ndarray: inpValues.append(np.atleast_1d(np.float(self._dataContainer['metadata'][var.split('|')[1]])))
              else: inpValues.append(np.atleast_1d(self._dataContainer['metadata'][var.split('|')[1]]))
            else: print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '-> metadata '+var.split('|')[1]+' not compatible with CSV output.It is going to be outputted into Xml out')
      else:
        inpKeys   = self._dataContainer['inputs'].keys()
        inpValues = self._dataContainer['inputs'].values()
        outKeys   = self._dataContainer['outputs'].keys()
        outValues = self._dataContainer['outputs'].values()
        if len(self._dataContainer['metadata'].keys()) > 0:
          #write metadata as well_known_implementations
          for key,value in self._dataContainer['metadata'].items():
            if key in self.metaExclXml:
              if type(value) not in self.metatype:
                raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> metadata '+key+' not compatible with CSV output. Its type needs to be one of '+str(self.metatype))
              inpKeys.append(key)
              if type(value) != np.ndarray: inpValues.append(np.atleast_1d(np.float(value)))
              else: inpValues.append(np.atleast_1d(value))
      if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '.csv', 'wb')
      else: return

      #Print header
      myFile.write(b','.join([utils.toBytes(str(item)) for item in itertools.chain(inpKeys,outKeys)]))
      myFile.write(b'\n')
      #Print values
      for j in range(len(next(iter(itertools.chain(inpValues,outValues))))):
        myFile.write(b','.join([utils.toBytes(str(item[j])) for item in itertools.chain(inpValues,outValues)]))
        myFile.write(b'\n')
      myFile.close()
      self._createXMLFile(filenameLocal,'timepointset',inpKeys,outKeys)

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    '''override of the method in the base class Datas'''
    if stepID!=None: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> seeking to extract a history slice over an TimePointSet type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':
      if varID!=None:
        if self._dataParameters['hierarchical']: exec('extractedValue ='+varTyp +'(self.getHierParam(inOutType,nodeid,varName,serialize=False)[nodeid])')
        else: exec('extractedValue ='+varTyp +'(self.getParam(inOutType,varName)[varID])')
        return extractedValue
      #if varID!=None: exec ('return varTyp(self.getParam('+inOutType+','+varName+')[varID])')
      else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> trying to extract a scalar value from a time point set without an index')
    else:
      if self._dataParameters['hierarchical']:
        paramss = self.getHierParam(inOutType,nodeid,varName,serialize=True)
        extractedValue = np.zeros(len(paramss[nodeid]))
        for index in range(len(paramss[nodeid])): extractedValue[index] = paramss[nodeid][index]
        return extractedValue
      else: return self.getParam(inOutType,varName)

class History(Data):
  def acceptHierarchical(self):
    ''' Overwritten from baseclass'''
    return False

  def addSpecializedReadingSettings(self):
    '''
      This function adds in the _dataParameters dict the options needed for reading and constructing this class
    '''
    self._dataParameters['type'] = self.type # store the type into the _dataParameters dictionary
    try: sourceType = self._toLoadFromList[0].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      if(not self._dataParameters['history']): raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> In order to create a History data, history name must be provided')
      self._dataParameters['filter'] = 'whole'

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data History
      @ In, None
      @ Out, None
    '''
    for key in self._dataContainer['inputs'].keys():
      if (self._dataContainer['inputs'][key].size) != 1:
        raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The input parameter value, for key ' + key + ' has not a consistent shape for History ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(len(self._dataContainer['inputs'][key])))
    for key in self._dataContainer['outputs'].keys():
      if (self._dataContainer['outputs'][key].ndim) != 1:
        raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The output parameter value, for key ' + key + ' has not a consistent shape for History ' + self.name + '!! It should be an 1D array.' + '.Actual dimension is ' + str(self._dataContainer['outputs'][key].ndim))

  def _updateSpecializedInputValue(self,name,value,options=None):
    '''
      This function performs the updating of the values (input space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (1-D array)
      @ Out, None
    '''
    if name in self._dataContainer['inputs'].keys():
      self._dataContainer['inputs'].pop(name)
    if name not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(name)
    self._dataContainer['inputs'][name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def _updateSpecializedMetadata(self,name,value,options=None):
    '''
      This function performs the updating of the values (metadata) into this Data
      @ In,  name, string, parameter name (ex. probability)
      @ In,  value, whatever type, newer value
      @ Out, None
      NB. This method, if the metadata name is already present, replaces it with the new value. No appending here, since the metadata are dishomogenius and a common updating strategy is not feasable.
    '''
    self._dataContainer['metadata'][name] = copy.deepcopy(value)

  def _updateSpecializedOutputValue(self,name,value,options=None):
    '''
      This function performs the updating of the values (output space) into this Data
      @ In,  name, string, parameter name (ex. cladTemperature)
      @ In,  value, float, newer value (1-D array)
      @ Out, None
    '''
    if name in self._dataContainer['outputs'].keys():
      self._dataContainer['outputs'].pop(name)
    if name not in self._dataParameters['outParam']: self._dataParameters['outParam'].append(name)
    self._dataContainer['outputs'][name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal,options):
    '''
      This function prints a CSV file with the content of this class (Input and Output space)
      @ In,  filenameLocal, string, filename root (for example, 'homo_homini_lupus' -> the final file name is gonna be called 'homo_homini_lupus.csv')
      @ In,  options, dictionary, dictionary of printing options
      @ Out, None (a csv is gonna be printed)
    '''
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
    if 'variables' in options.keys():
      for var in options['variables']:
        if var.split('|')[0] == 'input':
          inpKeys.append(var.split('|')[1])
          inpValues.append(self._dataContainer['inputs'][var.split('|')[1]])
        if var.split('|')[0] == 'output':
          outKeys.append(var.split('|')[1])
          outValues.append(self._dataContainer['outputs'][var.split('|')[1]])
        if var.split('|')[0] == 'metadata':
          if var.split('|')[1] in self.metaExclXml:
            if type(self._dataContainer['metadata'][var.split('|')[1]]) not in self.metatype:
              raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> metadata '+var.split('|')[1]+' not compatible with CSV output. Its type needs to be one of '+str(self.metatype))
            inpKeys.append(var.split('|')[1])
            inpValues.append(np.atleast_1d(np.float(self._dataContainer['metadata'][var.split('|')[1]])))
          else: print(self.printTag+': ' +utils.returnPrintPostTag('Warning') + '-> metadata '+var.split('|')[1]+' not compatible with CSV output.It is going to be outputted into Xml out')
    else:
      inpKeys   = self._dataContainer['inputs'].keys()
      inpValues = self._dataContainer['inputs'].values()
      outKeys   = self._dataContainer['outputs'].keys()
      outValues = self._dataContainer['outputs'].values()
      if len(self._dataContainer['metadata'].keys()) > 0:
        #write metadata as well_known_implementations
        for key,value in self._dataContainer['metadata'].items():
          if key in self.metaExclXml:
            if type(value) not in self.metatype:
              raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> metadata '+key+' not compatible with CSV output. Its type needs to be one of '+str(self.metatype))
            inpKeys.append(key)
            inpValues.append(np.atleast_1d(np.float(value)))

    if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '.csv', 'wb')
    else: return

    #Create Input file
    #Print header
    myFile.write(b','.join([utils.toBytes(item) for item in  itertools.chain(inpKeys,['filename'])]))
    myFile.write(b'\n')
    #Print data
    myFile.write(b','.join([utils.toBytes(str(item[0])) for item in itertools.chain(inpValues,[[filenameLocal + '_0' + '.csv']])]))
    myFile.write(b'\n')
    myFile.close()
    #Create Output file
    myDataFile = open(filenameLocal + '_0' + '.csv', 'wb')
    #Print headers
    #Print time + output values
    myDataFile.write(b','.join([utils.toBytes(item) for item in outKeys]))
    myDataFile.write(b'\n')
    #Print data
    for j in range(next(iter(outValues)).size):
      myDataFile.write(b','.join([utils.toBytes(str(item[j])) for item in outValues]))
      myDataFile.write(b'\n')
    myDataFile.close()
    self._createXMLFile(filenameLocal,'history',inpKeys,outKeys)

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    '''override of the method in the base class Datas'''
    if varID!=None: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->seeking to extract a slice over number of parameters an History type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':
      if varName in self._dataParameters['inParam']: exec ('return varTyp(self.getParam('+inOutType+','+varName+')[0])')
      else:
        if stepID!=None and type(stepID)!=tuple: exec ('return self.getParam('+inOutType+','+varName+')['+str(stepID)+']')
        else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->To extract a scalar from an history a step id is needed. Variable: '+varName+', Data: '+self.name)
    else:
      if stepID==None : return self.getParam(inOutType,varName)
      elif stepID!=None and type(stepID)==tuple: return self.getParam(inOutType,varName)[stepID[0]:stepID[1]]
      else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->trying to extract variable '+varName+' from '+self.name+' the id coordinate seems to be incoherent: stepID='+str(stepID))


class Histories(Data):
  def acceptHierarchical(self):
    '''
      Overwritten from base class
    '''
    return True

  def addSpecializedReadingSettings(self):
    '''
      This function adds in the _dataParameters dict the options needed for reading and constructing this class
      @ In,  None
      @ Out, None
    '''
    if self._dataParameters['hierarchical']: self._dataParameters['type'] = 'History'
    else: self._dataParameters['type'] = self.type # store the type into the _dataParameters dictionary
    try: sourceType = self._toLoadFromList[0].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      self._dataParameters['filter'   ] = 'whole'

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data Histories
      @ In,  None
      @ Out, None
    '''
    try: sourceType = self._toLoadFromList[-1].type
    except AttributeError: sourceType = None

    if self._dataParameters['hierarchical']:
      for key in self._dataContainer['inputs'].keys():
        if (self._dataContainer['inputs'][key].size) != 1:
          raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The input parameter value, for key ' + key + ' has not a consistent shape for History in Histories ' + self.name + '!! It should be a single value since we are in hierarchical mode.' + '.Actual size is ' + str(len(self._dataContainer['inputs'][key])))
      for key in self._dataContainer['outputs'].keys():
        if (self._dataContainer['outputs'][key].ndim) != 1:
          raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The output parameter value, for key ' + key + ' has not a consistent shape for History in Histories ' + self.name + '!! It should be an 1D array since we are in hierarchical mode.' + '.Actual dimension is ' + str(self._dataContainer['outputs'][key].ndim))
    else:
      if('HDF5' == sourceType):
        eg = self._toLoadFromList[0].getEndingGroupNames()
        if(len(eg) != len(self._dataContainer['inputs'].keys())):
          raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> Number of Histories contained in Histories data ' + self.name + ' != number of loading sources!!! ' + str(len(eg)) + ' !=' + str(len(self._dataContainer['inputs'].keys())))
      else:
        if(len(self._toLoadFromList) != len(self._dataContainer['inputs'].keys())):
          raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> Number of Histories contained in Histories data ' + self.name + ' != number of loading sources!!! ' + str(len(self._toLoadFromList)) + ' !=' + str(len(self._dataContainer['inputs'].keys())))
      for key in self._dataContainer['inputs'].keys():
        for key2 in self._dataContainer['inputs'][key].keys():
          if (self._dataContainer['inputs'][key][key2].size) != 1:
            raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The input parameter value, for key ' + key2 + ' has not a consistent shape for History ' + key + ' contained in Histories ' +self.name+ '!! It should be a single value.' + '.Actual size is ' + str(len(self._dataContainer['inputs'][key][key2])))
      for key in self._dataContainer['outputs'].keys():
        for key2 in self._dataContainer['outputs'][key].keys():
          if (self._dataContainer['outputs'][key][key2].ndim) != 1:
            raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The output parameter value, for key ' + key2 + ' has not a consistent shape for History ' + key + ' contained in Histories ' +self.name+ '!! It should be an 1D array.' + '.Actual dimension is ' + str(self._dataContainer['outputs'][key][key2].ndim))

  def _updateSpecializedInputValue(self,name,value,options=None):
    '''
      This function performs the updating of the values (input space) into this Data
      @ In,  name, either 1) list (size = 2), name[0] == history number(ex. 1 or 2 etc) - name[1], parameter name (ex. cladTemperature)
                       or 2) string, parameter name (ex. cladTemperature) -> in this second case,the parameter is added in the last history (if not present),
                                                                             otherwise a new history is created and the new value is inserted in it
      @ In, value, newer value
      @ Out, None
    '''
    if (not isinstance(value,(float,int,bool,np.ndarray))):
      raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> Histories Data accepts only a numpy array (dim 1) or a single value for method <_updateSpecializedInputValue>. Got type ' + str(type(value)))
    if isinstance(value,np.ndarray):
      if value.size != 1: raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> Histories Data accepts only a numpy array of dim 1 or a single value for method <_updateSpecializedInputValue>. Size is ' + str(value.size))

    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'History' has been stored
      parent_id = None
      if type(name) == list:
        namep = name[1]
        if type(name[0]) == str: nodeid = name[0]
        else:
          if 'metadata' in options.keys():
            nodeid = options['metadata']['prefix']
            if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
          else:
            nodeid = options['prefix']
            if 'parent_id' in options.keys(): parent_id = options['parent_id']
      else:
        if 'metadata' in options.keys():
          nodeid = options['metadata']['prefix']
          if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
        else:
          nodeid = options['prefix']
          if 'parent_id' in options.keys(): parent_id = options['parent_id']
        namep = name
      if parent_id: tsnode = self.retrieveNodeInTreeMode(nodeid, parent_id)
      else:         tsnode = self.retrieveNodeInTreeMode(nodeid)
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if namep in self._dataContainer['inputs'].keys():
        self._dataContainer['inputs'].pop(name)
      if namep not in self._dataParameters['inParam']: self._dataParameters['inParam'].append(namep)
      self._dataContainer['inputs'][namep] = copy.deepcopy(np.atleast_1d(np.array(value)))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if type(name) == list:
        # there are info regarding the history number
        if name[0] in self._dataContainer['inputs'].keys():
          gethistory = self._dataContainer['inputs'].pop(name[0])
          popped = gethistory[name[1]]
          if name[1] in popped.keys():
            gethistory[name[1]] = copy.deepcopy(np.atleast_1d(np.array(value)))
            self._dataContainer['inputs'][name[0]] = copy.deepcopy(gethistory)
        else:
          self._dataContainer['inputs'][name[0]] = copy.deepcopy({name[1]:np.atleast_1d(np.array(value))})
      else:
        # no info regarding the history number => use internal counter
        if len(self._dataContainer['inputs'].keys()) == 0: self._dataContainer['inputs'][1] = copy.deepcopy({name:np.atleast_1d(np.array(value))})
        else:
          hisn = max(self._dataContainer['inputs'].keys())
          if name in list(self._dataContainer['inputs'].values())[-1]:
            hisn += 1
            self._dataContainer['inputs'][hisn] = {}
          self._dataContainer['inputs'][hisn][name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def _updateSpecializedMetadata(self,name,value,options=None):
    '''
      This function performs the updating of the values (metadata) into this Data
      @ In,  name, string, parameter name (ex. probability)
      @ In,  value, whatever type, newer value
      @ Out, None
      NB. This method, if the metadata name is already present, replaces it with the new value. No appending here, since the metadata are dishomogenius and a common updating strategy is not feasable.
    '''
    if options and self._dataParameters['hierarchical']:
      # we retrieve the node in which the specialized 'TimePoint' has been stored
      parent_id = None
      if type(name) == list:
        namep = name[1]
        if type(name[0]) == str: nodeid = name[0]
        else:
          if 'metadata' in options.keys():
            nodeid = options['metadata']['prefix']
            if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
          else:
            nodeid = options['prefix']
            if 'parent_id' in options.keys(): parent_id = options['parent_id']
      else:
        if 'metadata' in options.keys():
          nodeid = options['metadata']['prefix']
          if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
        else:
          nodeid = options['prefix']
          if 'parent_id' in options.keys(): parent_id = options['parent_id']
        namep = name
      if parent_id: tsnode = self.retrieveNodeInTreeMode(nodeid, parent_id)
      #if 'parent_id' in options.keys(): tsnode = self.retrieveNodeInTreeMode(options['prefix'], options['parent_id'])
      #else:                             tsnode = self.retrieveNodeInTreeMode(options['prefix'])
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'metadata':{}})
        self._dataContainer = tsnode.get('dataContainer')
      else:
        if 'metadata' not in self._dataContainer.keys(): self._dataContainer['metadata'] ={}
      if name in self._dataContainer['metadata'].keys(): self._dataContainer['metadata'][name] = copy.deepcopy(np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value))))
      else                                             : self._dataContainer['metadata'][name] = copy.deepcopy(np.atleast_1d(value))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if name in self._dataContainer['metadata'].keys():
        self._dataContainer['metadata'][name] = copy.deepcopy(np.concatenate((self._dataContainer['metadata'][name],np.atleast_1d(value))))
      else                                             : self._dataContainer['metadata'][name] = copy.deepcopy(np.atleast_1d(value))

  def _updateSpecializedOutputValue(self,name,value,options=None):
    '''
      This function performs the updating of the values (output space) into this Data
      @ In,  name, either 1) list (size = 2), name[0] == history number(ex. 1 or 2 etc) - name[1], parameter name (ex. cladTemperature)
                       or 2) string, parameter name (ex. cladTemperature) -> in this second case,the parameter is added in the last history (if not present),
                                                                             otherwise a new history is created and the new value is inserted in it
      @ Out, None
    '''
    if not isinstance(value,np.ndarray):
        raise NotConsistentData(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> Histories Data accepts only numpy array as type for method <_updateSpecializedOutputValue>. Got ' + str(type(value)))

    if options and self._dataParameters['hierarchical']:
      parent_id = None
      if type(name) == list:
        namep = name[1]
        if type(name[0]) == str: nodeid = name[0]
        else:
          if 'metadata' in options.keys():
            nodeid = options['metadata']['prefix']
            if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
          else:
            nodeid = options['prefix']
            if 'parent_id' in options.keys(): parent_id = options['parent_id']
      else:
        if 'metadata' in options.keys():
          nodeid = options['metadata']['prefix']
          if 'parent_id' in options['metadata'].keys(): parent_id = options['metadata']['parent_id']
        else:
          nodeid = options['prefix']
          if 'parent_id' in options.keys(): parent_id = options['parent_id']
        namep = name
      if parent_id: tsnode = self.retrieveNodeInTreeMode(nodeid, parent_id)

      # we store the pointer to the container in the self._dataContainer because checkConsistency acts on this
      self._dataContainer = tsnode.get('dataContainer')
      if not self._dataContainer:
        tsnode.add('dataContainer',{'inputs':{},'outputs':{}})
        self._dataContainer = tsnode.get('dataContainer')
      if namep in self._dataContainer['outputs'].keys():
        self._dataContainer['outputs'].pop(namep)
      if namep not in self._dataParameters['inParam']: self._dataParameters['outParam'].append(namep)
      self._dataContainer['outputs'][namep] = copy.deepcopy(np.atleast_1d(np.array(value)))
      self.addNodeInTreeMode(tsnode,options)
    else:
      if type(name) == list:
        # there are info regarding the history number
        if name[0] in self._dataContainer['outputs'].keys():
          gethistory = self._dataContainer['outputs'].pop(name[0])
          popped = gethistory[name[1]]
          if name[1] in popped.keys():
            gethistory[name[1]] = copy.deepcopy(np.atleast_1d(np.array(value)))
            self._dataContainer['outputs'][name[0]] = copy.deepcopy(gethistory)
        else:
          self._dataContainer['outputs'][name[0]] = copy.deepcopy({name[1]:np.atleast_1d(np.array(value))})
      else:
        # no info regarding the history number => use internal counter
        if len(self._dataContainer['outputs'].keys()) == 0: self._dataContainer['outputs'][1] = copy.deepcopy({name:np.atleast_1d(np.array(value))})
        else:
          hisn = max(self._dataContainer['outputs'].keys())
          if name in list(self._dataContainer['outputs'].values())[-1]:
            hisn += 1
            self._dataContainer['outputs'][hisn] = {}
          self._dataContainer['outputs'][hisn][name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal,options):
    '''
      This function prints a CSV file with the content of this class (Input and Output space)
      @ In,  filenameLocal, string, filename root (for example, 'homo_homini_lupus' -> the final file name is gonna be called 'homo_homini_lupus.csv')
      @ In,  options, dictionary, dictionary of printing options
      @ Out, None (a csv is gonna be printed)
    '''

    if self._dataParameters['hierarchical']:
      outKeys   = []
      inpKeys   = []
      inpValues = []
      outValues = []
      # retrieve a serialized of datas from the tree
      O_o = self.getHierParam('inout','*',serialize=True)
      for key in O_o.keys():
        inpKeys.append([])
        inpValues.append([])
        outKeys.append([])
        outValues.append([])
        if 'variables' in options.keys():
          for var in options['variables']:
            if var.split('|')[0] == 'input':
              inpKeys[-1].append(var.split('|')[1])
              axa = np.zeros(len(O_o[key]))
              for index in range(len(O_o[key])):
                axa[index] = O_o[key][index]['inputs'][var.split('|')[1]][0]
              inpValues[-1].append(axa)
            if var.split('|')[0] == 'output':
              outKeys[-1].append(var.split('|')[1])
              axa = O_o[key][0]['outputs'][var.split('|')[1]]
              for index in range(len(O_o[key])-1): axa = np.concatenate((axa,O_o[key][index+1]['outputs'][var.split('|')[1]]))
              outValues[-1].append(axa)
        else:
          inpKeys[-1] = O_o[key][0]['inputs'].keys()
          outKeys[-1] = O_o[key][0]['outputs'].keys()
          for var in O_o[key][0]['inputs'].keys():
            axa = np.zeros(len(O_o[key]))
            for index in range(len(O_o[key])): axa[index] = O_o[key][index]['inputs'][var][0]
            inpValues[-1].append(copy.deepcopy(axa))
          for var in O_o[key][0]['outputs'].keys():
            axa = O_o[key][0]['outputs'][var]
            for index in range(len(O_o[key])-1):
              axa = np.concatenate((axa,O_o[key][index+1]['outputs'][var]))
            outValues[-1].append(copy.deepcopy(axa))

        if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '_' + key + '.csv', 'wb')
        else: return
        myFile.write(b'Ending branch,'+utils.toBytes(key)+b'\n')
        myFile.write(b'branch #')
        for item in inpKeys[-1]:
          myFile.write(b',' + utils.toBytes(item))
        myFile.write(b'\n')
        # write the input paramters' values for each branch
        for i in range(inpValues[-1][0].size):
          myFile.write(utils.toBytes(str(i+1)))
          for index in range(len(inpValues[-1])):
            myFile.write(b',' + utils.toBytes(str(inpValues[-1][index][i])))
          myFile.write(b'\n')
        # write out keys
        myFile.write(b'\n')
        myFile.write(b'TimeStep #')
        for item in outKeys[-1]:
          myFile.write(b',' + utils.toBytes(item))
        myFile.write(b'\n')
        for i in range(outValues[-1][0].size):
          myFile.write(utils.toBytes(str(i+1)))
          for index in range(len(outValues[-1])):
            myFile.write(b',' + utils.toBytes(str(outValues[-1][index][i])))
          myFile.write(b'\n')
        myFile.close()
    else:
      #if not hierarchical
      #For histories, create an XML file, and multiple CSV
      #files.  The first CSV file has a header with the input names,
      #and a column for the filenames.  There is one CSV file for each
      #data line in the first CSV and they are named with the
      #filename.  They have the output names for a header, a column
      #for time, and the rest of the file is data for different times.
      inpValues = list(self._dataContainer['inputs'].values())
      outKeys   = self._dataContainer['outputs'].keys()
      outValues = list(self._dataContainer['outputs'].values())
      #Create Input file
      myFile = open(filenameLocal + '.csv','wb')
      for n in range(len(outKeys)):
        inpKeys_h   = []
        inpValues_h = []
        outKeys_h   = []
        outValues_h = []
        if 'variables' in options.keys():
          for var in options['variables']:
            if var.split('|')[0] == 'input':
              inpKeys_h.append(var.split('|')[1])
              inpValues_h.append(inpValues[n][var.split('|')[1]])
            if var.split('|')[0] == 'output':
              outKeys_h.append(var.split('|')[1])
              outValues_h.append(outValues[n][var.split('|')[1]])
        else:
          inpKeys_h   = list(inpValues[n].keys())
          inpValues_h = list(inpValues[n].values())
          outKeys_h   = list(outValues[n].keys())
          outValues_h = list(outValues[n].values())

        dataFilename = filenameLocal + '_'+ str(n) + '.csv'
        if len(inpKeys_h) > 0 or len(outKeys_h) > 0: myDataFile = open(dataFilename, 'wb')
        else: return #XXX should this just skip this iteration?
        #Write header for main file
        if n == 0:
          myFile.write(b','.join([utils.toBytes(item) for item in
                                  itertools.chain(inpKeys_h,['filename'])]))
          myFile.write(b'\n')
          self._createXMLFile(filenameLocal,'histories',inpKeys_h,outKeys_h)
        myFile.write(b','.join([utils.toBytes(str(item[0])) for item in
                                itertools.chain(inpValues_h,[[dataFilename]])]))
        myFile.write(b'\n')
        #Data file
        #Print time + output values
        myDataFile.write(b','.join([utils.toBytes(item) for item in outKeys_h]))
        if len(outKeys_h) > 0:
          myDataFile.write(b'\n')
          for j in range(outValues_h[0].size):
            myDataFile.write(b','.join([utils.toBytes(str(item[j])) for item in
                                    outValues_h]))
            myDataFile.write(b'\n')
        myDataFile.close()
      myFile.close()

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None,nodeid='root'):
    '''
      override of the method in the base class Datas
      @ In,  myType, string, unused
      @ In,  inOutType
      IMPLEMENT COMMENT HERE
    '''
    if varTyp!='numpy.ndarray':
      if varName in self._dataParameters['inParam']:
        if varID!=None: exec ('return varTyp(self.getParam('+inOutType+','+str(varID)+')[varName]')
        else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID to identify the history (varID missed)')
      else:
        if varID!=None:
          if stepID!=None and type(stepID)!=tuple: exec ('return varTyp(self.getParam('+inOutType+','+str(varID)+')[varName][stepID]')
          else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID of the input set used and a time coordinate (time or timeID missed or tuple)')
        else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID of the input set used (varID missed)')
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
          if stepID==None: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->more info needed trying to extract '+varName+' from data '+self.name)
          elif type(stepID)==tuple:
            if stepID[1]!=None:
              myOut=np.zeros((len(self.getOutParametersValues().keys()),stepID[1]-stepID[0]))
              for key in self.getOutParametersValues().keys():
                myOut[int(key),:]=self.getParam(inOutType,key)[varName][stepID[0]:stepID[1]]
            else: raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->more info needed trying to extract '+varName+' from data '+self.name)
          else:
            myOut=np.zeros(len(self.getOutParametersValues().keys()))
            for key in self.getOutParametersValues().keys():
              myOut[int(key)]=self.getParam(inOutType,key)[varName][stepID]
            return myOut

'''
 Interface Dictionary (factory) (private)
'''
__base                          = 'Data'
__interFaceDict                 = {}
__interFaceDict['TimePoint'   ] = TimePoint
__interFaceDict['TimePointSet'] = TimePointSet
__interFaceDict['History'     ] = History
__interFaceDict['Histories'   ] = Histories
__knownTypes                    = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
