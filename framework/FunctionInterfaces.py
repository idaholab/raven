'''
This module contains interfaces to import external functions
'''
#for future compatibility with Python 3--------------------------------------------------------------

#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseType import BaseType
import Datas
#Internal Modules End--------------------------------------------------------------------------------

class Function(BaseType):
  '''
  This class is the base class for different wrapper for external functions
  provide the tools to solve
  F(x)=0 where at least one of the component of the output space of a model (F is a vector as x)
  plus a series of inequality in the input space that are considered by the supportBoundingTest
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.__functionFile                  = ''
    self.varType                         = {}
    self.__actionDictionary              = {}
    self.__inputFromWhat                 = {}
    self.__inputFromWhat['dict']         = self.__inputFromDict
    self.__inputFromWhat['Data']         = self.__inputFromData    
    
  def readMoreXML(self,xmlNode):
    if 'file' in xmlNode.attrib.keys():
      self.functionFile = xmlNode.attrib['file']
      moduleName        = ''.join(xmlNode.attrib['file'].split('.')[:-1]) #remove the .py
      exec('import '+ moduleName)
      #here the methods in the imported file are brought inside the class
      for method in moduleName.__dict__.keys():
        if method == 'residualSign':
          self.__residualSign                                =  moduleName.__dict__['residualSign']
          self.__actionDictionary['residualSign']            = self.__residualSign
          self.__actionImplemented['residualSign']           = True
        else:self.__actionImplemented['residualSign']        = False
        if method == 'supportBoundingTest':
          self.__supportBoundingTest                         =  moduleName.__dict__['supportBoundingTest']
          self.__actionDictionary['supportBoundingTest']     = self.__supportBoundingTest
          self.__actionImplemented['supportBoundingTest']    = True
        else:self.__actionImplemented['supportBoundingTest'] = False
        if method == 'residuum':
          self.__residual                                    =  moduleName.__dict__['residuum']
          self.__actionDictionary['residuum']                = self.__residual
          self.__actionImplemented['residuum']               = True
        else:self.__actionImplemented['residuum']            = False
        if method == 'gradient':
          self.__gradient                                    =  moduleName.__dict__['gradient']
          self.__actionDictionary['gradient']                = self.__gradient
          self.__actionImplemented['gradient']               = True
        else:self.__actionImplemented['gradient']            = False
    else: raise IOError('No file name for the external function has been provided for external function '+self.name+' of type '+self.type)
    for child in xmlNode:
      if child.tag=='variable':
        self.__inVarValues[child.text] = None
        exec('self.'+child.text+' = self.inVarValues['+'child.text'+']')
        if 'type' in child.attrib.keys(): self.varType[child.text] = child.attrib['type']
        else                            : raise IOError('the type for the variable '+child.text+' is missed')
        
  def addInitParams(self,tempDict):
    '''
    This function is called from the base class to print some of the information inside the class.
    Whatever is permanent in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary. No information about values that change during the simulation are allowed
    @ In/Out tempDict: {'attribute name':value}
    '''
    tempDict['Module file name'                    ] = self.functionFile
    tempDict['The residuum is provided'            ] = self.__actionImplemented['residuum']
    tempDict['The sign of the residuum is provided'] = self.__actionImplemented['residualSign']
    tempDict['The gradient is provided'            ] = self.__actionImplemented['gradient']
    tempDict['The support bonding is provided'     ] = self.__actionImplemented['supportBoundingTest']
    for key in self.inVarValues.keys():
      tempDict['Variable:type'                     ] = key+':'+self.varType[key]

  def addCurrentSetting(self,tempDict):
    '''
    This function is called from the base class to print some of the information inside the class.
    Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary
    Function adds the current settings in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    for key in self.varType.keys():
      exec("tempDict['variable '+"+key+"+'has value'] = 'self.'"+key)
      exec("tempDict['variable '+"+key+"+' is of type'] = 'self.'"+self.varType[key])


  def __importValues(self,myInput):
    '''this makes available the variable values sent in as self.key'''
    if type(myInput)==dict         :self.__inputFromWhat['dict'](myInput)
    elif 'Data' in myInput.__base__:self.__inputFromWhat['Data'](myInput)
    else: raise 'Unknown type of input provided to the function '+str(self.name)

  def __inputFromData(self,inputData):
    '''
    This is meant to be used to collect the input from a Data. A conversion to the declared type of data is attempted by inputData.extractValue'''
    for key, myType in self.varType.items():
      exec('self.'+key+'=inputData.extractValue('+myType+','+key+')')

  def __inputFromDict(self,myInputDict):
    '''
    This is meant to be used to collect the input directly from a sampler generated input or simply from a generic dictionary
    In case the input come from a sampler the expected structure is myInputDict['SampledVars'][variable name] = value
    In case it is a generic dictionary the expected structure is myInputDict[variable name] = value
    '''
    if 'SampledVars' in myInputDict.keys(): inDict = myInputDict['SampledVars']
    else                                  : inDict = myInputDict
    for name, myType in self.varType.items():
      if name in inDict.keys():
        if myType == type(inDict[name]): exec("self."+name+"=inDict[name]")
        else: raise 'Not proper type for the variable '+name+' in external function '+self.name
      else: raise 'The input variable '+name+' in external function seems not to be passed in'

  def evaluate(self,what,myInput):
    '''return the result of the type of action described by 'what' '''
    self.__importValues(myInput)
    return self.__actionDictionary[what]
  
    
    
    
    
    
    
    
    
    
    
    
    
    

    
  