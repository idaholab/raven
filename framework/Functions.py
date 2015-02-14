'''
Created on Oct 20, 2014

@author: alfoa

This module contains interfaces to import external functions
'''
#for future compatibility with Python 3--------------------------------------------------------------

#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
import utils
from CustomCommandExecuter import execCommand
#import Datas
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
    self.__functionFile                  = ''                                # function file name
    self.__actionDictionary              = {}                                # action dictionary
    # dictionary of implemented actions
    self.__actionImplemented             = {'residuumSign':False,'supportBoundingTest':False,'residuum':False,'gradient':False}
    self.__inputVariables                = []                                # list of variables' names' given in input (xml)
    self.__inputFromWhat                 = {}                                # dictionary of input data type
    self.__inputFromWhat['dict']         = self.__inputFromDict
    self.__inputFromWhat['Data']         = self.__inputFromData
    self.printTag                        = utils.returnPrintTag('FUNCTIONS')
  def _readMoreXML(self,xmlNode,debug=False):
    if 'file' in xmlNode.attrib.keys():
      self.functionFile = xmlNode.attrib['file']
      if self.functionFile.endswith('.py') : moduleName = ''.join(self.functionFile.split('.')[:-1]) #remove the .py
      else: moduleName = self.functionFile
      importedModule = utils.importFromPath(moduleName)
      if not importedModule: raise IOError('Failed to import the module '+moduleName+' supposed to contain the function: '+self.name)
      #here the methods in the imported file are brought inside the class
      for method in importedModule.__dict__.keys():
        if method in ['__residuumSign__','__residuumSign','residuumSign',
                      '__supportBoundingTest__','__supportBoundingTest','supportBoundingTest',
                      '__residuum__','__residuum','residuum','__gradient__','__gradient','gradient']:
          if method in ['__residuumSign__','__residuumSign','residuumSign']:
            self.__residuumSign                                =  importedModule.__dict__[method]
            self.__actionDictionary['residuumSign' ]           = self.__residuumSign
            self.__actionImplemented['residuumSign']           = True
          if method in ['__supportBoundingTest__','__supportBoundingTest','supportBoundingTest']:
            self.__supportBoundingTest                         =  importedModule.__dict__[method]
            self.__actionDictionary['supportBoundingTest' ]    = self.__supportBoundingTest
            self.__actionImplemented['supportBoundingTest']    = True
          if method in ['__residuum__','__residuum','residuum']:
            self.__residuum                                    =  importedModule.__dict__[method]
            self.__actionDictionary['residuum' ]               = self.__residuum
            self.__actionImplemented['residuum']               = True
          if method in ['__gradient__','__gradient','gradient']:
            self.__gradient                                    =  importedModule.__dict__[method]
            self.__actionDictionary['gradient']                = self.__gradient
            self.__actionImplemented['gradient']               = True
        else:
          #custom
          self.__actionDictionary[method]                    = importedModule.__dict__[method]
          self.__actionImplemented[method]                   = True
    else: raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> No file name for the external function has been provided for external function '+self.name+' of type '+self.type)
    cnt = 0
    for child in xmlNode:
      if child.tag=='variable':
        execCommand('self.'+child.text+' = None',self=self)
        self.__inputVariables.append(child.text)
        cnt +=1
        if len(child.attrib.keys()) > 0: raise IOError( self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> variable block in the definition of the function '+self.name + ' should not have any attribute!')
    if cnt == 0: raise IOError( self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> not variable found in the definition of the function '+self.name)

  def addInitParams(self,tempDict):
    '''
    This function is called from the base class to print some of the information inside the class.
    Whatever is permanent in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary. No information about values that change during the simulation are allowed
    @ In/Out tempDict: {'attribute name':value}
    '''
    tempDict['Module file name'                    ] = self.functionFile
    tempDict['The residuum is provided'            ] = self.__actionImplemented['residuum']
    tempDict['The sign of the residuum is provided'] = self.__actionImplemented['residuumSign']
    tempDict['The gradient is provided'            ] = self.__actionImplemented['gradient']
    tempDict['The support bonding is provided'     ] = self.__actionImplemented['supportBoundingTest']
    for key,value in enumerate(self.__actionImplemented):
      if key not in ['residualSign','supportBoundingTest','residual','gradient']: tempDict['Custom Function'] = value
  def addCurrentSetting(self,tempDict):
    '''
    This function is called from the base class to print some of the information inside the class.
    Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary
    Function adds the current settings in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict
    '''
    for key in self.__inputVariables: execCommand("object['variable "+str(key)+" has value']=self."+key,self=self,object=tempDict)

  def __importValues(self,myInput):
    '''this makes available the variable values sent in as self.key'''
    if type(myInput)==dict         :self.__inputFromWhat['dict'](myInput)
    elif 'Data' in myInput.__base__:self.__inputFromWhat['Data'](myInput)
    else: raise Exception (self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> Unknown type of input provided to the function '+str(self.name))

  def __inputFromData(self,inputData):
    '''
    This is meant to be used to collect the input from a Data. A conversion to the declared type of data is attempted by inputData.extractValue'''
    for key in self.__inputVariables:
      #exec('self.'+key+'=inputData.extractValue(myType,key)')
      ##### TEMPORARY FIXXXXXXXX - ALIAS NEEDED#######
      print('FIXME: Alias are already in place why we have still the fixme (once done see also if the loop should contain the myType???')
      foundperfectly = False
      for index in range(len(inputData.dataParameters['inParam'])):
        if key == inputData.dataParameters['inParam'][index]: foundperfectly = True
      if not foundperfectly:
        for index in range(len(inputData.dataParameters['outParam'])):
          if key == inputData.dataParameters['outParam'][index]: foundperfectly = True
      if foundperfectly: execCommand('self.'+key+'=object.extractValue(myType,"'+key+'")',self=self,object=inputData)
      if not foundperfectly:
        semifound = False
        for index in range(len(inputData.dataParameters['inParam'])):
          if key in inputData.dataParameters['inParam'][index]:
            similarVariable = inputData.dataParameters['inParam'][index]
            semifound = True
        if not semifound:
          for index in range(len(inputData.dataParameters['outParam'])):
            if key in inputData.dataParameters['outParam'][index]:
              similarVariable = inputData.dataParameters['outParam'][index]
              semifound = True
        if semifound: execCommand('self.'+key+'=object.extractValue(myType,"'+similarVariable+'")',self=self,object=inputData)

  def __inputFromDict(self,myInputDict):
    '''
    This is meant to be used to collect the input directly from a sampler generated input or simply from a generic dictionary
    In case the input come from a sampler the expected structure is myInputDict['SampledVars'][variable name] = value
    In case it is a generic dictionary the expected structure is myInputDict[variable name] = value
    '''
    if 'SampledVars' in myInputDict.keys(): inDict = myInputDict['SampledVars']
    else                                  : inDict = myInputDict
    for name in self.__inputVariables:
      if name in inDict.keys(): execCommand('self.'+name+'=object["'+name+'"]',self=self,object=inDict)
      else                    : raise Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The input variable '+name+' in external function seems not to be passed in')

  def evaluate(self,what,myInput):
    '''return the result of the type of action described by 'what' '''
    self.__importValues(myInput)

    if what not in self.__actionDictionary:
      raise IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR')
                    + '-> Method ' + what + ' not defined in ' + self.name)
    return self.__actionDictionary[what](self)

  def availableMethods(self):
    ''' Get a list of the callable methods this interface provides '''
    return self.__actionDictionary.keys()

  def parameterNames(self):
    ''' Get a list of the variables this function needs '''
    return self.__inputVariables[:]

'''
 Interface Dictionary (factory) (private)
'''

__base = 'function'
__interFaceDict = {}
__interFaceDict['External'] = Function
__knownTypes                = __interFaceDict.keys()


def knownTypes():
  return __knownTypes

def returnInstance(Type):
  '''This function return an instance of the request model type'''
  if Type in knownTypes():return __interFaceDict[Type]()
  else: raise NameError('not known '+__base+' type '+Type)
