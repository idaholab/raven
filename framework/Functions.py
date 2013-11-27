'''
This module contains interfaces to import external functions
'''
#for future compatibility with Python 3--------------------------------------------------------------

#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy
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
    self.name                            = ''
    self.__varType__                     = {}
    self.__actionDictionary__            = {}
    self.__actionImplemented__           = {}
    self.__inputFromWhat__               = {}
    self.__inputFromWhat__['dict']       = self.__inputFromDict__
    self.__inputFromWhat__['Data']       = self.__inputFromData__
    
  def readMoreXML(self,xmlNode):
    if 'file' in xmlNode.attrib.keys():
      self.functionFile = xmlNode.attrib['file']
      moduleName        = ''.join(xmlNode.attrib['file'].split('.')[:-1]) #remove the .py
      print('moduleName '+moduleName)
      exec('import '+ moduleName)
      #here the methods in the imported file are brought inside the class
      exec('methoList='+moduleName+'.__dict__.keys()')
      for method in methoList:
        if method == '__residualSign__':
          exec('self.__residualSign__                          ='+moduleName+'.__dict__["__residualSign__"]')
          self.__actionDictionary__['residualSign'    ]        = self.__residualSign__
          self.__actionImplemented__['residualSign'   ]        = True
        else:self.__actionImplemented__['residualSign']        = False
        if method == '__supportBoundingTest__':
          exec('self.__supportBoundingTest__                   ='+moduleName+'.__dict__["__supportBoundingTest__"]')
          self.__actionDictionary__['supportBoundingTest'    ] = self.__supportBoundingTest__
          self.__actionImplemented__['supportBoundingTest'   ] = True
        else:self.__actionImplemented__['supportBoundingTest'] = False
        if method == '__residuum__':
          exec('self.__residuum__                              ='+moduleName+'.__dict__["__residuum__"]')
          self.__actionDictionary__['residuum'    ]            = self.__residuum__
          self.__actionImplemented__['residuum'   ]            = True
        else:self.__actionImplemented__['residuum']            = False
        if method == '__gradient__':
          exec('self.__gradient__                              ='+moduleName+'.__dict__["__gradient__"]')
          self.__actionDictionary__['gradient'    ]            = self.__gradient__
          self.__actionImplemented__['gradient'   ]            = True
        else:self.__actionImplemented__['gradient']            = False
    else: raise IOError('No file name for the external function has been provided for external function '+self.name+' of type '+self.type)
    for child in xmlNode:
      if child.tag=='variable':
#        exec('self.'+child.text+' = self.inVarValues['+'child.text'+']')
        exec('self.'+child.text+' = None')
        if 'type' in child.attrib.keys(): self.__varType__[child.text] = child.attrib['type']
        else                            : raise IOError('the type for the variable '+child.text+' is missed')
    if len(self.__varType__.keys())==0: raise IOError('not variable found in the definition of the function '+self.name)
        
  def addInitParams(self,tempDict):
    '''
    This function is called from the base class to print some of the information inside the class.
    Whatever is permanent in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary. No information about values that change during the simulation are allowed
    @ In/Out tempDict: {'attribute name':value}
    '''
    tempDict['Module file name'                    ] = self.functionFile
    tempDict['The residuum is provided'            ] = self.__actionImplemented__['residuum']
    tempDict['The sign of the residuum is provided'] = self.__actionImplemented__['residualSign']
    tempDict['The gradient is provided'            ] = self.__actionImplemented__['gradient']
    tempDict['The support bonding is provided'     ] = self.__actionImplemented__['supportBoundingTest']
    for key in self.__varType__.keys():
      tempDict['Variable:type'                     ] = key+':'+self.__varType__[key]

  def addCurrentSetting(self,tempDict):
    '''
    This function is called from the base class to print some of the information inside the class.
    Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary
    Function adds the current settings in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    for key in self.__varType__.keys():
      exec("tempDict['variable '+key+' has value'] = self."+key)
      exec("tempDict['variable '+key+' is of type'] = self.__varType__[key]")


  def __importValues__(self,myInput):
    '''this makes available the variable values sent in as self.key'''
    if type(myInput)==dict         :self.__inputFromWhat__['dict'](myInput)
    elif 'Data' in [x.__name__ for x in myInput.__class__.__bases__]:self.__inputFromWhat__['Data'](myInput)
    else: raise Exception('Unknown type of input provided to the function '+str(self.name))

  def __inputFromData__(self,inputData):
    '''
    This is meant to be used to collect the input from a Data. A conversion to the declared type of data is attempted by inputData.extractValue'''
    for key, myType in self.__varType__.items():
      exec('self.'+key+'=inputData.extractValue(myType,key)')

  def __inputFromDict__(self,myInputDict):
    '''
    This is meant to be used to collect the input directly from a sampler generated input or simply from a generic dictionary
    In case the input come from a sampler the expected structure is myInputDict['SampledVars'][variable name] = value
    In case it is a generic dictionary the expected structure is myInputDict[variable name] = value
    '''
    if 'SampledVars' in myInputDict.keys(): inDict = myInputDict['SampledVars']
    else                                  : inDict = myInputDict
    for name, myType in self.__varType__.items():
      if name in inDict.keys():
        if myType.split('.')[-1] == type(inDict[name]).__name__: exec("self."+name+"=inDict[name]")
        else: raise Exception('Not proper type for the variable '+name+' in external function '+self.name)
      else: raise Exception('The input variable '+name+' in external function seems not to be passed in')

  def evaluate(self,what,myInput):
    '''return the result of the type of action described by 'what' '''
    self.__importValues__(myInput)
    toBeReturned=self.__actionDictionary__[what](self)
    print('toBeReturned'+str(toBeReturned))
    return toBeReturned
  
    
    
'''
 Interface Dictionary (factory) (private)
'''

__base = 'function'
__interFaceDict = {}
__interFaceDict['External'      ] = Function
__knownTypes                      = __interFaceDict.keys()


def knonwnTypes():
  return __knownTypes

def returnInstance(Type,debug=False):
  '''This function return an instance of the request model type'''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
  
    
    
    
    
    
    
    
    
    
    
    

    
  