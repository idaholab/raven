'''
Created on Mar 16, 2013
@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
from utils    import returnPrintTag

class BaseType(object):
  '''this is the base class for each general type used by the simulation'''
  
  def __init__(self):
    self.name             = ''      # name of this istance (alias)
    self.type             = ''      # specific type within this class
    self.debug            = False   #set up the debug status of the code
    self.globalAttributes = None    #this is a dictionary that contains parameters that are set at the level of the base classes defining the types    
    self._knownAttribute  = []      #this is a list of strings representing the allowed attribute in the xml input for the class
    self._knownAttribute += ['name','debug']
    self.printTag         = returnPrintTag('BaseType')

  def readXML(self,xmlNode,debug=False,globalAttributes=None):
    '''
    provide a basic reading capability from the xml input file for what is common to all types in the simulation than calls _readMoreXML
    that needs to be overloaded and used as API. Each type supported by the simulation should have: name (xml attribute), type (xml tag)
    '''
    if 'name' in xmlNode.attrib.keys(): self.name = xmlNode.attrib['name']
    else: raise IOError(self.printTag+':not found name for a '+self.__class__.__name__)
    self.type     = xmlNode.tag
    if self.globalAttributes!= None: self.globalAttributes = globalAttributes
    if 'debug' in xmlNode.attrib:
      if   xmlNode.attrib['debug'].lower() in ['true','t','yes'] : self.debug = True
      elif xmlNode.attrib['debug'].lower() in ['false','f','no']: self.debug = False
      else                                   : raise IOError('For the attribute debug '+ xmlNode.attrib['debug']+' is not a recognized keyword')
    else                                     : self.debug = debug
    self._readMoreXML(xmlNode)
    if self.debug:
      print(self.printTag+'------Reading Completed for:')
      self.printMe()

  def _readMoreXML(self,xmlNode):
    '''method to be overloaded to collect the additional input'''
    pass

  def whoAreYou(self):
    '''This is a generic interface that will return the type and name of any class that inherits this base class plus all the inherited classes'''
    tempDict          = {}
    tempDict['Class'] = '{0:15}'.format(self.__class__.__name__) +' from '+' '.join([str(base) for base in self.__class__.__bases__])
    tempDict['Type' ] = self.type
    tempDict['Name' ] = self.name
    return tempDict

  def myInitializzationParams(self):
    '''
    this is a generic interface that will return the name and value of the initialization parameters of any class that inherits this base class.
    In reality it is just empty and will fill the dictionary calling addInitParams that is the function to be overloaded used as API
    '''
    tempDict = {}
    self.addInitParams(tempDict)
    return tempDict

  def addInitParams(self,originalDict):
    '''function to be overloaded to inject the name and values of the initial parameters'''
    pass

  def myCurrentSetting(self):
    '''
    this is a generic interface that will return the name and value of the parameters that change during the simulation of any class that inherits this base class.
    In reality it is just empty and will fill the dictionary calling addCurrentSetting that is the function to be overloaded used as API
    '''
    tempDict = {}
    self.addCurrentSetting(tempDict)
    return tempDict

  def addCurrentSetting(self,originalDict):
    '''function to be overloaded to inject the name and values of the parameters that might change during the simulation'''
    pass

  def printMe(self):
    '''
    This is a generic interface that will print all the info for
    the instance of an object that inherit this class
    '''
    tempDict = self.whoAreYou()
    for key in tempDict.keys(): print('{0:15}: {1}'.format(key,str(tempDict[key])))
    tempDict = self.myInitializzationParams()
    print(self.printTag+'Initialization Parameters:')
    for key in tempDict.keys(): print('{0:15}: {1}'.format(key,str(tempDict[key])))
    tempDict = self.myCurrentSetting()
    print(self.printTag+'Current Setting:')
    for key in tempDict.keys(): print('{0:15}: {1}'.format(key,str(tempDict[key])))
    print('\n')
    
    
  
  
