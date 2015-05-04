'''
Created on Mar 16, 2013
@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#External Modules------------------------------------------------------------------------------------
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class BaseType(MessageHandler.MessageUser):
  '''this is the base class for each general type used by the simulation'''
  def __init__(self):
    self.name             = ''      # name of this istance (alias)
    self.type             = ''      # specific type within this class
    self.globalAttributes = {}      #this is a dictionary that contains parameters that are set at the level of the base classes defining the types
    self._knownAttribute  = []      #this is a list of strings representing the allowed attribute in the xml input for the class
    self._knownAttribute += ['name','localVerbosity']
    self.printTag         = 'BaseType'
    self.messageHandler   = None    # message handling object
    self.localVerbosity   = None    # local verbosity value

  def readXML(self,xmlNode,messageHandler,globalAttributes=None):
    '''
    provide a basic reading capability from the xml input file for what is common to all types in the simulation than calls _readMoreXML
    that needs to be overloaded and used as API. Each type supported by the simulation should have: name (xml attribute), type (xml tag)
    '''
    self.setMessageHandler(messageHandler)
    if 'name' in xmlNode.attrib.keys(): self.name = xmlNode.attrib['name']
    else: self.raiseAnError(IOError,'not found name for a '+self.__class__.__name__)
    self.type     = xmlNode.tag
    if self.globalAttributes!= None: self.globalAttributes = globalAttributes
    if 'verbosity' in xmlNode.attrib.keys():
      self.localVerbosity = xmlNode.attrib['verbosity']
    self._readMoreXML(xmlNode)
    self.raiseAMessage('------Reading Completed for:',verbosity='debug')
    self.printMe(self.localVerbosity)

  def _readMoreXML(self,xmlNode):
    '''method to be overloaded to collect the additional input'''
    pass

  def setMessageHandler(self,handler):
    if not isinstance(handler,MessageHandler.MessageHandler):
      raise IOError('Attempted to set the message handler for '+str(self)+' to '+str(handler))
    self.messageHandler = handler

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

  def printMe(self,verbosity=None):
    '''
    This is a generic interface that will print all the info for
    the instance of an object that inherit this class
    '''
    if verbosity==None: verbosity = self.getLocalVerbosity()
    tempDict = self.whoAreYou()
    for key in tempDict.keys(): self.raiseAMessage('{0:15}: {1}'.format(key,str(tempDict[key])),verbosity=verbosity)
    tempDict = self.myInitializzationParams()
    self.raiseAMessage('Initialization Parameters:')
    for key in tempDict.keys(): self.raiseAMessage('{0:15}: {1}'.format(key,str(tempDict[key])),verbosity=verbosity)
    tempDict = self.myCurrentSetting()
    self.raiseAMessage('Current Setting:')
    for key in tempDict.keys(): self.raiseAMessage('{0:15}: {1}'.format(key,str(tempDict[key])),verbosity=verbosity)
