"""
Created on Mar 16, 2013
@author: crisr
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#External Modules------------------------------------------------------------------------------------
import abc
import sys
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class BaseType(MessageHandler.MessageUser):
  """
    this is the base class for each general type used by the simulation
  """
  def __init__(self):
    self.name             = ''                  # name of this istance (alias)
    self.type             = type(self).__name__ # specific type within this class
    self.verbosity        = None
    self.globalAttributes = {}                  # this is a dictionary that contains parameters that are set at the level of the base classes defining the types
    self._knownAttribute  = []                  # this is a list of strings representing the allowed attribute in the xml input for the class
    self._knownAttribute += ['name','verbosity']
    self.printTag         = 'BaseType'
    self.messageHandler   = None    # message handling object

  def readXML(self,xmlNode,messageHandler,globalAttributes=None):
    """
    provide a basic reading capability from the xml input file for what is common to all types in the simulation than calls _readMoreXML
    that needs to be overloaded and used as API. Each type supported by the simulation should have: name (xml attribute), type (xml tag),
    verbosity (xml attribute)
    """
    self.setMessageHandler(messageHandler)
    if 'name' in xmlNode.attrib.keys(): self.name = xmlNode.attrib['name']
    else: self.raiseAnError(IOError,'not found name for a '+self.__class__.__name__)
    self.type     = xmlNode.tag
    if self.globalAttributes!= None: self.globalAttributes = globalAttributes
    if 'verbosity' in xmlNode.attrib.keys():
      self.verbosity = xmlNode.attrib['verbosity']
      self.raiseADebug('Set verbosity for '+str(self)+' to '+str(self.verbosity))
    #FIXME temporarily create an error to prevent users from using the 'debug' attribute - remove it by end of June 2015 (Sonat)
    if 'debug' in xmlNode.attrib.keys(): self.raiseAnError(IOError,'"debug" attribute found, but has been deprecated.  Please change it to "verbosity."  Remove this error by end of June 2015.')
    self._readMoreXML(xmlNode)
    self.raiseADebug('------Reading Completed for:')
    self.printMe()

  def _readMoreXML(self,xmlNode):
    """
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some variables based on the inputs got.
    @ In, xmlNode, xml.etree.ElementTree, XML element node that represents the portion of the input that belongs to this class
    @ Out, None
    """
    pass

  def setMessageHandler(self,handler):
    if not isinstance(handler,MessageHandler.MessageHandler):
      e=IOError('Attempted to set the message handler for '+str(self)+' to '+str(handler))
      print('\nERROR! Setting MessageHandler in BaseClass,',e,'\n')
      sys.exit(1)
    self.messageHandler = handler

  def whoAreYou(self):
    """This is a generic interface that will return the type and name of any class that inherits this base class plus all the inherited classes"""
    tempDict          = {}
    tempDict['Class'] = '{0:15}'.format(self.__class__.__name__) +' from '+' '.join([str(base) for base in self.__class__.__bases__])
    tempDict['Type' ] = self.type
    tempDict['Name' ] = self.name
    return tempDict

  def myInitializzationParams(self):
    """
    this is a generic interface that will return the name and value of the initialization parameters of any class that inherits this base class.
    In reality it is just empty and will fill the dictionary calling addInitParams that is the function to be overloaded used as API
    """
    tempDict = {}
    self.addInitParams(tempDict)
    return tempDict

  def addInitParams(self,originalDict):
    """function to be overloaded to inject the name and values of the initial parameters"""
    pass

  def myCurrentSetting(self):
    """
    this is a generic interface that will return the name and value of the parameters that change during the simulation of any class that inherits this base class.
    In reality it is just empty and will fill the dictionary calling addCurrentSetting that is the function to be overloaded used as API
    """
    tempDict = {}
    self.addCurrentSetting(tempDict)
    return tempDict

  def addCurrentSetting(self,originalDict):
    """function to be overloaded to inject the name and values of the parameters that might change during the simulation"""
    pass

  def printMe(self):
    """
    This is a generic interface that will print all the info for
    the instance of an object that inherit this class
    """
    tempDict = self.whoAreYou()
    for key in tempDict.keys(): self.raiseADebug('{0:15}: {1}'.format(key,str(tempDict[key])))
    tempDict = self.myInitializzationParams()
    self.raiseADebug('Initialization Parameters:')
    for key in tempDict.keys(): self.raiseADebug('{0:15}: {1}'.format(key,str(tempDict[key])))
    tempDict = self.myCurrentSetting()
    self.raiseADebug('Current Setting:')
    for key in tempDict.keys(): self.raiseADebug('{0:15}: {1}'.format(key,str(tempDict[key])))
