'''
Created on Mar 16, 2013
@author: crisr
'''

class BaseType:
  '''this is the base class for each general type used by the simulation'''
  def __init__(self):
    self.name    = ''      # name of this istance (alias)
    self.type    = ''      # specific type within this class
  def readXML(self,xmlNode):
    '''provide a basic reading capability from the xml input file
       for what is common to all types in the simulation than calls readMoreXML
       that needs to be overloaded and used as API
       Each type supported by the simulation should have:
       name (xml attribute), type (xml tag)'''
    try:    self.name = xmlNode.attrib['name']
    except: raise IOError('not found name for a '+self.__class__.__name__)
    try: self.type = xmlNode.tag
    except: raise IOError('not found type for the '+self.__class__.__name__+' named '+self.name)
    self.readMoreXML(xmlNode)
  def readMoreXML(self,xmlNode):
    '''method to be overloaded to collect the additional input'''
    pass
  def whoAreYou(self):
    '''this is a generic interface that will return the type
       and name of any class that inherits this base class 
       plus all the inherited classes'''
    tempDict = {}
    tempDict['class type'] = self.__class__.__name__ +' from'
    for base in self.__class__.__bases__:
      tempDict['class type'] += ' ' + base.__name__
    tempDict['type']       = self.type
    tempDict['name']       = self.name
    return tempDict
  def myInitializzationParams(self):
    '''this is a generic interface that will return the name and
       value of the initialization parameters of any class
       that inherits this base class.
       In reality it is just empty and will fill the dictionary calling addInitParams
        that is the function to be overloaded used as API'''
    tempDict = {}
    self.addInitParams(tempDict)
    return tempDict
  def addInitParams(self,originalDict):
    '''function to be overloaded to inject the name and values of the initial parameters'''
    pass
  def myCurrentSetting(self):
    '''this is a generic interface that will return the name and
       value of the parameters that change during the simulation 
       of any class that inherits this base class.
       In reality it is just empty and will fill the dictionary calling addCurrentSetting
        that is the function to be overloaded used as API'''
    tempDict = {}
    self.addCurrentSetting(tempDict)
    return tempDict
  def addCurrentSetting(self,originalDict):
    '''function to be overloaded to inject the name and values of the parameters that might change during the simulation'''
    pass
  def printMe(self):
    '''this is a generic interface that will print all the info for
       the instance of an object that inherit this class'''
    tempDict = self.whoAreYou()
    print('Class:'+str(tempDict['class type']))
    print('Type:'+tempDict['type'])
    print('name:'+tempDict['name'])
    tempDict = self.myInitializzationParams()
    print('Initialization Parameters:')
    for key in tempDict.keys():
      print(key+': '+str(tempDict[key]))
    tempDict = self.myCurrentSetting()
    print('Current Setting:')
    for key in tempDict.keys():
      print(key+': '+str(tempDict[key]))
    print('\n')
    
    
  
  