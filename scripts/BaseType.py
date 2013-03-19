'''
Created on Mar 16, 2013

@author: crisr
'''

class BaseType:
  '''
  this is the base class for each type of the simulation
  '''
  def __init__(self):
    self.name    = ''      # name of this istance (alias)
    self.type    = ''      # specific type within this class
  def readXML(self,xmlNode):
    try:    self.name = xmlNode.attrib['name']
    except: raise IOError('not found name for a '+self.__class__.__name__)
    try: self.type = xmlNode.tag
    except: raise IOError('not found type for the '+self.__class__.__name__+' named '+self.name)
    self.readMoreXML(xmlNode)
  def readMoreXML(self,xmlNode):
    pass
  def whoAreYou(self):
    tempDict = {}
    tempDict['class type'] = self.__class__.__name__+' from '+self.__class__.__bases__[0].__name__
    tempDict['type']       = self.type
    tempDict['name']       = self.name
    return tempDict
  def myInitializzationParams(self):
    tempDict = {}
    self.addInitParams(tempDict)
    return tempDict
  def addInitParams(self,originalDict):
    pass
  def myCurrentSetting(self):
    tempDict = {}
    self.addCurrentSetting(tempDict)
    return tempDict
  def addCurrentSetting(self,originalDict):
    pass
  def printMe(self):
    tempDict = self.whoAreYou()
    print('\nClass:'+str(tempDict['class type']))
    print('Type:'+tempDict['type'])
    print('name:'+tempDict['name'])
    tempDict = self.myInitializzationParams()
    print('\nInitialization Parameters:')
    for key in tempDict.keys():
      print(key+': '+str(tempDict[key]))
    tempDict = self.myCurrentSetting()
    print('\nCurrent Setting:')
    for key in tempDict.keys():
      print(key+': '+str(tempDict[key]))
    
    
    
  
  