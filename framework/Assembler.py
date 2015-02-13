'''
Created on Jan 20, 2015

@author: senrs
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#External Modules------------------------------------------------------------------------------------
import abc
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
from utils    import returnPrintTag, metaclass_insert, stringsThatMeanTrue, stringsThatMeanFalse
#Internal Modules End--------------------------------------------------------------------------------


class Assembler(object):
  """
  Assembler class is used as base class for all the objects that need, for initialization purposes,
  to get pointers (links) of other objects at the Simulation stage (Simulation.run() method)
  """
  def __init__(self):
    self.type               = self.__class__.__name__  # type
    self.name               = self.__class__.__name__  # name
    self.assemblerObjects   = {}                       # {MainClassName(e.g.Distributions):[class(e.g.Models),type(e.g.ROM),objectName]}
    self.requiredAssObjects = (False,([],[]))          # tuple. first entry boolean flag. True if the XML parser must look for objects;
                                                       # second entry tuple.first entry list of object can be retrieved, second entry multiplicity (-1,-2,-n means optional (max 1 object,2 object, no number limit))
    self.debug              = False


  def whatDoINeed(self):
    """
    This method is used mainly by the Simulation class at the Step construction stage.
    It is used for inquiring the class, which is implementing the method, about the kind of objects the class needs to
    be initialize.
    @ In , None, None
    @ Out, needDict, dictionary of objects needed (class:tuple(object type{if None, Simulation does not check the type}, object name))
    """
    if '_localWhatDoINeed' in dir(self):
       needDict = self._localWhatDoINeed()
    else:
      needDict = {}
    for val in self.assemblerObjects.values():
      for value  in val:
        if value[0] not in needDict.keys(): needDict[value[0]] = []
        needDict[value[0]].append((value[1],value[2]))
    return needDict

  def generateAssembler(self,initDict):
    """
    This method is used mainly by the Simulation class at the Step construction stage.
    It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
    It is an abstract method -> It must be implemented in the derived class!
    @ In , initDict, dictionary ({'mainClassName(e.g., DataBases):{specializedObjectName(e.g.,DataBaseForSystemCodeNamedWolf):ObjectInstance}'})
    @ Out, None, None
    """
    if '_localGenerateAssembler' in dir(self):
      self._localGenerateAssembler(initDict)
    for key, value in self.assemblerObjects.items():
      self.assemblerDict[key] =  []
      for interface in value:
         self.assemblerDict[key].append([interface[0],interface[1],interface[2],initDict[interface[0]][interface[2]]])

  def _readMoreXML(self,xmlNode):
    self.type = xmlNode.tag
    if 'name' in xmlNode.attrib: self.name = xmlNode.attrib['name']
    self.printTag = self.type.ljust(25)
    if 'debug' in xmlNode.attrib.keys():self.debug = bool(xmlNode.attrib['debug'])
    if self.requiredAssObject[0]:
        testObjects = {}
        for token in self.requiredAssObject[1][0]:
            testObjects[token] = 0
        found = False
        for subNode in xmlNode:
            for token in self.requiredAssObject[1][0]:
                if subNode.tag in token:
                    found = True
                    if 'class' not in subNode.attrib.keys(): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> In '+self.type+' PostProcessor ' + self.name+ ', block ' + subNode.tag + ' does not have the attribute class!!')
                    if  subNode.tag not in self.assemblerObjects.keys(): self.assemblerObjects[subNode.tag] = []
                    self.assemblerObjects[subNode.tag].append([subNode.attrib['class'],subNode.attrib['type'],subNode.text])
                    testObjects[token] += 1
        if not found:
            for tofto in self.requiredAssObject[1][0]:
                if not str(self.requiredAssObject[1][1][0]).strip().startswith('-'):
                    raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> the required object ' +tofto+ ' is missed in the definition of the '+self.type+' PostProcessor!')
        # test the objects found
        else:
            for cnt,tofto in enumerate(self.requiredAssObject[1][0]):
                numerosity = str(self.requiredAssObject[1][1][cnt])
                if numerosity.strip().startswith('-'):
                # optional
                    if tofto in testObjects.keys():
                        numerosity = numerosity.replace('-', '').replace('n',str(testObjects[tofto]))
                        if testObjects[tofto] != int(numerosity): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Only '+numerosity+' '+tofto+' object/s is/are optionally required. PostProcessor '+self.name + ' got '+str(testObjects[tofto]) + '!')
                else:
                # required
                    if tofto not in testObjects.keys(): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Required object/s "'+tofto+'" not found. PostProcessor '+self.name + '!')
                    else:
                        numerosity = numerosity.replace('n',str(testObjects[tofto]))
                        if testObjects[tofto] != int(numerosity): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Only '+numerosity+' '+tofto+' object/s is/are required. PostProcessor '+self.name + ' got '+str(testObjects[tofto]) + '!')
    if '_localReadMoreXML' in dir(self): self._localReadMoreXML(xmlNode)
