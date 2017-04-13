# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on Jan 20, 2015

@author: senrs
based on alfoa design

"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#External Modules------------------------------------------------------------------------------------
import abc
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
from utils import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------


class Assembler(MessageHandler.MessageUser):
  """
    Assembler class is used as base class for all the objects that need, for initialization purposes,
    to get pointers (links) of other objects at the Simulation stage (Simulation.run() method)
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    self.type               = self.__class__.__name__  # type
    self.name               = self.__class__.__name__  # name
    self.assemblerObjects   = {}                       # {MainClassName(e.g.Distributions):[class(e.g.Models),type(e.g.ROM),objectName]}
    # list. first entry boolean flag. True if the XML parser must look for objects;
    # second entry tuple.first entry list of object can be retrieved, second entry multiplicity (-1,-2,-n means optional (max 1 object,2 object, no number limit))
    self.requiredAssObject = [False,([],[])]
    self.assemblerDict      = {}                       # {'class':[['class','type','name',instance]]}}

  def whatDoINeed(self):
    """
      This method is used mainly by the Simulation class at the Step construction stage.
      It is used for inquiring the class, which is implementing the method, about the kind of objects the class needs to
      be initialize.
      @ In, None
      @ Out, needDict, dict, dictionary of objects needed (class:tuple(object type{if None, Simulation does not check the type}, object name))
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
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    if '_localGenerateAssembler' in dir(self): self._localGenerateAssembler(initDict)
    for key, value in self.assemblerObjects.items():
      self.assemblerDict[key] =  []
      for interface in value: self.assemblerDict[key].append([interface[0],interface[1],interface[2],initDict[interface[0]][interface[2]]])

  def _readAssemblerObjects(self,subXmlNode, found, testObjects):
    """
      This method is used to look for the assemble objects in an subNodes of an xmlNode
      @ In, subXmlNode, ET, the XML node that needs to be inquired
      @ In, found, dict, a dictionary that check if all the tokens (requested) are found
      @ In, testObjects, dict, a dictionary that contains the number of time a token (requested) has been found
      @ Out, returnObject, tuple, tuple(found, testObjects) containig in [0], found       ->  a dictionary that check if all the tokens (requested) are found ;
                                                                         [1], testObjects ->  a dictionary that contains the number of time a token (requested) has been found
    """
    for subNode in subXmlNode:
      for token in self.requiredAssObject[1][0]:
        if subNode.tag == token:
          found[token] = True
          if 'class' not in subNode.attrib.keys(): self.raiseAnError(IOError,'In '+self.type+' Object ' + self.name+ ', block ' + subNode.tag + ' does not have the attribute class!!')
          if  subNode.tag not in self.assemblerObjects.keys(): self.assemblerObjects[subNode.tag.strip()] = []
          self.assemblerObjects[subNode.tag.strip()].append([subNode.attrib['class'],subNode.attrib['type'],subNode.text.strip()])
          testObjects[token] += 1
    returnObject = found, testObjects
    return returnObject

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some variables based on the inputs got. This method is used to automatically generate the Assembler 'request'
      based on the input of the daughter class.
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this class
      @ Out, None
    """
    self.type = xmlNode.tag
    if 'name' in xmlNode.attrib: self.name = xmlNode.attrib['name']
    self.printTag = self.type
    if 'verbosity' in xmlNode.attrib.keys(): self.verbosity = xmlNode.attrib['verbosity']
    if self.requiredAssObject[0]:
      testObjects = {}
      for token in self.requiredAssObject[1][0]:
        testObjects[token] = 0
      found = dict.fromkeys(testObjects.keys(),False)
      found, testObjects = self._readAssemblerObjects(xmlNode, found, testObjects)
      for subNode in xmlNode: found, testObjects = self._readAssemblerObjects(subNode, found, testObjects)
      for token in self.requiredAssObject[1][0]:
        if not found[token] and not str(self.requiredAssObject[1][1][self.requiredAssObject[1][0].index(token)]).strip().startswith('-'): self.raiseAnError(IOError,'the required object ' +token+ ' is missed in the definition of the '+self.type+' Object! Required objects number are :'+str(self.requiredAssObject[1][1][self.requiredAssObject[1][0].index(token)]))
      # test the objects found
      else:
        for cnt,toObjectName in enumerate(self.requiredAssObject[1][0]):
          numerosity = str(self.requiredAssObject[1][1][cnt])
          if numerosity.strip().startswith('-'):
          # optional
            if toObjectName in testObjects.keys():
              if testObjects[toObjectName] is not 0:
                numerosity = numerosity.replace('-', '').replace('n',str(testObjects[toObjectName]))
                if testObjects[toObjectName] != int(numerosity): self.raiseAnError(IOError,'Only '+numerosity+' '+toObjectName+' object/s is/are optionally required. Block '+self.name + ' got '+str(testObjects[toObjectName]) + '!')
          else:
            # required
            if toObjectName not in testObjects.keys(): self.raiseAnError(IOError,'Required object/s "'+toObjectName+'" not found. Block '+self.name + '!')
            else:
              numerosity = numerosity.replace('n',str(testObjects[toObjectName]))
              if testObjects[toObjectName] != int(numerosity): self.raiseAnError(IOError,'Only '+numerosity+' '+toObjectName+' object/s is/are required. Block '+self.name + ' got '+str(testObjects[toObjectName]) + '!')
    if '_localReadMoreXML' in dir(self): self._localReadMoreXML(xmlNode)

  def addAssemblerObject(self,name,flag, newXmlFlg = None):
    """
      Method to add required assembler objects to the requiredAssObject dictionary.
      @ In, name, string, the node name to search for (e.g. Function, Model)
      @ In, flag, string, the number of nodes to look for (- means optional, n means any number).
                                          For example, "2" means 2 nodes of type "name" are required!
      @ In, newXmlFlg, boolean, optional, if passed in, the first entry of the tuple self.requiredAssObject is going to updated with the new value
                                          For example, if newXmlFlg == True, the self.requiredAssObject[0] is set to True
      @ Out, None
    """
    if newXmlFlg is not None: self.requiredAssObject[0] = newXmlFlg
    self.requiredAssObject[1][0].append(name)
    self.requiredAssObject[1][1].append(flag)

  def retrieveObjectFromAssemblerDict(self,objectMainClass,objectName):
    """
      Method to retrieve an object from the assembler
      @ In, objectName, str, the object name that needs to be retrieved
      @ In, objectMainClass, str, the object main Class name (e.g. Input, Model, etc.) of the object that needs to be retrieved
      @ Out, assemblerObject, instance, the instance requested (None if not found)
    """
    assemblerObject = None
    if objectMainClass in self.assemblerDict.keys():
      for assemblerObj in self.assemblerDict[objectMainClass]:
        if objectName == assemblerObj[2]:
          assemblerObject = assemblerObj[3]
          break
    return assemblerObject
