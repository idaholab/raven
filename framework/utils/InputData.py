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
Created on 2016-Jan-26

@author: cogljj

This a library for defining the data used and for reading it in.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import xml.etree.ElementTree as ET
from utils import utils,mathUtils

class InputType(object):
  """
    InputType is a class used to define input types, such as string or integer.
  """
  name = "unknown"
  xmlType = "???"
  needGenerating = False

  @classmethod
  def createClass(cls, name, xmlType, needGenerating = False):
    """
      Creates a new class for use as an input type.
      @ In, name, string, is the name of the input type.
      @ In, xmlType, string, is the xml name of the type.
      @ In, needGenerating, bool, optional, is true if the type needs to be generated.
      @ Out, None
    """

    ## Rename the class to something understandable by a developer
    cls.__name__ = str(name+'Spec')
    # register class name to module (necessary for pickling)
    globals()[cls.__name__] = cls

    cls.name = name
    cls.xmlType = xmlType
    cls.needGenerating = needGenerating

  @classmethod
  def getName(cls):
    """
      Returns the name of the class
      @ Out, getName, string, the name of the input
    """
    return cls.name

  @classmethod
  def getXMLType(cls):
    """
      Returns the xml type of the class
      @ Out, getXMLType, string, the xml type of the input
    """
    return cls.xmlType

  @classmethod
  def needsGenerating(cls):
    """
      Returns if this input needs generating.  If True, then needs to be generated when the xsd file is created by calling the generateXML function.
      @ Out, needsGenerating, bool, true if needs to be generated.
    """
    return cls.needGenerating

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to something else.  For example if value is
      actually an integer it would call int(value)
      @ In, value, string, the value to convert
      @ Out, convert, string or something else, the converted value
    """
    return value

#
#
#
#
class StringType(InputType):
  """
    A type for arbitrary string data.
  """
  pass

StringType.createClass("string","xsd:string")

#
#
#
#
class IntegerType(InputType):
  """
    A type for integer data.
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to an integer.
      @ In, value, string, the value to convert
      @ Out, convert, int, the converted value
    """
    return int(value)

IntegerType.createClass("integer","xsd:integer")

#
#
#
#
class FloatType(InputType):
  """
    A type for floating point data.
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to a float.
      @ In, value, string, the value to convert
      @ Out, convert, float, the converted value
    """
    return float(value)

FloatType.createClass("float","xsd:double")

#
#
#
#
class InterpretedListType(InputType):
  """
    A type for lists with unknown (but consistent) type; could be string, float, etc
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to a listi with string, integer, or float type.
      @ In, value, string, the value to convert
      @ Out, convert, list, the converted value
    """
    values = value.split(",")
    base = utils.partialEval(values[0].strip())
    # three possibilities: string, integer, or float
    if mathUtils.isAString(base):
      conv = str
    elif mathUtils.isAnInteger(base):
      conv = int
    else: #float
      conv = float
    return [conv(x.strip()) for x in values]

#Note, XSD's list type is split by spaces, not commas, so using xsd:string
InterpretedListType.createClass("stringtype","xsd:string")


#
#
#
#
class StringListType(InputType):
  """
    A type for string lists "1, abc, 3" -> ["1","abc","3"]
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to a string list.
      @ In, value, string, the value to convert
      @ Out, convert, list, the converted value
    """
    return [x.strip() for x in value.split(",")]

#Note, XSD's list type is split by spaces, not commas, so using xsd:string
StringListType.createClass("stringtype","xsd:string")


#
#
#
#
class FloatListType(InputType):
  """
    A type for float lists "1.1, 2.0, 3.4" -> [1.1, 2.0, 3.4]
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to a float list.
      @ In, value, string, the value to convert
      @ Out, convert, list, the converted value
    """
    return [float(x.strip()) for x in value.split(",")]

#Note, XSD's list type is split by spaces, not commas, so using xsd:string
FloatListType.createClass("stringtype","xsd:string")


#
#
#
#
class IntegerListType(InputType):
  """
    A type for integer lists "1, 2, 3" -> [1,2,3]
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to an integer list.
      @ In, value, string, the value to convert
      @ Out, convert, list, the converted value
    """
    return [int(x.strip()) for x in value.split(",")]

#Note, XSD's list type is split by spaces, not commas, so using xsd:string
IntegerListType.createClass("stringtype","xsd:string")


#
#
#
#
class EnumBaseType(InputType):
  """
    A type that allows a set list of strings
  """
  enumList = []

  @classmethod
  def createClass(cls, name, xmlType, enumList):
    """
      creates a new enumeration type.
      @ In, name, string, the name of the type
      @ In, xmlType, string, the name used for the xml type.
      @ In, enumList, [string], a list of allowable strings.
      @ Out, None
    """

    ## Rename the class to something understandable by a developer
    cls.__name__ = str(name+'Spec')
    # register class name to module (necessary for pickling)
    globals()[cls.__name__] = cls

    cls.name = name
    cls.xmlType = xmlType
    cls.needGenerating = True
    cls.enumList = enumList

  @classmethod
  def generateXML(cls, xsdNode):
    """
      Generates the xml data.
      @ In, xsdNode, xml.etree.ElementTree.Element, the element to add the new xml type to.
      @ Out, None
    """
    simpleType = ET.SubElement(xsdNode, 'xsd:simpleType')
    simpleType.set('name', cls.getXMLType())
    restriction = ET.SubElement(simpleType, 'xsd:restriction')
    restriction.set('base','xsd:string')
    for enum in cls.enumList:
      enumNode = ET.SubElement(restriction, 'xsd:enumeration')
      enumNode.set('value',enum)

#
#
#
#

class BoolType(EnumBaseType):
  """
    A type that allows True or False
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to a bool.
      @ In, value, string, the value to convert
      @ Out, convert, bool, the converted value
    """
    if value in utils.stringsThatMeanTrue():
      return True
    else:
      return False

boolTypeList = utils.stringsThatMeanTrue() + utils.stringsThatMeanFalse()
BoolType.createClass("bool","boolType", boolTypeList + [elm.capitalize() for elm in boolTypeList])

#
#
#
#
class Quantity:
  """
    A class that allows the quantity of a node to be specified.
    If python3.4+ is required, this should be switched to a Python 3.4 Enum.
  """
  zero_to_one = (0,1)
  zero_to_infinity = (0,2)
  one = (1,1)
  one_to_infinity = (1,2)

#
#
#
#
class ParameterInput(object):
  """
    This class is for a node for inputing parameters
  """
  name = "unknown"
  subs = set()
  subOrder = None
  parameters = {}
  contentType = None
  strictMode = True #If true, only allow parameters and subnodes that are listed

  def __init__(self):
    """
      create new instance.
      @ Out, None
    """
    self.parameterValues = {}
    self.subparts = []
    self.value = ""

  @classmethod
  def createClass(cls, name, ordered=False, contentType=None, baseNode=None, strictMode=True):
    """
      Initializes a new class.
      @ In, name, string, The name of the node.
      @ In, ordered, bool, optional, If True, then the subnodes are checked to make sure they are in the same order.
      @ In, contentType, InputType, optional, If not None, set contentType.
      @ In, baseNode, ParameterInput, optional, If not None, copy parameters and subnodes, subOrder, and contentType from baseNode.
      @ In, strictNode, bool, option, If True, then only allow paramters and subnodes that are specifically mentioned.
      @ Out, None
    """

    ## Rename the class to something understandable by a developer
    cls.__name__ = str(name+'Spec')
    # register class name to module (necessary for pickling)
    globals()[cls.__name__] = cls

    cls.name = name
    cls.strictMode = strictMode
    if baseNode is not None:
      #Make new copies of data from baseNode
      cls.parameters = dict(baseNode.parameters)
      cls.subs = set(baseNode.subs)
      if ordered:
        cls.subOrder = list(baseNode.subOrder)
      else:
        cls.subOrder = None
      if contentType is None:
        cls.contentType = baseNode.contentType
    else:
      cls.parameters = {}
      cls.subs = set()
      if ordered:
        cls.subOrder = []
      else:
        cls.subOrder = None
      cls.contentType = contentType

  @classmethod
  def setStrictMode(cls, strictMode):
    """
      Sets how strict the parsing is.  Stricter will throw more IOErrors.
      @ In, strictNode, bool, option, If True, then only allow paramters and subnodes that are specifically mentioned.
      @ Out, None
    """
    cls.strictMode = strictMode

  @classmethod
  def getName(cls):
    """
      Returns the name of this class
      @ Out, getName, string, the name of this class
    """
    return cls.name

  @classmethod
  def addParam(cls, name, param_type=StringType, required=False):
    """
      Adds a direct parameter to this class.  In XML this is an attribute.
      @ In, name, string, the name of the parameter
      @ In, param_type, subclass of InputType, optional, that specifies the type of the attribute.
      @ In, required, bool, optional, if True, this parameter is required.
      @ Out, None
    """
    cls.parameters[name] = {"type":param_type, "required":required}

  @classmethod
  def removeParam(cls, name, param_type=StringType, required=False):
    """
      Adds a direct parameter to this class.  In XML this is an attribute.
      @ In, name, string, the name of the parameter
      @ In, param_type, subclass of InputType, optional, that specifies the type of the attribute.
      @ In, required, bool, optional, if True, this parameter is required.
      @ Out, None
    """
    del cls.parameters[name]

  @classmethod
  def addSub(cls, sub, quantity=Quantity.zero_to_infinity):
    """
      Adds a subnode to this class.
      @ In, sub, subclass of ParameterInput, the subnode to allow
      @ In, quantity, value in Quantity, the number of this subnode to allow.
      @ Out, None
    """
    cls.subs.add(sub)
    if cls.subOrder is not None:
      cls.subOrder.append((sub,quantity))
    elif quantity != Quantity.zero_to_infinity:
      print("ERROR only zero to infinity is supported if Order==False ",
            sub.getName()," in ",cls.getName())

  @classmethod
  def removeSub(cls, sub, quantity=Quantity.zero_to_infinity):
    """
      Removes a subnode from this class.
      @ In, sub, subclass of ParameterInput, the subnode to allow
      @ In, quantity, value in Quantity, the number of this subnode to allow.
      @ Out, None
    """
    for have in cls.subs:
      if have.name == sub:
        toRemove = have
        break
    if cls.subOrder is not None:
      for entry in cls.subOrder:
        if entry[0] == toRemove:
          toRemoveOrd = entry
          break
      cls.subOrder.remove(toRemoveOrd)
    cls.subs.remove(toRemove)

  @classmethod
  def popSub(cls, subname):
    """
      Removes a subnode from this class, and returns it.
      @ In, subname, string, the name of the subnode to remove
      @ Out, poppedSub, subclass of ParameterInput, the removed subnode, or None if not found.
    """
    poppedSub = None
    for sub in cls.subs:
      if sub.getName() == subname:
        poppedSub = sub
    if poppedSub is not None:
      cls.subs.remove(poppedSub)
    else:
      return None
    if cls.subOrder is not None:
      toRemoveList = []
      for (sub,quantity) in cls.subOrder:
        if poppedSub == sub:
          toRemoveList.append((sub,quantity))
      for toRemove in toRemoveList:
        cls.subOrder.remove(toRemove)
    return poppedSub

  @classmethod
  def setContentType(cls, contentType):
    """
      Sets the content type for the node.
      @ In, contentType, subclass of InputType, the content type to use
      @ Out, None
    """
    cls.contentType = contentType

  def parseNode(self,node, errorList = None):
    """
      Parses the xml node and puts the results in self.parameterValues and
      self.subparts and self.value
      @ In, node, xml.etree.ElementTree.Element, The node to parse.
      @ In, errorList, list, if not None, put errors in errorList instead of throwing IOError.
      @ Out, None
    """
    def handleError(s):
      """
        Handles the error, either by throwing IOError or adding to the errorlist
        @ In, s, string, string describing error.
      """
      if errorList == None:
        raise IOError(s)
      else:
        errorList.append(s)

    if node.tag != self.name:
      #should this be an error or a warning? Or even that?
      #handleError('XML node "{}" != param spec name "{}"'.format(node.tag,self.name))
      print('Using param spec "{}" to read XML node "{}.'.format(self.name,node.tag))
    if self.contentType:
      self.value = self.contentType.convert(node.text)
    else:
      self.value = node.text
    for parameter in self.parameters:
      if parameter in node.attrib:
        param_type = self.parameters[parameter]["type"]
        self.parameterValues[parameter] = param_type.convert(node.attrib[parameter])
      elif self.parameters[parameter]["required"]:
        handleError("Required parameter " + parameter + " not in " + node.tag)
    if self.strictMode:
      for parameter in node.attrib:
        if not parameter in self.parameters:
          handleError(parameter + " not in attributes and strict mode on in "+node.tag)
    if self.subOrder is not None:
      subs = [sub[0] for sub in self.subOrder]
    else:
      subs = self.subs
    subNames = set()
    for sub in subs:
      subName = sub.getName()
      subNames.add(subName)
      for subNode in node.findall(subName):
        subInstance = sub()
        subInstance.parseNode(subNode, errorList)
        self.subparts.append(subInstance)
    if self.strictMode:
      for child in node:
        if child.tag not in subNames:
          handleError('Child "{}" not allowed as sub-element of "{}"'.format(child.tag,node.tag))

  def findFirst(self, name):
    """
      Finds the first subpart with name.  Note that if this node is not ordered,
      and there are multiple subparts with the name, it is undefined which node
      will be found first.
      @ In, name, string, the name of the node to search for
      @ Out, findFirst, ParameterInput, the first node found, or None if none found.
    """
    for sub in self.subparts:
      if sub.getName() == name:
        return sub
    return None

  @classmethod
  def generateXSD(cls, xsdNode, definedDict):
    """
      Generates the xsd information for this node.
      @ In, xsdNode, xml.etree.ElementTree.Element, the place to put the information.
      @ In and Out, definedDict, dict, A dictionary that stores which names have been defined in the XSD already.
      @ Out, None
    """
    #generate complexType
    complexType = ET.SubElement(xsdNode, 'xsd:complexType')
    complexType.set('name', cls.getName()+'_type')
    if len(cls.subs) > 0:
      #generate choice node
      if cls.subOrder is not None:
        listNode = ET.SubElement(complexType, 'xsd:sequence')
        subList = cls.subOrder
      else:
        listNode = ET.SubElement(complexType, 'xsd:choice')
        listNode.set('maxOccurs', 'unbounded')
        subList = [(sub,Quantity.zero_to_infinity) for sub in cls.subs]
      #generate subnodes
      #print(subList)
      for sub,quantity in subList:
        subNode = ET.SubElement(listNode, 'xsd:element')
        subNode.set('name', sub.getName())
        subNode.set('type', sub.getName()+'_type')
        if cls.subOrder is not None:
          if quantity == Quantity.zero_to_one:
            occurs = ('0','1')
          elif quantity == Quantity.zero_to_infinity:
            occurs = ('0','unbounded')
          elif quantity == Quantity.one:
            occurs = ('1','1')
          elif quantity == Quantity.one_to_infinity:
            occurs = ('1','unbounded')
          else:
            print("ERROR unexpected quantity ",quantity)
          subNode.set('minOccurs', occurs[0])
          subNode.set('maxOccurs', occurs[1])
        else:
          subNode.set('minOccurs', '0')
        if sub.getName() not in definedDict:
          definedDict[sub.getName()] = sub
          sub.generateXSD(xsdNode, definedDict)
        elif definedDict[sub.getName()] != sub:
          print("ERROR: multiple definitions ",sub.getName())
    else:
      if cls.contentType is not None:
        contentNode = ET.SubElement(complexType, 'xsd:simpleContent')
        extensionNode = ET.SubElement(contentNode, 'xsd:extension')
        dataType = cls.contentType
        extensionNode.set('base', dataType.getXMLType())
        if dataType.needsGenerating() and dataType.getName() not in definedDict:
          dataType.generateXML(xsdNode)
    #generate attributes
    for parameter in cls.parameters:
      attributeNode = ET.SubElement(complexType, 'xsd:attribute')
      parameterData = cls.parameters[parameter]
      attributeNode.set('name', parameter)
      dataType = parameterData["type"]
      if dataType.needsGenerating() and dataType.getName() not in definedDict:
        dataType.generateXML(xsdNode)
      attributeNode.set('type', dataType.getXMLType())
      if parameterData["required"]:
        attributeNode.set('use','required')

## TODO: We should normalize the names of the following two functions, since they
## do the same thing just on different input types.
## e.g., parameterInputFactory and EnumFactory
## makeClassSpecification and makeEnumSpecification

def parameterInputFactory(*paramList, **paramDict):
  """
    Creates a new ParameterInput class with the same parameters as ParameterInput.createClass
    @ In, same parameters as ParameterInput.createClass
    @ Out, newClass, ParameterInput, the newly created class.
  """
  class newClass(ParameterInput):
    """
      The new class to be created by the factory
    """
  newClass.createClass(*paramList, **paramDict)
  return newClass

def makeEnumType(name, xmlName, enumList):
  """
    Creates a new enum type that can be used as a content type.
    @ In, name, string, Name of the type
    @ In, xmlName, string, Name of the type used in the XSD file.
    @ In, enumList, list of strings, the possible values of the enumeration.
    @ Out, newEnum, InputData.EnumBaseType, the new enumeration type.
  """
  class newEnum(EnumBaseType):
    """
      the new enum to be created by the factory
    """

  newEnum.createClass(name, xmlName, enumList)
  return newEnum

def createXSD(outerElement):
  """
    Creates an XSD element.
    @ In, outerElement, xml.etree.ElementTree.Element, the outer most element in the xml.
    @ Out, outside, xml.etree.ElementTree.Element, a element that can be dumped to create an xsd file.
  """
  outside = ET.Element('xsd:schema')
  outside.set('xmlns:xsd', 'http://www.w3.org/2001/XMLSchema')
  ET.SubElement(outside, 'xsd:element', {'name':outerElement.getName(),
                                         'type':outerElement.getName()+'_type'})
  outerElement.generateXSD(outside, {})
  return outside


class RavenBase(ParameterInput):
  """
    This can be used as a base class for things that inherit from BaseType
  """
RavenBase.createClass("RavenBase", baseNode=None)
RavenBase.addParam("verbosity") #XXX should be enumeration
