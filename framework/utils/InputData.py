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
import re
from collections import OrderedDict
import xml.etree.ElementTree as ET
from utils import InputTypes
import textwrap

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
  subs = OrderedDict() #set()
  subOrder = None
  parameters = OrderedDict()
  contentType = None
  strictMode = True #If true, only allow parameters and subnodes that are listed
  description = '-- no description yet --'
  printPriority = None

  def __init__(self):
    """
      create new instance.
      @ Out, None
    """
    self.parameterValues = {}
    self.subparts = []
    self.value = ""

  @classmethod
  def createClass(cls, name, ordered=False, contentType=None, baseNode=None,
                  strictMode=True, descr=None, printPriority=None):
    """
      Initializes a new class.
      @ In, name, string, The name of the node.
      @ In, ordered, bool, optional, If True, then the subnodes are checked to make sure they are in the same order.
      @ In, contentType, InputTypes.InputType, optional, If not None, set contentType.
      @ In, baseNode, ParameterInput, optional, If not None, copy parameters and subnodes, subOrder, and contentType from baseNode.
      @ In, strictNode, bool, optional, If True, then only allow paramters and subnodes that are specifically mentioned.
      @ In, printPriority, int, optional, sets the priority for printing this node e.g. in the user
      manual. Lower is higher priority; priority 0 gets printed first. See generateLatex for details.
      @ Out, None
    """

    ## Rename the class to something understandable by a developer
    cls.__name__ = str(name+'Spec')
    # register class name to module (necessary for pickling)
    globals()[cls.__name__] = cls

    cls.name = name
    cls.strictMode = strictMode
    cls.description = descr if descr is not None else cls.description
    if printPriority is None:
      # TODO set printPriority based on required/not required, but we don't have this system yet.
      cls.printPriority = 200
    else:
      cls.printPriority = printPriority
    if baseNode is not None:
      #Make new copies of data from baseNode
      cls.parameters = dict(baseNode.parameters)
      cls.subs = OrderedDict.fromkeys(baseNode.subs)
      if ordered:
        cls.subOrder = list(baseNode.subOrder)
      else:
        cls.subOrder = None
      if contentType is None:
        cls.contentType = baseNode.contentType
    else:
      cls.parameters = {}
      cls.subs = OrderedDict() #set()
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
  def getSub(cls, name):
    """
      Returns the name of this class
      @ In, name, str, name of the sub to acquire
      @ Out, getSub, ParameterInput, class with corresponding sub
    """
    for sub in cls.subs:
      if sub.name == name:
        return sub
    return None

  @classmethod
  def addParam(cls, name, param_type=InputTypes.StringType, required=False, descr=None):
    """
      Adds a direct parameter to this class.  In XML this is an attribute.
      @ In, name, string, the name of the parameter
      @ In, param_type, subclass of InputType, optional, that specifies the type of the attribute.
      @ In, required, bool, optional, if True, this parameter is required.
      @ Out, None
    """
    cls.parameters[name] = {"type":param_type, "required":required,
                            'description':descr if descr is not None else '-- no description yet --'}

  @classmethod
  def removeParam(cls, name, param_type=InputTypes.StringType, required=False):
    """
      Adds a direct parameter to this class.  In XML this is an attribute.
      @ In, name, string, the name of the parameter
      @ In, param_type, subclass of InputTypes.InputType, optional, that specifies the type of the attribute.
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
    cls.subs[sub] = None
    if cls.subOrder is not None:
      cls.subOrder.append((sub, quantity))
    elif quantity != Quantity.zero_to_infinity:
      print("ERROR only zero to infinity is supported if Order==False ",
           sub.getName()," in ",cls.getName())

  @classmethod
  def removeSub(cls, sub):
    """
      Removes a subnode from this class.
      @ In, sub, subclass of ParameterInput, the subnode to allow
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
    cls.subs.pop(toRemove)

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
      cls.subs.pop(poppedSub)
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
      # TODO give the offending XML! Use has no idea where they went wrong.
      if errorList == None:
        raise IOError(s)
      else:
        errorList.append(s)

    # check specs vs tag name
    if node.tag != self.name:
      #should this be an error or a warning? Or even that?
      #handleError('XML node "{}" != param spec name "{}"'.format(node.tag,self.name))
      print('InputData: Using param spec "{}" to read XML node "{}.'.format(self.name,node.tag))

    # check content type
    if self.contentType:
      try:
        self.value = self.contentType.convert(node.text)
      except Exception as e:
        handleError(str(e))
    else:
      self.value = node.text

    # check attributes
    for parameter in self.parameters:
      if parameter in node.attrib:
        param_type = self.parameters[parameter]["type"]
        self.parameterValues[parameter] = param_type.convert(node.attrib[parameter])
      elif self.parameters[parameter]["required"]:
        handleError("Required parameter " + parameter + " not in " + node.tag)
    # if strict, force parameter checking
    if self.strictMode:
      for parameter in node.attrib:
        if not parameter in self.parameters:
          handleError(parameter + " not in attributes and strict mode on in "+node.tag)

    # handle ordering of subnodes
    if self.subOrder is not None:
      subs = OrderedDict.fromkeys([sub[0] for sub in self.subOrder])
    else:
      subs = self.subs
    # read in subnodes
    subNames = set()
    for sub in subs:
      subName = sub.getName()
      subNames.add(subName)
      for subNode in node.findall(subName):
        subInstance = sub()
        subInstance.parseNode(subNode, errorList)
        self.subparts.append(subInstance)
    if self.strictMode:
      nodeNames = set([child.tag for child in node])
      if nodeNames != subNames:
        # there are mismatches
        unknownChilds = list(nodeNames - subNames)
        if unknownChilds:
          handleError('Childs "[{}]" not allowed as sub-elements of "{}"'.format(", ".join(unknownChilds),node.tag))
        #TODO: keep this for the future. We need to implement in the InputData a way to set some nodes to be required
        #missingChilds =  list(subNames - nodeNames)
        #if missingChilds:
        #  handleError('Not found Childs "[{}]" as sub-elements of "{}"'.format(",".join(missingChilds),node.tag))

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

  def findAll(self, name):
    """
      Finds all the subparts with name.
      @ In, name, string, the name of the node to search for
      @ Out, findAll, list, matching nodes (may be empty)
    """
    return list(sub for sub in self.subparts if sub.getName() == name)

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
    if cls.subs:
      #generate choice node
      if cls.subOrder is not None:
        listNode = ET.SubElement(complexType, 'xsd:sequence')
        subList = cls.subOrder
      else:
        listNode = ET.SubElement(complexType, 'xsd:choice')
        listNode.set('maxOccurs', 'unbounded')
        subList = [(sub, Quantity.zero_to_infinity) for sub in cls.subs]
      #generate subnodes
      #print(subList)
      for sub, quantity in subList:
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
          print('DEBUGG defined:')
          import pprint
          pprint.pprint(definedDict)
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

  @classmethod
  def generateLatex(cls, recDepth=0):
    """
      Generates the user manual entry for this input spec.
      @ In, recDepth, int, optional, recursion depth of printing
      @ Out, msg, str, LaTeX string representation of user manual entry
    """
    name = cls.name
    desc = wrapText(cls.description, indent=doDent(recDepth, 1))
    msg = ''
    # if this is a main entity, use subsection instead of itemizing
    if recDepth == 0:
      # triple curly braces preserves one set of curls while replacing "n"
      msg += '\n\n\subsection{{{n}}}\n{d}\n'.format(n=name, d=desc)
    else:
      # since this is a sub-entity, it's part of a list
      msg += '{i}\\item \\xmlNode{{{n}}}:'.format(i=doDent(recDepth), n=name)
      # add the required text type if it exists
      if cls.contentType:
        msg += ' \\xmlDesc{{{t}}}, '.format(t=cls.contentType.generateLatexType())
      # add description
      msg += '\n{d}'.format(d=desc)
    # add parameter definitions, if any, tabbed in by 1
    msg += '\n' + cls.generateParamsLatex(recDepth+1)
    # add subnode definitions in order of printing priority
    if cls.subs:
      msg += '\n{i}The \\xmlNode{{{n}}} node recognizes the following subnodes:'.format(i=doDent(recDepth, 1), n=name)
      msg += '\n{i}\\begin{{itemize}}'.format(i=doDent(recDepth, 1))
      # order subs in printing priority
      printSubs = [(sub, sub.printPriority) for sub in cls.subs]
      printSubs = (x[0] for x in sorted(printSubs, key=lambda x: x[1])) # generator
      for sub in printSubs:
        msg += '\n{sub}'.format(sub=sub.generateLatex(recDepth=recDepth+2))
      msg += '{i}\\end{{itemize}}\n'.format(i=doDent(recDepth, 1))
    # TODO is this a good idea? -> disables underscores in math mode :(
    if recDepth == 0:
      # assure underscore is escaped, but not doubly
      msg = re.sub(r'(?<!\\)_', r'\_', msg)
    return msg

  @classmethod
  def generateParamsLatex(cls, recDepth):
    """
      Generates the LaTeX representation of the parameters (attributes) of this spec.
      @ In, recDepth, int, recursion depth for indentation purposes
      @ Out, msg, str, LaTeX parameters
    """
    msg = ''
    # if no parameters, nothing to do
    if not cls.parameters:
      return msg
    specName = cls.name
    if '_' in specName:
      # assure underscore is escaped, but not doubly
      specName = re.sub(r'(?<!\\)_', r'\_', specName)
    msg += '{i}The \\xmlNode{{{n}}} node recognizes the following parameters:'.format(i=doDent(recDepth), n=specName)
    msg += '\n{i}\\begin{{itemize}}'.format(i=doDent(recDepth, 1))
    for param, info in cls.parameters.items():
      # assure underscore is escaped, but not doubly
      name = re.sub(r'(?<!\\)_', r'\_', param)
      typ = info['type'].generateLatexType()
      req = 'required' if info['required'] else 'optional'
      desc = wrapText(info['description'], indent=doDent(recDepth, 3))
      msg += '\n{i}  \\item \\xmlAttr{{{n}}}: \\xmlDesc{{{t}, {r}}}, \n{d}'.format(i=doDent(recDepth, 1),
                                                                                   n=name,
                                                                                   t=typ,
                                                                                   r=req,
                                                                                   d=desc)
    msg += '\n{i}\\end{{itemize}}\n'.format(i=doDent(recDepth))
    return msg




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

def assemblyInputFactory(*paramList, **paramDict):
  """
    Creates a new ParameterInput class with the same parameters as ParameterInput.createClass
    Also adds the standard "type" and "class" params for assembly objects
    @ In, same parameters as ParameterInput.createClass
    @ Out, newClass, ParameterInput, the newly created class.
  """
  class newClass(ParameterInput):
    """
      The new class to be created by the factory
    """
  newClass.createClass(*paramList, **paramDict)
  newClass.addParam('class', param_type=InputTypes.StringType, required=True,
      descr=r"""RAVEN class for this entity (e.g. Samplers, Models, DataObjects)""")
  newClass.addParam('type', param_type=InputTypes.StringType, required=True,
      descr=r"""RAVEN type for this entity; a subtype of the class (e.g. MonteCarlo, Code, PointSet)""")
  return newClass

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

#
#
#
#
class RavenBase(ParameterInput):
  """
    This can be used as a base class for things that inherit from BaseType
  """
RavenBase.createClass("RavenBase", baseNode=None)
verbs = InputTypes.makeEnumType('verbosity', 'verbosityType', ['silent', 'quiet', 'all', 'debug'])
RavenBase.addParam("verbosity", param_type=verbs, descr='Desired verbosity of messages coming from this entity') #XXX should be enumeration


#
#
#
#
def doDent(d, p=0, style='  '):
  """
    Creates an indent based on desired level
    @ In, d, int, number of indents to add nominally
    @ In, p, int, number of additional indents
    @ In, style, str, optional, characters for indenting
    @ Out, dent, str, indent string
  """
  return style * (d + p)

def wrapText(text, indent, width=100):
  """
    Utility to wrap text to an appropriate length and indentation
    @ In, text, text to wrap
    @ In, indent, str, string to use as an indent
    @ In, width, int, optional, number of characters including indent to wrap at
    @ Out, msg, str, modified text
  """
  msg = textwrap.dedent(text)
  msg = textwrap.fill(msg, width=width, initial_indent=indent, subsequent_indent=indent)
  return msg
