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
Created on 2020-Jan-07

@author: talbpaul

This a library for defining the data. Split from InputData module.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import xml.etree.ElementTree as ET
from utils import utils, mathUtils

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

StringType.createClass("string", "xsd:string")

#
#
#
#
class StringNoLeadingSpacesType(InputType):
  """
    A type for arbitrary string data. This is equivalent
    to the StringType but in case of a comma delimiter, it
    removes the leading white spaces. For Example:
    - if value = 'varName', it return 'varName'
    - if value = 'varName1, varName2', it returns 'varName1,varName2'
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to an integer
      @ In, value, string, the value to convert
      @ Out, convert, string, the converted value
    """
    return ','.join([key.strip() for key in value.split(',')]) if ',' in value else value
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
    if mathUtils.isAString(value):
      value = float(value)
    return int(value)

IntegerType.createClass("integer", "xsd:integer")

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

FloatType.createClass("float", "xsd:double")

#
#
#
#
class FloatOrIntType(InputType):
  """
    A type for floating point or integer data.
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to a float or int.
      @ In, value, string, the value to convert
      @ Out, val, float or int, the converted value
    """
    try:
      val = int(value)
      return val
    except ValueError:
      val = float(value)
      return val

FloatOrIntType.createClass("float_or_int", "xsd:string")

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
    delim = ',' if ',' in value else None
    values = list(x.strip() for x in value.split(delim) if x.strip())
    base = utils.partialEval(values[0])
    # three possibilities: string, integer, or float
    if mathUtils.isAString(base):
      conv = str
    elif mathUtils.isAnInteger(base):
      conv = int
    else: #float
      conv = float
    return [conv(x.strip()) for x in values]

#Note, XSD's list type is split by spaces, not commas, so using xsd:string
InterpretedListType.createClass("interpreted", "xsd:string")


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
    delim = ',' if ',' in value else None
    return [x.strip() for x in value.split(delim) if x.strip()]

#Note, XSD's list type is split by spaces, not commas, so using xsd:string
StringListType.createClass("string_list", "xsd:string")


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
    # prefer commas, but allow spaces, to divide
    delim = ',' if ',' in value else None
    return [float(x.strip()) for x in value.split(delim) if x.strip()]

#Note, XSD's list type is split by spaces, not commas, so using xsd:string
FloatListType.createClass("float_list", "xsd:string")


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
    delim = ',' if ',' in value else None
    return [int(x.strip()) for x in value.split(delim) if x.strip()]

#Note, XSD's list type is split by spaces, not commas, so using xsd:string
IntegerListType.createClass("integer_list", "xsd:string")

#
#
#
#
class IntegerOrIntegerTupleType(InputType):
  """
    A type for integer "1" -> 1
    or integer tuples "1, 2, 3" -> (1,2,3)
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to an integer tuple.
      @ In, value, string, the value to convert
      @ Out, convertedValue, int or tuple, the converted value
    """
    convertedValue = tuple(int(x.strip()) for x in value.split(","))
    convertedValue = convertedValue[0] if len(convertedValue) == 1 else convertedValue
    return convertedValue

IntegerOrIntegerTupleType.createClass("integer_or_integer_list", "xsd:string")

#
#
#
#
class IntegerTupleType(InputType):
  """
    A type for integer tuples "1, 2, 3" -> (1,2,3)
  """

  @classmethod
  def convert(cls, value):
    """
      Converts value from string to an integer tuple.
      @ In, value, string, the value to convert
      @ Out, convertedValue, tuple, the converted value
    """
    convertedValue = tuple(int(x.strip()) for x in value.split(","))
    return convertedValue

IntegerTupleType.createClass("integer_list", "xsd:string")

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
  def convert(cls, value):
    """
      Error checking for reading in enum entries
      @ In, value, object, user-requested enum value
      @ Out, value, object, adjusted enum value
    """
    # TODO is this the right place for checking?
    ## TODO need to provide the offending XML node somehow ...
    ## TODO should these by caught and handled by the parseNode?
    if value not in cls.enumList:
      raise IOError('Value "{}" unrecognized! Expected one of {}.'.format(value, cls.enumList))
    return value

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
    if value.lower() in utils.stringsThatMeanTrue():
      return True
    elif value.lower() in utils.stringsThatMeanFalse():
      return False
    else:
      raise IOError('Unrecognized boolean value: "{}"! Expected "True" or "False", or similar.'.format(value))

boolTypeList = utils.stringsThatMeanTrue() + utils.stringsThatMeanFalse()
BoolType.createClass("bool", "boolType", boolTypeList + [elm.capitalize() for elm in boolTypeList])



def makeEnumType(name, xmlName, enumList):
  """
    Creates a new enum type that can be used as a content type.
    @ In, name, string, Name of the type
    @ In, xmlName, string, Name of the type used in the XSD file.
    @ In, enumList, list of strings, the possible values of the enumeration.
    @ Out, newEnum, EnumBaseType, the new enumeration type.
  """
  class newEnum(EnumBaseType):
    """
      the new enum to be created by the factory
    """

  newEnum.createClass(name, xmlName, enumList)
  return newEnum
