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
Created on 2016-Apr-7

@author: cogljj

This is common classes for testing the xsd program.
"""


from __future__ import division, print_function, unicode_literals, absolute_import
import xml.etree.ElementTree as ET

from utils import InputData

class Sub1Input(InputData.ParameterInput):
  """
    Test class
  """
  pass

Sub1Input.createClass("sub_1")
Sub1Input.setContentType(InputData.StringType)

class Sub2Input(InputData.ParameterInput):
  """
    Test class
  """
  pass

Sub2Input.createClass("sub_2")
Sub2Input.setContentType(InputData.StringType)

class Sub3Input(InputData.ParameterInput):
  """
    Test class
  """
  pass

Sub3Input.createClass("sub_3")
Sub3Input.setContentType(InputData.IntegerType)

Sub4Input = InputData.parameterInputFactory("sub_4", contentType=InputData.FloatType)

class InnerInput(InputData.ParameterInput):
  """
    Test class
  """
  pass

InnerInput.createClass("inner")

InnerInput.addParam("data_1")
InnerInput.addParam("int_value", InputData.IntegerType)
InnerInput.addParam("required_string", InputData.StringType, True)
InnerInput.addSub(Sub1Input)
InnerInput.addSub(Sub2Input)
InnerInput.addSub(Sub3Input)
InnerInput.addSub(Sub4Input)

class SubBool(InputData.ParameterInput):
  """
    Test class
  """
  pass

SubBool.createClass("sub_bool")
SubBool.setContentType(InputData.BoolType)

class OrderedInput(InputData.ParameterInput):
  """
    Test class
  """
  pass

OrderedInput.createClass("ordered", True)
OrderedInput.addSub(Sub1Input, InputData.Quantity.zero_to_one)
OrderedInput.addSub(Sub2Input, InputData.Quantity.one)
OrderedInput.addSub(Sub3Input, InputData.Quantity.one_to_infinity)
OrderedInput.addSub(SubBool, InputData.Quantity.zero_to_infinity)


class OuterInput(InputData.ParameterInput):
  """
    Test class
  """
  pass

OuterInput.createClass("outer", True)
OuterInput.addSub(InnerInput)
OuterInput.addSub(OrderedInput)
