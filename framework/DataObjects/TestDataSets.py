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
  This Module performs Unit Tests for the Distribution class.
  It can not be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import sys, os
import pickle as pk
import numpy as np
frameworkDir = os.path.dirname(os.path.abspath(os.path.join(sys.argv[0],'..')))

sys.path.append(frameworkDir)
from utils.utils import find_crow

find_crow(frameworkDir)

import XrDataObject # FIXME
import MessageHandler

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug'})

print (XrDataObject )
def createElement(tag,attrib=None,text=None):
  """
    Method to create a dummy xml element readable by the distribution classes
    @ In, tag, string, the node tag
    @ In, attrib, dict, optional, the attribute of the xml node
    @ In, text, str, optional, the dict containig what should be in the xml text
  """
  if attrib is None:
    attrib = {}
  if text is None:
    text = ''
  element = ET.Element(tag,attrib)
  element.text = text
  return element

results = {"pass":0,"fail":0}

def checkAnswer(comment,value,expected,tol=1e-10):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ Out, None
  """
  if abs(value - expected) > tol:
    print("checking answer",comment,value,"!=",expected)
    results["fail"] += 1
  else:
    results["pass"] += 1

#Test module methods #TODO
#print(Distributions.knownTypes())
#Test error
#try:
#  Distributions.returnInstance("unknown",'dud')
#except:
#  print("error worked")

#############
# Point Set #
#############

xml = createElement('DataSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b,c'))
xml.append(createElement('Output',text='x,y,z'))

# check construction
# check builtins
# check basic property getters

# check appending (add row)
# check add var (add column)
# check remove var (remove column)
# check remove sample (remove row)

# check slicing
# # var vals
# # realization vals
# # by meta
# check find-by-index
# check find-by-value (including find-by-meta)

# check write to CSV
# check write to netCDF
# check load from CSV
# check load from netCDF


print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.test_datasets</name>
    <author>talbpaul</author>
    <created>2017-10-20</created>
    <classesTested>DataSet</classesTested>
    <description>
       This test is a Unit Test for the DataSet classes.
    </description>
  </TestInfo>
"""
