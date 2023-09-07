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
Created on 2016-Oct-21

@author: cogljj

This is used to test the xml differ
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import sys,os

scriptDir = os.path.dirname(os.path.abspath(__file__))
ravenDir = os.path.dirname(os.path.dirname(os.path.dirname(scriptDir)))

sys.path.append(os.path.join(ravenDir,"scripts","TestHarness","testers"))
sys.path.append(os.path.join(ravenDir,"rook"))

results = {"pass":0, "fail": 0}

def checkAnswer(comment,value,expected):
  """
    This method is aimed to compare two values
    @ In, comment, string, a comment printed out if it fails
    @ In, value, any, the value to compare
    @ In, expected, any, the expected value
    @ Out, None
  """
  if value != expected:
    print("checking answer",comment,value,"!=",expected)
    results["fail"] += 1
  else:
    results["pass"] += 1

import XMLDiff

import xml.etree.ElementTree as ET

same,message = XMLDiff.compare_ordered_element(ET.fromstring("<test></test>"),
                                             ET.fromstring("<test></test>"))
checkAnswer("simple",same,True)

same,message = XMLDiff.compare_ordered_element(ET.fromstring("<test>Hello  World</test>"),
                                             ET.fromstring("<test>Hello World</test>"))

checkAnswer("whitespace",same,False)

same,message = XMLDiff.compare_ordered_element(ET.fromstring("<test>Hello  World</test>"),
                                             ET.fromstring("<test>Hello World</test>"),
                                             remove_whitespace=True)

checkAnswer("whitespace with remove",same,True)
same,message = XMLDiff.compare_ordered_element(ET.fromstring("<test></test>"),
                                             ET.fromstring("<test></test>"))
checkAnswer("simple",same,True)


same,message = XMLDiff.compare_unordered_element(ET.fromstring("<test>Hello  World</test>"),
                                             ET.fromstring("<test>Hello World</test>"))

checkAnswer("whitespace unordered",same,False)

same,message = XMLDiff.compare_unordered_element(ET.fromstring("<test>Hello  World</test>"),
                                             ET.fromstring("<test>Hello World</test>"),
                                             remove_whitespace=True)

checkAnswer("whitespace with remove unordered",same,True)

a_tree,b_tree,success = XMLDiff.ignore_subnodes_from_tree(
                          ET.ElementTree(ET.fromstring("<test> <a>0</a> <b>0</b> </test>")),
                          ET.ElementTree(ET.fromstring("<test> <a>0</a> <b>1</b> </test>")),
                          ignored_nodes=["./b"])
checkAnswer("ignoring child node in both trees",success,True)
same,message = XMLDiff.compare_unordered_element(a_tree.getroot(),b_tree.getroot())
checkAnswer("comparing roots after ignoring child node in both trees",same,True)

a_tree,b_tree,success = XMLDiff.ignore_subnodes_from_tree(
                          ET.ElementTree(ET.fromstring('<test><a n="0">1</a><a n="1"/></test>')),
                          ET.ElementTree(ET.fromstring('<test><a n="0">2</a><a n="1"/></test>')),
                          ignored_nodes=['./a[@n="0"]'])
checkAnswer("ignoring child node in both trees via attribute",success,True)
same,message = XMLDiff.compare_unordered_element(a_tree.getroot(),b_tree.getroot())
checkAnswer("comparing roots after ignoring child node in both trees via attribute",same,True)

a_tree,b_tree,success = XMLDiff.ignore_subnodes_from_tree(
                          ET.ElementTree(ET.fromstring('<test> <a/> <c/> </test>')),
                          ET.ElementTree(ET.fromstring('<test> <a/> <b/> </test>')),
                          ignored_nodes=['./b', './c'])
checkAnswer("ignoring multiple nodes exclusive to trees",success,True)
same,message = XMLDiff.compare_unordered_element(a_tree.getroot(),b_tree.getroot())
checkAnswer("comparing roots after ignoring multiple nodes exclusive to trees",same,True)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.test_xml_differ</name>
    <author>cogljj</author>
    <created>2016-10-21</created>
    <classesTested>None</classesTested>
    <description>
       This test is aimed to check the xml differ program
    </description>
    <revisions>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
      <revision author="sotogj" date="2023-09-07">Adding tests for ignoring nodes.</revision>
    </revisions>
  </TestInfo>
"""
