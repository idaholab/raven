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

same,message = XMLDiff.compareOrderedElement(ET.fromstring("<test></test>"),
                                             ET.fromstring("<test></test>"))
checkAnswer("simple",same,True)

same,message = XMLDiff.compareOrderedElement(ET.fromstring("<test>Hello  World</test>"),
                                             ET.fromstring("<test>Hello World</test>"))

checkAnswer("whitespace",same,False)

same,message = XMLDiff.compareOrderedElement(ET.fromstring("<test>Hello  World</test>"),
                                             ET.fromstring("<test>Hello World</test>"),
                                             remove_whitespace=True)

checkAnswer("whitespace with remove",same,True)
same,message = XMLDiff.compareOrderedElement(ET.fromstring("<test></test>"),
                                             ET.fromstring("<test></test>"))
checkAnswer("simple",same,True)


same,message = XMLDiff.compareUnorderedElement(ET.fromstring("<test>Hello  World</test>"),
                                             ET.fromstring("<test>Hello World</test>"))

checkAnswer("whitespace unordered",same,False)

same,message = XMLDiff.compareUnorderedElement(ET.fromstring("<test>Hello  World</test>"),
                                             ET.fromstring("<test>Hello World</test>"),
                                             remove_whitespace=True)

checkAnswer("whitespace with remove unordered",same,True)


sys.exit(results["fail"])
