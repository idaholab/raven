"""
  This Module performs Unit Tests for the utils methods
  It cannot be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os,sys
import numpy as np
frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import utils

print (utils)

results = {"pass":0,"fail":0}

def checkTrue(comment,value,expected):
  """
    Takes a boolean and checks it against True or False.
  """
  if value == expected:
    results["pass"] += 1
    return True
  else:
    print("checking answer",comment,value,"!=",expected)
    results["fail"] += 1
    return False

def checkAnswer(comment,value,expected,tol=1e-10,updateResults=True):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, updateResults, bool, optional, if True updates global results
    @ Out, None
  """
  if abs(value - expected) > tol:
    print("checking answer",comment,value,"!=",expected)
    if updateResults:
      results["fail"] += 1
    return False
  else:
    if updateResults:
      results["pass"] += 1
    return True

def checkArray(comment,check,expected,tol=1e-10):
  """
    This method is aimed to compare two arrays of floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, check, list, the value to compare
    @ In, expected, list, the expected value
    @ In, tol, float, optional, the tolerance
    @ Out, None
  """
  same=True
  if len(check) != len(expected):
    same=False
  else:
    for i in range(len(check)):
      same = same*checkAnswer(comment+'[%i]'%i,check[i],expected[i],tol,False)
  if not same:
    print("checking array",comment,"did not match!")
    results['fail']+=1
    return False
  else:
    results['pass']+=1
    return True

def checkType(comment,value,expected,updateResults=True):
  """
    This method compares the data type of two values
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, updateResults, bool, optional, if True updates global results
    @ Out, None
  """
  if type(value) != type(expected):
    print("checking type",comment,value,'|',type(value),"!=",expected,'|',type(expected))
    if updateResults:
      results["fail"] += 1
    return False
  else:
    if updateResults:
      results["pass"] += 1
    return True

# check getRelativeSortedListEntry
toPopulate = [0.8, 0.002, 0.0003, 0.9, 0.85, 0.799999999999, 0.90000001, 0.00029999999999]
#populating these in this order tests adding new entries to the front (0.0003), back (0.9), and middle (0.85),
#  as well as adding matches in the front (0.00029...), back (0.90...1), and middle (0.79...)
desired = [0.0003, 0.002, 0.8, 0.85, 0.9]
sortedList = []
for x in toPopulate:
  sortedList,index,match = utils.getRelativeSortedListEntry(sortedList,x,tol=1e-6)
checkArray('Maintaining sorted list',sortedList,desired)

print(results)

sys.exit(results["fail"])

"""
  <TestInfo>
    <name>framework.utils</name>
    <author>talbpaul</author>
    <created>2017-11-01</created>
    <classesTested>utils.utils</classesTested>
    <description>
       This test performs Unit Tests for the utils class.
       It cannot be considered part of the active code but of the regression test system
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
      <revision author="alfoa" date="2019-03-04">Moved methods isAString, isAFloat, isAInteger, isABoolean from mathUtils to utils</revision>
    </revisions>
  </TestInfo>
"""
