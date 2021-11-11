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
  This Module performs Unit Tests for the TreeStructure classes
  It cannot be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os,sys
import numpy as np

frameworkDir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
print('framework:',frameworkDir)
sys.path.append(frameworkDir)

from utils import TreeStructure as TS

results = {"pass":0,"fail":0}
#type comparison

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

def checkSame(comment,value,expected,updateResults=True):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, updateResults, bool, optional, if True updates global results
    @ Out, None
  """
  if value != expected:
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

##############
# Node Tests #
##############
# TODO not complete!

#test equivalency (eq, neq, hash)
## test all same are same
a = TS.HierarchicalNode('rightTag',valuesIn={'attrib1':1,'attrib2':'2'},text='sampleText')
b = TS.HierarchicalNode('rightTag',valuesIn={'attrib1':1,'attrib2':'2'},text='sampleText')
checkSame('Equivalency of nodes ==:',a==b,True)
checkSame('Equivalency of nodes !=:',a!=b,False)

## test different tag
b = TS.HierarchicalNode('diffTag',valuesIn={'attrib1':1,'attrib2':'2'},text='sampleText')
checkSame('Inequivalent tag ==:',a==b,False)
checkSame('Inequivalent tag !=:',a!=b,True)

## test different attribute name
b = TS.HierarchicalNode('rightTag',valuesIn={'attrib3':1,'attrib2':'2'},text='sampleText')
checkSame('Inequivalent value name ==:',a==b,False)
checkSame('Inequivalent value name !=:',a!=b,True)

## test different attribute value
b = TS.HierarchicalNode('rightTag',valuesIn={'attrib1':3,'attrib2':'2'},text='sampleText')
checkSame('Inequivalent value name ==:',a==b,False)
checkSame('Inequivalent value name !=:',a!=b,True)

## test different text value
b = TS.HierarchicalNode('rightTag',valuesIn={'attrib1':3,'attrib2':'2'},text='diffText')
checkSame('Inequivalent value name ==:',a==b,False)
checkSame('Inequivalent value name !=:',a!=b,True)

## test equivalent, only tags
a = TS.HierarchicalNode('rightTag')
b = TS.HierarchicalNode('rightTag')
checkSame('Equivalency only tag ==:',a==b,True)
checkSame('Equivalency only tag !=:',a!=b,False)

## test equivalent, only values
a = TS.HierarchicalNode('rightTag',valuesIn={'attrib1':1,'attrib2':'2'})
b = TS.HierarchicalNode('rightTag',valuesIn={'attrib1':1,'attrib2':'2'})
checkSame('Equivalency only values ==:',a==b,True)
checkSame('Equivalency only values !=:',a!=b,False)

## test equivalent, only text
a = TS.HierarchicalNode('rightTag',text='sampleText')
b = TS.HierarchicalNode('rightTag',text='sampleText')
checkSame('Equivalency only text ==:',a==b,True)
checkSame('Equivalency only text !=:',a!=b,False)


##############
# Tree Tests #
##############
# TODO

##################
# Metadata Tests #
##################

# construction
static = TS.StaticMetadataTree('myStaticData')
dynamic = TS.DynamicMetadataTree('myDynamicData','timeParam')

# test "dynamic" attribute set correctly
checkSame('Static "dynamic" property correctly set:',static.getrootnode().get('dynamic'),'False')
checkSame('Dynamic "dynamic" property correctly set:',dynamic.getrootnode().get('dynamic'),'True')

# test message handler works (implicit test, no error means success)
static.raiseADebug('Debug message in Static successful!')
dynamic.raiseADebug('Debug message in Dynamic successful!')
results['pass']+=2

#test adding scalar entries (implicit test, no error means success)
static.addScalar('myTarget','myMetric',3.14159)
results['pass']+=1
dynamic.addScalar('myTarget','myMetric',3.14159,pivotVal=0.1) #pivot value as float
results['pass']+=1
dynamic.addScalar('myTarget','myMetric',299792358,pivotVal='0.2') #pivot value as string
results['pass']+=1

#test finding pivotNode (dynamic only)
a = TS.HierarchicalNode('timeParam',valuesIn={'value':0.2})
b = dynamic._findPivot(dynamic.getrootnode(),0.2)
checkSame('Finding pivot node:',b,a)

#test finding targetNode
## static
a = TS.HierarchicalNode('myTarget')
b = static._findTarget(static.getrootnode(),'myTarget')
checkSame('Finding target (static):',b,a)
## dynamic
a = TS.HierarchicalNode('myTarget')
c = dynamic._findTarget(dynamic.getrootnode(),'myTarget',0.2)
checkSame('Finding target (dynamic):',c,a)

#test values recorded
checkAnswer('Recorded data (static):',b.findBranch('myMetric').text,3.14159)
c = dynamic._findTarget(dynamic.getrootnode(),'myTarget',0.1)
checkAnswer('Recorded data (dynamic 1):',c.findBranch('myMetric').text,3.14159)
c = dynamic._findTarget(dynamic.getrootnode(),'myTarget',0.2)
checkAnswer('Recorded data (dynamic 2):',c.findBranch('myMetric').text,299792358)

print('{0}ed: {2}, {1}ed: {3}'.format(*(list(str(r) for r in results.keys())+list(results.values()))))
sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.treeStructure</name>
    <author>talbpaul</author>
    <created>2016-11-01</created>
    <classesTested>utils.TreeStructure</classesTested>
    <description>
       This test performs Unit Tests for the TreeStructure classes
       It cannot be considered part of the active code but of the regression test system
    </description>
    <revisions>
      <revision author="talbpaul" date="2016-11-08">Relocated utils tests</revision>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
    </revisions>
  </TestInfo>
"""
