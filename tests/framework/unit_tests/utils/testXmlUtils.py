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
  This Module performs Unit Tests for the xmlUtils methods
  It cannot be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os,sys
import numpy as np
import xml.etree.ElementTree as ET

frameworkDir = os.path.normpath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import xmlUtils, utils

from MessageHandler import MessageHandler
mh = MessageHandler()
mh.initialize({})

print (xmlUtils)

results = {"pass":0,"fail":0}
#type comparison
elemType = type(ET.Element('dummy'))
treeType = type(ET.ElementTree())
#cleanup utilities
toRemove = []

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

def attemptFileClear(fName,later):
  """
    Attempts to remove the file.  If not possible, store it in "later".
    @ In, fName, string, name of file to remove
    @ In, later, list, list of files to remove later
    @ Out, later, list, list of files to remove later
  """
  try:
    os.remove(fName)
  except OSError:
    later.append(fName)
  return later

#establish test XML
xmlString = '<root ratr="root_attrib"><child catr1="child attrib 1" catr2="child attrib 2"><cchild ccatr="cc_attrib">cchildtext</cchild></child></root>'
inFileName = 'testXMLInput.xml'
open(inFileName,'w').write(xmlString)
xmlTree = ET.parse(inFileName)
toRemove = attemptFileClear(inFileName,toRemove)

# test prettify
pretty = utils.toString(xmlUtils.prettify(xmlTree))
prettyFileName = 'xml/testXMLPretty.xml'
open(prettyFileName,'w').writelines(pretty)
gold = ''.join(line.rstrip('\n\r') for line in open(os.path.join(os.path.dirname(__file__),'gold',prettyFileName),'r'))
test = ''.join(line.rstrip('\n\r') for line in open(                    prettyFileName ,'r'))
if gold==test:
  results['pass']+=1
  toRemove = attemptFileClear(prettyFileName,toRemove)
else:
  print('ERROR: Test of "pretty" failed!  See',prettyFileName,'(below) vs gold/',prettyFileName)
  print('( START',prettyFileName,')')
  for line in file(prettyFileName,'r'):
    print(line[:-1]) #omit newline
  print('( END',prettyFileName,')')
  results['fail']+=1

# test newNode
### test with tag, text, and multiple attributes
node = xmlUtils.newNode('tagname',text="text",attrib={'atr1':'atr1_text','atr2':'atr2_text'})
okay = True
#type
if type(node)!=elemType:
  okay = False
  print('Test of "newNode" failed!  Returned node was not an xml.etree.ElementTree!  Instead was:',type(node))
#tag
if node.tag!='tagname':
  okay = False
  print('ERROR: Test of "newNode" failed!  Tag should have been "tagname" but instead was "'+node.tag+'".')
#text
if node.text!='text':
  okay = False
  print('ERROR: Test of "newNode" failed!  Text should have been "text" but instead was "'+node.text+'".')
#attributes
if 'atr1' not in node.attrib.keys():
  okay = False
  print('ERROR: Test of "newNode" failed!  Did not find attribute "atr1" in keys:',node.attrib.keys())
else:
  if node.attrib['atr1']!='atr1_text':
    okay = False
    print('ERROR: Test of "newNode" failed! Attribute "atr1" should have been "atr1_text" but was',node.attrib['atr1'])
if 'atr2' not in node.attrib.keys():
  okay = False
  print('ERROR: Test of "newNode" failed!  Did not find attribute "atr2" in keys:',node.attrib.keys())
else:
  if node.attrib['atr2']!='atr2_text':
    okay = False
    print('ERROR: Test of "newNode" failed! Attribute "atr2" should have been "atr2_text" but was',node.attrib['atr2'])
if okay:
  results['pass']+=1
else:
  results['fail']+=1

# test newTree
tree = xmlUtils.newTree('newroot')
okay = True
if type(tree) != treeType:
  okay = False
  print('ERROR: Test of "newTree" failed!  Returned tree was not xml.etree.ElementTree.ElementTree, instead was',type(tree))
elif tree.getroot().tag != 'newroot':
  okay = False
  print('ERROR: Test of "newTree" failed!  Root of new tree should be "newroot" but instead got',tree.getroot().tag)
if okay:
  results['pass']+=1
else:
  results['fail']+=1

# test tag finder for xpath
inps = []
outs = []
exps = []
inps.append('root/child[@na:me]')
outs.append(xmlUtils.fixTagsInXpath(inps[0]))
exps.append('root/child[@na:me]') # no change, not illegal

inps.append('root/child[1cch:ild]')
outs.append(xmlUtils.fixTagsInXpath(inps[1]))
exps.append('root/child[_1cch.ild]') # prepend _, : to .

inps.append('root/child[cchi:ld="te:mp"]')
outs.append(xmlUtils.fixTagsInXpath(inps[2]))
exps.append('root/child[cchi.ld=\'te:mp\']') # cchi.ld, " to '

inps.append('root/chi:ld/cchild')
outs.append(xmlUtils.fixTagsInXpath(inps[3]))
exps.append('root/chi.ld/cchild') # : to .

inps.append('root/child[0]')
outs.append(xmlUtils.fixTagsInXpath(inps[4]))
exps.append('root/child[0]') # no change

for i in range(5):
  if outs[i] == exps[i]:
    results['pass']+=1
  else:
    results['fail']+=1
    print('ERROR: fixTagsInXpath #{}: expected "{}" but got "{}"'.format(i,exps[i],outs[i]))


# test findPath
###test successful find
found = xmlUtils.findPath(xmlTree.getroot(),'child/cchild')
okay = True
#  type
if type(found)!=elemType:
  okay = False
  print('ERROR: Test of "findPath" failed!  Returned node was not an xml.etree.ElementTree!  Instead was:',type(found))
elif found.tag!='cchild':
  okay = False
  print('ERROR: Test of "findPath" failed!  Returned node tag was not "cchild"!  Instead was:',found.tag)
if okay:
  results['pass']+=1
else:
  results['fail']+=1
###test not found
found = xmlUtils.findPath(xmlTree.getroot(),'child/cchild/notANodeInTheTree')
if found is not None:
  print('ERROR: Test of "findPath" failed!  No element should have been found, but found',found)
  results['fail']+=1
else:
  results['pass']+=1

for f in toRemove:
  if os.path.exists(f):
    try:
      os.remove(f)
    except OSError:
      print('WARNING: In cleaning up, could not remove file',f)

#test findPathEllipsesParents
found = xmlUtils.findPathEllipsesParents(xmlTree.getroot(),'child/cchild')
print ('ellipses')
print(xmlUtils.prettify(found,doc=True))
# TODO is there supposed to be a test here?

#test bad XML tags
# rule 1: only start with letter or underscore, can't start with xml
bads = ['xmlstart','0startnum','.startspec']
for bad in bads:
  fixed = xmlUtils.fixXmlTag(bad)
  if fixed == '_'+bad:
    results['pass']+=1
  else:
    print('ERROR: Fixing illegal XML tag "'+bad+'" FAILED:',fixed,'should be','_'+bad)
    results['fail']+=1
# rule 2: only contain letters, digits, hyphens, underscores, and periods
bads = ['spec@char','co:lon','!234','?<>, ']
rights = ['spec.char','co.lon','_.234','_.....']
for b,bad in enumerate(bads):
  fixed = xmlUtils.fixXmlTag(bad)
  if fixed == rights[b]:
    results['pass']+=1
  else:
    print('ERROR: Fixing illegal XML tag "'+bad+'" FAILED:',fixed,'should be',rights[b])
    results['fail']+=1
#don't fix what ain't broke
okay = ['abcd','ABCD','_xml','per.iod','hy-phen','under_score']
for ok in okay:
  fixed = xmlUtils.fixXmlTag(ok)
  if fixed == ok:
    results['pass']+=1
  else:
    print('ERROR: Fixing legal XML tag "'+ok+'" FAILED:',fixed,'should be',ok)
    results['fail']+=1


# test readExternalXML, use relative path
extFile = 'GoodExternalXMLFile.xml'
extNode = 'testMainNode'
cwd = os.path.join(os.path.dirname(__file__),'xml')
node = xmlUtils.readExternalXML(extFile,extNode,cwd)
strNode = """<testMainNode att=\"attrib1\">
  <firstSubNode>
    <firstFirstSubNode att=\"attrib1.1.1\">firstFirstSubText</firstFirstSubNode>
  </firstSubNode>
  <secondSubNode att=\"attrib1.2\">
    <secondFirstSubNode>secondFirstSubText</secondFirstSubNode>
  </secondSubNode>
</testMainNode>"""
if strNode != utils.toString(ET.tostring(node)):
  print('ERROR: loaded XML node:')
  print(ET.tostring(node))
  print(' ----- does not match expected:')
  print(strNode)
  results['fail']+=1
else:
  results['pass']+=1


# test expandExternalXML, two substitutions
strNode = """<root>
  <ExternalXML node="testMainNode" xmlToLoad="GoodExternalXMLFile.xml"/>
  <rootsub>
    <ExternalXML node="testMainNode" xmlToLoad="GoodExternalXMLFile.xml"/>
  </rootsub>
</root>
"""
root = ET.fromstring(strNode)
cwd = os.path.join(os.path.dirname(__file__),'xml')
xmlUtils.expandExternalXML(root,cwd)
correct = """<root>
  <testMainNode att="attrib1">
  <firstSubNode>
    <firstFirstSubNode att="attrib1.1.1">firstFirstSubText</firstFirstSubNode>
  </firstSubNode>
  <secondSubNode att="attrib1.2">
    <secondFirstSubNode>secondFirstSubText</secondFirstSubNode>
  </secondSubNode>
</testMainNode><rootsub>
    <testMainNode att="attrib1">
  <firstSubNode>
    <firstFirstSubNode att="attrib1.1.1">firstFirstSubText</firstFirstSubNode>
  </firstSubNode>
  <secondSubNode att="attrib1.2">
    <secondFirstSubNode>secondFirstSubText</secondFirstSubNode>
  </secondSubNode>
</testMainNode></rootsub>
</root>"""
if correct != utils.toString(ET.tostring(root)):
  print('ERROR: expanded XML node:')
  print(ET.tostring(root))
  print(' ----- does not match expected:')
  print(correct)
  results['fail']+=1
else:
  results['pass']+=1


# test StaticXmlElement
print('')
static = xmlUtils.StaticXmlElement('testRoot')
static.addScalar('newTarget','newMetric',42)
# check new structure was added
try:
  val = int(static.getRoot().find('newTarget').find('newMetric').text)
  if val == 42:
    results['pass']+=1
  else:
    print('ERROR: StaticXmlElement value failure: "newTarget.newMetric = {}"'.format(val))
    results['fail']+=1
except AttributeError:
  print('ERROR: StaticXmlElement could not find newTarget.newMetric!')
  results['fail']+=1

newTarget = static.getRoot().find('newTarget')
if newTarget is None:
  print('ERROR: StaticXmlElement new node missing: "newTarget"')
  results['fail']+=1
else:
  newMetric = newTarget.find('newMetric')
  if newMetric is None:
    print('ERROR: StaticXmlElement new node missing: "newTarget.newMetric"')
    results['fail']+=1
  else:
    val = int(newMetric.text)
    if val == 42:
      results['pass'] +=1
    else:
      print('ERROR: StaticXmlElement value failure: "newTarget.newMetric = {}"'.format(val))
      results['fail']+=1

static.addVector('newTarget','newVectorMetric',{'A':1,'B':2})
try:
  A = int(static.getRoot().find('newTarget').find('newVectorMetric').find('A').text)
  if A == 1:
    results['pass']+=1
  else:
    print('ERROR: StaticXmlElement value failure: "newTarget.newVectorMetric.A = {}"'.format(A))
    results['fail']+=1
except AttributeError:
  print('ERROR: StaticXmlElement could not find newTarget.newVectorMetric.A!')
  results['fail']+=1

try:
  B = int(static.getRoot().find('newTarget').find('newVectorMetric').find('B').text)
  if B == 2:
    results['pass']+=1
  else:
    print('ERROR: StaticXmlElement value failure: "newTarget.newVectorMetric.B = {}"'.format(B))
    results['fail']+=1
except AttributeError:
  print('ERROR: StaticXmlElement could not find newTarget.newVectorMetric.A!')
  results['fail']+=1


# test DynamicXmlElement
print('')
dynamic = xmlUtils.DynamicXmlElement('testRoot',pivotParam='Time')
values = {0.1: 1, 0.2:1, 0.3:2}
for t,v in values.items():
  dynamic.addScalar('newTarget', 'newMetric', v, t)

try:
  times = dynamic.getRoot().findall('Time')
  for node in times:
    t = float(node.attrib['value'])
    v = float(node.find('newTarget').find('newMetric').text)
    if v == values[t]:
      results['pass']+=1
    else:
      print('ERROR: DynamicXmlElement value failure: "Time {} newTarget.newMetric = {}"'.format(t,v))
      results['fail']+=1
except AttributeError:
  print('ERROR: DynamicXmlElement has missing elements!')
  results['fail']+=1

# for debugging:
#print(xmlUtils.prettify(dynamic._tree,addRavenNewlines=False))
###################
# Variable Groups #
###################

print('')
# all ( a, b, d)
# d (some a, some b)
# ab (a, b)
# ce (c, e)
# f is isolated
example = """<VariableGroups>
  <Group name="a">a1, a2, a3</Group>
  <Group name="b">b1, b2, b3</Group>
  <Group name="c">c1, c2, c3</Group>
  <Group name="d">a1, b1</Group>
  <Group name="e">e1, e2 ,e3</Group>
  <Group name="f">f1, f2, f3</Group>
  <Group name="ce">c, e</Group>
  <Group name="ab">a,b</Group>
  <Group name="abd">a,b,d</Group>
  <Group name="plus">a,c</Group>
  <Group name="minus">a,-d</Group>
  <Group name="intersect">a,^d</Group>
  <Group name="symmdiff">a,%d</Group>
  <Group name="symmrev">d,%a</Group>
</VariableGroups>"""
node = ET.fromstring(example)
groups = xmlUtils.readVariableGroups(node,mh,None)

# test contents
def testVarGroup(groups,g,right):
  got = groups[g].getVarsString()
  if got == right:
    results['pass']+=1
  else:
    print('ERROR: Vargroups group "{}" should be "{}" but got "{}"!'.format(g,right,got))
    results['fail']+=1

# note that order matters for these solutions!
testVarGroup(groups,'a','a1,a2,a3')             # a and b are first basic sets
testVarGroup(groups,'b','b1,b2,b3')             # a and b are first basic sets
testVarGroup(groups,'c','c1,c2,c3')             # c and e are second basic sets
testVarGroup(groups,'d','a1,b1')                # d is just a1 and b1, to test one-entry-per-variable
testVarGroup(groups,'e','e1,e2,e3')             # c and e are second basic sets
testVarGroup(groups,'f','f1,f2,f3')             # f is isolated; nothing else uses it
testVarGroup(groups,'ce','c1,c2,c3,e1,e2,e3')   # ce combines c and e
testVarGroup(groups,'ab','a1,a2,a3,b1,b2,b3')   # ab combines a and b
testVarGroup(groups,'abd','a1,a2,a3,b1,b2,b3')  # ab combines a and b
testVarGroup(groups,'plus','a1,a2,a3,c1,c2,c3') # plus is OR for a and c
testVarGroup(groups,'minus','a2,a3')            # minus is a but not d
testVarGroup(groups,'intersect','a1')           # intersect is AND for a, d
testVarGroup(groups,'symmdiff','a2,a3,b1')      # symdiff is XOR for a, d
testVarGroup(groups,'symmrev','b1,a2,a3')       # symmrev shows order depends on how variables are put in








print(results)




sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.xmlUtils</name>
    <author>talbpaul</author>
    <created>2016-11-01</created>
    <classesTested>utils.xmlUtils</classesTested>
    <description>
       This test performs Unit Tests for the xmlUtils class
       It cannot be considered part of the active code but of the regression test system
    </description>
    <revisions>
      <revision author="talbpaul" date="2016-11-08">Relocated utils tests</revision>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
    </revisions>
  </TestInfo>
"""
