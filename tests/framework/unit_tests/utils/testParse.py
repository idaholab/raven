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

#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

import os,sys
import xml.etree.ElementTree as ET

frameworkDir = os.path.normpath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)

import utils.TreeStructure as TS
from utils.utils import toString

results = {'failed':0,'passed':0}

def checkSameFile(a,b):
  genA = iter(a)
  genB = iter(b)
  same = True
  msg = []
  i = -1
  while True:
    i+=1
    try:
      al = next(genA)
    except StopIteration:
      try:
        next(genB) #see if B is done
        return False,msg + ['file '+str(b)+' has more lines thani '+str(a)]
      except StopIteration:
        #both are done
        return same,msg
    try:
      bl = next(genB)
    except StopIteration:
      return False,msg + ['file '+str(a)+' has more lines than '+str(b)]
    if al.rstrip('\n\r') != bl.rstrip('\n\r'):
      same = False
      print('al '+repr(al)+" != bl "+repr(bl))
      msg += ['line '+str(i)+' is not the same!']



#first test XML to XML
print('Testing XML to XML ...')
tree = TS.parse(open(os.path.join('parse','example_xml.xml'),'r'))
strTree = TS.tostring(tree)
fname = os.path.join('parse','fromXmltoXML.xml')
open(fname,'w').write(toString(strTree))
same,msg = checkSameFile(open(fname,'r'),open(os.path.join('gold',fname),'r'))
if same:
  results['passed']+=1
  print('  ... passed!')
else:
  results['failed']+=1
  print('  ... failures in XML to XML:')
  print('     ',msg[0])


print('Results:', list('%s: %i' %(k,v) for k,v in results.items()))
sys.exit(results['failed'])


"""
  <TestInfo>
    <name>framework.inputParsing</name>
    <author>talbpaul</author>
    <created>2017-11-01</created>
    <classesTested>utils.utils.TreeStructure</classesTested>
    <description>
       This test performs Unit Tests for the utils class (TreeStructure)
       It cannot be considered part of the active code but of the regression test system
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
  </TestInfo>
"""
