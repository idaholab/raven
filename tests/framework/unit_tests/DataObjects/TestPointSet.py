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
  This Module performs Unit Tests for the PointSet data objects.
  It can not be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import sys, os, copy
import pickle as pk
import numpy as np
import xarray as xr

# find location of crow, message handler
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4+['framework'])))
sys.path.append(frameworkDir)

from utils.utils import find_crow
find_crow(frameworkDir)
import MessageHandler

# find location of data objects
sys.path.append(os.path.join(frameworkDir,'DataObjects'))

import PointSet

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})

print('Module undergoing testing:')
print(PointSet)
print('')

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

def checkFloat(comment,value,expected,tol=1e-10,update=True):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ Out, res, bool, True if same
  """
  if np.isnan(value) and np.isnan(expected):
    res = True
  elif np.isnan(value) or np.isnan(expected):
    res = False
  else:
    res = abs(value - expected) <= tol
  if update:
    if not res:
      print("checking float",comment,'|',value,"!=",expected)
      results["fail"] += 1
    else:
      results["pass"] += 1
  return res

def checkTrue(comment,res,update=True):
  """
    This method is a pass-through for consistency and updating
    @ In, comment, string, a comment printed out if it fails
    @ In, res, bool, the tested value
    @ Out, res, bool, True if test
  """
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking bool",comment,'|',res,'is not True!')
      results["fail"] += 1
  return res

def checkSame(comment,value,expected,update=True):
  """
    This method is aimed to compare two identical things
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking string",comment,'|',value,"!=",expected)
      results["fail"] += 1
  return res

def checkArray(comment,first,second,dtype,tol=1e-10,update=True):
  """
    This method is aimed to compare two arrays
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ Out, res, bool, True if same
  """
  res = True
  if len(first) != len(second):
    res = False
    print("checking answer",comment,'|','lengths do not match:',len(first),len(second))
  else:
    for i in range(len(first)):
      if dtype == float:
        pres = checkFloat('',first[i],second[i],tol,update=False)
      elif dtype in (str,unicode):
        pres = checkSame('',first[i],second[i],update=False)
      if not pres:
        print('checking array',comment,'|','entry "{}" does not match: {} != {}'.format(i,first[i],second[i]))
        res = False
  if update:
    if res:
      results["pass"] += 1
    else:
      results["fail"] += 1
  return res

def checkRlz(comment,first,second,tol=1e-10,update=True):
  """
    This method is aimed to compare two realization
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ Out, res, bool, True if same
  """
  res = True
  if len(first) != len(second):
    res = False
    print("checking answer",comment,'|','lengths do not match:',len(first),len(second))
  else:
    for key,val in first.items():
      if isinstance(val,float):
        pres = checkFloat('',val,second[key],tol,update=False)
      elif isinstance(val,(str,unicode)):
        pres = checkSame('',val,second[key][0],update=False)
      elif isinstance(val,xr.DataArray):
        if isinstance(val.item(0),(float,int)):
          pres = (val - second[key]).sum()<1e-20 #necessary due to roundoff
        else:
          pres = val.equals(second[key])
      else:
        raise TypeError(type(val))
      if not pres:
        print('checking dict',comment,'|','entry "{}" does not match: {} != {}'.format(key,first[key],second[key]))
        res = False
  if update:
    if res:
      results["pass"] += 1
    else:
      results["fail"] += 1
  return res

def checkNone(comment,entry,update=True):
  """
    Tests if the entry identifies as None.
    @ In, comment, str, comment to print if failed
    @ In, entry, object, object to test
    @ In, update, bool, optional, if True then updates results
    @ Out, None
  """
  res = entry is None
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking answer",comment,'|','"{}" is not None!'.format(entry))
      results["fail"] += 1

def checkFails(comment,errstr,function,update=True,args=None,kwargs=None):
  """
    Tests if function fails as expected
    @ In, comment, str, comment to print if failed
    @ In, errstr, str, expected error string
    @ In, function, method, method to run
    @ In, update, bool, optional, if True then updates results
    @ In, args, list, arguments to function
    @ In, kwargs, dict, keywords arguments to function
    @ Out, res, bool, result (True if passed)
  """
  print('Error testing ...')
  if args is None:
    args = []
  if kwargs is None:
    kwargs = {}
  try:
    function(*args,**kwargs)
    res = False
    msg = 'Function call did not error!'
  except Exception as e:
    res = checkSame('',e.args[0],errstr,update=False)
    if not res:
      msg = 'Unexpected error message.  \n    Received: "{}"\n    Expected: "{}"'.format(e.args[0],errstr)
  if update:
    if res:
      results["pass"] += 1
      print(' ... end Error testing (PASSED)')
    else:
      print("checking error",comment,'|',msg)
      results["fail"] += 1
      print(' ... end Error testing (FAILED)')
  print('')
  return res

def formatRealization(rlz):
  """
    Converts types of each input.
    @ In, rlz, dict, var:val
    @ Out, rlz, dict, formatted
  """
  for k,v in rlz.items():
    rlz[k] = np.atleast_1d(v)

#Test module methods #TODO
#print(Distributions.knownTypes())
#Test error
#try:
#  Distributions.returnInstance("unknown",'dud')
#except:
#  print("error worked")

######################################
#            CONSTRUCTION            #
######################################
xml = createElement('PointSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b'))
xml.append(createElement('Output',text='x,z'))

# check construction
data = PointSet.PointSet()
# inputs, outputs
checkSame('DataSet __init__ name',data.name,'PointSet')
checkSame('DataSet __init__ print tag',data.printTag,'PointSet')
checkNone('DataSet __init__ _data',data._data)
checkNone('DataSet __init__ _collector',data._collector)

# check initialization
data.messageHandler = mh
data._readMoreXML(xml)
checkArray('DataSet __init__ inp',data._inputs,['a','b'],str)
checkArray('DataSet __init__ out',data._outputs,['x','z'],str)
checkArray('DataSet __init__ all',data._orderedVars,['a','b','x','z'],str)
checkNone('DataSet __init__ _data',data._data)
checkNone('DataSet __init__ _collector',data._collector)


######################################
#    SAMPLING AND APPENDING DATA     #
######################################
# append some data to get started
# NOTE histories are currently disabled pending future work
data.addExpectedMeta(['prefix'])
rlz0 = {'a': 1.0,
        'b': 2.0,
        'x': 4.0,
        'prefix': 'first',
       }
rlz1 = {'a' :11.0,
        'b': 12.0,
        'x': 14.0,
        'z': 16.0,
        'prefix': 'second',
       }
rlz2 = {'a' :21.0,
        'b': 22.0,
        'x': 24.0,
        'z': 26.0,
        'prefix': 'third',
       }
formatRealization(rlz0)
formatRealization(rlz1)
formatRealization(rlz2)
# test missing data
checkFails('PointSet addRealization err','Provided realization does not have all requisite values for object \"PointSet\": \"z\"',data.addRealization,args=[rlz0])
rlz0['z'] = 6.0
formatRealization(rlz0)
# test appending
data.addRealization(rlz0)
# get realization by index, from collector
checkRlz('PointSet append 0',data.realization(index=0),rlz0)
# try to access the inaccessible
checkFails('PointSet inaccessible index check','PointSet: Requested index "1" but only have 1 entries (zero-indexed)!',data.realization,kwargs={'index':1})
# add more data
data.addRealization(rlz1)
data.addRealization(rlz2)
# get realization by index
checkRlz('PointSet append 1 idx 0',data.realization(index=0),rlz0)
checkRlz('PointSet append 1 idx 1',data.realization(index=1),rlz1)
checkRlz('PointSet append 1 idx 2',data.realization(index=2),rlz2)
######################################
#      GET MATCHING REALIZATION      #
######################################
m,match = data.realization(matchDict={'a':11.0})
checkSame('PointSet append 1 match index',m,1)
checkRlz('PointSet append 1 match',match,rlz1)
idx,rlz = data.realization(matchDict={'x':1.0})
checkSame('PointSet find bogus match index',idx,3)
checkNone('PointSet find bogus match',rlz)

######################################
#        COLLAPSING DATA SET         #
######################################
# collapse dataset
data.asDataset()
# check sample tag IDs
checkArray('PointSet first collapse sample IDs',data._data['RAVEN_sample_ID'].values,[0,1,2],float)
# check time coordinate
# check values for scalars "a"
checkArray('PointSet first collapse "a"',data._data['a'].values,[1.0,11.0,21.0],float)
checkArray('PointSet first collapse "prefix"',data._data['prefix'].values,['first','second','third'],str)
# TODO test "getting" data from _data instead of _collector

######################################
#     SAMPLING AFTER COLLAPSING      #
######################################
# take new samples to test simultaneous collector and data
rlz3 = {'a': 31.0,
        'b': 32.0,
        'x': 34.0,
        'z': 36.0,
        'prefix': 'fourth',
       }
formatRealization(rlz3)
data.addRealization(rlz3)
checkRlz('PointSet append 2 idx 0',data.realization(index=3),rlz3)
# TODO test reading from both main and collector

data.asDataset()
# check new sample IDs
checkArray('PointSet first collapse sample IDs',data._data['RAVEN_sample_ID'].values,[0,1,2,3],float)
# check new "a"
checkArray('PointSet first collapse "a"',data._data['a'].values,[1.0,11.0,21.0,31.0],float)
# check string prefix
checkArray('PointSet first collapse "prefix"',data._data['prefix'].values,['first','second','third','fourth'],str)

######################################
#         GENERAL META DATA          #
######################################
# add scalar metadata
data.addMeta('TestPP',{'firstVar':{'scalarMetric1':10.0,
                                   'scalarMetric2':'20',
                                   'vectorMetric':{'a':1,'b':'2','c':u'3','d':4.0}
                                   },
                       'secondVar':{'scalarMetric1':100.}
                      })
# directly test contents, without using API
checkSame('Metadata top level entries',len(data._meta),2)
treePP = data._meta['TestPP'].tree.getroot()
checkSame('Metadata TestPP',treePP.tag,'TestPP')
first,second = (c for c in treePP) # TODO always same order?

checkSame('Metadata TestPP/firstVar tag',first.tag,'firstVar')
sm1,vm,sm2 = (c for c in first) # TODO always same order?
checkSame('Metadata TestPP/firstVar/scalarMetric1 tag',sm1.tag,'scalarMetric1')
checkSame('Metadata TestPP/firstVar/scalarMetric1 value',sm1.text,'10.0')
checkSame('Metadata TestPP/firstVar/scalarMetric2 tag',sm2.tag,'scalarMetric2')
checkSame('Metadata TestPP/firstVar/scalarMetric2 value',sm2.text,'20')
checkSame('Metadata TestPP/firstVar/vectorMetric tag',vm.tag,'vectorMetric')
for child in vm:
  if child.tag == 'a':
    checkSame('Metadata TestPP/firstVar/vectorMetric/a value',child.text,'1')
  elif child.tag == 'b':
    checkSame('Metadata TestPP/firstVar/vectorMetric/b value',child.text,'2')
  elif child.tag == 'c':
    checkSame('Metadata TestPP/firstVar/vectorMetric/c value',child.text,'3')
  elif child.tag == 'd':
    checkSame('Metadata TestPP/firstVar/vectorMetric/d value',child.text,'4.0')
  else:
    checkTrue('Unexpected node in TestPP/firstVar/vectorMetric nodes: '+child.text,False)

checkSame('Metadata TestPP/secondVar tag',second.tag,'secondVar')
checkSame('Metadata TestPP/secondVar entries',len(second),1)
child = second[0]
checkSame('Metadata TestPP/secondVar/scalarMetric1 tag',child.tag,'scalarMetric1')
checkSame('Metadata TestPP/secondVar/scalarMetric1 value',child.text,'100.0')

treeDS = data._meta['DataSet'].tree.getroot()
checkSame('Metadata DataSet',treeDS.tag,'DataSet')
checkSame('Metadata DataSet entries',len(treeDS),1) # 2
general = treeDS[:][0]
print('general:',general)
checkSame('Metadata DataSet/general tag',general.tag,'general')
checkSame('Metadata DataSet/general entries',len(general),4)
inputs,pointwise_meta,outputs,sampleTag = general[:]
checkSame('Metadata DataSet/general/inputs tag',inputs.tag,'inputs')
checkSame('Metadata DataSet/general/inputs value',inputs.text,'a,b')#,c')
checkSame('Metadata DataSet/general/outputs tag',outputs.tag,'outputs')
checkSame('Metadata DataSet/general/outputs value',outputs.text,'x,z') # 'y'
checkSame('Metadata DataSet/general/pointwise_meta tag',pointwise_meta.tag,'pointwise_meta')
checkSame('Metadata DataSet/general/pointwise_meta value',pointwise_meta.text,'prefix')
checkSame('Metadata DataSet/general/sampleTag tag',sampleTag.tag,'sampleTag')
checkSame('Metadata DataSet/general/sampleTag value',sampleTag.text,'RAVEN_sample_ID')

# use getters to access contents (using API)
meta = data.getMeta(pointwise=True,general=True)
checkArray('Metadata get keys',sorted(meta.keys()),['DataSet','TestPP','prefix'],str)
# fail to find pointwise in general
checkFails('Metadata get missing general','Some requested keys could not be found in the requested metadata: set([u\'prefix\'])',data.getMeta,kwargs=dict(keys=['prefix'],general=True))
# fail to find general in pointwise
checkFails('Metadata get missing general','Some requested keys could not be found in the requested metadata: set([u\'DataSet\'])',data.getMeta,kwargs=dict(keys=['DataSet'],pointwise=True))
# TODO more value testing, easier "getting" of specific values


######################################
#        READ/WRITE FROM FILE        #
######################################
# to netCDF
# NOTE: due to a cool little seg fault error in netCDF4 versions less than 1.3.1, we cannot test it currently.
# Leaving implementation for the future.
#netname = 'PointSetUnitTest.nc'
#data.write(netname,style='netcdf',format='NETCDF4') # WARNING this will fail if netCDF4 not installed
#checkTrue('Wrote to netcdf',os.path.isfile(netname))
## read fresh from netCDF
#dataNET = PointSet.PointSet()
#dataNET.messageHandler = mh
#dataNET.load(netname,style='netcdf')
# validity of load is checked below, in ACCESS USING GETTERS section

# to CSV
## test writing to file
csvname = 'PointSetUnitTest'
data.write(csvname,style='CSV',**{'what':'a,b,c,x,y,z,RAVEN_sample_ID,prefix'})
## test metadata written
correct = ['<DataObjectMetadata name="PointSet">',
'  <TestPP type="Static">',
'    <firstVar>',
'      <scalarMetric1>10.0</scalarMetric1>',
'      <vectorMetric>',
'        <a>1</a>',
'        <c>3</c>',
'        <b>2</b>',
'        <d>4.0</d>',
'      </vectorMetric>',
'      <scalarMetric2>20</scalarMetric2>',
'    </firstVar>',
'    <secondVar>',
'      <scalarMetric1>100.0</scalarMetric1>',
'    </secondVar>',
'  </TestPP>',
'  ',
'  <DataSet type="Static">',
'    <general>',
'      <inputs>a,b</inputs>',
'      <pointwise_meta>prefix</pointwise_meta>',
'      <outputs>x,z</outputs>',
'      <sampleTag>RAVEN_sample_ID</sampleTag>',
'    </general>',
'  </DataSet>',
'  ',
'</DataObjectMetadata>']
# read in XML
lines = open(csvname+'.xml','r').readlines()
# remove line endings
for l,line in enumerate(lines):
  lines[l] = line.rstrip(os.linesep).rstrip('\n')
# check
checkArray('CSV XML',lines,correct,str)
## read from CSV/XML
### create the data object
xml = createElement('PointSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b'))
xml.append(createElement('Output',text='x,z'))
dataCSV = PointSet.PointSet()
dataCSV.messageHandler = mh
dataCSV._readMoreXML(xml)
### load the data (with both CSV, XML)
dataCSV.load(csvname,style='CSV')
for var in data.getVars():
  if isinstance(data.getVarValues(var).item(0),(float,int)):
    checkTrue('CSV var {}'.format(var),(dataCSV._data[var] - data._data[var]).sum()<1e-20) #necessary due to roundoff
  else:
    checkTrue('CSV var {}'.format(var),bool((dataCSV._data[var] == data._data[var]).prod()))

### try also without the XML file
os.remove(csvname+'.xml')
dataCSV.reset()
dataCSV.load(csvname,style='CSV')
for var in data.getVars():
  if isinstance(data.getVarValues(var).item(0),(float,int)):
    checkTrue('CSV var {}'.format(var),(dataCSV._data[var] - data._data[var]).sum()<1e-20) #necessary due to roundoff
  else:
    checkTrue('CSV var {}'.format(var),bool((dataCSV._data[var] == data._data[var]).prod()))

# clean up temp files
os.remove(csvname+'.csv')

######################################
#        ACCESS USING GETTERS        #
######################################
# test contents of data in parallel (base, netcdf, csv)
# by index
checkRlz('PointSet full origin idx 1',data.realization(index=1),rlz1)
#checkRlz('PointSet full netcdf idx 1',dataNET.realization(index=1),rlz1)
checkRlz('PointSet full csvxml idx 1',dataCSV.realization(index=1),rlz1)
# by match
idx,rlz = data.realization(matchDict={'prefix':'third'})
checkSame('PointSet full origin match idx',idx,2)
checkRlz('PointSet full origin match',rlz,rlz2)
#idx,rlz = dataNET.realization(matchDict={'prefix':'third'})
#checkSame('PointSet full netcdf match idx',idx,2)
#checkRlz('PointSet full netCDF match',rlz,rlz2)
idx,rlz = dataCSV.realization(matchDict={'prefix':'third'})
checkSame('PointSet full csvxml match idx',idx,2)
checkRlz('PointSet full csvxml match',rlz,rlz2)
# TODO metadata checks?

## remove files, for cleanliness (comment out to debug)
#dataNET._data.close()
#os.remove(netname) # if this is a problem because of lazy loading, force dataNET to load completely

######################################
#         SELECTIVE SAMPLING         #
######################################
data = PointSet.PointSet()
xml = createElement('PointSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a'))
xml.append(createElement('Output',text='x'))
data._readMoreXML(xml)
data.messageHandler = mh

rlz0 = {'a': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'x': np.array([1.1, 1.2, 1.3, 1.4, 1.7]),
        'time':np.array([1e-6,2e-6,3e-6,4e-6,5e-6]),
       }
data.setPivotParams(dict((var,['time']) for var in data.vars)) # this would normally be set through the input xml
# sampling the last entry (default)
data.addRealization(rlz0)
checkRlz('PointSet selective default',data.realization(index=0),{'a':0.5,'x':1.7})
# sampling arbitrary row
data.setSelectiveInput('inputRow',2)
data.setSelectiveOutput('outputRow',1)
data.addRealization(rlz0)
checkRlz('PointSet selective default',data.realization(index=1),{'a':0.3,'x':1.2})
# sampling value match
data.setSelectiveInput('inputPivotValue',1.5e-6)
data.setSelectiveOutput('outputPivotValue',4e-6)
data.addRealization(rlz0)
checkRlz('PointSet selective default',data.realization(index=2),{'a':0.2,'x':1.4})
# sampling operator in output, last in input
data._pivotParam = 'time'
data.setSelectiveInput('inputRow',-1)
data.setSelectiveOutput('operator','mean')
data.addRealization(rlz0)
checkRlz('PointSet selective default',data.realization(index=3),{'a':0.5,'x':1.34})


# TODO more exhaustive tests are needed, but this is sufficient for initial work.

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
