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
import xarray as xr
frameworkDir = os.path.dirname(os.path.abspath(os.path.join(sys.argv[0],'..')))

sys.path.append(frameworkDir)
from utils.utils import find_crow

find_crow(frameworkDir)

import XDataSet
import MessageHandler

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})

print('Module undergoing testing:')
print (XDataSet )
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
    @ In, update, bool, optional, if False then don't update results counter
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
    @ In, update, bool, optional, if False then don't update results counter
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
    @ In, update, bool, optional, if False then don't update results counter
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
    @ In, update, bool, optional, if False then don't update results counter
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

def checkRlz(comment,first,second,tol=1e-10,update=True,skip=None):
  """
    This method is aimed to compare two realization
    @ In, comment, string, a comment printed out if it fails
    @ In, first, dict, the first dict, the "calculated" value -> should be as obtained from the data object
    @ In, second, dict, the second dict, the "expected" value -> should be as a realization submitted
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ In, skip, list, optional, keywords not to check
    @ Out, res, bool, True if same
  """
  if skip is None:
    skip = []
  res = True
  if abs(len(first) - len(second)) > len(skip):
    res = False
    print("checking answer",comment,'|','lengths do not match:',len(first),len(second))
  else:
    for key,val in first.items():
      if key in skip:
        continue
      if isinstance(val,float):
        pres = checkFloat('',val,second[key][0],tol,update=False)
      elif isinstance(val,(str,unicode)):
        pres = checkSame('',val,second[key][0],update=False)
      elif isinstance(val,np.ndarray):
        if isinstance(val[0],(float,int)):
          pres = (val - second[key]).sum()<1e-20 #necessary due to roundoff
        else:
          pres = val == second[key]
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
    Checks if entry is None.
    @ In, comment, string, a comment printed out if it fails
    @ In, entry, object, to test if against None
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if None
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
    Checks if expected error occurs
    @ In, comment, string, a comment printed out if it fails
    @ In, errstr, str, expected fail message
    @ In, function, method, method to run to test for failure
    @ In, update, bool, optional, if False then don't update results counter
    @ In, args, list, arguments to pass to function
    @ In, kwargs, dict, keyword arguments to pass to function
    @ Out, res, bool, True if failed as expected
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

######################################
#            CONSTRUCTION            #
######################################
xml = createElement('DataSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b,c'))
xml.append(createElement('Output',text='x,y,z'))
xml.append(createElement('Index',attrib={'var':'time'},text='c,y'))

# check construction
data = XDataSet.DataSet()
# inputs, outputs
checkSame('DataSet __init__ name',data.name,'DataSet')
checkSame('DataSet __init__ print tag',data.printTag,'DataSet')
checkNone('DataSet __init__ _data',data._data)
checkNone('DataSet __init__ _collector',data._collector)

# check initialization
data.messageHandler = mh
data._readMoreXML(xml)
# NOTE histories are currently disabled pending future work (c,y are history vars)
checkArray('DataSet __init__ inp',data._inputs,['a','b','c'],str)
checkArray('DataSet __init__ out',data._outputs,['x','y','z'],str)
checkArray('DataSet __init__ all',data._allvars,['a','b','c','x','y','z'],str)
checkNone('DataSet __init__ _data',data._data)
checkNone('DataSet __init__ _collector',data._collector)


######################################
#    SAMPLING AND APPENDING DATA     #
######################################
# test ND construction
vals = np.array([[1.0,1.1,1.2],[2.0,2.1,2.2]])
right = xr.DataArray(vals,dims=['x','time'],coords={'time':[1e-6,2e-6,3e06],'x':[1e-3,2e-3]})
dims = ['x','time']
coords = {'time':[1e-6,2e-6,3e06],'x':[1e-3,2e-3]}
test = data.constructNDSample(vals,dims,coords)
checkTrue('ND instance construction',test.equals(right))

# append some data to get started
# TODO expand this to ND not just History
data.addExpectedMeta(['prefix'])
rlz0 = {'a': 1.0,
        'b': 2.0,
        'c': np.array([3.0, 3.1, 3.2]),
        'x': 4.0,
        'y': np.array([5.0, 5.1, 5.2]),
        'prefix': 'first',
        'time':np.array([3.1e-6,3.2e-6,3.3e-6]),
       }
rlz1 = {'a' :11.0,
        'b': 12.0,
        'c': [13.0, 13.1, 13.2],
        'x': 14.0,
        'y': [15.0, 15.1, 15.2],
        'z': 16.0,
        'prefix': 'second',
        'time':[13.1e-6,13.2e-6,13.3e-6],
       }
rlz2 = {'a' :21.0,
        'b': 22.0,
        'c': [23.0, 23.1, 23.2],
        'x': 24.0,
        'y': [25.0, 25.1, 25.2],
        'z': 26.0,
        'prefix': 'third',
        'time':[23.1e-6,23.2e-6,23.3e-6],
       }
formatRealization(rlz0)
formatRealization(rlz1)
formatRealization(rlz2)
# test missing data
rlzMissing = dict(rlz0)
rlz0['z'] = 6.0
formatRealization(rlz0)
checkFails('DataSet addRealization err missing','Provided realization does not have all requisite values: \"z\"',data.addRealization,args=[rlzMissing])
# bad formatting
rlzFormat = dict(rlz0)
rlzFormat['c'] = list(rlzFormat['c'])
checkFails('DataSet addRealization err format','Realization was not formatted correctly! See warnings above.',data.addRealization,args=[rlzFormat])
# test appending
data.addRealization(dict(rlz0))


# get realization by index, from collector
checkRlz('Dataset append 0',data.realization(index=0),rlz0,skip=['time'])
# try to access the inaccessible
checkFails('DataSet get nonexistant realization by index','Requested index \"1\" but only have 1 entries (zero-indexed)!',data.realization,kwargs={'index':1})
# add more data
data.addRealization(dict(rlz1))
data.addRealization(dict(rlz2))
# get realization by index
checkRlz('Dataset append 1 idx 0',data.realization(index=0),rlz0,skip=['time'])
checkRlz('Dataset append 1 idx 1',data.realization(index=1),rlz1,skip=['time'])
checkRlz('Dataset append 1 idx 2',data.realization(index=2),rlz2,skip=['time'])
######################################
#      GET MATCHING REALIZATION      #
######################################
m,match = data.realization(matchDict={'a':11.0})
checkSame('Dataset append 1 match index',m,1)
checkRlz('Dataset append 1 match',match,rlz1,skip=['time'])
idx,rlz = data.realization(matchDict={'x':1.0})
checkSame('Dataset find bogus match index',idx,3)
checkNone('Dataset find bogus match',rlz)
# TODO more checks on reading collector, writing to file, etc

######################################
#        COLLAPSING DATA SET         #
######################################
# collapse dataset
data.asDataset()
# check sample tag IDs
checkArray('Dataset first collapse sample IDs',data._data['RAVEN_sample_ID'].values,[0,1,2],float)
# check time coordinate
times = [ 3.1e-6, 3.2e-6, 3.3e-6,
         13.1e-6,13.2e-6,13.3e-6,
         23.1e-6,23.2e-6,23.3e-6]
checkArray('Dataset first collapse "time"',data._data['time'].values,times,float)
# check values for scalars "a"
checkArray('Dataset first collapse "a"',data._data['a'].values,[1.0,11.0,21.0],float)
# check values for timeset "c"
c = np.array(
    [           [ 3.0, 3.1, 3.2]+[np.nan]*6,
     [np.nan]*3+[13.0,13.1,13.2]+[np.nan]*3,
     [np.nan]*6+[23.0,23.1,23.2]           ])
checkArray('Dataset first collapse "c" 0',data._data['c'].values[0],c[0],float)
checkArray('Dataset first collapse "c" 1',data._data['c'].values[1],c[1],float)
checkArray('Dataset first collapse "c" 2',data._data['c'].values[2],c[2],float)
# check values for timeset "y"
y = np.array(
    [           [ 5.0, 5.1, 5.2]+[np.nan]*6,
     [np.nan]*3+[15.0,15.1,15.2]+[np.nan]*3,
     [np.nan]*6+[25.0,25.1,25.2]           ])
checkArray('Dataset first collapse "y" 0',data._data['y'].values[0],y[0],float)
checkArray('Dataset first collapse "y" 1',data._data['y'].values[1],y[1],float)
checkArray('Dataset first collapse "y" 2',data._data['y'].values[2],y[2],float)
# check values for metadata prefix (unicode, not float)
checkArray('Dataset first collapse "prefix"',data._data['prefix'].values,['first','second','third'],str)

# get dimensions
checkSame('Dataset getDimensions "None" num entries',len(data.getDimensions()),7)
checkArray('Dataset getDimensions "None" entry "c"',data.getDimensions()['c'],['time'],str)
checkArray('Dataset getDimensions "c"',data.getDimensions('c')['c'],['time'],str)
checkSame('Dataset getDimensions "inp" num entries',len(data.getDimensions('input')),3)
checkArray('Dataset getDimensions "inp" entry "c"',data.getDimensions('input')['c'],['time'],str)
checkSame('Dataset getDimensions "out" num entries',len(data.getDimensions('output')),3)
checkArray('Dataset getDimensions "out" entry "y"',data.getDimensions('output')['y'],['time'],str)
checkSame('Dataset getDimensions "dummy" num entries',len(data.getDimensions('dummy')),1)
######################################
#     SAMPLING AFTER COLLAPSING      #
######################################
# take a couple new samples to test simultaneous collector and data
# use the same time stamps as rlz0 to test same coords
rlz3 = {'a' :31.0,
        'b': 32.0,
        'c': [33.0, 33.1, 33.2],
        'x': 34.0,
        'y': [35.0, 35.1, 35.2],
        'z': 36.0,
        'prefix': 'fourth',
        'time':[ 3.1e-6, 3.2e-6, 3.3e-6],
       }
formatRealization(rlz3)
data.addRealization(dict(rlz3))
# get new entry (should be in the collector, but we shouldn't care)
checkRlz('Dataset append 2 idx 3',data.realization(index=3),rlz3,skip=['time'])
# make sure old entry is still there
checkRlz('Dataset append 2 idx 1',data.realization(index=1),rlz1,skip=['time'])
# test grabbing negative indices
checkRlz('Dataset append 2 idx -1',data.realization(index=-1),rlz3,skip=['time'])
checkRlz('Dataset append 2 idx -3',data.realization(index=-3),rlz1,skip=['time'])

data.asDataset()
# check new sample IDs
checkArray('Dataset first collapse sample IDs',data._data['RAVEN_sample_ID'].values,[0,1,2,3],float)
# "times" should not have changed
times = [ 3.1e-6, 3.2e-6, 3.3e-6,
         13.1e-6,13.2e-6,13.3e-6,
         23.1e-6,23.2e-6,23.3e-6]
#checkArray('Dataset first collapse "time"',data._data['time'].values,times,float)
# check new "a"
checkArray('Dataset first collapse "a"',data._data['a'].values,[1.0,11.0,21.0,31.0],float)
# check new "c"
c = np.array(
    [           [ 3.0, 3.1, 3.2]+[np.nan]*6,
     [np.nan]*3+[13.0,13.1,13.2]+[np.nan]*3,
     [np.nan]*6+[23.0,23.1,23.2]           ,
                [33.0,33.1,33.2]+[np.nan]*6])
checkArray('Dataset post collapse "c" 0',data._data['c'].values[0],c[0],float)
checkArray('Dataset post collapse "c" 1',data._data['c'].values[1],c[1],float)
checkArray('Dataset post collapse "c" 2',data._data['c'].values[2],c[2],float)
checkArray('Dataset post collapse "c" 3',data._data['c'].values[3],c[3],float)
# check string prefix
checkArray('Dataset post collapse "prefix"',data._data['prefix'].values,['first','second','third','fourth'],str)

# check removing variable from collector and data
rlz4 = {'a' :41.0,
        'b': 42.0,
        'c': [43.0, 43.1, 43.2],
        'x': 44.0,
        'y': [45.0, 45.1, 45.2],
        'z': 46.0,
        'prefix': 'five',
        'time':[ 4.1e-6, 4.2e-6, 4.3e-6],
       }
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
checkSame('Metadata DataSet entries',len(treeDS),2)
dims,general = treeDS[:]
checkSame('Metadata DataSet/dims tag',dims.tag,'dims')
checkSame('Metadata DataSet/dims entries',len(dims),2)
y,c = dims[:]
checkSame('Metadata DataSet/dims/y tag',y.tag,'y')
checkSame('Metadata DataSet/dims/y value',y.text,'time')
checkSame('Metadata DataSet/dims/c tag',c.tag,'c')
checkSame('Metadata DataSet/dims/c value',c.text,'time')
checkSame('Metadata DataSet/general tag',general.tag,'general')
checkSame('Metadata DataSet/general entries',len(general),4)
inputs,pointwise_meta,outputs,sampleTag = general[:]
checkSame('Metadata DataSet/general/inputs tag',inputs.tag,'inputs')
checkSame('Metadata DataSet/general/inputs value',inputs.text,'a,b,c')
checkSame('Metadata DataSet/general/outputs tag',outputs.tag,'outputs')
checkSame('Metadata DataSet/general/outputs value',outputs.text,'x,y,z')
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
# check that poorly-aligned set checks out as such
checkTrue('Check misaligned data is not aligned',not data.checkIndexAlignment())
# check aligned data too
xml = createElement('DataSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a'))
xml.append(createElement('Output',text='b'))
xml.append(createElement('Index',attrib={'var':'t'},text='b'))
dataAlign = XDataSet.DataSet()
dataAlign.messageHandler = mh
dataAlign._readMoreXML(xml)
rlz = {'a':np.array([1.9]),
       'b':np.array([3.4, 2.4, 6.5]),
       't':np.array([0.4, 0.9, 10])}
dataAlign.addRealization(rlz)
rlz = {'a':np.array([7.9]),
       'b':np.array([0.3, -0.8, 9.7]),
       't':np.array([0.4, 0.9, 10])}
dataAlign.addRealization(rlz)
checkTrue('Check aligned data is aligned', dataAlign.checkIndexAlignment('t'))

######################################
#        READ/WRITE FROM FILE        #
######################################
# to netCDF
netname = 'DataSetUnitTest.nc'
data.write(netname,style='netcdf',format='NETCDF4') # WARNING this will fail if netCDF4 not installed
checkTrue('Wrote to netcdf',os.path.isfile(netname))
## read fresh from netCDF
dataNET = XDataSet.DataSet()
dataNET.messageHandler = mh
dataNET.load(netname,style='netcdf')
# validity of load is checked below, in ACCESS USING GETTERS section
## remove files, for cleanliness (comment out to debug)
os.remove(netname) # if this is a problem because of lazy loading, force dataNET to load completely

# to CSV
## test writing to file
csvname = 'DataSetUnitTest'
data.write(csvname,style='CSV',**{'what':'a,b,c,x,y,z,RAVEN_sample_ID,prefix'})
## test metadata written
correct = ['<DataObjectMetadata name="DataSet">',
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
'    <dims>',
'      <y>time</y>',
'      <c>time</c>',
'    </dims>',
'    <general>',
'      <inputs>a,b,c</inputs>',
'      <pointwise_meta>prefix</pointwise_meta>',
'      <outputs>x,y,z</outputs>',
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
dataCSV = XDataSet.DataSet()
dataCSV.messageHandler = mh
dataCSV.load(csvname,style='CSV')
for var in data.getVars():
  if isinstance(data.getVarValues(var).item(0),(float,int)):
    checkTrue('CSV var {}'.format(var),(dataCSV._data[var] - data._data[var]).sum()<1e-20) #necessary due to roundoff
  else:
    checkTrue('CSV var {}'.format(var),bool((dataCSV._data[var] == data._data[var]).prod()))

# clean up temp files
os.remove(csvname+'.csv')
os.remove(csvname+'.xml')


######################################
#        ACCESS USING GETTERS        #
######################################
# test contents of data in parallel
# by index
checkRlz('Dataset full origin idx 1',data.realization(index=1),rlz1,skip=['time'])
checkRlz('Dataset full netcdf idx 1',dataNET.realization(index=1),rlz1,skip=['time'])
checkRlz('Dataset full csvxml idx 1',dataCSV.realization(index=1),rlz1,skip=['time'])
# by match
idx,rlz = data.realization(matchDict={'prefix':'third'})
checkSame('Dataset full origin match idx',idx,2)
checkRlz('Dataset full origin match',rlz,rlz2,skip=['time'])
idx,rlz = dataNET.realization(matchDict={'prefix':'third'})
checkSame('Dataset full netcdf match idx',idx,2)
checkRlz('Dataset full netCDF match',rlz,rlz2,skip=['time'])
idx,rlz = dataCSV.realization(matchDict={'prefix':'third'})
checkSame('Dataset full csvxml match idx',idx,2)
checkRlz('Dataset full csvxml match',rlz,rlz2,skip=['time'])
# TODO metadata checks?


######################################
#        ADDING NEW VARIABLE         #
######################################
f = np.array([9., 19., 29., 39.])
data.addVariable('f',f)
rlzAdd = dict(rlz2)
rlzAdd['f'] = np.atleast_1d(29.)
checkArray('Dataset add variable column',data.asDataset()['f'].values,f,float)
checkRlz('Dataset add variable rlz 2',data.realization(index=2),rlzAdd,skip='time')


######################################
#           SLICE BY INDEX           #
######################################
slices = data.sliceByIndex('time')
checkFloat('Index slicing "time" [2] "time"',slices[2]['time'].item(0),3.3e-6)
checkArray('Index slicing "time" [2] "a"',slices[2]['a'].values,[1.0, 11.0, 21.0, 31.0],float)
checkArray('Index slicing "time" [2] "c"',slices[2]['c'].values,[3.2, np.nan, np.nan, 33.2],float)

slices = data.sliceByIndex('RAVEN_sample_ID')
checkFloat('Index slicing sampleTag [3] sampleTag',slices[3]['RAVEN_sample_ID'].item(0),3)
checkFloat('Index slicing sampleTag [3] "a"',slices[3]['a'].values,31.0,float)
checkArray('Index slicing sampleTag [3] "c"',slices[3]['c'].values,[33.0,33.1,33.2,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],float)

######################################
#        CONSTRUCT FROM DICT         #
######################################
seed = {}
# vector variable, 10 entries with arbitrary lengths
seed['b'] = np.array([ np.array([1.00]),
                       np.array([1.10, 1.11]),
                       np.array([1.20, 1.21, 1.22]),
                       np.array([1.30, 1.31, 1.32, 1.33]),
                       np.array([1.40, 1.41, 1.42, 1.43, 1.44]),
                       np.array([1.50, 1.51, 1.52, 1.53, 1.54, 1.55]),
                       np.array([1.60, 1.61, 1.62, 1.63, 1.64, 1.65, 1.66]),
                       np.array([1.70, 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77]),
                       np.array([1.80, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88]),
                       np.array([1.90, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99])
                       ])
# coordinate, as vector
seed['t'] = np.array([ np.linspace(0,1,1),
                       np.linspace(0,1,2),
                       np.linspace(0,1,3),
                       np.linspace(0,1,4),
                       np.linspace(0,1,5),
                       np.linspace(0,1,6),
                       np.linspace(0,1,7),
                       np.linspace(0,1,8),
                       np.linspace(0,1,9),
                       np.linspace(0,1,10) ])
# set up data object
xml = createElement('DataSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a'))
xml.append(createElement('Output',text='b'))
xml.append(createElement('Index',attrib={'var':'t'},text='b'))
data = XDataSet.DataSet()
data.messageHandler = mh
data._readMoreXML(xml)
# load with insufficient values
checkFails('Load from dict missing variable','Variables are missing from "source" that are required for this data object: set([u\'a\'])',data.load,args=[seed],kwargs=dict(style='dict',dims=data.getDimensions()))
# add a scalar variable, 10 entries
seed['a'] = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
# load properly
data.load(seed,style='dict',dims=data.getDimensions())
# test contents
checkArray('load from dict "a"',data.asDataset()['a'].values,seed['a'],float)
checkArray('load from dict "b"[3]',data.asDataset().isel(True,RAVEN_sample_ID=3)['b'].dropna('t').values,seed['b'][3],float)
rlz = data.realization(index=2)
checkFloat('load from dict rlz 2 "a"',rlz['a'],1.2)
checkArray('load from dict rlz 2 "b"',rlz['b'].values,[1.2,1.21,1.22],float)


# Here I am testing the functionality that converts the dataObject into a dict
convertedDict = data.asDataset(outType='dict')
# check that the dictionary entries are the same
checkArray('asDict "a"',seed['a'],convertedDict['data']['a'],float)
checkArray('asDict "b[0]"',convertedDict['data']['b'][0],seed['b'][0],float)
checkArray('asDict "b[4]"',convertedDict['data']['b'][4],seed['b'][4],float)
checkArray('asDict "b[9]"',convertedDict['data']['b'][9],seed['b'][9],float)
checkArray('asDict "t[0]"',convertedDict['data']['t'][0],seed['t'][0],float)
checkArray('asDict "t[4]"',convertedDict['data']['t'][4],seed['t'][4],float)
checkArray('asDict "t[9]"',convertedDict['data']['t'][9],seed['t'][9],float)
checkSame('asDict dims "a"',convertedDict['dims']['a'],[])
checkSame('asDict dims "b"',convertedDict['dims']['b'],['t'])
# TODO check metadata?
# double-check there's no errors using this to construct a new dataset (full loop)
xml = createElement('DataSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a'))
xml.append(createElement('Output',text='b'))
xml.append(createElement('Index',attrib={'var':'t'},text='b'))
dataRe = XDataSet.DataSet()
dataRe.messageHandler = mh
dataRe._readMoreXML(xml)
dataRe.load(convertedDict['data'],style='dict',dims=convertedDict['dims'])
# use exact same tests as originally loading from dict, but for dataRe
checkArray('load from dict "a"',dataRe.asDataset()['a'].values,seed['a'],float)
checkArray('load from dict "b"[3]',dataRe.asDataset().isel(True,RAVEN_sample_ID=3)['b'].dropna('t').values,seed['b'][3],float)
rlz = dataRe.realization(index=2)
checkFloat('load from dict rlz 2 "a"',rlz['a'],1.2)
checkArray('load from dict rlz 2 "b"',rlz['b'].values,[1.2,1.21,1.22],float)

######################################
#        REMOVING VARIABLES          #
######################################
# first, add a sample so the variable is also in the collector
rlz = {'a':np.atleast_1d(2.0), 'b':np.array([2.0,2.1,2.2]), 't':np.linspace(0,1,3)}
data.addRealization(rlz)
del rlz['b']
del rlz['t']
checkArray('Remove variable starting vars',data.getVars(),['a','b'],str)
data.remove(variable='b')
checkArray('Remove variable remaining vars',data.getVars(),['a'],str)
checkRlz('Remove variable rlz -1',data.realization(index=-1),rlz)
# collapse and re-check
data.asDataset()
checkArray('Remove variable remaining vars',data.getVars(),['a'],str)
checkRlz('Remove variable rlz -1',data.realization(index=-1),rlz)
# check we can add a new realization
data.addRealization({'a':np.array([2.1]), 't':np.array([0])})


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
