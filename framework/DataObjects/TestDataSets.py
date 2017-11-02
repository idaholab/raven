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

import XrDataObject # FIXME
import MessageHandler

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})

print('Module undergoing testing:')
print (XrDataObject )
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
        pres = checkSame('',val,second[key],update=False)
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
  res = entry is None
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking answer",comment,'|','"{}" is not None!'.format(entry))
      results["fail"] += 1

def checkFails(comment,errstr,function,update=True,args=None,kwargs=None):
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
xml = createElement('DataSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b'))
xml.append(createElement('Output',text='x,z'))
#xml.append(createElement('Input',text='a,b,c'))
#xml.append(createElement('Output',text='x,y,z'))

# check construction
data = XrDataObject.DataSet()
# inputs, outputs
checkSame('DataSet __init__ name',data.name,'DataSet')
checkSame('DataSet __init__ print tag',data.printTag,'DataSet')
checkNone('DataSet __init__ _data',data._data)
checkNone('DataSet __init__ _collector',data._collector)

# check initialization
data.messageHandler = mh
data._readMoreXML(xml)
# NOTE histories are currently disabled pending future work (c,y are history vars)
checkArray('DataSet __init__ inp',data._inputs,['a','b'],str)
checkArray('DataSet __init__ out',data._outputs,['x','z'],str)
checkArray('DataSet __init__ all',data._allvars,['a','b','x','z'],str)
#checkArray('DataSet __init__ inp',data._inputs,['a','b','c'],str)
#checkArray('DataSet __init__ out',data._outputs,['x','y','z'],str)
#checkArray('DataSet __init__ all',data._allvars,['a','b','c','x','y','z'],str)
checkNone('DataSet __init__ _data',data._data)
checkNone('DataSet __init__ _collector',data._collector)


######################################
#    SAMPLING AND APPENDING DATA     #
######################################
# append some data to get started
# NOTE histories are currently disabled pending future work
rlz0 = {'a': 1.0,
        'b': 2.0,
        #'c': xr.DataArray([3.0, 3.1, 3.2],dims=['time'],coords={'time':[3.1e-6,3.2e-6,3.3e-6]}),
        'x': 4.0,
        #'y': xr.DataArray([5.0, 5.1, 5.2],dims=['time'],coords={'time':[5.1e-6,5.2e-6,5.3e-6]}),
        'prefix': 'first',
       }
rlz1 = {'a' :11.0,
        'b': 12.0,
        #'c': xr.DataArray([13.0, 13.1, 13.2],dims=['time'],coords={'time':[13.1e-6,13.2e-6,13.3e-6]}),
        'x': 14.0,
        #'y': xr.DataArray([15.0, 15.1, 15.2],dims=['time'],coords={'time':[15.1e-6,15.2e-6,15.3e-6]}),
        'z': 16.0,
        'prefix': 'second',
       }
rlz2 = {'a' :21.0,
        'b': 22.0,
        #'c': xr.DataArray([23.0, 23.1, 23.2],dims=['time'],coords={'time':[23.1e-6,23.2e-6,23.3e-6]}),
        'x': 24.0,
        #'y': xr.DataArray([25.0, 25.1, 25.2],dims=['time'],coords={'time':[25.1e-6,25.2e-6,25.3e-6]}),
        'z': 26.0,
        'prefix': 'third',
       }
# test missing data
checkFails('DataSet addRealization err','Provided realization does not have all requisite values: \"z\"',data.addRealization,args=[rlz0])
rlz0['z'] = 6.0
# test appending
data.addRealization(rlz0)
# get realization by index, from collector
checkRlz('Dataset append 0',data.realization(index=0,readCollector=True),rlz0)
# try to access the inaccessible
try:
  data.realization(index=1,readCollector=True)
  print('Checking error Dataset append 0 | Accessed inaccessible index!')
  results['fail'] += 1
except AssertionError:
  results['pass'] += 1
# add more data
data.addRealization(rlz1)
data.addRealization(rlz2)
# get realization by index
checkRlz('Dataset append 1 idx 0',data.realization(index=0,readCollector=True),rlz0)
checkRlz('Dataset append 1 idx 1',data.realization(index=1,readCollector=True),rlz1)
checkRlz('Dataset append 1 idx 2',data.realization(index=2,readCollector=True),rlz2)
######################################
#      GET MATCHING REALIZATION      #
######################################
m,match = data.realization(matchDict={'a':11.0},readCollector=True)
checkSame('Dataset append 1 match index',m,1)
checkRlz('Dataset append 1 match',match,rlz1)
idx,rlz = data.realization(matchDict={'x':1.0},readCollector=True)
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
times = [ 3.1e-6, 3.2e-6, 3.3e-6, 5.1e-6, 5.2e-6, 5.3e-6,
         13.1e-6,13.2e-6,13.3e-6,15.1e-6,15.2e-6,15.3e-6,
         23.1e-6,23.2e-6,23.3e-6,25.1e-6,25.2e-6,25.3e-6]
#checkArray('Dataset first collapse "time"',data._data['time'].values,times,float)
# check values for scalars "a"
checkArray('Dataset first collapse "a"',data._data['a'].values,[1.0,11.0,21.0],float)
# check values for timeset "c"
#c = np.array(
#    [            [ 3.0, 3.1, 3.2]+[np.nan]*15,
#     [np.nan]* 6+[13.0,13.1,13.2]+[np.nan]*9,
#     [np.nan]*12+[23.0,23.1,23.2]+[np.nan]*3])
#checkArray('Dataset first collapse "c" 0',data._data['c'].values[0],c[0],float)
#checkArray('Dataset first collapse "c" 1',data._data['c'].values[1],c[1],float)
#checkArray('Dataset first collapse "c" 2',data._data['c'].values[2],c[2],float)
# check values for timeset "y"
#y = np.array(
#    [[np.nan]* 3+[ 5.0, 5.1, 5.2]+[np.nan]*12,
#     [np.nan]* 9+[15.0,15.1,15.2]+[np.nan]*6,
#     [np.nan]*15+[25.0,25.1,25.2]           ])
#checkArray('Dataset first collapse "y" 0',data._data['y'].values[0],y[0],float)
#checkArray('Dataset first collapse "y" 1',data._data['y'].values[1],y[1],float)
#checkArray('Dataset first collapse "y" 2',data._data['y'].values[2],y[2],float)
# check values for metadata prefix (unicode, not float)
checkArray('Dataset first collapse "prefix"',data._data['prefix'].values,['first','second','third'],str)
# TODO test "getting" data from _data instead of _collector

######################################
#     SAMPLING AFTER COLLAPSING      #
######################################
# take a couple new samples to test simultaneous collector and data
# use the same time stamps as rlz0 to test same coords
rlz3 = {'a' :31.0,
        'b': 32.0,
#        'c': xr.DataArray([33.0, 33.1, 33.2],dims=['time'],coords={'time':[ 3.1e-6, 3.2e-6, 3.3e-6]}),
        'x': 34.0,
#        'y': xr.DataArray([35.0, 35.1, 35.2],dims=['time'],coords={'time':[ 5.1e-6, 5.2e-6, 5.3e-6]}),
        'z': 36.0,
        'prefix': 'fourth',
       }
data.addRealization(rlz3)
checkRlz('Dataset append 2 idx 0',data.realization(index=0,readCollector=True),rlz3)
# TODO test reading from both main and collector

data.asDataset()
# check new sample IDs
checkArray('Dataset first collapse sample IDs',data._data['RAVEN_sample_ID'].values,[0,1,2,3],float)
# "times" should not have changed
#times = [ 3.1e-6, 3.2e-6, 3.3e-6, 5.1e-6, 5.2e-6, 5.3e-6,
#         13.1e-6,13.2e-6,13.3e-6,15.1e-6,15.2e-6,15.3e-6,
#         23.1e-6,23.2e-6,23.3e-6,25.1e-6,25.2e-6,25.3e-6]
#checkArray('Dataset first collapse "time"',data._data['time'].values,times,float)
# check new "a"
checkArray('Dataset first collapse "a"',data._data['a'].values,[1.0,11.0,21.0,31.0],float)
# check new "c"
#c = np.array(
#    [            [ 3.0, 3.1, 3.2]+[np.nan]*15,
#     [np.nan]* 6+[13.0,13.1,13.2]+[np.nan]*9,
#     [np.nan]*12+[23.0,23.1,23.2]+[np.nan]*3,
#                 [33.0,33.1,33.2]+[np.nan]*15])
#checkArray('Dataset first collapse "c" 0',data._data['c'].values[0],c[0],float)
#checkArray('Dataset first collapse "c" 1',data._data['c'].values[1],c[1],float)
#checkArray('Dataset first collapse "c" 2',data._data['c'].values[2],c[2],float)
#checkArray('Dataset first collapse "c" 3',data._data['c'].values[3],c[3],float)
# check string prefix
checkArray('Dataset first collapse "prefix"',data._data['prefix'].values,['first','second','third','fourth'],str)


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
#dims,general = treeDS[:]
#checkSame('Metadata DataSet/dims tag',dims.tag,'dims')
#checkSame('Metadata DataSet/dims entries',len(dims),2)
#y,c = dims[:]
#checkSame('Metadata DataSet/dims/y tag',y.tag,'y')
#checkSame('Metadata DataSet/dims/y value',y.text,'time')
#checkSame('Metadata DataSet/dims/c tag',c.tag,'c')
#checkSame('Metadata DataSet/dims/c value',c.text,'time')
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
netname = 'DataSetUnitTest.nc'
data.write(netname,style='netcdf',format='NETCDF4') # WARNING this will fail if netCDF4 not installed
checkTrue('Wrote to netcdf',os.path.isfile(netname))
## read fresh from netCDF
dataNET = XrDataObject.DataSet()
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
'    <general>',
'      <inputs>a,b</inputs>',
'      <pointwise_meta>prefix</pointwise_meta>',
'      <outputs>x,z</outputs>',
'      <sampleTag>RAVEN_sample_ID</sampleTag>',
'    </general>',
'  </DataSet>',
'  ',
'</DataObjectMetadata>']
#'    <dims>',
#'      <y>time</y>',
#'      <c>time</c>',
#'    </dims>',
# read in XML
lines = file(csvname+'.xml','r').readlines()
# remove line endings
for l,line in enumerate(lines):
  lines[l] = line.rstrip(os.linesep).rstrip('\n')
# check
checkArray('CSV XML',lines,correct,str)
## read from CSV/XML
dataCSV = XrDataObject.DataSet()
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
checkRlz('Dataset full origin idx 1',data.realization(index=1),rlz1)
checkRlz('Dataset full netcdf idx 1',dataNET.realization(index=1),rlz1)
checkRlz('Dataset full csvxml idx 1',dataCSV.realization(index=1),rlz1)
# by match
idx,rlz = data.realization(matchDict={'prefix':'third'})
checkSame('Dataset full origin match idx',idx,2)
checkRlz('Dataset full origin match',rlz,rlz2)
idx,rlz = dataNET.realization(matchDict={'prefix':'third'})
checkSame('Dataset full netcdf match idx',idx,2)
checkRlz('Dataset full netCDF match',rlz,rlz2)
idx,rlz = dataCSV.realization(matchDict={'prefix':'third'})
checkSame('Dataset full csvxml match idx',idx,2)
checkRlz('Dataset full csvxml match',rlz,rlz2)
# TODO metadata checks?


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
