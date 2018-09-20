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
  This Module performs Unit Tests for the HistorySet data objects.
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

# find location of crow, message handler
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4+['framework'])))
sys.path.append(frameworkDir)

from utils.utils import find_crow
find_crow(frameworkDir)
import MessageHandler

# find location of data objects
sys.path.append(os.path.join(frameworkDir,'DataObjects'))

import HistorySet

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})

print('Module undergoing testing:')
print(HistorySet )
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
      if isinstance(val,(float,int)):
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
xml = createElement('HistorySet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b'))
xml.append(createElement('Output',text='x,y'))
options = createElement('options')
options.append(createElement('pivotParameter',text='Timelike'))
xml.append(options)

# check construction
data = HistorySet.HistorySet()
# inputs, outputs
checkSame('HistorySet __init__ name',data.name,'HistorySet')
checkSame('HistorySet __init__ print tag',data.printTag,'HistorySet')
checkNone('HistorySet __init__ _data',data._data)
checkNone('HistorySet __init__ _collector',data._collector)

# check initialization
data.messageHandler = mh
data._readMoreXML(xml)
# NOTE histories are currently disabled pending future work (c,y are history vars)
checkArray('HistorySet __init__ inp',data._inputs,['a','b'],str)
checkArray('HistorySet __init__ out',data._outputs,['x','y'],str)
checkArray('HistorySet __init__ all',data._orderedVars,['a','b','x','y'],str)
checkNone('HistorySet __init__ _data',data._data)
checkNone('HistorySet __init__ _collector',data._collector)


######################################
#    SAMPLING AND APPENDING DATA     #
######################################
# append some data to get started
data.addExpectedMeta(['prefix'])
rlz0 = {'a': 1.0,
        'b': 2.0,
        'y': [5.0, 5.1, 5.2],
        'prefix': 'first',
        'Timelike':[3.1e-6,3.2e-6,3.3e-6],
       }
rlz1 = {'a' :11.0,
        'b': 12.0,
        'x': [14.0, 14.1, 14.2],
        'y': [15.0, 15.1, 15.2],
        'prefix': 'second',
        'Timelike':[13.1e-6,13.2e-6,13.3e-6],
       }
rlz2 = {'a' :21.0,
        'b': 22.0,
        'x': [24.0, 24.1, 24.2],
        'y': [25.0, 25.1, 25.2],
        'prefix': 'third',
        'Timelike':[23.1e-6,23.2e-6,23.3e-6],
       }
# NOTE: raven only takes np.array as entries right now, so format will convert floats and lists to np.array
formatRealization(rlz0)
formatRealization(rlz1)
formatRealization(rlz2)
# test missing data
rlzMissing = dict(rlz0)
rlz0['x'] = [4.0, 4.1, 4.2]
formatRealization(rlz0)
checkFails('HistorySet addRealization err missing','Provided realization does not have all requisite values for object \"HistorySet\": \"x\"',data.addRealization,args=[rlzMissing])
# bad formatting
rlzFormat = dict(rlz0)
rlzFormat['b'] = list(rlzFormat['b'])
checkFails('HistorySet addRealization err format','Realization was not formatted correctly for \"HistorySet\"! See warnings above.',data.addRealization,args=[rlzFormat])
# test appending
data.addRealization(dict(rlz0))


# get realization by index, from collector
checkRlz('HistorySet append 0',data.realization(index=0),rlz0,skip=['Timelike'])
# try to access the inaccessible
checkFails('HistorySet get nonexistant realization by index','HistorySet: Requested index "1" but only have 1 entries (zero-indexed)!',data.realization,kwargs={'index':1})
# add more data
data.addRealization(dict(rlz1))
data.addRealization(dict(rlz2))
# get realization by index
checkRlz('HistorySet append 1 idx 0',data.realization(index=0),rlz0,skip=['Timelike'])
checkRlz('HistorySet append 1 idx 1',data.realization(index=1),rlz1,skip=['Timelike'])
checkRlz('HistorySet append 1 idx 2',data.realization(index=2),rlz2,skip=['Timelike'])

######################################
#      GET MATCHING REALIZATION      #
######################################
m,match = data.realization(matchDict={'a':11.0})
checkSame('HistorySet append 1 match index',m,1)
checkRlz('HistorySet append 1 match',match,rlz1,skip=['Timelike'])
idx,rlz = data.realization(matchDict={'b':0.0})
checkSame('HistorySet find bogus match index',idx,3)
checkNone('HistorySet find bogus match',rlz)

######################################
#        COLLAPSING DATA SET         #
######################################
# collapse dataset
data.asDataset()
# check sample tag IDs
checkArray('HistorySet first collapse sample IDs',data._data['RAVEN_sample_ID'].values,[0,1,2],float)
# check time coordinate
times = [ 3.1e-6, 3.2e-6, 3.3e-6,
         13.1e-6,13.2e-6,13.3e-6,
         23.1e-6,23.2e-6,23.3e-6]
checkArray('HistorySet first collapse "Timelike"',data._data['Timelike'].values,times,float)
# check values for scalars "a"
checkArray('HistorySet first collapse "a"',data._data['a'].values,[1.0,11.0,21.0],float)
# check values for timeset "c"
x = np.array(
    [           [ 4.0, 4.1, 4.2]+[np.nan]*6,
     [np.nan]*3+[14.0,14.1,14.2]+[np.nan]*3,
     [np.nan]*6+[24.0,24.1,24.2]           ])
checkArray('HistorySet first collapse "x" 0',data._data['x'].values[0],x[0],float)
checkArray('HistorySet first collapse "x" 1',data._data['x'].values[1],x[1],float)
checkArray('HistorySet first collapse "x" 2',data._data['x'].values[2],x[2],float)
# check values for timeset "y"
y = np.array(
    [           [ 5.0, 5.1, 5.2]+[np.nan]*6,
     [np.nan]*3+[15.0,15.1,15.2]+[np.nan]*3,
     [np.nan]*6+[25.0,25.1,25.2]           ])
checkArray('HistorySet first collapse "y" 0',data._data['y'].values[0],y[0],float)
checkArray('HistorySet first collapse "y" 1',data._data['y'].values[1],y[1],float)
checkArray('HistorySet first collapse "y" 2',data._data['y'].values[2],y[2],float)
# check values for metadata prefix (unicode, not float)
checkArray('HistorySet first collapse "prefix"',data._data['prefix'].values,['first','second','third'],str)
# TODO test "getting" data from _data instead of _collector
######################################
#     SAMPLING AFTER COLLAPSING      #
######################################
# take a couple new samples to test simultaneous collector and data
# use the same time stamps as rlz0 to test same coords
rlz3 = {'a' :31.0,
        'b': 32.0,
        'x': [34.0, 34.1, 34.2],
        'y': [35.0, 35.1, 35.2],
        'prefix': 'fourth',
        'Timelike':[ 3.1e-6, 3.2e-6, 3.3e-6],
       }
formatRealization(rlz3)
data.addRealization(dict(rlz3))
# get new entry (should be in the collector, but we shouldn't care)
checkRlz('HistorySet append 2 idx 3',data.realization(index=3),rlz3,skip=['Timelike'])
# make sure old entry is still there
checkRlz('HistorySet append 2 idx 1',data.realization(index=1),rlz1,skip=['Timelike'])
# test grabbing negative indices
checkRlz('HistorySet append 2 idx -1',data.realization(index=-1),rlz3,skip=['Timelike'])
checkRlz('HistorySet append 2 idx -3',data.realization(index=-3),rlz1,skip=['Timelike'])

data.asDataset()
# check new sample IDs
checkArray('HistorySet first collapse sample IDs',data._data['RAVEN_sample_ID'].values,[0,1,2,3],float)
# "times" should not have changed
times = [ 3.1e-6, 3.2e-6, 3.3e-6,
         13.1e-6,13.2e-6,13.3e-6,
         23.1e-6,23.2e-6,23.3e-6]
checkArray('Dataset first collapse "Timelike"',data._data['Timelike'].values,times,float)
# check new "a"
checkArray('HistorySet first collapse "a"',data._data['a'].values,[1.0,11.0,21.0,31.0],float)
# check new "c"
x = np.array(
    [           [ 4.0, 4.1, 4.2]+[np.nan]*6,
     [np.nan]*3+[14.0,14.1,14.2]+[np.nan]*3,
     [np.nan]*6+[24.0,24.1,24.2]           ,
                [34.0,34.1,34.2]+[np.nan]*6])
checkArray('HistorySet post collapse "x" 0',data._data['x'].values[0],x[0],float)
checkArray('HistorySet post collapse "x" 1',data._data['x'].values[1],x[1],float)
checkArray('HistorySet post collapse "x" 2',data._data['x'].values[2],x[2],float)
checkArray('HistorySet post collapse "x" 3',data._data['x'].values[3],x[3],float)
# check string prefix
checkArray('HistorySet post collapse "prefix"',data._data['prefix'].values,['first','second','third','fourth'],str)


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
checkSame('Metadata HistorySet',treeDS.tag,'DataSet')
checkSame('Metadata HistorySet entries',len(treeDS),2)
dims,general = treeDS[:]
checkSame('Metadata HistorySet/dims tag',dims.tag,'dims')
checkSame('Metadata HistorySet/dims entries',len(dims),2)
y,x = dims[:]
checkSame('Metadata HistorySet/dims/x tag',x.tag,'x')
checkSame('Metadata HistorySet/dims/x value',x.text,'Timelike')
checkSame('Metadata HistorySet/dims/y tag',y.tag,'y')
checkSame('Metadata HistorySet/dims/y value',y.text,'Timelike')
checkSame('Metadata HistorySet/general tag',general.tag,'general')
checkSame('Metadata HistorySet/general entries',len(general),4)
inputs,pointwise_meta,outputs,sampleTag = general[:]
checkSame('Metadata HistorySet/general/inputs tag',inputs.tag,'inputs')
checkSame('Metadata HistorySet/general/inputs value',inputs.text,'a,b')
checkSame('Metadata HistorySet/general/outputs tag',outputs.tag,'outputs')
checkSame('Metadata HistorySet/general/outputs value',outputs.text,'x,y')
checkSame('Metadata HistorySet/general/pointwise_meta tag',pointwise_meta.tag,'pointwise_meta')
checkSame('Metadata HistorySet/general/pointwise_meta value',pointwise_meta.text,'prefix')
checkSame('Metadata HistorySet/general/sampleTag tag',sampleTag.tag,'sampleTag')
checkSame('Metadata HistorySet/general/sampleTag value',sampleTag.text,'RAVEN_sample_ID')

# use getters to access contents (using API)
meta = data.getMeta(pointwise=True,general=True)
checkArray('Metadata get keys',sorted(meta.keys()),['DataSet','TestPP','prefix'],str)
# fail to find pointwise in general
checkFails('Metadata get missing general','Some requested keys could not be found in the requested metadata: set([u\'prefix\'])',data.getMeta,kwargs=dict(keys=['prefix'],general=True))
# fail to find general in pointwise
checkFails('Metadata get missing general','Some requested keys could not be found in the requested metadata: set([u\'HistorySet\'])',data.getMeta,kwargs=dict(keys=['HistorySet'],pointwise=True))
# TODO more value testing, easier "getting" of specific values


######################################
#        READ/WRITE FROM FILE        #
######################################
# to netCDF
# NOTE: due to a cool little seg fault error in netCDF4 versions less than 1.3.1, we cannot test it currently.
# Leaving implementation for the future.
#netname = 'HistorySetUnitTest.nc'
#data.write(netname,style='netcdf',format='NETCDF4') # WARNING this will fail if netCDF4 not installed
#checkTrue('Wrote to netcdf',os.path.isfile(netname))
## read fresh from netCDF
#dataNET = HistorySet.HistorySet()
#dataNET.messageHandler = mh
#dataNET.load(netname,style='netcdf')
# validity of load is checked below, in ACCESS USING GETTERS section

# to CSV
## test writing to file
csvname = 'HistorySetUnitTest'
data.write(csvname,style='CSV',**{'what':'a,b,c,x,y,z,RAVEN_sample_ID,prefix'})
## test metadata written
correct = ['<DataObjectMetadata name="HistorySet">',
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
'      <y>Timelike</y>',
'      <x>Timelike</x>',
'    </dims>',
'    <general>',
'      <inputs>a,b</inputs>',
'      <pointwise_meta>prefix</pointwise_meta>',
'      <outputs>x,y</outputs>',
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
xml = createElement('HistorySet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b'))
xml.append(createElement('Output',text='x,y'))
options = createElement('options')
options.append(createElement('pivotParameter',text='Timelike'))
xml.append(options)
dataCSV = HistorySet.HistorySet()
dataCSV.messageHandler = mh
dataCSV._readMoreXML(xml)
### load the data (with both CSV, XML)
dataCSV.load(csvname,style='CSV')
for var in data.getVars():
  if isinstance(data.getVarValues(var).item(0),(float,int)):
    checkTrue('CSV var {}'.format(var),(dataCSV._data[var] - data._data[var]).sum()<1e-20) #necessary due to roundoff
  else:
    checkTrue('CSV var {}'.format(var),bool((dataCSV._data[var] == data._data[var]).prod()))

### also try without the XML metadata file, just the CSVs
# get rid of the xml file
os.remove(csvname+'.xml')
dataCSV.reset()
dataCSV.load(csvname,style='CSV')
for var in data.getVars():
  if isinstance(data.getVarValues(var).item(0),(float,int)):
    checkTrue('CSV var {}'.format(var),(dataCSV._data[var] - data._data[var]).sum()<1e-20) #necessary due to roundoff
  else:
    checkTrue('CSV var {}'.format(var),bool((dataCSV._data[var] == data._data[var]).prod()))

# clean up remaining temp files
os.remove(csvname+'.csv')
os.remove(csvname+'_0.csv')
os.remove(csvname+'_1.csv')
os.remove(csvname+'_2.csv')
os.remove(csvname+'_3.csv')


######################################
#        ACCESS USING GETTERS        #
######################################
# test contents of data in parallel
# by index
checkRlz('HistorySet full origin idx 1',data.realization(index=1),rlz1,skip=['Timelike'])
#checkRlz('HistorySet full netcdf idx 1',dataNET.realization(index=1),rlz1,skip=['Timelike'])
checkRlz('HistorySet full csvxml idx 1',dataCSV.realization(index=1),rlz1,skip=['Timelike'])
# by match
idx,rlz = data.realization(matchDict={'prefix':'third'})
checkSame('HistorySet full origin match idx',idx,2)
checkRlz('HistorySet full origin match',rlz,rlz2,skip=['Timelike'])
#idx,rlz = dataNET.realization(matchDict={'prefix':'third'})
#checkSame('HistorySet full netcdf match idx',idx,2)
#checkRlz('HistorySet full netCDF match',rlz,rlz2,skip=['Timelike'])
idx,rlz = dataCSV.realization(matchDict={'prefix':'third'})
checkSame('HistorySet full csvxml match idx',idx,2)
checkRlz('HistorySet full csvxml match',rlz,rlz2,skip=['Timelike'])
# TODO metadata checks?

## remove files, for cleanliness (comment out to debug)
#dataNET._data.close()
#os.remove(netname) # if this is a problem because of lazy loading, force dataNET to load completely

######################################
#        NO INPUT SPACE CASE         #
######################################
# if there are no scalar elements in the dataobject,
# there used to be a bug in addRealization that resuted
# in the wrong shape for adding rows to the collector.
# This section against that bug.
xml = createElement('HistorySet',attrib={'name':'test'})
xml.append(createElement('Output',text='x,y'))
options = createElement('options')
options.append(createElement('pivotParameter',text='Timelike'))
xml.append(options)

data = HistorySet.HistorySet()
data.messageHandler = mh
data._readMoreXML(xml)
rlz = {'x': np.array([1, 2, 3]),
       'y': np.array([4, 5, 6]),
       'Timelike': np.array([0.1, 0.2, 0.3])}
data.addRealization(rlz)
# check contents
rlz0 = data.realization(index=0)
checkRlz('No input space',rlz0,rlz,skip='Timelike')



######################################
#           ASYNC HISTORIES          #
######################################
xml = createElement('HistorySet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b'))
xml.append(createElement('Output',text='x,y'))
data = HistorySet.HistorySet()
data.messageHandler = mh
data._readMoreXML(xml)
rlz1 = {'a': np.array([1.0]),
        'b': np.array([2.0]),
        'x': np.array([1.0, 2.0, 3.0]),
        'y': np.array([6.0, 7.0, 8.0]),
        'time': np.array([0.0, 0.1, 0.2])}

rlz2 = {'a': np.array([11.0]),
        'b': np.array([12.0]),
        'x': np.array([11.0, 12.0]),
        'y': np.array([16.0, 17.0]),
        'time': np.array([0.05, 0.15])}

# ADD when EnsembleModel for times series is done
#rlzFoulty = {'a': np.array([11.0]),
#             'b': np.array([12.0]),
#             'x': np.array([11.0]),
#             'y': np.array([16.0]),
#             'time': np.array([0.05, 0.15])}

data.addRealization(rlz1)
data.addRealization(rlz2)
# check collection in realizations, in collector
checkRlz('Adding asynchronous histories, collector[0]',data.realization(index=0),rlz1,skip=['time'])
checkRlz('Adding asynchronous histories, collector[1]',data.realization(index=1),rlz2,skip=['time'])
# check stored in collector, not in synced histories
idx = data._orderedVars.index('time')
times = data._collector[:,idx]
checkArray('Asynchronous histories, collector, time[0]',times[0],rlz1['time'],float)
checkArray('Asynchronous histories, collector, time[1]',times[1],rlz2['time'],float)
# check as dataset, just for kicks
data.asDataset()
checkRlz('Adding asynchronous histories, dataset[0]',data.realization(index=0),rlz1,skip=['time'])
checkRlz('Adding asynchronous histories, dataset[1]',data.realization(index=1),rlz2,skip=['time'])
# ADD when EnsembleModel for times series is done
# check expected error in case index and index-dependent variable have different shape
#data = HistorySet.HistorySet()
#data.messageHandler = mh
#data._readMoreXML(xml)
#checkFails('Expected error foulty realization (index/variable no matching shape), rlzFoulty', "SyntaxError: Realization was not formatted correctly", data.addRealization, args=(rlzFoulty,))

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.test_datasets</name>
    <author>talbpaul</author>
    <created>2017-10-20</created>
    <classesTested>HistorySet</classesTested>
    <description>
       This test is a Unit Test for the HistorySet classes.
    </description>
  </TestInfo>
"""

