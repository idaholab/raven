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
  This Module performs Unit Tests for the Realization objects.
"""

import os, sys

# find location of crow, message handler
ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(ravenDir)

from ravenframework import Realizations

print('Module undergoing testing:')
print(Realizations.Realization)
print('')

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
      elif dtype.__name__ in ('str','unicode'):
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
      elif type(val).__name__ in ('str','unicode','str_','unicode_'):
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



#######
#
# Quacks like a dict
#
rlz = Realizations.Realization()

# setitem
a = 3
b = 42
pi = 3.14159
rlz['a'] = a
rlz['b'] = b
rlz['pi'] = pi
rlz[5] = 'c'


# membership, contains
for member in ['a', 'b', 'pi', 5]:
  if not member in rlz:
    print(f'checking member "{member}", got False expected True!')
    results['fail'] += 1
  else:
    results['pass'] += 1

for nonmember in ['d', 2, 1.618, 'values']:
  if nonmember in rlz:
    print(f'checking member "{nonmember}", got True expected False!')
    results['fail'] += 1
  else:
    results['pass'] += 1


# getitem
gb = rlz['b']
if not gb == 42:
  print(f'checking getitem "b", got "{gb}" expected "{b}"!')
  results['fail'] += 1
else:
  results['pass'] += 1

ga = rlz['a']
if not ga == 3:
  print(f'checking getitem "a", got "{ga}" expected "{a}"!')
  results['fail'] += 1
else:
  results['pass'] += 1

gp = rlz['pi']
if not gp == 3.14159:
  print(f'checking getitem "pi", got "{gp}" expected "{pi}"!')
  results['fail'] += 1
else:
  results['pass'] += 1

g5 = rlz[5]
if not g5 == 'c':
  print(f'checking getitem "pi", got "{g5}" expected "c"!')
  results['fail'] += 1
else:
  results['pass'] += 1


# get
ga = rlz.get('a')
if not ga == 3:
  print(f'checking get "a", got "{ga}" expected "{a}"!')
  results['fail'] += 1
else:
  results['pass'] += 1

gd = rlz.get('d', 15)
if not gd == 15:
  print(f'checking get default, got "{gd}" expected "{15}"!')
  results['fail'] += 1
else:
  results['pass'] += 1


# len
if not len(rlz) == 4:
  print(f'checking len, got "{len(rlz)}" expected "{3}"!')
  results['fail'] += 1
else:
  results['pass'] += 1


# delitem
del rlz['b']
if 'b' in rlz:
  print('checking del, failed to remove "b"!')
  results['fail'] += 1
else:
  results['pass'] += 1


# iter
expk = ['a', 'pi', 5]
for i, k in enumerate(rlz):
  if k != expk[i]:
    print(f'checking iter[{i}], got "{k}" expected "{expk[i]}"!')
    results['fail'] += 1
  else:
    results['pass'] += 1


# keys
for i, k in enumerate(rlz.keys()):
  if k != expk[i]:
    print(f'checking keys[{i}], got "{k}" expected "{expk[i]}"!')
    results['fail'] += 1
  else:
    results['pass'] += 1


# values
expv = [3, 3.14159, 'c']
for i, v in enumerate(rlz.values()):
  if v != expv[i]:
    print(f'checking values[{i}], got "{v}" expected "{expv[i]}"!')
    results['fail'] += 1
  else:
    results['pass'] += 1


# items
for i, (k, v) in enumerate(rlz.items()):
  if (k != expk[i]) or (v != expv[i]):
    print(f'checking items[{i}], got "({k}, {v})" expected ("{expk[i]}, {expv[i]}")!')
    results['fail'] += 1
  else:
    results['pass'] += 1


# update
new = {'a': 30,    # update old entry
       'b': 420,   # add back old entry in new position
       5: 'c2',     # update old entry
       'new': 372} # new entry
rlz.update(new)
expk = ['a',    'pi',    5, 'b', 'new']
expv = [ 30, 3.14159, 'c2', 420,   372]
for i, (k, v) in enumerate(rlz.items()):
  if (k != expk[i]) or (v != expv[i]):
    print(f'checking update[{i}], got "({k}, {v})" expected ("{expk[i]}, {expv[i]}")!')
    results['fail'] += 1
  else:
    results['pass'] += 1

# pop
val = rlz.pop(5)
if val != 'c2':
  print(f'checking pop[5], got "{val}" expected "c")!')
  results['fail'] += 1
else:
  results['pass'] += 1
if 5 in rlz:
  print('checking pop[5], failed to remove 5!')
  results['fail'] += 1
else:
  results['pass'] += 1



#######
#
# Results
#

print(results)


sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.test_realization</name>
    <author>talbpaul</author>
    <created>2024-10-23</created>
    <classesTested>Realization</classesTested>
    <description>
       This test is a Unit Test for the Realization class.
    </description>
  </TestInfo>
"""
