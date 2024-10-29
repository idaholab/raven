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
  print(f'checking getitem "5", got "{g5}" expected "c"!')
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
