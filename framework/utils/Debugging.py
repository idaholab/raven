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
Tools used to debug performance within RAVEN
Not intended to be used during analysis runs.
talbpaul, 2020-11
"""
import sys
from numbers import Number
from collections import Set, Mapping, deque

# these types have no depth, so should not be searched for subitems
zero_depth_bases = (str, bytes, Number, range, bytearray)

def checkSizesWalk(obj, r=0, prename='', tol=1e4):
  """
    Walks through "obj" recursively, identifying the sizes of all its members,
    and printing results for members larger than "tol" in bytes.
    Not sure if this works for swigged objects as expected.
    @ In, obj, object, object to analyze
    @ In, r, int, recursion depth
    @ In, prename, str, chain name of parents of obj
    @ In, tol, float, minimum size of object to enable printing
    @ Out, None
  """
  size = getsize(obj)
  if size < tol:
    return
  new_tol = min(0.5 * size, tol)
  print('  '*r + f'-> {prename} ({type(obj)}): {size:1.1e}' )
  if isinstance(obj, zero_depth_bases):
    pass
  elif isinstance(obj, (tuple, list, Set, deque)):
    for i, k in enumerate(obj):
      checkSizesWalk(k, r+1, f'{prename}[{i}]', tol=new_tol)
  elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
    for k, v in obj.items():
      checkSizesWalk(v, r+1, f'{prename}[{k}]', tol=new_tol)
  if hasattr(obj, '__dict__'):
    for k, v in obj.__dict__.items():
      checkSizesWalk(v, r+1, f'{prename}.{k}', tol=new_tol)
  if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
    for i, s in enumerate(obj.__slots__):
      if hasattr(obj, s):
        checkSizesWalk(getattr(obj, s), r+1, f'{prename}.<{i}>', tol=new_tol)

def getsize(obj_0):
  """
    Recursively iterate to sum size of object & members.
    @ In, obj_0, object, object to recurse through
    @ Out, size, int, size of object in bytes
  """
  _seen_ids = set()
  def inner(obj):
    """
      Evaluate size of object only considering unique members
      @ In, obj, object, object to recurse through
      @ Out, size, int, size of object in bytes
    """
    obj_id = id(obj)
    if obj_id in _seen_ids:
      return 0
    _seen_ids.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, zero_depth_bases):
      pass # bypass remaining control flow and return
    elif isinstance(obj, (tuple, list, Set, deque)):
      size += sum(inner(i) for i in obj)
    elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
      size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
    # Check for custom object instances - may subclass above too
    if hasattr(obj, '__dict__'):
      size += inner(vars(obj))
    if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
      size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
    return size
  return inner(obj_0)
