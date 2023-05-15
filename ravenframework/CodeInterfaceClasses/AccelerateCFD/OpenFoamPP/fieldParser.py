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
Created on August 31, 2020

@author: Andrea Alfonsi

comments: parser for field data of open foam outputfile
"""
import os
import struct
import numpy as np

def getFileContent(fn):
  """
    get the content from the file
    @ In, fn, str, file name
    @ Out, getFileContent, None or Str, the content of the file, None if the file does not exist
  """
  content = None
  if not os.path.exists(fn):
    print("Can not open file " + fn)
  else:
    with open(fn, "rb") as f:
      content = f.readlines()
  return content

def parseFieldAll(fn):
  """
    parse internal field, extract data to numpy.array
    @ In, fn, str, file name
    @ Out, parseFieldAll, np.array, numpy array of internal field and boundary
  """
  content = getFileContent(fn)
  if content is not None:
    return parseInternalFieldContent(content), parseBoundaryContent(content)
  else:
    return None

def parseInternalField(fn):
  """
    parse internal field, extract data to numpy.array
    @ In, fn, str, file name
    @ Out, parseInternalField, np.array, numpy array of internal field
  """
  content = getFileContent(fn)
  if content is not None:
    return parseInternalFieldContent(content)
  else:
    return None

def parseInternalFieldContent(content):
  """
    parse internal field from content
    @ In, content, list, contents of lines
    @ Out, parseInternalFieldContent, np.array, numpy array of internal field
  """
  isBinary = isBinaryFormat(content)
  for ln, lc in enumerate(content):
    if lc.startswith(b'internalField'):
      if b'nonuniform' in lc:
        return parseDataNonuniform(content, ln, len(content), isBinary)
      elif b'uniform' in lc:
        return parseDataUniform(content[ln])
        break
  return None

def parseBoundaryField(fn):
  """
    parse internal field, extract data to numpy.array
    @ In, fn, str, file name
    @ Out, parseBoundaryField, np.array, numpy array of boundary field
  """
  content = getFileContent(fn)
  if content is not None:
    return parseBoundaryContent(content)
  else:
    return None

def parseBoundaryContent(content):
  """
    parse each boundary from boundaryField
    @ In, content, list, contents of lines
    @ Out, parseBoundaryContent, np.array, numpy array of boundary content
  """
  data = {}
  isBinary = isBinaryFormat(content)
  bd = splitBoundaryContent(content)
  for boundary, (n1, n2) in bd.items():
    pd = {}
    n = n1
    while True:
      lc = content[n]
      if b'nonuniform' in lc:
        v = parseDataNonuniform(content, n, n2, isBinary)
        pd[lc.split()[0]] = v
        if not isBinary:
          n += len(v) + 4
        else:
          n += 3
        continue
      elif b'uniform' in lc:
        pd[lc.split()[0]] = parseDataUniform(content[n])
      n += 1
      if n > n2:
        break
    data[boundary] = pd
  return data

def parseDataUniform(line):
  """
    parse uniform data from a line
    @ In, line, str,  a line include uniform data, eg. "value           uniform (0 0 0);"
    @ Out, data, float, the data
  """
  if b'(' in line:
    return np.array([float(x) for x in line.split(b'(')[1].split(b')')[0].split()])
  return float(line.split(b'uniform')[1].split(b';')[0])

def parseDataNonuniform(content, n, n2, isBinary):
  """
    parse nonuniform data from lines
    @ In, content, list, contents of lines
    @ In, n, int, line number
    @ In, n2, int, last line number
    @ In, isBinary, bool, binary format or not
    @ Out, data, np.array, the data
  """
  num = int(content[n + 1])
  if not isBinary:
    if b'scalar' in content[n]:
      data = np.array([float(x) for x in content[n + 3:n + 3 + num]])
    else:
      data = np.array([parseDataUniform(ln) for ln in content[n + 3:n + 3 + num]], dtype=float)
  else:
    nn = 1
    if b'vector' in content[n]:
      nn = 3
    elif b'symmtensor' in content[n]:
      nn = 6
    elif b'tensor' in content[n]:
      nn = 9
    buf = b''.join(content[n+2:n2+1])
    vv = np.array(struct.unpack('{}d'.format(num*nn),
                                    buf[struct.calcsize('c'):num*nn*struct.calcsize('d')+struct.calcsize('c')]))
    if nn > 1:
      data = vv.reshape((num, nn))
    else:
      data = vv
  return data

def splitBoundaryContent(content):
  """
    split each boundary from boundaryField
    @ In, content, list, contents of lines
    @ Out, bd, dict, boundary and its content range
  """
  bd = {}
  n = 0
  inBoundaryField = False
  inPatchField = False
  currentPath = ''
  while True:
    lc = content[n]
    if lc.startswith(b'boundaryField'):
      inBoundaryField = True
      if content[n+1].startswith(b'{'):
        n += 2
        continue
      elif content[n+1].strip() == b'' and content[n+2].startswith(b'{'):
        n += 3
        continue
      else:
        print('no { after boundaryField')
        break
    if inBoundaryField:
      if lc.rstrip() == b'}':
        break
      if inPatchField:
        if lc.strip() == b'}':
          bd[currentPath][1] = n-1
          inPatchField = False
          currentPath = ''
        n += 1
        continue
      if lc.strip() == b'':
        n += 1
        continue
      currentPath = lc.strip()
      if content[n+1].strip() == b'{':
        n += 2
      elif content[n+1].strip() == b'' and content[n+2].strip() == b'{':
        n += 3
      else:
        print('no { after boundary patch')
        break
      inPatchField = True
      bd[currentPath] = [n,n]
      continue
    n += 1
    if n > len(content):
      if inBoundaryField:
        print('error, boundaryField not end with }')
      break
  return bd

def isBinaryFormat(content, maxline=20):
  """
    parse file header to judge the format is binary or not
    @ In, content, list, contents of lines
    @ In, maxline, int, optional, maximum lines to parse
    @ Out, isBinaryFormat, bool, optional, binary format or not
  """
  for lc in content[:maxline]:
    if b'format' in lc:
      if b'binary' in lc:
        return True
      return False
  return False
