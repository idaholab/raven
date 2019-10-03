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
Created on June 18, 2018

@author: talbpaul
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3-------------------------------------------

import os
import sys
from collections import OrderedDict

# location of raven
updateScript = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','.ravenrc'))

def loadRC(filename):
  """
    Loads RC file.
    @ In, filename, string, fullpath to .ravenrc file
    @ Out, arguments, dict, {key:arg} ordered dict of entries
  """
  arguments = OrderedDict() # {key:arg}
  with open(filename,'r') as f:
    for l,line in enumerate(f):
      line = line.strip()
      # skip empty, comment lines
      if len(line) == 0 or line.startswith('#'):
        continue
      # check consistency
      if len(line.split('=')) != 2:
        raise IOError('In raven/.ravenrc, expecting key,arg pairs as "key = arg" on each line! On line {} got: "{}"'.format(l,line))
      key,arg = line.split('=')
      arguments[key.strip()] = arg.strip()
  return arguments

def writeRC(filename,entries):
  """
    Writes RC file.
    @ In, filename, str, fullpath to .ravenrc file
    @ In, entries, dict, entries to write
    @ Out, None
  """
  with open(filename,'w') as f:
    for key,val in entries.items():
      f.writelines('{} = {}\n'.format(key,val))

if __name__ == '__main__':
  # write or read mode?
  if '--write' in sys.argv:
    sys.argv.remove('--write')
    mode = 'write'
  elif '--read' in sys.argv:
    sys.argv.remove('--read')
    mode = 'read'
  else:
    print('ERROR: Neither "--read" nor "--write" present in update_install_data.py arguments!')
    sys.exit(1)
  if mode == 'write':
    # collect arguments to change
    toChange = {}
    for c,cla in enumerate(sys.argv):
      # skip the python script name
      if c == 0:
        continue
      # this list could expand later, for now we just need the two.
      if cla == '--conda-defs':
        toChange['CONDA_DEFS'] = sys.argv[c+1]
      elif cla == '--RAVEN_LIBS_NAME':
        toChange['RAVEN_LIBS_NAME'] = sys.argv[c+1]
      elif cla == '--python-command':
        toChange['PYTHON_COMMAND'] = sys.argv[c+1]
      elif cla == '--installation-manager':
        toChange['INSTALLATION_MANAGER'] = sys.argv[c+1]
      else:
        # all keys should start with --, ignore the arguments
        if cla.startswith('--'):
          print('Unrecognized argument:',cla)
    # load existing settings, if any
    if os.path.isfile(updateScript):
      RC = loadRC(updateScript)
    else:
      RC = {}
    # update RC dict
    RC.update(toChange)
    # write the new file
    writeRC(updateScript,RC)
  elif mode == 'read':
    # load RC values
    if os.path.isfile(updateScript):
      RC = loadRC(updateScript)
    else:
      RC = {}
    # print values for requested argument
    for entry in sys.argv[1:]:
      print(RC.get(entry,''))




