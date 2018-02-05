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
import os, sys, json

#find TreeStructure module
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,os.pardir,'framework')))

from reader_old_HDF5_format import OldHDF5Database
from h5py_interface_creator import hdf5Database

import MessageHandler

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})


if __name__=='__main__':
  if len(sys.argv) != 3:
      raise IOError('Expected two argument, the filename of the database to convert and the new filename, but instead got %i: %s' %(len(sys.argv)-1,sys.argv[1:]))
  oldDataBase = sys.argv[1]
  newDataBase = sys.argv[2]
  sameFileName = oldDataBase == newDataBase
  if sameFileName:
    raise IOError('The filenames must be different!!!')
  if not os.path.isfile(oldDataBase):
    raise IOError('ERROR: File not found:',oldDataBase)
  os.path.dirname(oldDataBase)
  oldDatabase = OldHDF5Database("old_database", os.path.dirname(oldDataBase),os.path.basename(oldDataBase))
  newDatabase = hdf5Database("new_database", os.path.dirname(newDataBase), mh, os.path.basename(newDataBase), False)
  historyNames = oldDatabase.retrieveAllHistoryNames()
  
  for hist in historyNames:
    rlz = {}
    # retrieve old data
    histData = oldDatabase.retrieveHistory(hist, filterHist='whole')
    # construct rlz dictionary
    rlz = dict(zip(histData[1]['inputSpaceHeaders'],histData[1]['inputSpaceValues']))
    for varIndex,outputKey in enumerate(histData[1]['outputSpaceHeaders']):
      rlz[outputKey] = histData[0][:,varIndex]
    metadata = histData[1]['metadata'][-1]
    rlz.update(metadata)
    if 'SampledVarsPb' in metadata:
      for var, value in metadata['SampledVarsPb'].items():
        rlz['ProbabilityWeight-'+var.strip()] = value
    # now we have the rlz and we add it in the new database
    newDatabase.addGroup(rlz)
  newDatabase.closeDatabaseW()
  print("CONVERSION PERFORMED!")
