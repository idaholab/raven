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
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,os.pardir,'ravenframework')))

from reader_hdf5_from_Feb_2018_to_Oct_2021 import AfterFeb2018ToOct2021HDF5Database
from reader_hdf5_prior_Feb_2018 import PriorFeb2018HDF5Database
from h5py_interface_creator import hdf5Database

import MessageHandler


mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})


if __name__=='__main__':
  if len(sys.argv) != 4:
      raise IOError('Expected 3 arguments, the filename of the database to convert, the new filename and the version of the old database, but instead got %i: %s' %(len(sys.argv)-1,sys.argv[1:]))
  oldDataBase = sys.argv[1]
  newDataBase = sys.argv[2]
  hdf5Version = sys.argv[3].strip()

  sameFileName = oldDataBase == newDataBase
  if sameFileName:
    raise IOError('The filenames must be different!!!')
  if not os.path.isfile(oldDataBase):
    raise IOError('ERROR: File not found:',oldDataBase)

  if hdf5Version not in ['Jan2018','Oct2021','v2.1']:
    raise IOError('ERROR: Only version available are :',str(['Jan2018','Oct2021','v2.1']))

  if hdf5Version == 'Jan2018':
    oldDatabase = PriorFeb2018HDF5Database("old_database", os.path.dirname(oldDataBase), os.path.basename(oldDataBase))
  elif hdf5Version == 'Oct2021':
    oldDatabase = AfterFeb2018ToOct2021HDF5Database("old_database", os.path.dirname(oldDataBase), os.path.basename(oldDataBase), True)
  elif hdf5Version == 'v2.1':
    oldDatabase = hdf5Database("old_database", os.path.dirname(oldDataBase), os.path.basename(oldDataBase), True)

  newDatabase = hdf5Database("new_database", os.path.dirname(newDataBase), os.path.basename(newDataBase), False)
  historyNames = oldDatabase.retrieveAllHistoryNames()
  for hist in historyNames:
    rlz = {}
    print("retrieving history '{}' from database '{}'" .format(hist, os.path.basename(oldDataBase)))
    # retrieve old data
    if hdf5Version == 'Jan2018':
      histData = oldDatabase.retrieveHistory(hist, filterHist='whole')
    else:
      histData = oldDatabase._getRealizationByName(hist,options = {'reconstruct':True})
    # construct rlz dictionary
    if hdf5Version == 'Jan2018':
      rlz = dict(zip(histData[1]['inputSpaceHeaders'],histData[1]['inputSpaceValues']))
      for varIndex,outputKey in enumerate(histData[1]['outputSpaceHeaders']):
        rlz[outputKey] = histData[0][:,varIndex]
      metadata = histData[1]['metadata'][-1]
      rlz.update(metadata)
      if 'SampledVarsPb' in metadata:
        for var, value in metadata['SampledVarsPb'].items():
          rlz['ProbabilityWeight-'+var.strip()] = value
    else:
      rlz = histData[0]
    # now we have the rlz and we add it in the new database
    print("re-adding  history '{}' into database '{}'".format(hist,os.path.basename(newDataBase)))
    newDatabase.addGroup(rlz)
  newDatabase.closeDatabaseW()
  print("CONVERSION PERFORMED!")
