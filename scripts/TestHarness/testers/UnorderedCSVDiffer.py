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
from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os

def isANumber(x):
  '''Checks if x can be converted to a float.
  @ In, x, a variable or value
  @ Out, bool, True if x can be converted to a float.
  '''
  try:
    float(x)
    return True
  except ValueError:
    return False

class UnorderedCSVDiffer:
  """ Used for comparing a bunch of xml files.
  """
  def __init__(self, test_dir, out_files,relative_error=1e-10,absolute_check=False):
    """ Create an UnorderedCSVDiffer class
    test_dir:
    out_files:
    and test_dir + gold + out_files
    @ In, test_dir, the directory where the test takes place
    @ In, out_files, the files to be compared.  They will be in test_dir + out_files
    @ In, *args, unused.
    @ Out, None.
    """
    self.__out_files = out_files
    self.__messages = ""
    self.__same = True
    self.__test_dir = test_dir
    self.__check_absolute_values = absolute_check
    #self.__options = args
    self.__rel_err = relative_error

  def diff(self):
    """ Run the comparison.
    returns (same,messages) where same is true if all the
    csv files are the same, and messages is a string with all the
    differences.
    @ In, None
    @ Out, (bool,string), (same) and (messages)
    """
    # read in files
    for out_file in self.__out_files:
      test_filename = os.path.join(self.__test_dir,out_file)
      gold_filename = os.path.join(self.__test_dir, 'gold', out_file)
      files_read = False
      if not os.path.exists(test_filename):
        self.__same = False
        self.__messages += '\nTest file does not exist: '+test_filename
      elif not os.path.exists(gold_filename):
        self.__same = False
        self.__messages += '\nGold file does not exist: '+gold_filename
      else:
        files_read = True
      if files_read:
        testHead, testData = self.loadCSV(test_filename)
        goldHead, goldData = self.loadCSV(gold_filename)
        #match headers
        #this if can check the data in whatever column order
        #if testHead != goldHead:
        #  goldHeadList = goldHead.split(',')
        #  testHeadList = testHead.split(',')
        #  newTestData = np.zeros(np.array(testData).shape)
        #  oldTestData = np.array(testData)
        #  for cnt,goldVar in enumerate(goldHeadList):newTestData[:,cnt] = oldTestData[:,testHeadList.index(goldVar)]
        #  testData = newTestData.tolist()
        #  testHead = goldHead
        if testHead != goldHead:
          self.__same = False
          self.__messages+='\nHeaders are not the same!...\n...Test: %s\n...Gold: %s' %(testHead,goldHead)
        #in case we want to allow flexible header order, you'd have to change the data order as well before checking!
        #  ...but here is the header search part.
        #while len(testHead)>0:
        #  toFind = testHead.pop()
        #  if toFind in goldHead: goldHead.remove(toFind)
        #  else:
        #    self.__messages += '\nHeader in Test but not in Gold: '+toFind
        #    self.__same = False
        #    break
        #if len(goldHead)>0:
        #  self.__messages += '\nHeader in Gold but not in Test: '+toFind
        #  self.__same = False
        #match body
        if not self.__same: self.__messages+='\nSince headers of csv are not the same, values will not be compared.'
        else:
          if len(testData) != len(goldData):
            self.__same = False
            self.__messages+='\nTest has %i rows, but Gold has %i rows.' %(len(testData),len(goldData))
            return (self.__same,self.__messages.strip())
          while len(testData)>0:
            #take out the first row
            datarow = testData.pop()
            #search for match in gold
            found = False
            for g,goldrow in enumerate(goldData):
              #establish a baseline magnitude
              denom = sum(g if type(g)==float else 0 for g in goldrow)
              if denom == 0: denom = 1.0 #protection from div by zero
              allfound = True
              for d,g in zip(datarow,goldrow):
                if type(d) != type(g): allfound = False
                if type(d) == float:
                  if not self.__check_absolute_values:
                    check = abs(d-g)
                  else:
                    check = abs(abs(d)-abs(g))
                  #div by 0 error handling
                  if abs(g)>1e-15: check/=abs(g)
                  if check > self.__rel_err:
                    allfound = False
                elif type(d) == str:
                  allFound = d == g
              # if sum(abs(d-g)/g for d,g in zip(datarow,goldrow)) < self.__rel_err: #match found -> old method, div by 0 error
              if allfound:
                goldData.remove(goldrow)
                found = True
                break
            if not found:
              self.__same = False
              self.__messages+='\nRow in Test not found in Gold: %s' %str(datarow).strip('[]')
          if len(goldData)>0:
            self.__same = False
            for row in goldData:
              self.__messages+='\nRow in Gold not found in Test: %s' %str(row).strip('[]')
    return (self.__same,self.__messages)

  def loadCSV(self,filename):
    """Method to load CSVs in lieu of using Numpy's method.
    @ In, filename, string file name to load
    @ Out, (list of string,list of lists of floats), (header row) and the body data
    """
    f = file(filename,'r')
    header = f.readline()
    data=[]
    for l,line in enumerate(f):
      if line.strip()=='': continue #sometimes a newline at and of file)
      #if all values are floats, this works great
      try: data.append(list(float(e) for e in line.strip().split(',')))
      #otherwise, we need to take it one entry at a time
      except ValueError:
        toAppend = []
        for e in line.strip().split(','):
          #if it's float, make it so
          try: e = float(e)
          #otherwise, leave it string
          except ValueError: pass
          toAppend.append(e)
        data.append(toAppend)
    return header.strip(),data
