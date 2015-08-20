from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os

num_tol = 1e-10 #effectively zero for our purposes


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
  def __init__(self, test_dir, out_files,*args):
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
    self.__options = args

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
      if not os.path.exists(test_filename):
        self.__same = False
        self.__messages += '\nTest file does not exist: '+test_filename
      elif not os.path.exists(gold_filename):
        self.__same = False
        self.__messages += '\nGold file does not exist: '+gold_filename
      else:
        files_read = True
      testHead, testData = self.loadCSV(test_filename)
      goldHead, goldData = self.loadCSV(gold_filename)
      #match headers
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
            denom = sum(goldrow)
            if denom == 0: denom = 1.0 #protection from div by zero
            allfound = True
            for d,g in zip(datarow,goldrow):
              check = abs(d-g)
              #div by 0 error handling
              if abs(g)>1e-15: check/=abs(g)
              if check > num_tol:
                allfound = False
            # if sum(abs(d-g)/g for d,g in zip(datarow,goldrow)) < num_tol: #match found -> old method, div by 0 error
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
      data.append(list(float(e) for e in line.strip().split(',')))
    return header.strip(),data
