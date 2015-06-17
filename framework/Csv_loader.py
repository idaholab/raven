"""
Created on Feb 7, 2013
@author: alfoa
This python module performs the loading of
data from csv files
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class CsvLoader(MessageHandler.MessageUser):
  def __init__(self,messageHandler):
    """
    Constructor
    """
    self.all_out_param      = False # all output parameters?
    self.field_names        = []    #
    self.all_field_names    = []
    self.type               = 'CsvLoader'
    self.printTag           = self.type
    self.messageHandler     = messageHandler

  def loadCsvFile(self,filein):
    """
    Function to load a csv file into a numpy array (2D)
    It also retrieves the headers
    The format of the csv must be:
    STRING,STRING,STRING,STRING
    FLOAT ,FLOAT ,FLOAT ,FLOAT
    ...
    FLOAT ,FLOAT ,FLOAT ,FLOAT
    @ In, filein, string -> Input file name (absolute path)
    @ Out, data, numpy.ndarray -> the loaded data
    """
    # open file
    myFile = open (filein,'rb')
    # read the field names
    head = myFile.readline().decode()
    self.all_field_names = head.split(',')
    for index in range(len(self.all_field_names)): self.all_field_names[index] = self.all_field_names[index].strip()
    # load the table data (from the csv file) into a numpy nd array
    data = np.loadtxt(myFile,dtype='float',delimiter=',',ndmin=2)
    # close file
    myFile.close()
    return data

  def getFieldNames(self):
    """
    @ In, None
    @ Out, field_names, list -> field names' list
    Function to get actual field names (desired output parameter keywords)
    """
    return self.field_names

  def getAllFieldNames(self):
    """
    Function to get all field names found in the csv file
    @ In, None
    @ Out, all_field_names, list -> list of field names (headers)
    """
    return self.all_field_names

  def csvLoadData(self,filein,options):
    """
    General interface function to call the private methods for loading the different dataObjects!
    @ In, filein, csv file name
    @ In, options, dictionary of options
    """
    if   options['type'] == 'TimePoint':    return self.__csvLoaderForTimePoint(filein[0],options)
    elif options['type'] == 'TimePointSet': return self.__csvLoaderForTimePointSet(filein,options)
    elif options['type'] == 'History':      return self.__csvLoaderForHistory(filein[0],options)
    elif options['type'] == 'Histories':
      listhist_in  = {}
      listhist_out = {}
      for index in xrange(len(filein)):
        tupleVar = self.__csvLoaderForHistory(filein[index],options)
        # dictionary of dictionary key = i => ith history ParameterValues dictionary
        listhist_in[index]  = tupleVar[0]
        listhist_out[index] = tupleVar[1]
        del tupleVar
      return(listhist_in,listhist_out)
    else:
      self.raiseAnError(IOError,'Type ' + options['type'] + 'unknown')

  def __csvLoaderForTimePoint(self,filein,options):
    """
    loader for time point data type
    @ In, filein, file name
    @ In, options, dictionary of options:
          time, time
          inParam, input Parameters
          outParam, output Parameters
          inputTs, time-step from which the input parameters need to be taken
          SampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
    """
    #load the data into the numpy array
    inParam, outParam, inputTs = options['inParam'], options['outParam'], options['inputTs'] if 'inputTs' in options.keys() else None
    SampledVars, time = options['SampledVars'] if 'SampledVars' in options.keys() else None, options['time'] if 'time' in options.keys() else None
    data = self.loadCsvFile(filein)
    if 'all' in outParam: self.all_out_param  = True
    else                : self.all_out_param = False
    if (time == 'end') or (not time):
      time_end = True
      time_float = -1.0
    else:
      # convert the time in float
      time_end = False
      time_float = float(time)
    ints = int(inputTs) if inputTs else 0
    if ints > data[:,0].size -1  and ints != -1: self.raiseAnError(IOError,'inputTs is greater than number of actual ts in file '+ str(filein) + '!')
    inDict, outDict = {}, {}
    self.field_names = self.all_field_names if self.all_out_param else outParam
    #fill input param dictionary
    for key in inParam:
      ix = self.all_field_names.index(key) if key in self.all_field_names else None
      if ix != None:
        inDict[key] = np.atleast_1d(np.array(data[ints,ix]))
      else:
        if SampledVars != None:
          if key in SampledVars.keys(): inDict[key], ix = np.atleast_1d(SampledVars[key]), 0
      if ix == None: self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    # fill output param dictionary
    # time end case
    if time_end:
      last_row = data[:,0].size - 1
      if self.all_out_param:
        for key in self.all_field_names: outDict[key] = np.atleast_1d(np.array(data[last_row,self.all_field_names.index(key)]))
      else:
        for key in outParam:
          if key in self.all_field_names: outDict[key] = np.atleast_1d(np.array(data[last_row,self.all_field_names.index(key)]))
          else: self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    else:
      for i in data:
        if data[i,0] >= time_float and time_float >= 0.0:
          if i-1 >= 0: previous_time = data[i-1,0]
          else:        previous_time = data[i,0]
          actual_time   = data[i,0]
          if self.all_out_param:
            for key in self.all_field_names:
              if(actual_time == previous_time): outDict[key] = np.atleast_1d(np.array((data[i,self.all_field_names.index(key)]  - time_float) / actual_time))
              else:
                actual_value   = data[i,self.all_field_names.index(key)]
                previous_value = data[i-1,self.all_field_names.index(key)]
                outDict[key] = np.atleast_1d(np.array((actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)))
          else:
            for key in outParam:
              if key in self.all_field_names:
                if actual_time == previous_time: outDict[key] = np.atleast_1d(np.array((data[i,self.all_field_names.index(key)]  - time_float) / actual_time))
                else:
                  actual_value   = data[i,self.all_field_names.index(key)]
                  previous_value = data[i-1,self.all_field_names.index(key)]
                  outDict[key] = np.atleast_1d(np.array((actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)))
              else: self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    return (inDict,outDict)

  def __csvLoaderForTimePointSet(self,filesin,options):
    """
    loader for time point set data type
    @ In, filein, file name
    @ In, options, dictionary of options:
          time, time
          inParam, input Parameters
          outParam, output Parameters
          inputTs, time-step from which the input parameters need to be taken
          SampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
    """
    inParam, outParam, inputTs = options['inParam'], options['outParam'], options['inputTs'] if 'inputTs' in options.keys() else None
    SampledVars, time = options['SampledVars'] if 'SampledVars' in options.keys() else None, options['time'] if 'time' in options.keys() else None
    if 'all' in outParam:
      self.all_out_param  = True
    else:
      self.all_out_param = False
    if (time == 'end') or (not time):
      time_end = True
      time_float = -1.0
    else:
      # convert the time in float
      time_end = False
      time_float = float(time)
    if inputTs: ints = int(inputTs)
    else: ints = 0
    inDict, outDict = {}, {}

    for i in range(len(filesin)):
      #load the data into the numpy array
      data = self.loadCsvFile(filesin[i])
      if ints > data[:,0].size -1  and ints != -1: self.raiseAnError(IOError,'inputTs is greater than number of actual ts in file '+ str(filesin[i]) + '!')
      if i == 0:
        if(self.all_out_param): self.field_names = self.all_field_names
        else: self.field_names = outParam
      #fill input param dictionary
      for key in inParam:
        ix = self.all_field_names.index(key) if key in self.all_field_names else None
        if ix != None:
          inDict[key] = np.atleast_1d(np.array(data[ints,ix]))
        else:
          if SampledVars != None:
            if key in SampledVars.keys(): inDict[key], ix = np.atleast_1d(SampledVars[key]), 0
        if ix == None: self.raiseAnError(IOError,"the parameter " + key + " has not been found")
      # time end case
      if time_end:
        last_row = data[:,0].size - 1
        if self.all_out_param:
          for key in self.all_field_names:
            if i == 0:
              #create numpy array
              outDict[key] = np.zeros(len(filesin))
            outDict[key][i] = data[last_row,self.all_field_names.index(key)]
        else:
          for key in outParam:
            if key in self.all_field_names:
              if i == 0:
                #create numpy array
                outDict[key] = np.zeros(len(filesin))
              outDict[key][i] = data[last_row,self.all_field_names.index(key)]
            else:
              self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      else:

        for i in data:
          if data[i,0] >= time_float and time_float >= 0.0:
            if i-1 >= 0:
              previous_time = data[i-1,0]
            else:
              previous_time = data[i,0]
            actual_time   = data[i,0]
            if self.all_out_param:
              for key in self.all_field_names:
                if(actual_time == previous_time):
                  if i == 0:
                    #create numpy array
                    outDict[key] = np.zeros(np.shape(len(filesin)))

                  outDict[key][i] = (data[i,self.all_field_names.index(key)]  - time_float) / actual_time
                else:
                  if i == 0: outDict[key] = np.zeros(np.shape(len(filesin)))
                  actual_value   = data[i,self.all_field_names.index(key)]
                  previous_value = data[i-1,self.all_field_names.index(key)]
                  outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)
            else:
              for key in outParam:
                if key in self.all_field_names:
                  if(actual_time == previous_time):
                    if i == 0:
                      #create numpy array
                      outDict[key] = np.zeros(np.shape(len(filesin)))
                    outDict[key][i] = (data[i,self.all_field_names.index(key)]  - time_float) / actual_time
                  else:
                    if i == 0: outDict[key] = np.zeros(np.shape(len(filesin)))
                    actual_value   = data[i,self.all_field_names.index(key)]
                    previous_value = data[i-1,self.all_field_names.index(key)]
                    outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)
                else:
                  self.raiseAnError(IOError,"the parameter " + key + " has not been found")
      del data
    return (inDict,outDict)

  def __csvLoaderForHistory(self,filein,options):
    """
    loader for history data type
    @ In, filein, file name
    @ In, options, dictionary of options:
          time, time
          inParam, input Parameters
          outParam, output Parameters
          inputTs, time-step from which the input parameters need to be taken
          SampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
    """
    inParam, outParam, inputTs = options['inParam'], options['outParam'], options['inputTs'] if 'inputTs' in options.keys() else None
    SampledVars, time = options['SampledVars'] if 'SampledVars' in options.keys() else None, options['time'] if 'time' in options.keys() else None
    #load the data into the numpy array
    data = self.loadCsvFile(filein)

    time_float = []

    if 'all' in outParam: self.all_out_param  = True
    else                : self.all_out_param = False

    if time:
      if 'all' in time: time_all = True
      else:
        time_all = False
        time_float = [float(x) for x in time]
    else: time_all = True

    if inputTs: ints = int(inputTs)
    else: ints = 0
    if ints > data[:,0].size-1  and ints != -1: self.raiseAnError(IOError,'inputTs is greater than number of actual ts in file '+ str(filein) + '!')
    inDict, outDict = {}, {}

    if(self.all_out_param): self.field_names = self.all_field_names
    else: self.field_names = outParam
    #fill input param dictionary
    for key in inParam:
      ix = self.all_field_names.index(key) if key in self.all_field_names else None
      if ix != None:
        inDict[key] = np.atleast_1d(np.array(data[ints,ix]))
      else:
        if SampledVars != None:
          if key in SampledVars.keys(): inDict[key], ix = np.atleast_1d(SampledVars[key]), 0
      if ix == None: self.raiseAnError(IOError,"the parameter " + key + " has not been found")

    # time all case
    if time_all:
      if self.all_out_param:
        for key in self.all_field_names:
          outDict[key] = data[:,self.all_field_names.index(key)]
      else:
        for key in outParam:
          if key in self.all_field_names:
            outDict[key] = data[:,self.all_field_names.index(key)]
          else:
            self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    else:
      # it will be implemented when we decide a strategy about time filtering
      ## for now it is a copy paste of the time_all case
      if self.all_out_param:
        for key in self.all_field_names:
          outDict[key] = data[:,self.all_field_names.index(key)]
      else:
        for key in outParam:
          if key in self.all_field_names:
            outDict[key] = data[:,self.all_field_names.index(key)]
          else:
            self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    return (inDict,outDict)
