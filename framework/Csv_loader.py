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
from scipy.interpolate import interp1d
import copy
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
    if   options['type'] == 'Point'   : return self.__csvLoaderForPoint(filein[0],options)
    elif options['type'] == 'PointSet': return self.__csvLoaderForPointSet(filein,options)
    elif options['type'] == 'History' : return self.__csvLoaderForHistory(filein[0],options)
    elif options['type'] == 'HistorySet':
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

  def __csvLoaderForPoint(self,filein,options):
    """
    loader for point data type
    @ In, filein, file name
    @ In, options, dictionary of options:
          outputPivotVal, output value at which the outputs need to be collected
          inParam, input Parameters
          outParam, output Parameters
          inputRow, outputPivotVal-step from which the input parameters need to be taken
          SampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
              <inputRow>
              <outputRow>
              <operator>
              <pivotParameter>
              <outputPivotValue>
              <outputPivotValue>
              <inputPivotValue>
    """
    inParam, outParam, inputRow, outputRow                 = options['inParam'], options['outParam'], options.get('inputRow',None), options.get('outputRow',None)
    SampledVars, inputPivotVal, outputPivotVal, operator   = options.get('SampledVars',None), options.get('inputPivotValue',None), options.get('outputPivotValue',None), options.get('operator',None)
    pivotParameter                                         = options.get('pivotParameter',None)

    if 'all' in outParam: self.all_out_param = True
    else                : self.all_out_param = False

    if outputPivotVal != None:
      if 'end' in outputPivotVal: outputPivotVal_end = True
      else:
        outputPivotVal_end, outputPivotVal = False,  float(outputPivotVal)
    else: outputPivotVal_end = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None : inputRow  = int(inputRow)
    if outputRow != None: outputRow = int(outputRow)
    inDict, outDict = {}, {}

    #load the data into the numpy array
    data = self.loadCsvFile(filein)
    if pivotParameter != None:
      pivotIndex = self.all_field_names.index(pivotParameter) if pivotParameter in self.all_field_names else None
      if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in file '+ str(filein) + '!')
    else:
      pivotIndex = self.all_field_names.index("time") if "time" in self.all_field_names else None
      # if None...default is 0
      if pivotIndex == None: pivotIndex = 0
    if inputRow > data[:,0].size-1  and inputRow != -1: self.raiseAnError(IOError,'inputRow is greater than number of actual rows in file '+ str(filein) + '!')

    if(self.all_out_param): self.field_names = self.all_field_names
    else: self.field_names = outParam

    #fill input param dictionary
    for key in inParam:
      ix = self.all_field_names.index(key) if key in self.all_field_names else None
      if ix != None:
        if inputPivotVal != None:
          if float(inputPivotVal) > np.max(data[:,pivotIndex]) or float(inputPivotVal) < np.min(data[:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in file '+ str(filein) + '!')
          inDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,ix], kind='linear')(float(inputPivotVal))))
        else: inDict[key] = np.atleast_1d(np.array(data[inputRow,ix]))
      else:
        if SampledVars != None:
          if key in SampledVars.keys(): inDict[key], ix = copy.deepcopy(np.atleast_1d(np.array(SampledVars[key]))), 0
      if ix == None: self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    # outputPivotVal end case
    if outputPivotVal_end:
      last_row = data[:,0].size - 1
      if self.all_out_param:
        for key in self.all_field_names:
          outDict[key] = np.atleast_1d(np.array(data[last_row,self.all_field_names.index(key)]))
      else:
        for key in outParam:
          if key in self.all_field_names:
            outDict[key] = np.atleast_1d(np.array(data[last_row,self.all_field_names.index(key)]))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    elif outputRow != None:
      if outputRow > data[:,0].size-1  and outputRow != -1: self.raiseAnError(IOError,'outputRow is greater than number of actual rows in file '+ str(filein) + '!')
      if self.all_out_param:
        for key in self.all_field_names:
          outDict[key] = np.atleast_1d(np.array(data[outputRow,self.all_field_names.index(key)]))
      else:
        for key in outParam:
          if key in self.all_field_names:
            outDict[key] = np.atleast_1d(np.array(data[outputRow,self.all_field_names.index(key)]))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    elif operator != None:
      if operator not in ['max','min','average']: self.raiseAnError(IOError,'operator unknown. Available are min,max,average')
      if self.all_out_param:
        for key in self.all_field_names:
          if operator == 'max'    : outDict[key] = np.atleast_1d(np.array(np.max(data[:,self.all_field_names.index(key)])))
          if operator == 'min'    : outDict[key] = np.atleast_1d(np.array(np.min(data[:,self.all_field_names.index(key)])))
          if operator == 'average': outDict[key] = np.atleast_1d(np.array(np.average(data[:,self.all_field_names.index(key)])))
      else:
        for key in outParam:
          if key in self.all_field_names:
            if operator == 'max'    : outDict[key] = np.atleast_1d(np.array(np.max(data[:,self.all_field_names.index(key)])))
            if operator == 'min'    : outDict[key] = np.atleast_1d(np.array(np.min(data[:,self.all_field_names.index(key)])))
            if operator == 'average': outDict[key] = np.atleast_1d(np.array(np.average(data[:,self.all_field_names.index(key)])))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    else:
      if self.all_out_param:
        for key in self.all_field_names:
          outDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,self.all_field_names.index(key)], kind='linear')(outputPivotVal)))
      else:
        for key in outParam:
          if key in self.all_field_names: outDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,self.all_field_names.index(key)], kind='linear')(outputPivotVal)))
          else                          : self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    return (inDict,outDict)

  def __csvLoaderForPointSet(self,filesin,options):
    """
    loader for outputPivotVal point set data type
    @ In, filein, file name
    @ In, options, dictionary of options:
          outputPivotVal, outputPivotVal
          inParam, input Parameters
          outParam, output Parameters
          inputRow, outputPivotVal-step from which the input parameters need to be taken
          SampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
              <inputRow>
              <outputRow>
              <operator>
              <pivotParameter>
              <outputPivotValue>
              <outputPivotValue>
              <inputPivotValue>
    """
    inParam, outParam, inputRow, outputRow                 = options['inParam'], options['outParam'], options.get('inputRow',None), options.get('outputRow',None)
    SampledVars, inputPivotVal, outputPivotVal, operator   = options.get('SampledVars',None), options.get('inputPivotValue',None), options.get('outputPivotValue',None), options.get('operator',None)
    pivotParameter                                         = options.get('pivotParameter',None)

    if 'all' in outParam: self.all_out_param = True
    else                : self.all_out_param = False

    if outputPivotVal != None:
      if 'end' in outputPivotVal: outputPivotVal_end = True
      else:
        outputPivotVal_end, outputPivotVal = False,  float(outputPivotVal)
    else: outputPivotVal_end = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None : inputRow  = int(inputRow)
    if outputRow != None: outputRow = int(outputRow)
    inDict, outDict = {}, {}

    for i in range(len(filesin)):
      #load the data into the numpy array
      data = self.loadCsvFile(filesin[i])
      if pivotParameter != None:
        pivotIndex = self.all_field_names.index(pivotParameter) if pivotParameter in self.all_field_names else None
        if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in file '+ str(filesin[i]) + '!')
      else:
        pivotIndex = self.all_field_names.index("time") if "time" in self.all_field_names else None
        # if None...default is 0
        if pivotIndex == None: pivotIndex = 0
      if inputRow > data[:,0].size-1  and inputRow != -1: self.raiseAnError(IOError,'inputRow is greater than number of actual rows in file '+ str(filesin[i]) + '!')

      if i == 0:
        if(self.all_out_param): self.field_names = self.all_field_names
        else: self.field_names = outParam

      #fill input param dictionary
      for key in inParam:
        if i == 0: inDict[key] = np.zeros(len(filesin))
        ix = self.all_field_names.index(key) if key in self.all_field_names else None
        if ix != None:
          if inputPivotVal != None:
            if float(inputPivotVal) > np.max(data[:,pivotIndex]) or float(inputPivotVal) < np.min(data[:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in file '+ str(filesin[i]) + '!')
            inDict[key][i] = interp1d(data[:,pivotIndex], data[:,ix], kind='linear')(float(inputPivotVal))
          else: inDict[key][i] = data[inputRow,ix]
        else:
          if SampledVars != None:
            if key in SampledVars.keys(): inDict[key][i], ix = copy.deepcopy(SampledVars[key]), 0
        if ix == None: self.raiseAnError(IOError,"the parameter " + key + " has not been found")
      # outputPivotVal end case
      if outputPivotVal_end:
        last_row = data[:,0].size - 1
        if self.all_out_param:
          for key in self.all_field_names:
            if i == 0: outDict[key] = np.zeros(len(filesin))
            outDict[key][i] = data[last_row,self.all_field_names.index(key)]
        else:
          for key in outParam:
            if key in self.all_field_names:
              if i == 0: outDict[key] = np.zeros(len(filesin))
              outDict[key][i] = data[last_row,self.all_field_names.index(key)]
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      elif outputRow != None:
        if outputRow > data[:,0].size-1  and outputRow != -1: self.raiseAnError(IOError,'outputRow is greater than number of actual rows in file '+ str(filesin[i]) + '!')
        if self.all_out_param:
          for key in self.all_field_names:
            if i == 0: outDict[key] = np.zeros(len(filesin))
            outDict[key][i] = data[outputRow,self.all_field_names.index(key)]
        else:
          for key in outParam:
            if key in self.all_field_names:
              if i == 0: outDict[key] = np.zeros(len(filesin))
              outDict[key][i] = data[outputRow,self.all_field_names.index(key)]
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      elif operator != None:
        if operator not in ['max','min','average']: self.raiseAnError(IOError,'operator unknown. Available are min,max,average')
        if self.all_out_param:
          for key in self.all_field_names:
            if i == 0: outDict[key] = np.zeros(len(filesin))
            if operator == 'max'    : outDict[key][i] = np.max(data[:,self.all_field_names.index(key)])
            if operator == 'min'    : outDict[key][i] = np.min(data[:,self.all_field_names.index(key)])
            if operator == 'average': outDict[key][i] = np.average(data[:,self.all_field_names.index(key)])
        else:
          for key in outParam:
            if key in self.all_field_names:
              if i == 0: outDict[key] = np.zeros(len(filesin))
              if operator == 'max'    : outDict[key][i] = np.max(data[:,self.all_field_names.index(key)])
              if operator == 'min'    : outDict[key][i] = np.min(data[:,self.all_field_names.index(key)])
              if operator == 'average': outDict[key][i] = np.average(data[:,self.all_field_names.index(key)])
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      else:
        if self.all_out_param:
          for key in self.all_field_names:
            if i == 0: outDict[key] = np.zeros(len(filesin))
            outDict[key][i] = interp1d(data[:,pivotIndex], data[:,self.all_field_names.index(key)], kind='linear')(outputPivotVal)
        else:
          for key in outParam:
            if i == 0: outDict[key] = np.zeros(len(filesin))
            if key in self.all_field_names: outDict[key][i] = interp1d(data[:,pivotIndex], data[:,self.all_field_names.index(key)], kind='linear')(outputPivotVal)
            else                          : self.raiseAnError(IOError,"the parameter " + key + " has not been found")
      del data
    return (inDict,outDict)

  def __csvLoaderForHistory(self,filein,options):
    """
    loader for history data type
    @ In, filein, file name
    @ In, options, dictionary of options:
          outputPivotVal, outputPivotVal
          inParam, input Parameters
          outParam, output Parameters
          inputRow, outputPivotVal-step from which the input parameters need to be taken
          SampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
              <inputRow>
              <pivotParameter>
              <outputPivotValue>
              <outputPivotValue>
              <inputPivotValue>

    """
    inParam, outParam, inputRow                 = options['inParam'], options['outParam'], options.get('inputRow',None)
    SampledVars, inputPivotVal, outputPivotVal  = options.get('SampledVars',None), options.get('inputPivotValue',None), options.get('outputPivotValue',None)
    pivotParameter                              = options.get('pivotParameter',None)
    #load the data into the numpy array
    data = self.loadCsvFile(filein)
    if 'all' in outParam: self.all_out_param = True
    else                : self.all_out_param = False
    if pivotParameter != None:
      pivotIndex = self.all_field_names.index(pivotParameter) if pivotParameter in self.all_field_names else None
      if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in file '+ str(filein) + '!')
    else:
      pivotIndex = self.all_field_names.index("time") if "time" in self.all_field_names else None
      # if None...default is 0
      if pivotIndex == None: pivotIndex = 0

    if outputPivotVal != None:
      if 'all' in outputPivotVal: outputPivotVal_all = True
      else:
        outputPivotVal_all, outputPivotVal = False,  [float(x) for x in outputPivotVal.split()]
    else: outputPivotVal_all = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None: inputRow = int(inputRow)
    if inputRow > data[:,0].size-1  and inputRow != -1: self.raiseAnError(IOError,'inputRow is greater than number of actual rows in file '+ str(filein) + '!')
    inDict, outDict = {}, {}
    self.field_names = self.all_field_names if self.all_out_param else outParam

    #fill input param dictionary
    for key in inParam:
      ix = self.all_field_names.index(key) if key in self.all_field_names else None
      if ix != None:
        if inputPivotVal != None:
          if float(inputPivotVal) > np.max(data[:,pivotIndex]) or float(inputPivotVal) < np.min(data[:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in file '+ str(filein) + '!')
          inDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,ix], kind='linear')(float(inputPivotVal))))
        else: inDict[key] = np.atleast_1d(np.array(data[inputRow,ix]))
      else:
        if SampledVars != None:
          if key in SampledVars.keys(): inDict[key], ix = copy.deepcopy(np.atleast_1d(SampledVars[key])), 0
      if ix == None: self.raiseAnError(IOError,"the parameter " + key + " has not been found")

    # outputPivotVal all case
    if outputPivotVal_all:
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
      # pivot value
      if self.all_out_param:
        for key in self.all_field_names: outDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,self.all_field_names.index(key)], kind='linear')(outputPivotVal)))
      else:
        for key in outParam:
          if key in self.all_field_names: outDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,self.all_field_names.index(key)], kind='linear')(outputPivotVal)))
          else                          : self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    return (inDict,outDict)
