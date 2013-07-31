'''
Created on May 8, 2013

@author: mandd
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import numpy as np
from BaseType import BaseType
from Csv_loader import CsvLoader as ld
import DataBases
import matplotlib.pyplot as plt
#from hdf5_manager import hdf5Manager as AAFManager
#import h5py as h5

'''
  *************************
  *    OUTSTREAM CLASS    *
  *************************
'''

class OutStream(BaseType):
  def __init__(self):
    '''
      Init of Base class 
    '''
    BaseType.__init__(self)
    '''
      Root of the file name
    '''
    self.fileNameRoot   = None
    '''
      List of variables that must be displayed
    '''
    self.variables      = []
    '''
      List of source for retrieving data (for example, list of CSVs or HDF5)
    '''
    self.toLoadFromList = []
    '''
      List of Histories already loaded (working var)
    '''
    self.alreadyRead    = []
    '''
      Dictionary of histories
    '''
    self.histories      = {}

  '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
  '''
  def readMoreXML(self,xmlNode):
    var = xmlNode.find('variable').text
    var.replace(" ","")
    if var.lower() == 'all':
      self.variables = ['all']
    else:
      self.variables = var.split(',')
      
    try:
      var = xmlNode.find('fileNameRoot').text
      self.fileNameRoot = var.replace(" ","")
    except:
      pass

  '''
    Function adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
  '''
  def addInitParams(self,tempDict):
    for i in range(len(self.variables)):
      tempDict['Variables_'+str(i)] = self.variables[i]
    if self.fileNameRoot:
      tempDict['FileNameRoot'] = self.fileNameRoot
    return tempDict

  '''
    Function to finalize the outstream.Each outstream specialized class must implement it
    @ In, None
    @ Out, None 
  '''
  def finalize(self):
    pass 

  '''
    Function to add a new output source (for example a CSV file or a HDF5 object)
    @ In, toLoadFrom, source object
    @ Out, None 
  '''
  def addOutput(self,toLoadFrom):
    # this function adds the file name/names to the
    # filename list
    print('toLoadFrom :')
    print(toLoadFrom)
    
    self.toLoadFromList.append(toLoadFrom)
    return
#  def getInpParametersValues(self):
#    return self.inpParametersValues  
#
#  def getOutParametersValues(self):
#    return self.outParametersValues 

  '''
    Function to retrieve histories from th toLoadFromList object
    @ In, None
    @ Out, None 
  '''
  def retrieveHistories(self):
    '''
      Check type of source
    '''
    try:
      if self.toLoadFromList[0].type == "HDF5":
        '''
          HDF5 database
        '''
        '''
          Retrieve ending histories' names from the database
        '''
        endGroupNames = self.toLoadFromList[0].getEndingGroupNames()
        '''
          Retrieve the histories
        '''
        for index in xrange(len(endGroupNames)):
          if not endGroupNames[index] in self.alreadyRead:
            self.histories[endGroupNames[index]] = self.toLoadFromList[0].returnHistory({'history':endGroupNames[index],'filter':'whole'})
            self.alreadyRead.append(endGroupNames[index])
    except:
      ''' loading from file (csv) '''
      '''
        Retrieve histories from CSV files
      '''
      for index in xrange(len(self.toLoadFromList)):
        groupname = self.toLoadFromList[index].split('~')[1]
        if not groupname in self.alreadyRead:
           # open file
          myFile = open (self.toLoadFromList[index],'rb')
          # read the field names
          all_field_names = myFile.readline().split(',')
          # load the table data (from the csv file) into a numpy nd array 
          data = np.loadtxt(myFile,dtype='float',delimiter=',',ndmin=2)
          # close file
          myFile.close()  
          self.histories[groupname] = (data,{'headers':all_field_names})
          self.alreadyRead.append(groupname)
    return

  '''
    Function to get a Parameter in this function
    @ In, typeVar : Variable type (string)
    @ In, keyword: Keyword to retrieve 
    @ Out,param  : Requested parameter
  '''
  def getParam(self,typeVar,keyword):
    pass

'''
  Specialized OutStream class ScreenPlot: Show data on the screan
'''
class ScreenPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)

  '''
    Function to add a new output source (for example a CSV file or a HDF5 object)
    @ In, toLoadFrom, source object
    @ Out, None 
  '''
  def addOutput(self,toLoadFrom):
    # this function adds the file name/names to the
    # filename list
    print('toLoadFrom :')
    print(toLoadFrom)
    '''
      Append loading object in the list
    '''
    self.toLoadFromList.append(toLoadFrom)
    '''
      Retrieve histories
      NB. The finalization of the class is performed here since we want to have the screen output 
          during the calculation runs
    '''
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    '''
      Retrieve the headers
    '''
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    '''
      Find where the time evolution is stored
    '''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    '''
      Plot the requested histories
    '''
    for index in xrange(len(headers)):
      if headers[index].lower() != 'time':
        if not self.variables[0]=='all':
          if headers[index] in self.variables:
            plot_it = True
          else:
            plot_it = False  
        else:    
          plot_it = True
        if plot_it:
          plt.figure(index)
          plt.xlabel(headers[timeLoc])
          plt.ylabel(headers[index])
          plt.title('Plot of histories')
          for key in self.histories:
            plt.plot(self.histories[key][0][:,timeLoc],self.histories[key][0][:,index])
    plt.draw()
    #plt.show()

  '''
    Function to finalize the ScreenPlot. In this case it is not needed
    @ In, None
    @ Out, None 
  '''
  def finalize(self):
    pass

'''
  Specialized OutStream class PdfPlot: Create of PDF of data
'''
class PdfPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)

  '''
    Function to finalize the PdfPlot. It creates the PDF output
    @ In, None
    @ Out, None 
  '''
  def finalize(self):
    '''
      Retrieve histories
    '''
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    '''
      Retrieve headers
    '''
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    '''
      Find where the time evolution is stored
    '''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    '''
      Create the PDF
    '''
    for index in xrange(len(headers)):
      if headers[index].lower() != 'time':
        if not self.variables[0]=='all':
          if headers[index] in self.variables:
            plot_it = True
          else:
            plot_it = False  
        else:    
          plot_it = True
        if plot_it:
          plt.figure()
          plt.xlabel(headers[timeLoc])
          plt.ylabel(headers[index])
          plt.title('Plot of histories')
          for key in self.histories:
            plt.plot(self.histories[key][0][:,timeLoc],self.histories[key][0][:,index])
        fileName = self.fileNameRoot + '.pdf'
        plt.savefig(fileName, dpi=fig.dpi)
    return

'''
  Specialized OutStream class PngPlot: Create of PNG picture of data
'''
class PngPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)

  '''
    Function to finalize the PngPlot. It creates the PNG output
    @ In, None
    @ Out, None 
  '''
  def finalize(self):
    '''
      Retrieve Histories
    '''
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    '''
      Retrieve headers
    '''
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    '''
      Find where the time evolution is stored
    '''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    '''
      Create the PNG output file
    '''
    for index in xrange(len(headers)):
      if headers[index].lower() != 'time':
        if not self.variables[0]=='all':
          if headers[index] in self.variables:
            plot_it = True
          else:
            plot_it = False  
        else:    
          plot_it = True
        if plot_it:
          plt.figure()
          plt.xlabel(headers[timeLoc])
          plt.ylabel(headers[index])
          plt.title('Plot of histories')
          for key in self.histories:
            plt.plot(self.histories[key][0][:,timeLoc],self.histories[key][0][:,index])
        fileName = self.fileNameRoot + '.png'
        fig.savefig(fileName, dpi=fig.dpi)
    return

'''
  Specialized OutStream class JpegPlot: Create of JPEG picture of data
'''
class JpegPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)  

  '''
    Function to finalize the JpegPlot. It creates the PNG output
    @ In, None
    @ Out, None 
  '''
  def finalize(self):
    '''
      Retrieve Histories
    '''
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    '''
      Retrieve headers
    '''
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    '''
      Find where the time evolution is stored
    '''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    '''
      Create the JPEG output file
    '''
    for index in xrange(len(headers)):
      if headers[index].lower() != 'time':
        if not self.variables[0]=='all':
          if headers[index] in self.variables:
            plot_it = True
          else:
            plot_it = False  
        else:    
          plot_it = True
        if plot_it:
          plt.figure()
          plt.xlabel(headers[timeLoc])
          plt.ylabel(headers[index])
          plt.title('Plot of histories')
          for key in self.histories:
            plt.plot(self.histories[key][0][:,timeLoc],self.histories[key][0][:,index])
        fileName = self.fileNameRoot + '.jpeg'
        fig.savefig(fileName, dpi=fig.dpi)
    return

'''
  Specialized OutStream class EpsPlot: Create of EPS picture of data
'''
class EpsPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)  

  '''
    Function to finalize the EpsPlot. It creates the EPS output
    @ In, None
    @ Out, None 
  '''
  def finalize(self):
    '''
      Retrieve Histories
    '''
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    '''
      Retrieve headers
    '''
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    '''
      Find where the time evolution is stored
    '''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    '''
      Create the EPS output file
    '''
    for index in xrange(len(headers)):
      if headers[index].lower() != 'time':
        if not self.variables[0]=='all':
          if headers[index] in self.variables:
            plot_it = True
          else:
            plot_it = False  
        else:    
          plot_it = True
        if plot_it:
          plt.figure()
          plt.xlabel(headers[timeLoc])
          plt.ylabel(headers[index])
          plt.title('Plot of histories')
          for key in self.histories:
            plt.plot(self.histories[key][0][:,timeLoc],self.histories[key][0][:,index])
        fileName = self.fileNameRoot + '.eps'
        fig.savefig(fileName, dpi=fig.dpi)
    return

'''
  function used to generate a OutStream class
  @ In, Type : OutStream type
  @ Out,Instance of the Specialized OutStream class
'''
def returnInstance(Type):
  base = 'OutStream'
  InterfaceDict = {}
  InterfaceDict['Screen'   ]    = ScreenPlot
  InterfaceDict['Pdf'      ]    = PdfPlot
  InterfaceDict['Png'      ]    = PngPlot
  InterfaceDict['Jpeg'     ]    = JpegPlot
  InterfaceDict['Eps'      ]    = EpsPlot
  try:
    if Type in InterfaceDict.keys():
      return InterfaceDict[Type]()
  except:
    raise NameError('not known '+base+' type'+Type)
