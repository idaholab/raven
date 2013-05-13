'''
Created on May 8, 2013

@author: mandd
'''
import xml.etree.ElementTree as ET
import numpy as np
from BaseType import BaseType
from Csv_loader import CsvLoader as ld
import DataSets
import matplotlib.pyplot as plt
#from hdf5_manager import hdf5Manager as AAFManager
#import h5py as h5

class OutStream(BaseType):
  def __init__(self):
    BaseType.__init__(self)
    self.fileNameRoot   = None
    self.variables      = []
    self.toLoadFromList = []
    self.alreadyRead    = []
    self.histories      = {}
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

  def addInitParams(self,tempDict):
    for i in range(len(self.variables)): 
      tempDict['Variables_'+str(i)] = self.variables[i]
    if self.fileNameRoot:
      tempDict['FileNameRoot'] = self.fileNameRoot
    return tempDict

  def finalize(self):
    pass 
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
  def retrieveHistories(self):
    try:
      if self.toLoadFromList[0].type == "HDF5":
        endGroupNames = self.toLoadFromList[0].getEndingGroupNames()
        
        for index in xrange(len(endGroupNames)):
          if not endGroupNames[index] in self.alreadyRead:
            self.histories[endGroupNames[index]] = self.toLoadFromList[0].returnHistory({'history':endGroupNames[index],'filter':'whole'})
            self.alreadyRead.append(endGroupNames[index])
    except:
      # loading from file (csv)
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

  def getParam(self,typeVar,keyword):
    pass
  
class ScreenPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)  

  def addOutput(self,toLoadFrom):
    # this function adds the file name/names to the
    # filename list
    print('toLoadFrom :')
    print(toLoadFrom)
    
    self.toLoadFromList.append(toLoadFrom)
    
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    
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
    
    
        
  def finalize(self):
    pass
#    try:
#      self.retrieveHistories()
#    except:  
#      OutStream.retrieveHistories(self)
#    headers = self.histories[self.alreadyRead[0]][1]['headers']
#    timeVar = ''
#    for i in xrange(len(headers)):
#      if headers[i].lower() == 'time':
#        #timeVar = headers.pop(i)
#        timeLoc = i
#        break
#    
#    for index in xrange(len(headers)):
#      if headers[index].lower() != 'time':
#        if not self.variables[0]=='all':
#          if headers[index] in self.variables:
#            plot_it = True
#          else:
#            plot_it = False  
#        else:    
#          plot_it = True
#        if plot_it:
#          plt.figure(index)
#          plt.xlabel(headers[timeLoc])
#          plt.ylabel(headers[index])
#          plt.title('Plot of histories')
#          for key in self.histories:
#            plt.plot(self.histories[key][0][:,timeLoc],self.histories[key][0][:,index])
#    plt.show()


class PdfPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)  
  
  def finalize(self):
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    
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
class PngPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)  
  
  def finalize(self):
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    
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
class JpegPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)  
  
  def finalize(self):
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    
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
class EpsPlot(OutStream):
  def __init__(self):
    OutStream.__init__(self)  
  
  def finalize(self):
    try:
      self.retrieveHistories()
    except:  
      OutStream.retrieveHistories(self)
    headers = self.histories[self.alreadyRead[0]][1]['headers']
    timeVar = ''
    for i in xrange(len(headers)):
      if headers[i].lower() == 'time':
        #timeVar = headers.pop(i)
        timeLoc = i
        break
    
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