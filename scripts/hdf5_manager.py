'''
Created on April 4, 2013

@author: alfoa
'''
import numpy as np
from h5py_interface_creator import hdf5Database as h5Data

class hdf5Manager:
    '''
    class to manage a or multiple h5py (hdf5) database/s,
    to build them and to retrieve attributes and values in them
    '''
    def __init__(self):
      '''
      Constructor
      '''
      # container of data bases object
      self.h5DataBases  = {}
      # number of data based actually available
      self.n_databases  = 0
      # map<string,(bool)> ie. database name, built or not
      self.areTheyBuilt = {}
      # map<string,(bool)> ie. database name, database already exists???
      self.doTheyAlreadyExist  = {}

    def addDataBase(self,name,type,exist=False):
      self.h5DataBases[name] = h5Data(name,type,exist)
      self.n_databases = self.n_databases + 1
      self.doTheyAlreadyExist[name] = exist
      self.areTheyBuilt[name] = False

    def removeDataBase(self,name):
      try:
        del self.h5DataBases[name]
        del self.doTheyAlreadyExist[name]
        del self.areTheyBuilt[name]
        self.n_databases = self.n_databases - 1
      except:
        pass
    
    def addGroup(self,name,attributes,loadFrom):
      self.h5DataBases[name].addGroup(attributes["group"],attributes,loadFrom)
      elf.areTheyBuilt[name] = True
      return
    
    def retrieveData(self,attributes):
      #time,inParam,outParam)
      if attributes["type"] == "TimePoint":
        data = self.__retrieveDataTimePoint(attributes)
      elif attributes["type"] == "TimePointSet":
        data = self.__retrieveDataTimePointSet(attributes)
      elif attributes["type"] == "History":
        data = self.__retrieveDataTimePointSet(attributes)
      elif attributes["type"] == "Histories":
        data = self.__retrieveDataTimePointSet(attributes)
      else:
        raise("Type" + attributes["type"] +" unknown.Caller: hdf5Manager.retrieveData") 
      return data
      
      
       