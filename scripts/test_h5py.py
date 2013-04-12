import os
from h5py_interface_creator import hdf5Database as h5Data
from hdf5_manager import hdf5Manager as h5Manager

if __name__ == '__main__':
  debug = True
  
  #######################################
  #           TEST DET TYPE             #   
  #######################################
  h5d = h5Data("HDF5_dataset_test","DET")
  
  attributes = {}
  source     = {}
  
  source["type"] = 'csv'
  source["name"] = 'test1.csv'
  
  attributes["parent"] = "/" 
  attributes["input_file"] = "test1.i" 
  attributes["branch_param"] = "CH1averageFT"
  attributes["conditional_prb"] = 1
  
  h5d.addGroup("test_1",attributes,source)
  
  source["name"] = 'test1_1.csv'
  
  attributes["parent"] = "/test_1" 
  attributes["input_file"] = "test1_1.i" 
  attributes["branch_param"] = "CH1averageFT"
  attributes["conditional_prb"] = 0.5
  
  h5d.addGroup("test_1.1",attributes,source)
  
  source["name"] = 'test1_2.csv'
  
  attributes["parent"] = "/test_1" 
  attributes["input_file"] = "test1_2.i" 
  attributes["branch_param"] = "CH1averageFT"
  attributes["conditional_prb"] = 0.5
  
  h5d.addGroup("test_1.2",attributes,source)  
  
  source["name"] = 'test1_1_1.csv'
  
  attributes["parent"] = "/test_1/test_1.1" 
  attributes["input_file"] = "test1_1_1.i" 
  attributes["branch_param"] = "CH1averageFT"
  attributes["conditional_prb"] = 0.25
  
  h5d.addGroup("test_1.1.1",attributes,source)  

  source["name"] = 'test1_1_2.csv'
  
  attributes["parent"] = "/test_1/test_1.1" 
  attributes["input_file"] = "test1_1_2.i" 
  attributes["branch_param"] = "CH1averageFT"
  attributes["conditional_prb"] = 0.25
  
  h5d.addGroup("test_1.1.2",attributes,source)  

  source["name"] = 'test1_2_1.csv'
  
  attributes["parent"] = "/test_1/test_1.2" 
  attributes["input_file"] = "test1_2_1.i" 
  attributes["branch_param"] = "CH1averageFT"
  attributes["conditional_prb"] = 0.25
  
  h5d.addGroup("test_1.2.1",attributes,source)  
  
  source["name"] = 'test1_2_2.csv'
  
  attributes["parent"] = "/test_1/test_1.2" 
  attributes["input_file"] = "test1_2_2.i" 
  attributes["branch_param"] = "CH1averageFT"
  attributes["conditional_prb"] = 0.25
  
  h5d.addGroup("test_1.2.2",attributes,source) 
  
  tup1 = h5d.retrieveHistory("test_1.2.2","whole")
  
  h5d.closeDataBaseW()
  
  tup2 = h5d.retrieveHistory("test_1.2.2",0)
  
  tup3 = h5d.retrieveHistory("test_1.2.2",1)
  
  tup4 = h5d.retrieveHistory("test_1","whole") 
  
  tup5 = h5d.retrieveHistory("test_1") 
  
  back = h5d.computeBack("test_1","test_1.2.2")
  
  result4 = tup4[0]
  
  h5d2 = h5Data("HDF5_dataset_test","DET","HDF5_dataset_test_DET.h5")
  
  tup9 = h5d.retrieveHistory("test_1","whole") 
  
  #######################################
  #       TEST MONTECARLO TYPE          #   
  #######################################  

  h5dMC = h5Data("HDF5_dataset_test","MC")
  
  attributes = {}
  source     = {}
  
  source["type"] = 'csv'
  source["name"] = 'test1.csv'
  
  attributes["parent"] = "/" 
  attributes["input_file"] = "test1.i" 
  
  h5dMC.addGroup("test_1",attributes,source)
  
  h5dMC.addGroup("test_2",attributes,source)
  
  h5dMC.addGroup("test_3",attributes,source)  

  tup6 = h5dMC.retrieveHistory("test_1")
  
  tup7 = h5dMC.retrieveHistory("test_2")
  
  tup8 = h5dMC.retrieveHistory("test_3")
  
  h5dMC.closeDataBaseW()
  ##########################
  # TEST DATA BASE MANAGER #
  ##########################
  
  hMang = h5Manager()
  
  hMang.addDataBase("testManager","MC")
  
  attributes["group"] = "test_1_manager"
  
  hMang.addGroup("testManager",attributes,source)
  
  attributes["group"] = "test_2_manager"
  
  hMang.addGroup("testManager",attributes,source)
  
  attributes["type"] = "TimePoint"
  attributes['outParam'] = 'all'
  attributes['time'] = 'end'
  attributes['inParam'] = []
  attributes['inParam'].append("CH1averageFT")
  attributes['d_name'] = "testManager"
  attributes["history"] = "test_1_manager"
  attributes["filter"]  =  "whole"
  
  time_point_tuple = hMang.retrieveData(attributes)
  del time_point_tuple
  attributes["history"] = "test_2_manager"
  time_point_tuple = hMang.retrieveData(attributes)
  
  print(time_point_tuple)
  
  print("success")
  
  
  
  
   