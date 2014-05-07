'''
Created on July 10, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

#import Datas
import numpy as np
import os
from utils import toString, toBytes

'''
  ********************************
  *  SPECIALIZED FILTER CLASSES  *
  ********************************
'''

class PrintCSV:
  '''
    PrintCSV filter class. It prints a CSV file loading data from a hdf5 database or other sources 
  '''
  def __init__(self):
    self.paramters = []

  def _readMoreXML(self,xmlNode):
    '''
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    '''
    param = xmlNode.text
    if(param.lower() != 'all'): self.paramters = param.strip().split(',')
    else: self.paramters.append(param) 
    return
  
  def collectOutput(self,finishedjob,output):
    # Check the input type 
    if(self.inObj.type == "HDF5"):
      #  Input source is a database (HDF5)
      #  Retrieve the ending groups' names
      endGroupNames = self.inObj.getEndingGroupNames()
      histories = {}

      #  Construct a dictionary of all the histories
      for index in range(len(endGroupNames)): histories[endGroupNames[index]] = self.inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
      try:
        # not yet implemented 
        outType = output.type
        print('FIXME: in filters the outType selection has not yet been implemented')
      except AttributeError:
        #  If file, split the strings and add the working directory if present
        for key in histories:
          #  Loop over histories
          #  Retrieve the metadata (posion 1 of the history tuple)
          attributes = histories[key][1]
          #  Construct the header in csv format (first row of the file)
          headers = b",".join([histories[key][1]['output_space_headers'][i] for i in 
                               range(len(attributes['output_space_headers']))])
          #  Construct history name
          hist = key
          #  If file, split the strings and add the working directory if present
          if self.workingDir:
            if os.path.split(output)[1] == '': output = output[:-1]
            splitted_1 = os.path.split(output)
            output = splitted_1[1]
          splitted = output.split('.')
          #  Create csv files' names
          addfile = splitted[0] + '_additional_info_' + hist + '.'+splitted[1]
          csvfilen = splitted[0] + '_' + hist + '.'+splitted[1]
          #  Check if workingDir is present and in case join the two paths
          if self.workingDir:
            addfile = os.path.join(self.workingDir,addfile)
            csvfilen = os.path.join(self.workingDir,csvfilen)
          
          #  Open the files and save the data
          with open(csvfilen, 'wb') as csvfile, open(addfile, 'wb') as addcsvfile:
            #  Add history to the csv file
            np.savetxt(csvfile, histories[key][0], delimiter=",",header=toString(headers))
            csvfile.write(b' \n')
            #  process the attributes in a different csv file (different kind of informations) 
            #  Add metadata to additional info csv file
            addcsvfile.write(b'# History Metadata, \n')
            addcsvfile.write(b'# ______________________________,' + b'_'*len(key)+b','+b'\n')
            addcsvfile.write(b'#number of parameters,\n')
            addcsvfile.write(toBytes(str(attributes['n_params']))+b',\n')
            addcsvfile.write(b'#parameters,\n') 
            addcsvfile.write(headers+b'\n') 
            addcsvfile.write(b'#parent_id,\n') 
            addcsvfile.write(toBytes(attributes['parent_id'])+b'\n') 
            addcsvfile.write(b'#start time,\n')
            addcsvfile.write(toBytes(str(attributes['start_time']))+b'\n')
            addcsvfile.write(b'#end time,\n')
            addcsvfile.write(toBytes(str(attributes['end_time']))+b'\n')
            addcsvfile.write(b'#number of time-steps,\n')
            addcsvfile.write(toBytes(str(attributes['n_ts']))+b'\n')
            if 'initiator_distribution' in attributes:
              init_dist = attributes['initiator_distribution']
              addcsvfile.write(b'#number of branches in this history,\n')
              addcsvfile.write(toBytes(str(len(init_dist)))+b'\n')
              string_work = ''
              for i in range(len(init_dist)):
                string_work_2 = ''
                for j in init_dist[i]: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','          
              addcsvfile.write(b'#initiator distributions,\n')
              addcsvfile.write(toBytes(string_work)+b'\n')
            if 'end_timestep' in attributes:
              string_work = ''
              end_ts = attributes['end_timestep']
              for i in xrange(len(end_ts)): string_work = string_work + str(end_ts[i]) + ','          
              addcsvfile.write('#end time step,\n')
              addcsvfile.write(str(string_work)+'\n')
            if 'branch_changed_param' in attributes:
              string_work = ''
              branch_changed_param = attributes['branch_changed_param']
              for i in range(len(branch_changed_param)):
                string_work_2 = ''
                for j in branch_changed_param[i]:
                  if not j: string_work_2 = string_work_2 + 'None' + ' '
                  else: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','          
              addcsvfile.write(b'#changed parameters,\n')
              addcsvfile.write(toBytes(str(string_work))+b'\n')
            if 'branch_changed_param_value' in attributes:
              string_work = ''
              branch_changed_param_value = attributes['branch_changed_param_value']
              for i in range(len(branch_changed_param_value)):
                string_work_2 = ''
                for j in branch_changed_param_value[i]:
                  if not j: string_work_2 = string_work_2 + 'None' + ' '
                  else: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','                          
              addcsvfile.write(b'#changed parameters values,\n')
              addcsvfile.write(toBytes(str(string_work))+b'\n')
            if 'conditional_prb' in attributes:
              string_work = ''
              cond_pbs = attributes['conditional_prb']
              for i in range(len(cond_pbs)):
                string_work_2 = ''
                for j in cond_pbs[i]:
                  if not j: string_work_2 = string_work_2 + 'None' + ' '
                  else: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','                
              addcsvfile.write(b'#conditional probability,\n')
              addcsvfile.write(toBytes(str(string_work))+b'\n')
            if 'PbThreshold' in attributes:
              string_work = ''
              pb_thresholds = attributes['PbThreshold']
              for i in range(len(pb_thresholds)):
                string_work_2 = ''
                for j in pb_thresholds[i]:
                  if not j: string_work_2 = string_work_2 + 'None' + ' '
                  else: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','
              addcsvfile.write(b'#Probability threshold,\n')
              addcsvfile.write(toBytes(str(string_work))+b'\n')
            addcsvfile.write(b' \n')
            
    elif(self.inObj.type == "Datas"):
      pass
    else:
      raise NameError ('Filter PrintCSV for input type ' + self.inObj.type + ' not yet implemented.')
  
  def finalizeFilter(self,inObj,workingDir=None):
    '''
     Function to finalize the filter => execute the filtering 
     @ In, inObj      : Input object (for example HDF5 object)
     @ In, workingDir : Working directory (where to store the csvs)
     @ Out, None      : Print of the CSV file
    '''
    self.inObj = inObj
    self.workingDir = workingDir
    return

'''
 Interface Dictionary (factory) (private)
'''
__base                          = 'Filter'
__interFaceDict                 = {}
__interFaceDict['PrintCSV']     = PrintCSV
__knownTypes                    = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnFilterInterface(Type):
  '''
    function used to generate a Filter class
    @ In, Type : Filter type
    @ Out,Instance of the Specialized Filter class
  '''  
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)  


  
