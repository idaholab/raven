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
from utils import *

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

  def readMoreXML(self,xmlNode):
    '''
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    '''
    param = ''
    param = xmlNode.text
    if(param.lower() != 'all'): self.paramters = param.strip().split(',')
    else: self.paramters.append(param) 
    return

  def finalizeFilter(self,inObj,outObj,workingDir=None):
    '''
     Function to finalize the filter => execute the filtering 
     @ In, inObj      : Input object (for example HDF5 object)
     @ In, outObj     : Output object (in this case is the csv file name) => string 
     @ In, workingDir : Working directory (where to store the csvs)
     @ Out, None      : Print of the CSV file
    '''
    
    # Check the input type 
    if(inObj.type == "HDF5"):

      #  Input source is a database (HDF5)
      
      #  Retrieve the ending groups' names
      endGroupNames = inObj.getEndingGroupNames()
      histories = {}

      #  Construct a dictionary of all the histories
      for index in range(len(endGroupNames)): histories[endGroupNames[index]] = inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
      
      try:
        # not yet implemented 
        outType = outObj.type
      except AttributeError:
#        splitted = outObj.split('.')
#        addfile = splitted[0] + '_additional_info.' + splitted[1]
#        with open(outObj, 'w') as csvfile, open(addfile, 'w') as addcsvfile:
        #  If file, split the strings and add the working directory if present
        for key in histories:
          #  Loop over histories
          #  Retrieve the metadata (posion 1 of the history tuple)
          attributes = histories[key][1]
          #  Construct the header in csv format (first row of the file)
          headers = b",".join([histories[key][1]['headers'][i] for i in 
                               range(len(attributes['headers']))])
          #  Construct history name
          hist = key
          #  If file, split the strings and add the working directory if present
          if workingDir:
            if os.path.split(outObj)[1] == '': outObj = outObj[:-1]
            splitted_1 = os.path.split(outObj)
            outObj = splitted_1[1]
          splitted = outObj.split('.')
          #  Create csv files' names
          addfile = splitted[0] + '_additional_info_' + hist + '.'+splitted[1]
          csvfilen = splitted[0] + '_' + hist + '.'+splitted[1]
          #  Check if workingDir is present and in case join the two paths
          if workingDir:
            addfile = os.path.join(workingDir,addfile)
            csvfilen = os.path.join(workingDir,csvfilen)
          
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
              addcsvfile.write('#number of branches in this history,\n')
              addcsvfile.write(str(len(init_dist))+'\n')
              string_work = ''
              for i in xrange(len(init_dist)):
                string_work_2 = ''
                for j in init_dist[i]: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','          
              addcsvfile.write('#initiator distributions,\n')
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
              for i in xrange(len(branch_changed_param)):
                string_work_2 = ''
                for j in branch_changed_param[i]:
                  if not j: string_work_2 = string_work_2 + 'None' + ' '
                  else: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','          
              addcsvfile.write('#changed parameters,\n')
              addcsvfile.write(str(string_work)+'\n')
            if 'branch_changed_param_value' in attributes:
              string_work = ''
              branch_changed_param_value = attributes['branch_changed_param_value']
              for i in xrange(len(branch_changed_param_value)):
                string_work_2 = ''
                for j in branch_changed_param_value[i]:
                  if not j: string_work_2 = string_work_2 + 'None' + ' '
                  else: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','                          
              addcsvfile.write('#changed parameters values,\n')
              addcsvfile.write(str(string_work)+'\n')
            if 'conditional_prb' in attributes:
              string_work = ''
              cond_pbs = attributes['conditional_prb']
              for i in xrange(len(cond_pbs)):
                string_work_2 = ''
                for j in cond_pbs[i]:
                  if not j: string_work_2 = string_work_2 + 'None' + ' '
                  else: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','                
              addcsvfile.write('#conditional probability,\n')
              addcsvfile.write(str(string_work)+'\n')
            if 'Probability_threshold' in attributes:
              string_work = ''
              pb_thresholds = attributes['Probability_threshold']
              for i in xrange(len(pb_thresholds)):
                string_work_2 = ''
                for j in pb_thresholds[i]:
                  if not j: string_work_2 = string_work_2 + 'None' + ' '
                  else: string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','
              addcsvfile.write('#Probability threshold,\n')
              addcsvfile.write(str(string_work)+'\n')
            addcsvfile.write(b' \n')
            
    elif(inObj.type == "Datas"):
      pass
    else:
      raise NameError ('Filter PrintCSV for input type ' + inObj.type + ' not yet implemented.')

class Plot:
  '''
    Plot filter class. It plots histories from a database HDF5 or CSV file/s
  '''
  def __init__(self):
    self.paramters = []

  def readMoreXML(self,xmlNode):
    '''
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    '''
 #   param = ''
 #   param = xmlNode.text
    return

  def finalizeFilter(self,inObj,outObj,workingDir=None):
    '''
      Function to finalize the filter => execute the filtering 
      @ In, inObj      : Input object (for example HDF5 object)
      @ In, outObj     : Output object (Plot type)
      @ In, workingDir : Working directory (where to store the csvs)
    '''

    #  Check the input type 
    if(inObj.type == "HDF5"):
      #  Input source is a database (HDF5)

      #  Retrieve the ending groups' names
      endGroupNames = inObj.getEndingGroupNames()
      histories = {}
      #  Retrieve histories from HDF5 database
      for index in xrange(len(endGroupNames)):
        histories[endGroupNames[index]] = inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
      
    elif (inObj.type == "CSV"):
      #  not implemented yet
      pass
    else:
      raise NameError ('Filter Plot for input type ' + inObj.type + ' not yet implemented.')
    #  Plot the histories
    fig = [None]*len(endGroupNames)
    for i in range (len(endGroupNames)):
      fig[i]=plt.figure()
      plt.plot(histories[endGroupNames[1]],histories[endGroupNames[i]])
      plt.xlabel('Time')
      plt.ylabel(histories[key][1]['headers'][i])
      plt.title('Plot of history:', i)
      if (outObj.type == "screen"):
        plt.show()
      elif (outObj.type == "jpeg"):
        fileName=str(histories[endGroupNames[i]])+'.jpeg'
        fig[i].savefig(fileName,dpi=fig.dpi)  # dpi=fig.dpi is to keep same same figure rendering of show() also for savefig()
      elif (outObj.type == "png"):
        fileName=str(histories[endGroupNames[i]])+'.png'
        fig[i].savefig(fileName,dpi=fig.dpi)
      elif (outObj.type == "eps"):
        fileName=str(histories[endGroupNames[i]])+'.eps'
        fig[i].savefig(fileName,dpi=fig.dpi)        
      elif (outObj.type == "pdf"):
        fileName=str(histories[endGroupNames[i]])+'.pdf'
        fig[i].savefig(fileName,dpi=fig.dpi)        
      else:
        raise NameError ('Filter Plot for output type ' + outObj.type + ' not implemented.')  
    return


'''
 Interface Dictionary (factory) (private)
'''
__base                          = 'Filter'
__interFaceDict                 = {}
__interFaceDict['PrintCSV'] = PrintCSV
__interFaceDict['Plot'    ] = Plot
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


  
