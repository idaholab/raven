'''
Created on July 10, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import Datas
import numpy as np

class PrintCSV:
  def __init__(self):
    self.paramters = []

  def readMoreXML(self,xmlNode):
    param = ''
    param = xmlNode.text
    if(param.lower() != 'all'):
      self.paramters = param.strip().split(',')
    else:
      self.paramters.append(param) 
    return

  def finalizeFilter(self,inObj,outObj):
    # check the input type
    if(inObj.type == "HDF5"):
      endGroupNames = inObj.getEndingGroupNames()
      histories = {}
      for index in xrange(len(endGroupNames)):
        histories[endGroupNames[index]] = inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
      
      try:
        outType = outObj.type
        #not yet implemented
      except:
#        splitted = outObj.split('.')
#        addfile = splitted[0] + '_additional_info.' + splitted[1]
#        with open(outObj, 'w') as csvfile, open(addfile, 'w') as addcsvfile:
        for key in histories:
          headers = ''
          attributes = histories[key][1]
          for i in xrange(len(attributes['headers'])):
            headers = headers + histories[key][1]['headers'][i] + ','
          try:
            hist = ''
            hist = key
            hist = hist.replace(',','_') 
          except:
            hist = key
          splitted = outObj.split('.')
          addfile = splitted[0] + '_additional_info_' + hist + '.'+splitted[1]
          csvfilen = splitted[0] + '_' + hist + '.'+splitted[1]
          with open(csvfilen, 'w') as csvfile, open(addfile, 'w') as addcsvfile:            
            np.savetxt(csvfile, histories[key][0], delimiter=",",header=headers,comments='history,' + hist +'\n')
            csvfile.write(' '+'\n')
            #process the attributes in a different csv file (different kind of informations)
            addcsvfile.write('history,'+hist+','+'\n')
            addcsvfile.write('________________________________,' + '_'*len(key)+','+'\n')
            addcsvfile.write('number of parameters,'+str(attributes['n_params'])+'\n')
            addcsvfile.write('parameters,'+headers+'\n') 
            addcsvfile.write('parent,'+str(attributes['parent_id'])+'\n') 
            addcsvfile.write('start time,'+str(attributes['start_time'])+'\n')
            addcsvfile.write('end time,'+str(attributes['end_time'])+'\n')
            addcsvfile.write('number of time-steps,'+str(attributes['n_ts'])+'\n')
            try:
              init_dist = attributes['initiator_distribution']
              addcsvfile.write('number of branches in this history,'+str(len(init_dist))+'\n')
              string_work = ''
              for i in xrange(len(init_dist)):
                string_work_2 = ''
                for j in init_dist[i]:
                  string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','          
              addcsvfile.write('initiator distributions,'+str(string_work)+'\n')
            except:
              pass
            try:
              string_work = ''
              end_ts = attributes['end_timestep']
              for i in xrange(len(end_ts)):
                string_work = string_work + str(end_ts[i]) + ','          
              addcsvfile.write('end time step,'+str(string_work)+'\n')
            except:
              pass             
            try:
              string_work = ''
              branch_changed_param = attributes['branch_changed_param']
              for i in xrange(len(branch_changed_param)):
                string_work_2 = ''
                for j in branch_changed_param[i]:
                  if not j:
                    string_work_2 = string_work_2 + 'None' + ' '
                  else:
                    string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','          
              addcsvfile.write('changed parameters,'+str(string_work)+'\n')
            except:
              pass
            try:
              string_work = ''
              branch_changed_param_value = attributes['branch_changed_param_value']
              for i in xrange(len(branch_changed_param_value)):
                string_work_2 = ''
                for j in branch_changed_param_value[i]:
                  if not j:
                    string_work_2 = string_work_2 + 'None' + ' '
                  else:
                    string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','                          
              addcsvfile.write('changed parameters values,'+str(string_work)+'\n')
            except:
              pass
            try:
              string_work = ''
              cond_pbs = attributes['conditional_prb']
              for i in xrange(len(cond_pbs)):
                string_work_2 = ''
                for j in cond_pbs[i]:
                  if not j:
                    string_work_2 = string_work_2 + 'None' + ' '
                  else:
                    string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','                
              addcsvfile.write('conditional probability,'+str(string_work)+'\n')
            except:
              pass
            try:
              string_work = ''
              pb_thresholds = attributes['Probability_threshold']
              for i in xrange(len(pb_thresholds)):
                string_work_2 = ''
                for j in pb_thresholds[i]:
                  if not j:
                    string_work_2 = string_work_2 + 'None' + ' '
                  else:
                    string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','
              addcsvfile.write('Probability threshold,'+str(string_work)+'\n')
            except:
              pass            
            addcsvfile.write(' '+'\n')
            
    elif(inObj.type == "Datas"):
      pass
    else:
      raise NameError ('Filter PrintCSV for input type ' + inObj.type + ' not yet implemented.')

class Plot:
  def __init__(self):
    self.paramters = []
    
  def readMoreXML(self,xmlNode):
    param = ''
    param = xmlNode.text
    return
  
  def finalizeFilter(self,inObj,outObj):    
    if(inObj.type == "HDF5"):
      endGroupNames = inObj.getEndingGroupNames()
      histories = {}
      for index in xrange(len(endGroupNames)):
        histories[endGroupNames[index]] = inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
      
    elif (inObj.type == "CSV"):
      # do something
      pass
    else:
      raise NameError ('Filter Plot for input type ' + inObj.type + ' not yet implemented.')
    
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
        
def returnFilterInterface(Type):
  base = 'Filter'
  filterInterfaceDict = {}
  filterInterfaceDict['PrintCSV'] = PrintCSV
  filterInterfaceDict['Plot'] = Plot
  try: return filterInterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)

  