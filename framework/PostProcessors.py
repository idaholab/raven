'''
Created on July 10, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

#import Datas
import numpy as np
import scipy as sci
import scipy.stats as stat
import os
from utils import toString, toBytes, metaclass_insert
import copy
#Internal Modules------------------------------------------------------------------------------------
import abc
#Internal Modules End--------------------------------------------------------------------------------

'''
  ***************************************
  *  SPECIALIZED PostProcessor CLASSES  *
  ***************************************
'''

class BasePostProcessor:
  def __init__(self): self.type =''
  def initialize(self, runInfo, inputs, externalFunction) : self.externalFunction = externalFunction
  def _readMoreXML(self,xmlNode): 
    self.type = xmlNode.tag
    self._localReadMoreXML(xmlNode) 
  def inputToInternal(self,currentInput): return [(copy.deepcopy(currentInput))]
  def run(self, Input): pass
  
class PrintCSV(BasePostProcessor):
  '''
    PrintCSV PostProcessor class. It prints a CSV file loading data from a hdf5 database or other sources 
  '''
  def __init__(self):
    BasePostProcessor.__init__(self)
    self.paramters  = ['all']
    self.inObj      = None
    self.workingDir = None
  
  def inputToInternal(self,currentInput): return [(currentInput)]
  
  def initialize(self, runInfo, inputs, externalFunction):
    BasePostProcessor.initialize(self, runInfo, inputs, externalFunction)
    self.workingDir               = os.path.join(runInfo['WorkingDir'],runInfo['stepName']) #generate current working dir
    runInfo['TempWorkingDir']     = self.workingDir
    try:                            os.mkdir(self.workingDir)
    except:                         print('POST-PROCESSOR: Warning -> current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
  
  def _localReadMoreXML(self,xmlNode):
    '''
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    '''
    for child in xmlNode:
      if child.tag == 'parameters':
        param = child.text
        if(param.lower() != 'all'): self.paramters = param.strip().split(',')
        else: self.paramters[param]
  
  def collectOutput(self,finishedjob,output):
    # Check the input type 
    if(self.inObj.type == "HDF5"):
      #  Input source is a database (HDF5)
      #  Retrieve the ending groups' names
      endGroupNames = self.inObj.getEndingGroupNames()
      histories = {}

      #  Construct a dictionary of all the histories
      for index in range(len(endGroupNames)): histories[endGroupNames[index]] = self.inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
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
      raise NameError ('PostProcessor PrintCSV for input type ' + self.inObj.type + ' not yet implemented.')
  
  def run(self, Input): # inObj,workingDir=None):
    '''
     Function to finalize the filter => execute the filtering 
     @ Out, None      : Print of the CSV file
    '''
    self.inObj = Input


  
class BasicStatistics(BasePostProcessor):
  '''
    BasicStatistics filter class. It computes all the most popular statistics
  '''
  def __init__(self):
    BasePostProcessor.__init__(self)
    self.parameters        = {}
    self.acceptedCalcParam = ['covariance','pearson','expectedValue','sigma','variance','kurtois','median','percentile','skewness']
    self.what              = self.acceptedCalcParam 
  
  def inputToInternal(self,currentInput): 
    # each post processor knows how to handle the coming inputs. The BasicStatistics postprocessor accept all the input type (files (csv only), hdf5 and datas
    if type(currentInput) == dict:
      if 'targets' in currentInput.keys(): return
    inputDict = {'targets':{},'metadata':{}}
    try: inType = currentInput.type
    except: 
      if type(currentInput) in [str,bytes,unicode]: inType = "file"
      else: raise IOError('POSTPROC: Error -> BasicStatistics postprocessor accepts files,HDF5,Data(s) only! Got '+ str(type(currentInput)))
    if inType == 'file':
      if currentInput.endswith('csv'): pass
    if inType == 'HDF5': pass # to be implemented
    if inType in ['TimePointSet']: 
      for targetP in self.parameters['targets']: 
        if   targetP in currentInput.getParaKeys('input' ): inputDict['targets'][targetP] = currentInput.getParam('input' ,targetP)
        elif targetP in currentInput.getParaKeys('output'): inputDict['targets'][targetP] = currentInput.getParam('output',targetP)
      inputDict['metadata'] = currentInput.getAllMetadata()        
    # to be added  
    return inputDict
  
  def initialize(self, runInfo, inputs, externalFunction = None):
    BasePostProcessor.initialize(self, runInfo, inputs, externalFunction)
  
  def _localReadMoreXML(self,xmlNode):
    '''
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    '''
    for child in xmlNode:
      if child.tag =="what": 
        self.what = child.text
        if self.what == 'all': self.what = self.acceptedCalcParam
        else:
          for whatc in self.what.split(','):
            if whatc not in self.acceptedCalcParam: raise IOError('POSTPROC: Error -> BasicStatistics postprocessor asked unknown operation ' + whatc + '. Available '+str(self.acceptedCalcParam))
          self.what = self.what.split(',')
      if child.tag =="parameters": self.parameters['targets'] = child.text.split(',')

        
        
  def collectOutput(self,finishedjob,output):
    #output
    pass
  
  def run(self, InputIn): # inObj,workingDir=None):
    '''
     Function to finalize the filter => execute the filtering 
     @ In , dictionary       : dictionary of data to process 
     @ Out, dictionary       : Dictionary with results
    '''
    Input  = self.inputToInternal(InputIn)
    outputDict = {}   
    if 'ProbabilityWeight' not in Input['metadata'].keys(): 
      print('POSTPROC: Warning -> BasicStatistics postprocessor can not compute expectedValue without ProbabilityWeights. Use unit weight')
      pbweights = 1.0
    else: pbweights = Input['metadata']['ProbabilityWeight']  
    outputDict['expectedValue'] = {}
    globPb = np.sum(Input['metadata']['ProbabilityWeight'])
    for targetP in self.parameters['targets']: outputDict['expectedValue'][targetP]= np.sum(np.multiply(pbweights,Input['targets'][targetP]))/globPb             
    for what in self.what:
      if what == 'sigma':
        #sigma
        outputDict[what] = {}
        for targetP in self.parameters['targets']:
          if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
          else                                      : N = Input['targets'][targetP].size 
          outputDict[what][targetP] = (np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1)**0.5
      if what == 'variance':
        #variance
        outputDict[what] = {}
        for targetP in self.parameters['targets']:
          if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
          else                                      : N = Input['targets'][targetP].size 
          outputDict[what][targetP] = np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1
      if what == 'kurtois':
        #kurtois
        outputDict[what] = {}
        for targetP in self.parameters['targets']:
          if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
          else                                      : N = Input['targets'][targetP].size 
          outputDict[what][targetP] = -3.0 + (np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**4)*(N-1)**-1)/(np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1)**2
      if what == 'median':
        #variance
        outputDict[what] ={}
        for targetP in self.parameters['targets'  ]: outputDict[what][targetP]  = np.median(Input['targets'][targetP]  )
      if what == 'pearson':   
        #pearson matrix 
        feat = np.zeros((len(Input['targets'].keys()),Input['targets'].values()[0].size))
        cnt = 0
        for targetP in self.parameters['targets'  ]: 
          feat[cnt,:] = Input['targets'][targetP][:] 
          cnt += 1  
        outputDict[what] = np.corrcoef(feat)
      if what == 'covariance':   
        #cov matrix 
        feat = np.zeros((len(Input['targets'].keys()),Input['targets'].values()[0].size))
        cnt = 0
        for targetP in self.parameters['targets'  ]: 
          feat[cnt,:] = Input['targets'][targetP][:] 
          cnt += 1  
        outputDict[what] = np.cov(feat)
      if what == 'percentile':
        outputDict[what+'_5%']  ={} 
        outputDict[what+'_95%'] ={} 
        for targetP in self.parameters['targets'  ]: 
          outputDict[what+'_5%'][targetP]  = np.percentile(Input['targets'][targetP],5)
          outputDict[what+'_95%'][targetP]  = np.percentile(Input['targets'][targetP],95)
      if what == 'skewness':
        outputDict[what] = {}
        for targetP in self.parameters['targets'  ]: 
          outputDict[what][targetP] = stat.skew(Input['targets'][targetP])
          outputDict[what][targetP] = stat.skew(Input['targets'][targetP])  
    # print on screen
    print('POSTPROC: BasicStatistics pp outputs')
    for targetP in self.parameters['targets']:
      print('        *************'+'*'*len(targetP)+'***')
      print('        * Variable * ' + targetP +'  *')
      print('        *************'+'*'*len(targetP)+'***')
      for what in outputDict.keys():
        if what not in ['covariance','pearson']:
          print('              ','**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***')
          print('              ','* '+what+' * ' + '%.8E' % outputDict[what][targetP]+'  *')  
          print('              ','**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***') 
    maxLenght = max(len(max(self.parameters['targets'], key=len))+5,16)
    if 'covariance' in outputDict.keys():    
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght,'*         Covariance        *')
      print(' '*maxLenght,'*****************************')

      print(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in self.parameters['targets']]))
      for index in range(len(self.parameters['targets'])):
        print(self.parameters['targets'][index] + ' '*(maxLenght-len(self.parameters['targets'][index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['covariance'][index]]))  
    if 'pearson' in outputDict.keys():
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght,'*          Pearson          *')
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in self.parameters['targets']]))
      for index in range(len(self.parameters['targets'])):
        print(self.parameters['targets'][index] + ' '*(maxLenght-len(self.parameters['targets'][index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['pearson'][index]]))        
    return outputDict
  
'''
 Interface Dictionary (factory) (private)
'''
__base                             = 'Filter'
__interFaceDict                    = {}
__interFaceDict['PrintCSV'       ] = PrintCSV
__interFaceDict['BasicStatistics'] = BasicStatistics
__knownTypes                       = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  '''
    function used to generate a Filter class
    @ In, Type : Filter type
    @ Out,Instance of the Specialized Filter class
  '''  
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)  


  
