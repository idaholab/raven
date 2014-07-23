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
from sklearn import tree
import scipy.stats as stat
import os
from glob import glob
import imp
import math
import inspect
import copy as cp
from utils import toString, toBytes, metaclass_insert, first
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
  def __init__(self):
    self.type =''     # pp type
    self.name = None  # pp name
    self.externalFunction = None
    self.debug = False
  def initialize(self, runInfo, inputs, initDict) : 
    if 'externalFunction' in initDict.keys(): self.externalFunction = initDict['externalFunction']
    self.inputs           = inputs
  def _readMoreXML(self,xmlNode):
    self.type = xmlNode.tag
    self.name = xmlNode.attrib['name']
    if 'debug' in xmlNode.attrib.keys():self.debug = bool(xmlNode.attrib['debug'])
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

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
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
        print(attributes)
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
    self.parameters        = {}                                                                                                      #parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.acceptedCalcParam = ['covariance','NormalizedSensitivity','sensitivity','pearson','expectedValue','sigma','variationCoefficient','variance','skewness','kurtois','median','percentile']  # accepted calculation parameters
    self.what              = self.acceptedCalcParam                                                                                  # what needs to be computed... default...all
    self.methodsToRun      = []                                                                                                      # if a function is present, its outcome name is here stored... if it matches one of the known outcomes, the pp is going to use the function to compute it
    #self.goalFunction.evaluate('residuumSign',tempDict)

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
#     # now we check if the sampler that genereted the samples are from adaptive... in case... create the grid
      if inputDict['metadata'].keys().count('SamplerType') > 0: pass
        #if inputDict['metadata']['SamplerType'] == 'Adaptive':
        #pass
        
    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']

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
      if child.tag =="parameters"   : self.parameters['targets'] = child.text.split(',')
      if child.tag =="methodsToRun" : self.methodsToRun          = child.text.split(',')

  def collectOutput(self,finishedjob,output):
    #output
    if finishedjob.returnEvaluation() == -1: raise Exception("POSTPROC: ERROR -> No available Output to collect (Run probabably is not finished yet)")
    outputDict = finishedjob.returnEvaluation()[1]
    methodToTest = []
    for key in self.methodsToRun:
      if key not in self.acceptedCalcParam: methodToTest.append(key)
    
    with open(os.path.join(self.__workingDir,self.name+'_out.txt'), 'wb') as basicStatdump:
      basicStatdump.write('POSTPROC: BasicStatistics '+str(self.name)+'pp outputs\n')
      for targetP in self.parameters['targets']:
        basicStatdump.write('        *************'+'*'*len(targetP)+'***\n')
        basicStatdump.write('        * Variable * '+ targetP +'  *\n')
        basicStatdump.write('        *************'+'*'*len(targetP)+'***\n')
        for what in outputDict.keys():
          if what not in ['covariance','pearson','NormalizedSensitivity','sensitivity'] + methodToTest:
            basicStatdump.write('              '+'**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***\n')
            basicStatdump.write('              '+'* '+what+' * ' + '%.8E' % outputDict[what][targetP]+'  *\n')
            basicStatdump.write('              '+'**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***\n')
      maxLenght = max(len(max(self.parameters['targets'], key=len))+5,16)
      if 'covariance' in outputDict.keys():
        basicStatdump.write(' '*maxLenght+'*****************************\n')
        basicStatdump.write(' '*maxLenght+'*         Covariance        *\n')
        basicStatdump.write(' '*maxLenght+'*****************************\n')

        basicStatdump.write(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in self.parameters['targets']])+'\n')
        for index in range(len(self.parameters['targets'])):
          basicStatdump.write(self.parameters['targets'][index] + ' '*(maxLenght-len(self.parameters['targets'][index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['covariance'][index]])+'\n')
      if 'pearson' in outputDict.keys():
        basicStatdump.write(' '*maxLenght+'*****************************\n')
        basicStatdump.write(' '*maxLenght+'*          Pearson          *\n')
        basicStatdump.write(' '*maxLenght+'*****************************\n')
        basicStatdump.write(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in self.parameters['targets']])+'\n')
        for index in range(len(self.parameters['targets'])):
          basicStatdump.write(self.parameters['targets'][index] + ' '*(maxLenght-len(self.parameters['targets'][index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['pearson'][index]])+'\n')
      if 'sensitivity' in outputDict.keys():
        basicStatdump.write(' '*maxLenght+'*****************************\n')
        basicStatdump.write(' '*maxLenght+'*        Sensitivity        *\n')
        basicStatdump.write(' '*maxLenght+'*****************************\n')
        basicStatdump.write(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in self.parameters['targets']])+'\n')
        for index in range(len(self.parameters['targets'])):
          basicStatdump.write(self.parameters['targets'][index] + ' '*(maxLenght-len(self.parameters['targets'][index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['pearson'][index]])+'\n')
      if 'NormalizedSensitivity' in outputDict.keys():
        basicStatdump.write(' '*maxLenght+'*****************************\n')
        basicStatdump.write(' '*maxLenght+'*   Normalized Sensitivity  *\n')
        basicStatdump.write(' '*maxLenght+'*****************************\n')
        basicStatdump.write(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in self.parameters['targets']])+'\n')
        for index in range(len(self.parameters['targets'])):
          basicStatdump.write(self.parameters['targets'][index] + ' '*(maxLenght-len(self.parameters['targets'][index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['pearson'][index]])+'\n')

      if self.externalFunction:
        basicStatdump.write(' '*maxLenght+'+++++++++++++++++++++++++++++\n')
        basicStatdump.write(' '*maxLenght+'+ OUTCOME FROM EXT FUNCTION +\n')
        basicStatdump.write(' '*maxLenght+'+++++++++++++++++++++++++++++\n')
        for what in self.methodsToRun:
          if what not in self.acceptedCalcParam:
            basicStatdump.write('              '+'**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***\n')
            basicStatdump.write('              '+'* '+what+' * ' + '%.8E' % outputDict[what]+'  *\n')
            basicStatdump.write('              '+'**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***\n')

  def run(self, InputIn):
    '''
     Function to finalize the filter => execute the filtering
     @ In , dictionary       : dictionary of data to process
     @ Out, dictionary       : Dictionary with results
    '''
    Input  = self.inputToInternal(InputIn)
    outputDict = {}
    if 'ProbabilityWeight' not in Input['metadata'].keys():
      if Input['metadata'].keys().count('SamplerType') > 0:
        if Input['metadata']['SamplerType'][0] != 'MC' : print('POSTPROC: Warning -> BasicStatistics postprocessor can not compute expectedValue without ProbabilityWeights. Use unit weight')
      else: print('POSTPROC: Warning -> BasicStatistics postprocessor can not compute expectedValue without ProbabilityWeights. Use unit weight')
      pbweights = 1.0
      globPb = 1.0*len(Input['targets'][self.parameters['targets'][0]])
    else: 
      pbweights = Input['metadata']['ProbabilityWeight']  
      globPb = np.sum(Input['metadata']['ProbabilityWeight'])

    if self.externalFunction:
      # there is an external function
      for what in self.methodsToRun: 
        outputDict[what] = self.externalFunction.evaluate(what,Input['targets'])      
        # check if "what" corresponds to an internal method
        if what in self.acceptedCalcParam:
          if what not in ['pearson','covariance','NormalizedSensitivity','sensitivity']:
            if type(outputDict[what]) != dict: raise IOError('POSTPROC: ERROR -> BasicStatistics postprocessor: You have overwritten the "'+what+'" method through an external function, it must be a dictionary!!')
          else:
            if type(outputDict[what]) != np.ndarray: raise IOError('POSTPROC: ERROR -> BasicStatistics postprocessor: You have overwritten the "'+what+'" method through an external function, it must be a numpy.ndarray!!')
            if len(outputDict[what].shape) != 2:     raise IOError('POSTPROC: ERROR -> BasicStatistics postprocessor: You have overwritten the "'+what+'" method through an external function, it must be a 2D numpy.ndarray!!')
    # if here because the user could have overwritten the method through the external function
    if 'expectedValue' not in outputDict.keys(): outputDict['expectedValue'] = {}
    
    for targetP in self.parameters['targets']:
      if targetP not in outputDict['expectedValue'].keys():
        if Input['metadata'].keys().count('ProbabilityWeight') > 0:
          outputDict['expectedValue'][targetP]= np.sum(np.multiply(Input['metadata']['ProbabilityWeight'],Input['targets'][targetP]))/np.sum(Input['metadata']['ProbabilityWeight'])
        else: outputDict['expectedValue'][targetP]= np.sum(np.multiply(pbweights,Input['targets'][targetP]))/globPb

    for what in self.what:
      if what not in outputDict.keys(): outputDict[what] = {}
      if what == 'sigma':
        #sigma
        for targetP in self.parameters['targets']:
          if targetP not in outputDict[what].keys():
            if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
            else                                      : N = Input['targets'][targetP].size
            if Input['metadata'].keys().count('ProbabilityWeight') > 0:
              outputDict[what][targetP] = np.average(((Input['targets'][targetP]-outputDict['expectedValue'][targetP])**2)**0.5, weights=Input['metadata']['ProbabilityWeight'])
            else: outputDict[what][targetP] = (np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1.0)**0.5
      if what == 'variance':
        #variance
        for targetP in self.parameters['targets']:
          if targetP not in outputDict[what].keys():
            if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
            else                                      : N = Input['targets'][targetP].size
            if Input['metadata'].keys().count('ProbabilityWeight') > 0:
              outputDict[what][targetP] = np.average(((Input['targets'][targetP]-outputDict['expectedValue'][targetP])**2), weights=Input['metadata']['ProbabilityWeight'])
            else: outputDict[what][targetP] = (np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1)**0.5
      if what == 'variationCoefficient':
        #coefficient of variation (sigma/mu)
        for targetP in self.parameters['targets']:
          if targetP not in outputDict[what].keys():
            if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
            else                                      : N = Input['targets'][targetP].size
            if Input['metadata'].keys().count('ProbabilityWeight') > 0: sigma = np.average((Input['targets'][targetP]-outputDict['expectedValue'][targetP])**2, weights=Input['metadata']['ProbabilityWeight'])**0.5
            else: sigma = (np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1.0)**0.5
            outputDict[what][targetP] = copy.deepcopy(sigma/outputDict['expectedValue'][targetP])
      if what == 'kurtois':
        #kurtois
        for targetP in self.parameters['targets']:
          if targetP not in outputDict[what].keys():
            if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
            else                                      : N = Input['targets'][targetP].size
            if Input['metadata'].keys().count('ProbabilityWeight') > 0:
              sigma = np.average((Input['targets'][targetP]-outputDict['expectedValue'][targetP])**2, weights=Input['metadata']['ProbabilityWeight'])**0.5
              outputDict[what][targetP] = np.average(((Input['targets'][targetP]-outputDict['expectedValue'][targetP])**4), weights=Input['metadata']['ProbabilityWeight'])/sigma**4
            else: 
              outputDict[what][targetP] = -3.0 + (np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**4)*(N-1)**-1)/(np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1)**2
      if what == 'skewness':
        #skewness
        for targetP in self.parameters['targets']:
          if targetP not in outputDict[what].keys():
            if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
            else                                      : N = Input['targets'][targetP].size
            if Input['metadata'].keys().count('ProbabilityWeight') > 0:
              sigma = np.average((Input['targets'][targetP]-outputDict['expectedValue'][targetP])**2, weights=Input['metadata']['ProbabilityWeight'])**0.5
              outputDict[what][targetP] = np.average((((Input['targets'][targetP]-outputDict['expectedValue'][targetP])/sigma)**3), weights=Input['metadata']['ProbabilityWeight'])
            else: 
              outputDict[what][targetP] = (np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**3)*(N-1)**-1)/(np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1)**1.5
      if what == 'median':
        #median
        if targetP not in outputDict[what].keys():
          for targetP in self.parameters['targets'  ]: outputDict[what][targetP]  = np.median(Input['targets'][targetP]  )
      if what == 'pearson':
        #pearson matrix
        if targetP not in outputDict[what].keys():
          feat = np.zeros((len(Input['targets'].keys()),first(Input['targets'].values()).size))
          for cnt, targetP in enumerate(self.parameters['targets']):
            feat[cnt,:] = Input['targets'][targetP][:]
          outputDict[what] = np.corrcoef(feat)
      if what == 'covariance':
        #cov matrix
        if targetP not in outputDict[what].keys():
          feat = np.zeros((len(Input['targets'].keys()),first(Input['targets'].values()).size))
          for cnt, targetP in enumerate(self.parameters['targets']):
            feat[cnt,:] = Input['targets'][targetP][:]
          outputDict[what] = np.cov(feat)
      if what == 'sensitivity':
        #sensitivity matrix
        if 'covariance' not in outputDict.keys():  
          if targetP not in outputDict[what].keys():
            feat = np.zeros((len(Input['targets'].keys()),first(Input['targets'].values()).size))
            for cnt, targetP in enumerate(self.parameters['targets']):
              feat[cnt,:] = Input['targets'][targetP][:]
            covarianceM = np.cov(feat)
        else: covarianceM = outputDict['covariance']
        if 'sigma' not in outputDict.keys() and 'variance' not in outputDict.keys():  
          varianceS = {}
          for targetP in self.parameters['targets']:
            if targetP not in outputDict[what].keys():
              if type(Input['targets'][targetP]) == list: N = len(Input['targets'][targetP])
              else                                      : N = Input['targets'][targetP].size
              if Input['metadata'].keys().count('ProbabilityWeight') > 0:
                varianceS[targetP] = np.average(((Input['targets'][targetP]-outputDict['expectedValue'][targetP])**2), weights=Input['metadata']['ProbabilityWeight'])
              else: varianceS[targetP] = (np.sum((np.asarray(Input['targets'][targetP]) - outputDict['expectedValue'][targetP])**2)*(N-1)**-1.0)
        else: 
          if 'sigma' in outputDict.keys(): 
            for targetP in self.parameters['targets']: varianceS[targetP] = outputDict['sigma'][targetP]**2
          else: varianceS = outputDict['variance']
        outputDict[what] = np.zeros(covarianceM.shape)
        for cnt,targetP in enumerate(self.parameters['targets']):  outputDict[what][cnt,:] = copy.deepcopy(covarianceM[cnt,:]/varianceS[targetP])   
      if what == 'NormalizedSensitivity':
        #sensitivity matrix
        if 'covariance' not in outputDict.keys():  
          if targetP not in outputDict[what].keys():
            feat = np.zeros((len(Input['targets'].keys()),first(Input['targets'].values()).size))
            for cnt, targetP in enumerate(self.parameters['targets']):
              feat[cnt,:] = Input['targets'][targetP][:]
            covarianceM = np.cov(feat)
        else: covarianceM = outputDict['covariance']
        outputDict[what] = np.zeros(covarianceM.shape)
        expectedValues = np.zeros(len(self.parameters['targets']))
        for cnt,targetP in enumerate(self.parameters['targets']): expectedValues[cnt] = outputDict['expectedValue'][targetP]
        for cnt,targetP in enumerate(self.parameters['targets']):  outputDict[what][cnt,:] = copy.deepcopy(covarianceM[cnt,:]*expectedValues/outputDict['expectedValue'][targetP])   
   
      if what == 'percentile':
        outputDict.pop(what)
        if what+'_5%'  not in outputDict.keys(): outputDict[what+'_5%']  ={}
        if what+'_95%' not in outputDict.keys(): outputDict[what+'_95%'] ={}
        for targetP in self.parameters['targets'  ]:
          if targetP not in outputDict[what+'_5%'].keys():
            outputDict[what+'_5%'][targetP]  = np.percentile(Input['targets'][targetP],5)
          if targetP not in outputDict[what+'_95%'].keys():
            outputDict[what+'_95%'][targetP]  = np.percentile(Input['targets'][targetP],95)

    # print on screen
    print('POSTPROC: BasicStatistics '+str(self.name)+'pp outputs')
    methodToTest = []
    for key in self.methodsToRun:
      if key not in self.acceptedCalcParam: methodToTest.append(key)
    for targetP in self.parameters['targets']:
      print('        *************'+'*'*len(targetP)+'***')
      print('        * Variable * ' + targetP +'  *')
      print('        *************'+'*'*len(targetP)+'***')
      for what in outputDict.keys():
        if what not in ['covariance','pearson','NormalizedSensitivity','sensitivity'] + methodToTest:
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
    if 'sensitivity' in outputDict.keys():
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght,'*        Sensitivity        *')
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in self.parameters['targets']]))
      for index in range(len(self.parameters['targets'])):
        print(self.parameters['targets'][index] + ' '*(maxLenght-len(self.parameters['targets'][index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['pearson'][index]]))
    if 'NormalizedSensitivity' in outputDict.keys():
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght,'*   Normalized Sensitivity  *')
      print(' '*maxLenght,'*****************************')
      print(' '*maxLenght+''.join([str(item) + ' '*(maxLenght-len(item)) for item in self.parameters['targets']]))
      for index in range(len(self.parameters['targets'])):
        print(self.parameters['targets'][index] + ' '*(maxLenght-len(self.parameters['targets'][index])) + ''.join(['%.8E' % item + ' '*(maxLenght-14) for item in outputDict['pearson'][index]]))

    if self.externalFunction:
      print(' '*maxLenght,'+++++++++++++++++++++++++++++')
      print(' '*maxLenght,'+ OUTCOME FROM EXT FUNCTION +')
      print(' '*maxLenght,'+++++++++++++++++++++++++++++')
      for what in self.methodsToRun:
        if what not in self.acceptedCalcParam:
          print('              ','**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***')
          print('              ','* '+what+' * ' + '%.8E' % outputDict[what]+'  *')
          print('              ','**'+'*'*len(what)+ '***'+6*'*'+'*'*8+'***')
    return outputDict
#
#
#
#

class LoadCsvIntoInternalObject(BasePostProcessor):
  '''
    LoadCsvIntoInternalObject pp class. It is in charge of loading CSV files into one of the internal object (Data(s) or HDF5)
  '''
  def __init__(self):
    BasePostProcessor.__init__(self)
    self.sourceDirectory = None
    self.listOfCsvFiles = []

  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']
    if '~' in self.sourceDirectory               : self.sourceDirectory = os.path.expanduser(self.sourceDirectory)
    if not os.path.isabs(self.sourceDirectory)   : self.sourceDirectory = os.path.normpath(os.path.join(self.__workingDir,self.sourceDirectory))
    if not os.path.exists(self.sourceDirectory)  : raise IOError("POSTPROC: ERROR -> The directory indicated for PostProcessor "+ self.name + "does not exist. Path: "+self.sourceDirectory)
    for _dir,_,_ in os.walk(self.sourceDirectory): self.listOfCsvFiles.extend(glob(os.path.join(_dir,"*.csv")))
    if len(self.listOfCsvFiles) == 0             : raise IOError("POSTPROC: ERROR -> The directory indicated for PostProcessor "+ self.name + "does not contain any csv file. Path: "+self.sourceDirectory)
    self.listOfCsvFiles.sort()
    
  def inputToInternal(self,currentInput): return self.listOfCsvFiles

  def _localReadMoreXML(self,xmlNode):
    '''
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    '''
    for child in xmlNode:
      if child.tag =="directory": self.sourceDirectory = child.text
    if not self.sourceDirectory: raise IOError("POSTPROC: ERROR -> The PostProcessor "+ self.name + "needs a directory for loading the csv files!")

  def collectOutput(self,finishedjob,output):
    #output
    '''collect the output file in the output object'''
    for index,csvFile in enumerate(self.listOfCsvFiles):
      
      attributes={"prefix":str(index),"input_file":self.name,"type":"csv","name":os.path.join(self.sourceDirectory,csvFile)}
      metadata = finishedjob.returnMetadata()
      if metadata:
        for key in metadata: attributes[key] = metadata[key]
      try:                   output.addGroup(attributes,attributes)
      except AttributeError:
        output.addOutput(os.path.join(self.sourceDirectory,csvFile),attributes)
        if metadata:
          for key,value in metadata.items(): output.updateMetadata(key,value,attributes)

  def run(self, InputIn):  return self.listOfCsvFiles
#
#
#
#

class LimitSurface(BasePostProcessor):
  '''
    LimitSurface filter class. It computes the limit surface associated to a dataset
  '''

  def __init__(self):
    BasePostProcessor.__init__(self)
    self.parameters        = {}               #parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.surfPoint         = None             #coordinate of the points considered on the limit surface
    self.testMatrix        = None             #This is the n-dimensional matrix representing the testing grid
    self.oldTestMatrix     = None             #This is the test matrix to use to store the old evaluation of the function
    self.functionValue     = {}               #This a dictionary that contains np vectors with the value for each variable and for the goal function
    self.ROM               = None
    self.subGridTol        = 1.0e-4
     
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
  
  def _griddataInterface(self,action,data,target):
    m = len(list(data.keys()))
    if target in data.keys(): m -= 1
    n = len( data[  list(data.keys())[0]   ]  )
    dataMatrix = np.zeros((n,m))
    if action=='train':
      self._KDTreeMappingList = []
      cnt = 0
      for key in data.keys(): 
          if key == target: targetValues = data[key]
          else:
            self._KDTreeMappingList.append(key)
            dataMatrix[:,cnt] = data[key]
            cnt+=1
      self._tree = tree.DecisionTreeClassifier()
      self._tree.fit(dataMatrix,targetValues)
    elif action == 'evaluate':
      for key in data.keys(): dataMatrix[:,self._KDTreeMappingList.index(key)] = data[key]
      return self._tree.predict(dataMatrix)
  
  def initialize(self, runInfo, inputs, initDict):
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = copy.deepcopy(runInfo['WorkingDir'])
    indexes = [-1,-1]
    for index,inp in enumerate(self.inputs):
      if type(inp) in [str,bytes,unicode]: raise IOError('POSTPROC: Error -> LimitSurface PostProcessor only accepts Data(s) as inputs!') 
      if inp.type in ['TimePointSet','TimePoint']: indexes[0] = index
    if indexes[0] == -1: raise IOError('POSTPROC: Error -> LimitSurface PostProcessor needs a TimePoint or TimePointSet as INPUT!!!!!!')
    else:
      # check if parameters are contained in the data
      inpKeys = self.inputs[indexes[0]].getParaKeys("inputs")
      outKeys = self.inputs[indexes[0]].getParaKeys("outputs")
      self.paramType ={}
      for param in self.parameters['targets']: 
        if param not in inpKeys+outKeys: raise IOError('POSTPROC: Error -> LimitSurface PostProcessor: The param '+ param+' not contained in Data '+self.inputs[indexes[0]].name +' !')
        if param in inpKeys: self.paramType[param] = 'inputs'
        else:                self.paramType[param] = 'outputs'
    # check if a ROM is present
    if 'ROM' in initDict.keys():
      if initDict['ROM']: indexes[1] = 1          
    if indexes[1] == -1:
      class ROM(object):
        def __init__(self,cKDTreeInterface,target):
          self.amItrained = False
          self._cKDTreeInterface = cKDTreeInterface
          self.target = target
        def train(self,trainSet,):
          self._cKDTreeInterface('train',trainSet,self.target)
          self.amItrained = True
        def evaluate(self,coordinateVect): return self._cKDTreeInterface('evaluate',coordinateVect,self.target)
        def confidence(self,coordinateVect): return self._cKDTreeInterface('confidence',coordinateVect,self.target)[0]
      self.ROM = ROM(self._griddataInterface,self.externalFunction.name)
    else: self.ROM = initDict['ROM'] 
    self.nVar        = len(self.parameters['targets'])         #Total number of variables
    stepLenght        = self.subGridTol**(1./float(self.nVar)) #build the step size in 0-1 range such as the differential volume is equal to the tolerance
    self.axisName     = []                                     #this list is the implicit mapping of the name of the variable with the grid axis ordering self.axisName[i] = name i-th coordinate
    self.gridVectors  = {}
    #here we build lambda function to return the coordinate of the grid point depending if the tolerance is on probability or on volume
    stepParam = lambda x: [stepLenght*(max(self.inputs[indexes[0]].getParam(self.paramType[x],x))-min(self.inputs[indexes[0]].getParam(self.paramType[x],x))),
                                       min(self.inputs[indexes[0]].getParam(self.paramType[x],x)),
                                       max(self.inputs[indexes[0]].getParam(self.paramType[x],x))]

    #moving forward building all the information set
    pointByVar = [None]*self.nVar                              #list storing the number of point by cooridnate
    #building the grid point coordinates
    for varId, varName in enumerate(self.parameters['targets']):
      self.axisName.append(varName)
      [myStepLenght, start, end]  = stepParam(varName)
      start                      += 0.5*myStepLenght
      self.gridVectors[varName]   = np.arange(start,end,myStepLenght)
      pointByVar[varId]           = np.shape(self.gridVectors[varName])[0]
    self.gridShape                = tuple   (pointByVar)          #tuple of the grid shape
    self.testGridLenght           = np.prod (pointByVar)          #total number of point on the grid
    self.testMatrix               = np.zeros(self.gridShape)      #grid where the values of the goalfunction are stored
    self.oldTestMatrix            = np.zeros(self.gridShape)      #swap matrix fro convergence test
    self.gridCoorShape            = tuple(pointByVar+[self.nVar]) #shape of the matrix containing all coordinate of all points in the grid
    self.gridCoord                = np.zeros(self.gridCoorShape)  #the matrix containing all coordinate of all points in the grid
    #filling the coordinate on the grid
    myIterator = np.nditer(self.gridCoord,flags=['multi_index'])
    while not myIterator.finished:
      coordinateID  = myIterator.multi_index[-1]
      axisName      = self.axisName[coordinateID]
      valuePosition = myIterator.multi_index[coordinateID]
      self.gridCoord[myIterator.multi_index] = self.gridVectors[axisName][valuePosition]
      myIterator.iternext()
    self.axisStepSize = {}
    for varName in self.parameters['targets']:
      self.axisStepSize[varName] = np.asarray([self.gridVectors[varName][myIndex+1]-self.gridVectors[varName][myIndex] for myIndex in range(len(self.gridVectors[varName])-1)])
           
      
      
  def _localReadMoreXML(self,xmlNode):
    '''
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode    : Xml element node
      @ Out, None
    '''
    for child in xmlNode:
      if child.tag =="parameters"   : self.parameters['targets'] = child.text.split(',')
      if child.tag =="tollerance"   : self.subGridTol          = float(child.text)

  def collectOutput(self,finishedjob,output):
    #output
    if finishedjob.returnEvaluation() == -1: raise Exception("POSTPROC: ERROR -> No available Output to collect (Run probabably is not finished yet)")
    limitSurf = finishedjob.returnEvaluation()[1]
    if limitSurf[0]!=None:
      for varName in output.getParaKeys('inputs'):
        for varIndex in range(len(self.axisName)):
          if varName == self.axisName[varIndex]:
            output.removeInputValue(varName)
            for value in limitSurf[0][:,varIndex]: output.updateInputValue(varName,copy.copy(value))    
      output.removeOutputValue('OutputPlaceOrder')
      for value in limitSurf[1]: output.updateOutputValue('OutputPlaceOrder',copy.copy(value)) 

  def run(self, InputIn): # inObj,workingDir=None):
    '''
     Function to finalize the filter => execute the filtering
     @ In , dictionary       : dictionary of data to process
     @ Out, dictionary       : Dictionary with results
    '''
    #Input  = self.inputToInternal(InputIn)
    print('Initiate training')
    self.functionValue.update(InputIn.getParametersValues('input'))
    self.functionValue.update(InputIn.getParametersValues('output'))
    #recovery the index of the last function evaluation performed
    if self.externalFunction.name in self.functionValue.keys(): indexLast = len(self.functionValue[self.externalFunction.name])-1
    else                                                  : indexLast = -1
    #index of last set of point tested and ready to perform the function evaluation
    
    indexEnd  = len(self.functionValue[self.axisName[0]])-1
    tempDict  = {}
    if self.externalFunction.name in self.functionValue.keys():
      self.functionValue[self.externalFunction.name] = np.append( self.functionValue[self.externalFunction.name], np.zeros(indexEnd-indexLast))
    else: self.functionValue[self.externalFunction.name] = np.zeros(indexEnd+1)
    
    for myIndex in range(indexLast+1,indexEnd+1):
      for key, value in self.functionValue.items(): tempDict[key] = copy.deepcopy(value[myIndex])       
      #self.hangingPoints= self.hangingPoints[    ~(self.hangingPoints==np.array([tempDict[varName] for varName in self.axisName])).all(axis=1)     ][:]
      self.functionValue[self.externalFunction.name][myIndex] =  self.externalFunction.evaluate('residuumSign',tempDict)
      if abs(self.functionValue[self.externalFunction.name][myIndex]) != 1.0: raise Exception("POSTPROC: ERROR -> LimitSurface: the function evaluation of the residuumSign method needs to return a 1 or -1!")
      if self.externalFunction.name in InputIn.getParaKeys('inputs'): InputIn.self.updateInputValue (self.externalFunction.name,self.functionValue[self.externalFunction.name][myIndex])
      if self.externalFunction.name in InputIn.getParaKeys('output'): InputIn.self.updateOutputValue(self.externalFunction.name,self.functionValue[self.externalFunction.name][myIndex])
    if np.sum(self.functionValue[self.externalFunction.name]) == float(len(self.functionValue[self.externalFunction.name])) or np.sum(self.functionValue[self.externalFunction.name]) == -float(len(self.functionValue[self.externalFunction.name])):
      raise Exception("POSTPROC: ERROR -> LimitSurface: all the Function evaluations brought to the same result (No Limit Surface has been crossed...). Increase or change the data set!") 

    #printing----------------------
    if self.debug: print('POSTPROC: Message -> LimitSurface: Mapping of the goal function evaluation performed')
    if self.debug:
      print('POSTPROC: Print -> LimitSurface: Already evaluated points and function values:')
      keyList = list(self.functionValue.keys())
      print(','.join(keyList))
      for index in range(indexEnd+1):
        print(','.join([str(self.functionValue[key][index]) for key in keyList]))
    #printing----------------------
    tempDict = {}
    for name in self.axisName: tempDict[name] = self.functionValue[name]
    tempDict[self.externalFunction.name] = self.functionValue[self.externalFunction.name]
    self.ROM.train(tempDict)
    print('POSTPROC: Message -> LimitSurface: Training performed')
    if self.debug: print('POSTPROC: Message -> LimitSurface: Training finished')                                   
    np.copyto(self.oldTestMatrix,self.testMatrix)                                #copy the old solution for convergence check
    self.testMatrix.shape     = (self.testGridLenght)                            #rearrange the grid matrix such as is an array of values
    self.gridCoord.shape      = (self.testGridLenght,self.nVar)                  #rearrange the grid coordinate matrix such as is an array of coordinate values
    tempDict ={}
    for  varId, varName in enumerate(self.axisName): tempDict[varName] = self.gridCoord[:,varId]
    self.testMatrix[:]        = self.ROM.evaluate(tempDict)                      #get the prediction on the testing grid
    self.testMatrix.shape     = self.gridShape                                   #bring back the grid structure
    self.gridCoord.shape      = self.gridCoorShape                               #bring back the grid structure
    if self.debug: print('POSTPROC: Message -> LimitSurface: Prediction performed')
    #here next the points that are close to any change are detected by a gradient (it is a pre-screener)
    toBeTested = np.squeeze(np.dstack(np.nonzero(np.sum(np.abs(np.gradient(self.testMatrix)),axis=0))))
    #printing----------------------
    if self.debug:
      print('POSTPROC: Print -> LimitSurface:  Limit surface candidate points')
      for coordinate in np.rollaxis(toBeTested,0):
        myStr = ''
        for iVar, varnName in enumerate(self.axisName): myStr +=  varnName+': '+str(coordinate[iVar])+'      '
        print('POSTPROC: LimitSurface: ' + myStr+'  value: '+str(self.testMatrix[tuple(coordinate)]))
    #printing----------------------
    #check which one of the preselected points is really on the limit surface
    listsurfPoint = []
    myIdList      = np.zeros(self.nVar)
    for coordinate in np.rollaxis(toBeTested,0):
      myIdList[:] = copy.deepcopy(coordinate)
      if int(self.testMatrix[tuple(coordinate)])<0: #we seek the frontier sitting on the -1 side
        for iVar in range(self.nVar):
          if coordinate[iVar]+1<self.gridShape[iVar]: #coordinate range from 0 to n-1 while shape is equal to n
            myIdList[iVar]+=1
            if self.testMatrix[tuple(myIdList)]>=0:
              listsurfPoint.append(copy.copy(coordinate))
              break
            myIdList[iVar]-=1
          if coordinate[iVar]>0:
            myIdList[iVar]-=1
            if self.testMatrix[tuple(myIdList)]>=0:
              listsurfPoint.append(copy.copy(coordinate))
              break
            myIdList[iVar]+=1
    #printing----------------------
    if self.debug:
      print('POSTPROC: Print -> LimitSurface: Limit surface points:')
      for coordinate in listsurfPoint:
        myStr = ''
        for iVar, varnName in enumerate(self.axisName): myStr +=  varnName+': '+str(coordinate[iVar])+'      '
        print('POSTPROC: LimitSurface: ' + myStr+'  value: '+str(self.testMatrix[tuple(coordinate)]))
    #printing----------------------

    #if the number of point on the limit surface is > than zero than save it
    outputPlaceOrder = np.zeros(len(listsurfPoint))
    if len(listsurfPoint)>0:
      self.surfPoint = np.ndarray((len(listsurfPoint),self.nVar))
      for pointID, coordinate in enumerate(listsurfPoint):
        self.surfPoint[pointID,:] = self.gridCoord[tuple(coordinate)]
        outputPlaceOrder[pointID] = pointID

    return self.surfPoint,outputPlaceOrder




'''
 Interface Dictionary (factory) (private)
'''
__base                                       = 'PostProcessor'
__interFaceDict                              = {}
__interFaceDict['PrintCSV'                 ] = PrintCSV
__interFaceDict['BasicStatistics'          ] = BasicStatistics
__interFaceDict['LoadCsvIntoInternalObject'] = LoadCsvIntoInternalObject
__interFaceDict['LimitSurface'             ] = LimitSurface
__knownTypes                                 = __interFaceDict.keys()

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
