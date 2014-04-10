'''
Created on Jun 8, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy

from utils import toString

class RavenInterface:
  '''this class is used as part of a code dictionary to specialize Model.Code for RAVEN'''
  def generateCommand(self,inputFiles,executable):
    '''seek which is which of the input files and generate According the running command'''
    if inputFiles[0].endswith('.i'): index = 0
    else: index = 1
    outputfile = 'out~'+os.path.split(inputFiles[index])[1].split('.')[0]
    executeCommand = (executable+' -i '+os.path.split(inputFiles[index])[1]+' Output/postprocessor_csv=true' +
    ' Output/file_base='+ outputfile)
    return executeCommand,outputfile

  def appendLoadFileExtension(self,fileRoot):
    '''  '''
    return fileRoot + '.csv'

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    '''this generate a new input file depending on which sampler has been chosen'''
    import MOOSEparser
    self._samplersDictionary                          = {}
    self._samplersDictionary['MonteCarlo'           ] = self.monteCarloForRAVEN
    self._samplersDictionary['Grid'                 ] = self.gridForRAVEN
    self._samplersDictionary['LHS'                  ] = self.latinHyperCubeForRAVEN
    self._samplersDictionary['DynamicEventTree'     ] = self.dynamicEventTreeForRAVEN
    self._samplersDictionary['StochasticCollocation'] = self.stochasticCollocationForRAVEN
    if currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    parser = MOOSEparser.MOOSEparser(currentInputFiles[index])
    modifDict = self._samplersDictionary[samplerType](**Kwargs)
    parser.modifyOrAdd(modifDict,False)
    temp = str(oriInputFiles[index][:])
    newInputFiles = copy.deepcopy(currentInputFiles)
    #TODO fix this? storing unwieldy amounts of data in 'prefix'
    if type(Kwargs['prefix']) in [str,type("")]:#Specifing string type for python 2 and 3
      newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],Kwargs['prefix']+"~"+os.path.split(temp)[1]))
    else:
      newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],str(Kwargs['prefix'][1][0])+"~"+os.path.split(temp)[1]))
    parser.printInput(newInputFiles[index])
    return newInputFiles

  def stochasticCollocationForRAVEN(self,**Kwargs):
    if 'prefix' not in Kwargs['prefix']:
      raise IOError('a counter is (currently) needed for the StochColl sampler for RAVEN')
    listDict = []
    varValDict = Kwargs['vars'] #come in as a string of a list, need to re-list
    for key in varValDict.keys():
      modifDict={}
      modifDict['name']=key.split(':')
      modifDict['value']=varValDict[key]
      #print('interface: set',key.split(':'),'to',varValDict[key])
      listDict.append(modifDict)
      del modifDict
    return listDict

  def monteCarloForRAVEN(self,**Kwargs):
    if 'prefix' in Kwargs: counter = Kwargs['prefix']
    else: raise IOError('a counter is needed for the Monte Carlo sampler for RAVEN')
    if 'initial_seed' in Kwargs:
      init_seed = Kwargs['initial_seed']
    else:
      init_seed = 1
    listDict = []
    modifDict = {}
    modifDict['name'] = ['Distributions']
    RNG_seed = int(counter) + int(init_seed) - 1
    modifDict[b'RNG_seed'] = str(RNG_seed)
    listDict.append(modifDict)
    return listDict

  def dynamicEventTreeForRAVEN(self,**Kwargs):
    listDict = []
    # Check the initiator distributions and add the next threshold
    if 'initiator_distribution' in Kwargs.keys():
      for i in range(len(Kwargs['initiator_distribution'])):
        modifDict = {}
        modifDict['name'] = ['Distributions',Kwargs['initiator_distribution'][i]]
        modifDict['ProbabilityThreshold'] = Kwargs['PbThreshold'][i]
        listDict.append(modifDict)
        del modifDict
    # add the initial time for this new branch calculation
    if 'start_time' in Kwargs.keys():
      if Kwargs['start_time'] not in ['Initial',b'Initial']:
        modifDict = {}
        st_time = Kwargs['start_time']
        modifDict['name'] = ['Executioner']
        modifDict['start_time'] = st_time
        listDict.append(modifDict)
        del modifDict
    # create the restart file name root from the parent branch calculation
    # in order to restart the calc from the last point in time
    if 'end_ts' in Kwargs.keys():
      #if Kwargs['end_ts'] != 0 or Kwargs['end_ts'] == 0:

      if Kwargs['start_time'] not in ['Initial',b'Initial']:
        modifDict = {}
        end_ts_str = str(Kwargs['end_ts'])
        if(Kwargs['end_ts'] <= 9999):
          n_zeros = 4 - len(end_ts_str)
          for i in range(n_zeros):
            end_ts_str = "0" + end_ts_str
        splitted = Kwargs['outfile'].split('~')
        output_parent = splitted[0] + '~' + toString(Kwargs['parent_id']) + '~' + splitted[1]
        #restart_file_base = output_parent + "_restart_" + end_ts_str
        restart_file_base = output_parent + "_cp/" + end_ts_str
        modifDict['name'] = ['Executioner']
        modifDict['restart_file_base'] = restart_file_base
        print('CODE INTERFACE: Restart file name base is "' + restart_file_base + '"')
        listDict.append(modifDict)
        del modifDict
    # max simulation time (if present)
    if 'end_time' in Kwargs.keys():
      modifDict = {}
      end_time = Kwargs['end_time']
      modifDict['name'] = ['Executioner']
      modifDict['end_time'] = end_time
      listDict.append(modifDict)
      del modifDict

    modifDict = {}
    modifDict['name'] = ['Output']
    modifDict['num_checkpoint_files'] = 1
    listDict.append(modifDict)
    del modifDict
    # in this way we erase the whole block in order to neglect eventual older info
    # remember this "command" must be added before giving the info for refilling the block
    modifDict = {}
    modifDict['name'] = ['RestartInitialize']
    modifDict['special'] = set(['erase_block'])
    listDict.append(modifDict)
    del modifDict
    # check and add the variables that have been changed by a distribution trigger
    # add them into the RestartInitialize block
    if 'branch_changed_param' in Kwargs.keys():
      if Kwargs['branch_changed_param'][0] not in ('None',b'None'):
        for i in range(len(Kwargs['branch_changed_param'])):
          modifDict = {}
          modifDict['name'] = ['RestartInitialize',Kwargs['branch_changed_param'][i]]
          modifDict['value'] = Kwargs['branch_changed_param_value'][i]
          listDict.append(modifDict)
          del modifDict
    return listDict

  def __genBasePointSampler(self,**Kwargs):
    """Figure out which distributions need to be handled by
    the grid or LHS samplers by modifying distributions in the .i file.
    Let the regular moose point sampler take care of the rest.
    Returns (distributions,listDict) where listDict is the
    start of the listDict that tells how to modify the input, and
    distributions is a dictionary with keys that are the 'variable name'
    and values of [computedValue,distribution name in .i file]
    Note that the key has "<distribution>" in front of the variable name.
    The actual variable can be gotten from the full key by:
    key[len('<distribution>'):]
    TODO This should check that the distributions in the .i file (if
    they exist) are consistent with the ones in the .xml file.
    TODO For variables, it should add them to the .csv file.
    """
    #print("Kwargs",Kwargs,"SampledVars",Kwargs["SampledVars"])
    distributionKeys = [key for key in Kwargs["SampledVars"] if key.startswith("<distribution>")]
    distributions = {}
    for key in distributionKeys:
      distributionName = Kwargs['distributionName'][key]
      distributionType = Kwargs['distributionType'][key]
      distributions[key] = [Kwargs["SampledVars"].pop(key),distributionName,
                            distributionType]
    mooseApp = MooseBasedAppInterface()
    listDict = mooseApp.pointSamplerForMooseBasedApp(**Kwargs)
    return distributions,listDict

  __FrameworkToRavenCDistNames = {'Uniform':'UniformDistribution',
                                  'Normal':'NormalDistribution',
                                  'Gamma':'GammaDistribution',
                                  'Beta':'BetaDistribution',
                                  'Triangular':'TriangularDistribution',
                                  'Poisson':'PoissonDistribution',
                                  'Binomial':'BinomialDistribution',
                                  'Bernoulli':'BernoulliDistribution',
                                  'Logistic':'LogisticDistribution',
                                  'Exponential':'ExponentialDistribution',
                                  'LogNormal':'LogNormalDistribution',
                                  'Weibull':'WeibullDistribution'  }

  def gridForRAVEN(self,**Kwargs):
    """Uses point sampler to generate variable points, and
    modifies distributions to be a zerowidth (constant) distribution
    at the grid point.
    """
    distributions,listDict = self.__genBasePointSampler(**Kwargs)
    for key in distributions.keys():
      distName, distType = distributions[key][1:3]
      listDict.append({'name':['Distributions',distName],
                       'special':set(['assert_match']),
                       'type':self.__FrameworkToRavenCDistNames[distType]})
      listDict.append({'name':['Distributions',distName],'special':set(['erase_block'])})
      listDict.append({'name':['Distributions',distName],'value':distributions[key][0]})
      listDict.append({'name':['Distributions',distName],'type':'ConstantDistribution'})
    #print("listDict",listDict,"distributions",distributions,"Kwargs",Kwargs)
    return listDict

  def latinHyperCubeForRAVEN(self,**Kwargs):
    """Uses point sampler to generate variable points, and truncates
    distribution to be inside of the latin hyper cube upper and lower
    bounds.
    """
    distributions,listDict = self.__genBasePointSampler(**Kwargs)
    for key in distributions.keys():
      distName, distType = distributions[key][1:3]
      listDict.append({'name':['Distributions',distName],
                       'special':set(['assert_match']),
                       'type':self.__FrameworkToRavenCDistNames[distType]})
      listDict.append({'name':['Distributions',distName],
                       'xMax':Kwargs['upper'][key]})
      listDict.append({'name':['Distributions',distName],
                       'xMin':Kwargs['lower'][key]})
    #print("listDict",listDict,"distributions",distributions)
    return listDict

class MooseBasedAppInterface:
  '''this class is used as part of a code dictionary to specialize Model.Code for RAVEN'''
  def generateCommand(self,inputFiles,executable):
    '''seek which is which of the input files and generate According the running command'''
    if inputFiles[0].endswith('.i'): index = 0
    else: index = 1
    outputfile = 'out~'+os.path.split(inputFiles[index])[1].split('.')[0]
    executable_tail = os.path.split(executable)[-1]
    #XXX What is the proper test to see if we should use Outputs or Output?
    print('FIXME: Fix this if statement when r7_moose gets consistent with the other moose-base-app for Output')
    if executable_tail.count('RAVEN') !=0 or executable_tail.count('r7_moose') !=0:
      executeCommand = (executable+' -i '+os.path.split(inputFiles[index])[1]+' Output/postprocessor_csv=true' +
                        ' Output/file_base='+ outputfile)
    else:
      executeCommand = (executable+' -i '+os.path.split(inputFiles[index])[1] +
                        ' Outputs/file_base='+ outputfile)
    return executeCommand,outputfile

  def appendLoadFileExtension(self,fileRoot):
    '''  '''
    return fileRoot + '.csv'

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    '''this generate a new input file depending on which sampler has been chosen'''
    import MOOSEparser
    self._samplersDictionary                          = {}
    self._samplersDictionary['MonteCarlo'           ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['Grid'                 ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['LHS'                  ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['DynamicEventTree'     ] = self.dynamicEventTreeForMooseBasedApp
    self._samplersDictionary['StochasticCollocation'] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['Adaptive']              = self.pointSamplerForMooseBasedApp
    if currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    parser = MOOSEparser.MOOSEparser(currentInputFiles[index])
    modifDict = self._samplersDictionary[samplerType](**Kwargs)
    modifDict.append(copy.deepcopy({'name':['Outputs'],'special':set(['erase_block'])}))
    modifDict.append(copy.deepcopy({'name':['Outputs'],'exodus':'true'}))
    modifDict.append(copy.deepcopy({'name':['Outputs'],'interval':'1'}))
    modifDict.append(copy.deepcopy({'name':['Outputs'],'output_initial':'true'}))
    modifDict.append(copy.deepcopy({'name':['Outputs'],'csv':'true'}))
    modifDict.append(copy.deepcopy({'name':['Output'], 'special':set(['erase_block'])}))
    parser.modifyOrAdd(modifDict,False)
    temp = str(oriInputFiles[index][:])
    newInputFiles = copy.deepcopy(currentInputFiles)
    #TODO fix this? storing unwieldy amounts of data in 'prefix'
    if type(Kwargs['prefix']) in [str,type("")]:#Specifing string type for python 2 and 3
      newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],Kwargs['prefix']+"~"+os.path.split(temp)[1]))
    else:
      newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],str(Kwargs['prefix'][1][0])+"~"+os.path.split(temp)[1]))
    parser.printInput(newInputFiles[index])
    return newInputFiles

  def pointSamplerForMooseBasedApp(self,**Kwargs):
    listDict  = []
    modifDict = {}
    # the position in, eventually, a vector variable is not available yet...
    # the MOOSEparser needs to be modified in order to accept this variable type
    # for now the position (i.e. ':' at the end of a variable name) is discarded
    for var in Kwargs['SampledVars']:
        if 'alias' in Kwargs.keys():
          # for understending the alias system, plase check module Models.py (class Code)
          if var in Kwargs['alias'].keys():
            key = Kwargs['alias'][var].split(':')
            varname = var
        else:
          key = var.split(':')
          varname = key[0]
        modifDict = {}
        modifDict['name'] = []
        modifDict['name'] = key[0].split('|')[:-1]
        modifDict[key[0].split('|')[-1]] = Kwargs['SampledVars'][var]
        listDict.append(modifDict)
        del modifDict
        if 'raven' not in Kwargs['executable'].lower():
          listDict.append({'name':['AuxVariables',varname],'family':'SCALAR'})
          listDict.append({'name':['AuxVariables',varname],'initial_condition':Kwargs['SampledVars'][var]})
          listDict.append({'name':['Postprocessors',varname],'type':'ScalarVariable'})
          listDict.append({'name':['Postprocessors',varname],'variable':varname})
    return listDict

  def dynamicEventTreeForMooseBasedApp(self,**Kwargs):
    raise IOError('dynamicEventTreeForMooseBasedApp not yet implemented')
    listDict = []
    return listDict



class Relap5Interface:
  '''this class is used a part of a code dictionary to specialize Model.Code for RELAP5-3D Version 4.0.3'''
  def generateCommand(self,inputFiles,executable):
    '''seek which is which of the input files and generate According the running command'''
    if inputFiles[0].endswith('.i'): index = 0
    else: index = 1
    outputfile = 'out~'+os.path.split(inputFiles[index])[1].split('.')[0]
    #   executeCommand will consist of a simple RELAP script that runs relap for inputfile
    #   extracts data and stores in csv file format
    executeCommand = (executable+' '+os.path.split(inputFiles[index])[1]+' ' +
    outputfile)
    return executeCommand,outputfile

  def appendLoadFileExtension(self,fileRoot):
    '''  '''
    return fileRoot + '.csv'

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    '''this generate a new input file depending on which sampler is chosen'''
    import RELAPparser
    self._samplersDictionary                          = {}
    self._samplersDictionary['MonteCarlo'           ] = self.pointSamplerForRELAP5
    self._samplersDictionary['Grid'                 ] = self.pointSamplerForRELAP5
    self._samplersDictionary['LHS'                  ] = self.pointSamplerForRELAP5
    self._samplersDictionary['DynamicEventTree'     ] = self.DynamicEventTreeForRELAP5
    self._samplersDictionary['StochasticCollocation'] = self.pointSamplerForRELAP5
    if currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    parser = RELAPparser.RELAPparser(currentInputFiles[index])
    modifDict = self._samplersDictionary[samplerType](**Kwargs)
    parser.modifyOrAdd(modifDict,True)
    temp = str(oriInputFiles[index][:])
    newInputFiles = copy.deepcopy(currentInputFiles)
    newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],Kwargs['prefix']+"~"+os.path.split(temp)[1]))
    parser.printInput(newInputFiles[index])
    return newInputFiles

  def pointSamplerForRELAP5(self,**Kwargs):
    modifDict = {}
    for keys in Kwargs['SampledVars']:
      key = keys.split(':')
      if len(key) > 1:    Kwargs['SampledVars'][keys]['position'] = int(key[1])
      else: Kwargs['SampledVars'][keys]['position'] = 0
      modifDict[key[0]]=Kwargs['SampledVars'][keys]
    return modifDict

  def DynamicEventTreeForRELAP5(self,**Kwargs):
    raise IOError('DynamicEventTreeForRELAP not yet implemented')
    listDict = []
    return listDict



class ExternalTest:
  def generateCommand(self,inputFiles,executable):
    return '', ''

  def findOutputFile(self,command):
    return ''

'''
 Interface Dictionary (factory) (private)
'''
__base                          = 'Code'
__interFaceDict                 = {}
__interFaceDict['RAVEN'        ] = RavenInterface
__interFaceDict['MooseBasedApp'] = MooseBasedAppInterface
__interFaceDict['ExternalTest' ] = ExternalTest
__interFaceDict['RELAP5'       ] = Relap5Interface
__knownTypes                    = list(__interFaceDict.keys())

def knonwnTypes():
  return __knownTypes

def returnCodeInterface(Type):
  '''this allow to the code(model) class to interact with a specific
     code for which the interface is present in the CodeInterfaces module'''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
