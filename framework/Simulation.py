'''
Created on Feb 19, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import os
import copy
#from Driver import self.debug
import Steps
import Datas
import Samplers
import Models
import Tests
import Distributions
import DataBases
import OutStreams
from JobHandler import JobHandler

class Simulation:
  '''This is a class that contain all the object needed to run the simulation'''
  def __init__(self,inputfile,frameworkDir):
    self.debug=True
    '''this dictionary contains the general info to run the simulation'''
    self.runInfoDict = {}
    self.runInfoDict['SimulationFile'    ] = inputfile
    self.runInfoDict['ScriptDir'         ] = os.path.join(os.path.dirname(frameworkDir),"scripts")
    self.runInfoDict['FrameworkDir'      ] = frameworkDir
    self.runInfoDict['WorkingDir'        ] = ''
    self.runInfoDict['TempWorkingDir'    ] = ''
    self.runInfoDict['ParallelCommand'   ] = ''
    self.runInfoDict['ParallelProcNumb'  ] = 1
    self.runInfoDict['ThreadingCommand'  ] = ''
    self.runInfoDict['ThreadingProcessor'] = 1
    self.runInfoDict['numNode'           ] = 1
    self.runInfoDict['batchSize'         ] = 1
    self.runInfoDict['quequingSoftware'  ] = ''
    self.runInfoDict['procByNode'        ] = 1
    self.runInfoDict['numProcByRun'      ] = 1
    self.runInfoDict['totNumbCores'      ] = 1
    self.runInfoDict['stepName'          ] = 1
    self.runInfoDict['precommand'        ] = ''
    self.runInfoDict['mode'              ] = ''
    self.runInfoDict['expectedTime'      ] = '10:00:00'
    '''the step to run the simulation in sequence'''
    self.stepSequenceList = []
    '''
      there is one dictionary for each type in the simulation
      the keys in the dictionary are the user provided name for all the needed types
      they point to an instance of the class
    '''
    self.stepsDict         = {}
    self.dataDict          = {}
    self.samplersDict      = {}
    self.modelsDict        = {}
    self.testsDict         = {}
    self.DistributionsDict = {}
    self.dataBasesDict     = {}
    self.OutStreamsDict    = {}
    self.filesDict         = {} #this is different, it just return the absolute path of the file
    '''list of supported quequing software:'''
    self.knownQuequingSoftware = []
    self.knownQuequingSoftware.append('None')
    self.knownQuequingSoftware.append('PBS Professional')
    '''
      Class Dictionary
      when a new function is added to the simulation this dictionary need to be expanded
    '''
    self.addWhatDict  = {}
    self.addWhatDict['Steps'         ] = Steps.returnInstance
    self.addWhatDict['Datas'         ] = Datas.returnInstance
    self.addWhatDict['Samplers'      ] = Samplers.returnInstance
    self.addWhatDict['Models'        ] = Models.returnInstance
    self.addWhatDict['Tests'         ] = Tests.returnInstance
    self.addWhatDict['Distributions' ] = Distributions.returnInstance
    self.addWhatDict['DataBases'     ] = DataBases.returnInstance
    self.addWhatDict['OutStreams'    ] = OutStreams.returnInstance
    '''
      Mapping between a class type and the dictionary containing the instances for the simulation
    '''
    self.whichDict = {}
    self.whichDict['Steps'        ] = self.stepsDict
    self.whichDict['Datas'        ] = self.dataDict
    self.whichDict['Samplers'     ] = self.samplersDict
    self.whichDict['Models'       ] = self.modelsDict
    self.whichDict['Tests'        ] = self.testsDict
    self.whichDict['RunInfo'      ] = self.runInfoDict
    self.whichDict['Files'        ] = self.filesDict
    self.whichDict['Distributions'] = self.DistributionsDict
    self.whichDict['DataBases'    ] = self.dataBasesDict
    self.whichDict['OutStreams'   ] = self.OutStreamsDict
    self.jobHandler = JobHandler()

  def XMLread(self,xmlNode):
    '''read the general input info to set up the calculation environment'''
    for child in xmlNode:
      if child.tag in self.whichDict.keys():
        Type = child.tag
        if Type != 'RunInfo':
          for childChild in child:
            if childChild.attrib['name'] != None:
              name = childChild.attrib['name']
              self.whichDict[Type][name] = self.addWhatDict[Type](childChild.tag)
              # Call the object readXML function
              self.whichDict[Type][name].readXML(childChild)
#              if self.debug: self.whichDict[Type][name].printMe()
            else: raise IOError('not found name attribute for one '+Type)
        else: self.readRunInfo(child)
      else: raise IOError('the '+child.tag+' is not among the known simulation components '+ET.tostring(child))
    if not os.path.exists(self.runInfoDict['WorkingDir']):
      os.makedirs(self.runInfoDict['WorkingDir'])
    os.chdir(self.runInfoDict['WorkingDir'])

  def readRunInfo(self,xmlNode):
    '''reads the xml input file for the RunInfo block'''
    for element in xmlNode:
      print(element.tag)
      if   element.tag == 'WorkingDir'        :
        temp_name = element.text
        if os.path.isabs(temp_name):
          self.runInfoDict['WorkingDir'        ] = element.text
        else:
          self.runInfoDict['WorkingDir'        ] = os.path.abspath(element.text)
      elif element.tag == 'ParallelCommand'   : self.runInfoDict['ParallelCommand'   ] = element.text.strip()
      elif element.tag == 'quequingSoftware'  : self.runInfoDict['quequingSoftware'  ] = element.text.strip()
      elif element.tag == 'ThreadingCommand'  : self.runInfoDict['ThreadingCommand'  ] = element.text.strip()
      elif element.tag == 'ThreadingProcessor': self.runInfoDict['ThreadingProcessor'] = int(element.text)
      elif element.tag == 'numNode'           : self.runInfoDict['numNode'           ] = int(element.text)
      elif element.tag == 'procByNode'        : self.runInfoDict['procByNode'        ] = int(element.text)
      elif element.tag == 'numProcByRun'      : self.runInfoDict['numProcByRun'      ] = int(element.text)
      elif element.tag == 'totNumbCores'      : self.runInfoDict['totNumbCores'      ] = int(element.text)
      elif element.tag == 'ParallelProcNumb'  : self.runInfoDict['ParallelProcNumb'  ] = int(element.text)
      elif element.tag == 'batchSize'         : self.runInfoDict['batchSize'         ] = int(element.text)
      elif element.tag == 'MaxLogFileSize'    : self.runInfoDict['MaxLogFileSize'    ] = int(element.text)
      elif element.tag == 'precommand'        : self.runInfoDict['precommand'        ] = element.text.strip()
      elif element.tag == 'mode'              : self.runInfoDict['mode'              ] = element.text.strip().lower()
      elif element.tag == 'expectedTime'      : self.runInfoDict['expectedTime'      ] = element.text.strip()
      elif element.tag == 'Sequence':
        for stepName in element.text.split(','):
          self.stepSequenceList.append(stepName.strip())
      elif element.tag == 'Files':
        for fileName in element.text.split(','):
          self.filesDict[fileName] = fileName.strip()
      
    self.runInfoDict['numProcByRun'] = self.runInfoDict['ParallelProcNumb']*self.runInfoDict['ThreadingProcessor']
    self.runInfoDict['totNumbCores'] = self.runInfoDict['numProcByRun']*self.runInfoDict['batchSize']
    for key in self.filesDict.keys():
      if os.path.split(key)[0] == '': self.filesDict[key] = os.path.join(self.runInfoDict['WorkingDir'],key)
      elif not os.path.isabs(key):self.filesDict[key] = os.path.abspath(key)
    #export to the job handler the environmental variables
    if self.runInfoDict['mode'] == 'pbs':
      self.runInfoDict['precommand'] = "pbsdsh -v -n %INDEX1% -- %SCRIPT_DIR%/remote_runner2.sh out_%CURRENT_ID% %WORKING_DIR% "+self.runInfoDict['precommand']
    self.jobHandler.initialize(self.runInfoDict)

  def printDicts(self):
    '''utility function capable to print a summary of the dictionaries'''
    def prntDict(Dict):
      for key in Dict:
        print(key+'= '+str(Dict[key]))
    prntDict(self.runInfoDict)
    prntDict(self.stepsDict)
    prntDict(self.dataDict)
    prntDict(self.samplersDict)
    prntDict(self.modelsDict)
    prntDict(self.testsDict)
    prntDict(self.filesDict)
    prntDict(self.dataBasesDict)
    prntDict(self.OutStreamsDict)
    prntDict(self.addWhatDict)
    prntDict(self.whichDict)

  def run(self):
    '''run the simulation'''
    if self.debug: print('entering in the run')
    for stepName in self.stepSequenceList:                #loop over the the steps
      stepInstance = self.stepsDict[stepName]             #retrieve the instance of the step
      self.runInfoDict['stepName'] = stepName             #provide the name of the step to runInfoDict
      if self.debug: print('starting a step of type: '+stepInstance.type+', with name: '+stepInstance.name+' '+''.join((['-']*40)))
      inputDict = {}                    #initialize the input dictionary
      inputDict['Input' ] = []          #set the Input to an empty list
      inputDict['Output'] = []          #set the Output to an empty list
      for [key,b,c,d] in stepInstance.parList: #fill the take a a step input dictionary
#        if self.debug: print(a+' is:')
        #print([key,b,c,d])
        if key == 'Input':
          print('this:',b,d)
          print(self.whichDict[b].keys())
          inputDict[key].append(self.whichDict[b][d])
#          if self.debug: print('type '+b+', and name: '+ str(self.whichDict[b][d])+'\n')
        elif key == 'Output':
          inputDict[key].append(self.whichDict[b][d])
#          if self.debug: self.whichDict[b][d].printMe()
        else:
          #Create extra dictionary entry
          inputDict[key] = self.whichDict[b][d]
#          if self.debug: self.whichDict[b][d].printMe()
      inputDict['jobHandler'] = self.jobHandler
      if 'Sampler' in inputDict.keys():
        inputDict['Sampler'].fillDistribution(self.DistributionsDict)
      stepInstance.takeAstep(inputDict)
      
      
      
      
#checks to be added: no same name within a data general class
#cross check existence of needed data

