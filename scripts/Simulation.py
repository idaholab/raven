'''
Created on Feb 19, 2013

@author: crisr
'''
import xml.etree.ElementTree as ET
import os
import copy
#from Driver import debug
import Steps
import Datas
import Samplers
import Models
import Tests
import Distributions
from JobHandler import JobHandler

def prntDict(Dict):
  for key in Dict:
    print(key+'= '+str(Dict[key]))

class Simulation:
  '''
  This is a class that contain all the object needed to run the simulation
  '''
  def __init__(self,inputfile):
    #this dictionary contains the general info to run the simulation
    self.runInfoDict = {}
    self.runInfoDict['SimulationFile'    ] = inputfile
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

    #the step to run the simulation in sequence
    self.stepSequenceList = []
    #there is one dictionary for each type in the simulation
    #the keys in the dictionary are the user provided name for all the needed types
    #they point to an instance of the class
    self.stepsDict    = {}
    self.dataDict     = {}
    self.samplersDict = {}
    self.modelsDict   = {}
    self.testsDict    = {}
    self.DistributionsDict = {}
    self.filesDict    = {} #this is different return the absolute path of the file
    #list of supported quequing software:
    self.knownQuequingSoftware = []
    self.knownQuequingSoftware.append('None')
    self.knownQuequingSoftware.append('PBS Professional')
    #Class Dictionary
    #when a new function is added to the simulation this dictionary need to be expanded
    self.addWhatDict  = {}
    self.addWhatDict['Steps'   ] = Steps.returnInstance
    self.addWhatDict['Datas'   ] = Datas.returnInstance
    self.addWhatDict['Samplers'] = Samplers.returnInstance
    self.addWhatDict['Models'  ] = Models.returnInstance
    self.addWhatDict['Tests'   ] = Tests.returnInstance
    self.addWhatDict['Distributions'   ] = Distributions.returnInstance
    #Mapping between a class type and the dictionary containing the istances for the simulation
    self.whichDict = {}
    self.whichDict['Steps'        ] = self.stepsDict
    self.whichDict['Datas'        ] = self.dataDict
    self.whichDict['Samplers'     ] = self.samplersDict
    self.whichDict['Models'       ] = self.modelsDict
    self.whichDict['Tests'        ] = self.testsDict
    self.whichDict['RunInfo'      ] = self.runInfoDict
    self.whichDict['Files'        ] = self.filesDict
    self.whichDict['Distributions'] = self.DistributionsDict
    #initialize the job handler (it should become a demon)
    self.jobHandler = JobHandler()
 
  def XMLread(self,xmlNode):
    '''
    read the general input info to set up the calculation environment
    '''
    for child in xmlNode:
      if child.tag in self.whichDict.keys():
        Type = child.tag
        if Type != 'RunInfo':
          for childChild in child:
            if childChild.attrib['name'] != None:
              name = childChild.attrib['name']
              self.whichDict[Type][name] = self.addWhatDict[Type](childChild.tag)
              self.whichDict[Type][name].readXML(childChild)
              self.whichDict[Type][name].printMe()
              print('end')
            else:
                raise IOError('not found name attribute for one '+Type)
        else:
          self.addRunInfo(child)
      else:
        raise IOError('the '+child.tag+' is not among the known simulation components')
    os.chdir(self.runInfoDict['WorkingDir'])
  
  def addRunInfo(self,xmlNode):
    for element in xmlNode:
      if element.tag == 'ParallelInfo':
        self.runInfoDict['ParallelCommand'] = element.text()
        try: self.runInfoDict['ParallelProcNumb'] = element['processor']
        except: raise IOError('to run in parallel i need the number of processors')
      elif element.tag == 'ThreadingInfo':
        self.runInfoDict['ThreadingCommand'] = element.text()
        try:    self.runInfoDict['ThreadingProcessor'] = element['processor']
        except: raise IOError('to run in threaded i need the number of processors')
      elif element.tag == 'QuequingInfo':
        self.runInfoDict['quequingSoftware'] = element['Software']
        if self.runInfoDict['quequingSoftware'] in self.knownQuequingSoftware:
          self.runInfoDict['numNode']    = element['numNode']
          self.runInfoDict['procByNode'] = element['procByNode']
          self.runInfoDict['batchSize'] = element['batchSize']
        else:
          raise IOError('not known Quequing software')
      elif element.tag == 'Sequence':
        for stepName in element.text.split(','):
          self.stepSequenceList.append(stepName)
      elif element.tag == 'Files':
        for fileName in element.text.split(','):
          self.filesDict[fileName] = fileName
      elif element.tag == 'WorkingDir':
        self.runInfoDict['WorkingDir'] = element.text
        print(element.text)

    self.runInfoDict['numProcByRun'] = self.runInfoDict['ParallelProcNumb']*self.runInfoDict['ThreadingProcessor']
    self.runInfoDict['totNumbCores'] = self.runInfoDict['numProcByRun']*self.runInfoDict['batchSize']
    for key in self.filesDict.keys():
      print ('qui '+key)
      print(os.path.split(key))
      if os.path.split(key)[0] == '':
        self.filesDict[key] = os.path.join(self.runInfoDict['WorkingDir'],key)
      elif not os.path.isabs(key):
        self.filesDict[key] = os.path.abspath(key)
    #export to the job handler the environmental variables
    self.jobHandler.initialize(self.runInfoDict)

      
  
  def printDicts(self):
    prntDict(self.runInfoDict)
    prntDict(self.stepsDict)
    prntDict(self.dataDict)
    prntDict(self.samplersDict)
    prntDict(self.modelsDict)
    prntDict(self.testsDict)
    prntDict(self.filesDict)
    prntDict(self.addWhatDict)
    prntDict(self.whichDict)

  def run(self):
    '''
    run the simulation
    '''
    print(self.runInfoDict['ParallelProcNumb'])
    for stepName in self.stepSequenceList:
      stepInstance = self.stepsDict[stepName]
      print('starting a step of type: '+stepInstance.type+', with name: '+stepInstance.name)
      inputDict = {}
      for [a,b,c,d] in stepInstance.parList:
        print(a+' is of type: '+b+', subtype: '+c+', and has name: '+d)
        inputDict[a] = self.whichDict[b][d]
      inputDict['jobHandler'] = self.jobHandler
      if 'Sampler' in inputDict.keys():
        inputDict['Sampler'].fillDistribution(self.DistributionsDict)
      print(inputDict)
      stepInstance.takeAstep(inputDict)
      
      
      
      
#checks to be added: no same name within a data general class
#cross check existence of needed data

