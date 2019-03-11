# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on April 14, 2014

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
import relapdata
import shutil
import re
from math import *
from CodeInterfaceBaseClass import CodeInterfaceBase
import RELAPparser


class Relap5(CodeInterfaceBase):
  """
    this class is used a part of a code dictionary to specialize Model.Code for RELAP5-3D Version 4.0.3
  """

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    found = False
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    parser = RELAPparser.RELAPparser(oriInputFiles[index].getAbsFile())
    cards = []
    for operator in self.operators:
      cards += operator['cards']
    if len(cards) > 0:
      cardValues = parser.retrieveCardValues(list(set(cards)))
      for cnt in range(len(self.operators)):
        self.operators[cnt]['cardsValues'] = {card:cardValues[card] for card in self.operators[cnt]['cards']}

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      Only one option is possible. You can choose here, if multi-deck mode is activated, from which deck you want to load the results
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    self.outputDeck = -1 # default is the last deck!
    self.operators  = []
    for child in xmlNode:
      if child.tag == 'outputDeckNumber':
        try:
          self.outputDeck = int(child.text)
        except ValueError:
          raise ValueError("can not convert outputDeckNumber to integer!!!! Got "+ child.text)
      elif child.tag == 'operator':
        operator = {}
        if 'variables' not in child.attrib:
          raise ValueError('ERROR in "RELAP5 Code Interface": "variables" attribute must be inputted in the <operator> XML node' )

        operator['vars'] = [var.strip() for var in child.attrib['variables'].split(",")]

        cards = child.find("cards")
        expression = child.find("expression")
        if cards is None or expression is None:
          raise IOError('ERROR in "RELAP5 Code Interface": <' +'cards' if cards is None else 'expression'+  '> node must be inputted within the <operator> XML node' )
        operator['cards'] = [self._convertVariablNameInInfo(card.strip()) for card in cards.text.split(",")]

        expression = expression.text.strip()
        operator['expression'] = copy.copy(expression)
        # now we check if the inputted expression is a valid python expression
        if '%card%' not in expression:
          raise IOError('ERROR in "RELAP5 Code Interface": <expression> node must contain the token "%card%"!' )
        expression = expression.replace("%card%","1.0")
        for var in operator['vars']:
          expression = expression.replace(var,"1.0")
        try:
          eval(expression)
        except Exception as e:
          raise IOError('ERROR in "RELAP5 Code Interface": inputted <expression> is not valid! Exception:'+str(e) )
        self.operators.append(operator)

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None, preExec=None):
    """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the auxiliary input file variables the user can specify in the input (e.g. under the node < Code >< fileargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    outputfile = 'out~'+inputFiles[index].getBase()
    if clargs:
      addflags = clargs['text']
    else:
      addflags = ''
    #commandToRun = executable + ' -i ' + inputFiles[index].getFilename() + ' -o ' + outputfile  + '.o' + ' -r ' + outputfile  + '.r' + addflags
    commandToRun = executable + ' -i ' + inputFiles[index].getFilename() + ' -o ' + outputfile  + '.o ' +  addflags
    commandToRun = commandToRun.replace("\n"," ")
    commandToRun  = re.sub("\s\s+" , " ", commandToRun )
    returnCommand = [('parallel',commandToRun)], outputfile
    return returnCommand

  def finalizeCodeOutput(self,command,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
    """
    outfile = os.path.join(workingDir,output+'.o')
    outputobj=relapdata.relapdata(outfile,self.outputDeck)
    if outputobj.hasAtLeastMinorData():
      outputobj.writeCSV(os.path.join(workingDir,output+'.csv'))
    else:
      raise IOError('Relap5 output file '+ command.split('-o')[0].split('-i')[-1].strip()+'.o' + ' does not contain any minor edits. It might be crashed!')

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
      (e.g. in RELAP5 would be the keyword "********")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = True
    goodWord  = ["Transient terminated by end of time step cards","Transient terminated by trip"]
    try:
      outputToRead = open(os.path.join(workingDir,output+'.o'),"r")
    except:
      return failure
    readLines = outputToRead.readlines()

    for goodMsg in goodWord:
      if any(goodMsg in x for x in readLines[-20:]):
        failure = False
    return failure

  def _evaluateOperators(self,**Kwargs):
    """
      Method to evaluate the operators
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, None
    """
    for operator in self.operators:
      expression = copy.copy(operator['expression'])
      for var in operator['vars']:
        if var not in Kwargs['SampledVars']:
          raise ValueError('The variable "'+var+'" has not been found among the  SampledVars')
        expression = expression.replace(var,str(Kwargs['SampledVars'][var]))
      expr = copy.copy(expression)
      for card in operator['cards']:
        expr = expr.replace("%card%",operator['cardsValues'][card])
        try:
          Kwargs['SampledVars'][card] = eval(expr)
        except Exception as e:
          raise IOError('ERROR in "RELAP5 Code Interface": inputted <expression> is not valid! Exception:'+str(e) )

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      this generate a new input file depending on which sampler is chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    self._samplersDictionary                = {}
    if 'dynamiceventtree' in str(samplerType).lower():
      self._samplersDictionary[samplerType] = self.DynamicEventTreeForRELAP5
    else:
      self._samplersDictionary[samplerType] = self.pointSamplerForRELAP5
    if len(self.operators) > 0:
      self._evaluateOperators(**Kwargs)
    found = False
    for index, inputFile in enumerate(currentInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    parser = RELAPparser.RELAPparser(currentInputFiles[index].getAbsFile())
    metadataToTransfer = Kwargs.get("metadataToTransfer",None)
    if metadataToTransfer is not None:
      sourceID = metadataToTransfer.get("sourceID",None)
      if sourceID is not None:
        # search for restrt file
        sourcePath = os.path.join(currentInputFiles[index].getPath(),"../",sourceID)
        rstrtFile = None
        for fileToCheck in os.listdir(sourcePath):
          if fileToCheck.strip() == 'restrt' or fileToCheck.strip().endswith(".r"):
            rstrtFile = fileToCheck
        if rstrtFile is None:
          raise IOError("metadataToTransfer|sourceID has been provided but no restart file has been found!")
        sourceFile = os.path.join(sourcePath, rstrtFile)
        try:
          shutil.copy(sourceFile, currentInputFiles[index].getPath())
        except:
          raise IOError('not able to copy restart file from "'+sourceFile+'" to "'+currentInputFiles[index].getPath()+'"')
      else:
        raise IOError('the only metadtaToTransfer that is available in RELAP5 is "sourceID". Got instad: '+', '.join(metadataToTransfer.keys()))

    if 'None' not in str(samplerType):
      modifDict = self._samplersDictionary[samplerType](**Kwargs)
      parser.modifyOrAdd(modifDict,True)

    parser.printInput(currentInputFiles[index])
    return currentInputFiles

  def _convertVariablNameInInfo(self, variableName):
    """
      @ In, variableName, string, the variable name to be converted
      @ Out, (deck, card, word), tuple , the converted variable (deck #, card #, word #)
    """
    if type(variableName).__name__ == 'tuple':
      return variableName

    key = variableName.split(':')
    multiDeck = key[0].split("|")
    card, deck, word = key[0], 1, (key[-1] if len(key) >1 else 0)
    if len(multiDeck) > 1:
      card = multiDeck[1]
      deck = multiDeck[0]
    try:
      deck = int(deck)
    except ValueError:
      raise IOError("RELAP5 interface: activated multi-deck/case approach but the deck number is not an integer (first word followed by '|' symbol). Got "+str(deck))
    try:
      word = int(word)
    except ValueError:
      raise IOError("RELAP5 interface: word number is not an integer (first word followed by '|' symbol). Got "+str(word))

    return (deck, card, word)


  def pointSamplerForRELAP5(self,**Kwargs):
    """
      This method is used to create a list of dictionaries that can be interpreted by the input Parser
      in order to change the input file based on the information present in the Kwargs dictionary.
      This is specific for Point samplers (Grid, Stratified, Monte Carlo, etc.).
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, listDict, list, list of dictionaries used by the parser to change the input file
    """
    listDict = []
    modifDict = {}
    deckList = {1:{}}
    deckActivated = False
    for keys in Kwargs['SampledVars']:
      deck, card, word = self._convertVariablNameInInfo(keys)
      deckActivated = deck > 1
      if deck not in deckList:
        deckList[deck] = {}
      if card not in deckList[deck]:
        deckList[deck][card] = [{'position':word,'value':Kwargs['SampledVars'][keys]}]
      else:
        deckList[deck][card].append({'position':word,'value':Kwargs['SampledVars'][keys]})

      if deck is None:
        # check if other variables have been defined with a deck ID, in case...error out
        if deckActivated:
          raise IOError("If the multi-deck/case approach gets activated, all the variables need to provide a DECK ID. E.g. deckNumber|card|word ! Wrong variable is "+card)
    modifDict['decks']=deckList
    listDict.append(modifDict)
    return listDict

  def DynamicEventTreeForRELAP5(self,**Kwargs):
    """
      This method is used to create a list of dictionaries that can be interpreted by the input Parser
      in order to change the input file based on the information present in the Kwargs dictionary.
      This is specific for DET-based samplers.
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, listDict, list, list of dictionaries used by the parser to change the input file
    """
    listDict =[]
    deckList={1:{}}   #  List of decks with the cards to be modified in RELAP5 Input File
    deckActivated = False
    # Check the initiator distributions and add the next threshold
    if 'initiator_distribution' in Kwargs.keys():
      for i in range(len(Kwargs['initiator_distribution'])):
        modifDict = {}
        modifDict['name'] = ['Distributions',Kwargs['initiator_distribution'][i]]
        modifDict['ProbabilityThreshold'] = Kwargs['PbThreshold'][i]
        listDict.append(modifDict)
        del modifDict
    # add the initial time for this new branch calculation
    if 'startTime' in Kwargs.keys():
      if Kwargs['startTime'] != 'Initial':
        modifDict = {}
        st_time = Kwargs['startTime']
        modifDict['name'] = ['Executioner']
        modifDict['startTime'] = st_time
        listDict.append(modifDict)
        del modifDict
    # create the restart file name root from the parent branch calculation
    # in order to restart the calc from the last point in time
    if 'end_ts' in Kwargs.keys():
      #if Kwargs['end_ts'] != 0 or Kwargs['end_ts'] == 0:
      if str(Kwargs['startTime']) != 'Initial':
        modifDict = {}
        #restart_parent = Kwargs['RAVEN_parentID']+'~restart.r'
        #new_restart = Kwargs['prefix']+'~restart.r'
        #shutil.copyfile(restart_parent,new_restart)
        modifDict['name'] = ['Executioner']
        #modifDict['restart_file_base'] = new_restart
        #print('CODE INTERFACE: Restart file name base is "' + new_restart + '"')
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
    modifDict['num_restart_files'] = 1
    listDict.append(modifDict)
    del modifDict
    # in this way we erase the whole block in order to neglect eventual older info
    # remember this "command" must be added before giving the info for refilling the block
    modifDict = {}
    modifDict['name'] = ['RestartInitialize']
    modifDict['erase_block'] = True
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
    modifDict={}
    for keys in Kwargs['SampledVars']:
      key = keys.split(':')
      deck = None
      multiDeck = key[0].split("|")
      if len(multiDeck) > 1:
        card = multiDeck[1]
        deck = multiDeck[0]
        try:
          deck = int(deck)
        except:
          raise IOError("RELAP5 interface: activated multi-deck/case approach but the deck number is not an integer (first word followed by '|' symbol). Got "+str(deck))
        deckActivated = True
        if deck not in deckList.keys():
          deckList[deck] = {}
      else:
        deck = 1
        card = key[0]
      if len(key) > 1:

        if Kwargs['startTime'] != 'Initial':
          if card not in deckList[deck].keys():
            deckList[deck][card] = [{'position':int(key[1]),'value':float(Kwargs['SampledVars'][keys])}]
          else:
            deckList[deck][card].append({'position':int(key[1]),'value':float(Kwargs['SampledVars'][keys])})
        else:
          if card not in deckList[deck].keys():
            deckList[deck][card] = [{'position':int(key[1]),'value':Kwargs['SampledVars'][keys]}]
          else:
            deckList[deck][card].append({'position':int(key[1]),'value':Kwargs['SampledVars'][keys]})
      else:
        if Kwargs['startTime'] != 'Initial':
          if card not in deckList[deck].keys():
            deckList[deck][card]=[{'position':0,'value':float(Kwargs['SampledVars'][keys])}]
          else:
            deckList[deck][card].append({'position':0,'value':float(Kwargs['SampledVars'][keys])})
        else:
          if card not in deckList[deck].keys():
            deckList[deck][card]=[{'position':0,'value':float(Kwargs['SampledVars'][keys])}]
          else:
            deckList[deck][card].append({'position':0,'value':float(Kwargs['SampledVars'][keys])})
      if deck is None:
        # check if other variables have been defined with a deck ID, in case...error out
        if deckActivated:
          raise IOError("If the multi-deck/case approach gets activated, all the variables need to provide a DECK ID. E.g. deckNumber|card|word ! Wrong variable is "+card)
    modifDict['decks']=deckList
    if 'aux_vars' in Kwargs.keys():
      for keys in Kwargs['aux_vars']:
        deck = None
        key = keys.split(':')
        multiDeck = key[0].split("|")
        if len(multiDeck) > 1:
          card = multiDeck[1]
          deck = multiDeck[0]
          try:
            deck = int(deck)
          except:
            raise IOError("RELAP5 interface: activated multi-deck/case approach but the deck number is not an integer (first word followed by '|' symbol). Got "+str(deck))
          deckActivated = True
          if deck not in deckList.keys():
            deckList[deck] = {}
        else:
          card = key[0]
        if len(key) > 1:
          if card not in deckList[deck].keys():
            deckList[deck][card]=[{'position':int(key[1]),'value':Kwargs['aux_vars'][keys]}]
          else:
            deckList[deck][card].append({'position':int(key[1]),'value':Kwargs['aux_vars'][keys]})
        else:
          if card not in deckList[deck].keys():
            deckList[deck][card]=[{'position':0,'value':Kwargs['aux_vars'][keys]}]
          else:
            deckList[deck][card].append({'position':0,'value':Kwargs['aux_vars'][keys]})
        if deck is None:
          # check if other variables have been defined with a deck ID, in case...error out
          if deckActivated:
            raise IOError("If the multi-deck/case approach gets activated, all the variables need to provide a DECK ID. E.g. deckNumber|card|word ! Wrong variable is "+card)
        modifDict['cards']=deckList
    listDict.append(modifDict)
    del modifDict
    return listDict
