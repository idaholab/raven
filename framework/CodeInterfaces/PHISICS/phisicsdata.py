"""
Created on July 25th, 2017
@author: rouxpn
"""

import os
import sys
import re 
import csv
import xml.etree.ElementTree as ET 

class phisicsdata():
  """
    This class parses the phisics output of interest. The output of interest are placed in the a csv file. 
  """

  def __init__(self, instantOutputFile, workingDir, mrtauBoolean, jobTitle, mrtauFileNameDict, numberOfMPI):
    """
      read the phisics output
      @ In, output, string (Instant output)
      @ In, workingDir, string
      @ In, mrtauBoolean, Boolean (True if mrtau standalone mode, False is instant coupled to mrtau)
      @ Out, keff, string, keff coming from Instant output
      @ Out, errorKeff, string error on keff coming from Instant output
      @ Out, reactionRateInfo, dictionary, reaction rates coming from Instant output. label and values included
      @ Out, fissionMatrixInfo, dictionary, fisison matrix coming from instant. label and values included 
      @ Out, fluxLabelList, list, list of labels relative to the flux cell coming from instant output 
      @ Out, fluxList, list, list of flux values coming from Instant output 
      @ Out, matFluxLabelList, list, list of labels relative to the flux within specific material. 
      @ Out, matFluxList, list, list of flux values within specific materials coming from Instant
      @ Out, depLabelList, list, list of labels relative to the depletion info, coming from mrtau output
      @ Out, depList, list, list of depletion values (i.e number densities as a function of time) coming from the mrtau Output
      @ Out, timeStepList, list of time steps coming from mrtau output
      @ Out, decayLabelList, list, list of labels relative to decay heat, coming from Instant output 
      @ Out, decayList, list, list of decay heat values coming from the Instant output 
      @ Out, timeStepIndex, integers, integers pointing to the index of the timeStep of interest
      @ Out, matchedTimeSteps, list, list of time step matching both mrtau input and instant output
    """
    instantOutputFileMPI = []
    mrtauOutputFileMPI   = []
    instantDict          = {}
    mrtauDict            = {}    
    markerList           = ['Fission matrices of','Scattering matrices of','Multigroup solver ended!'] 
    self.mrtauBoolean    = mrtauBoolean
    
    for mpi in xrange (0,numberOfMPI):
      instantOutputFileMPI.append(instantOutputFile+'-'+str(mpi))
      mrtauOutputFileMPI.append(os.path.join(workingDir,mrtauFileNameDict['atoms_csv'].split('.')[0]+'-'+str(mpi)+'.csv'))
    self.materialsDict    = self.locateMaterialInFile(numberOfMPI, workingDir, instantOutputFileMPI)      
    
    instantOutput         = [instantOutputFileMPI[0], mrtauOutputFileMPI, 'Dpl_INSTANT_'+jobTitle+'_flux_mat.csv', 'scaled_xs.xml']
    mrtauOutput           = [mrtauOutputFileMPI,      mrtauFileNameDict['decay_heat']                                         ]

    self.instantCSVOutput       = os.path.join(workingDir, instantOutput[2])
    self.decayHeatMrtauOutput   = os.path.join(workingDir, mrtauOutput[1])         
    self.pathToPhisicsOutput    = os.path.join(workingDir, instantOutput[0])
    pathToXSOutput              = os.path.join(workingDir, instantOutput[3])
    self.perturbationNumber     = self.getPertNumber(workingDir)
    self.cleanUp(workingDir, jobTitle)
    
    if self.mrtauBoolean == False:  
      mrtauTimeSteps       = self.getMrtauInstantTimeSteps(instantOutput[1])
      instantTimeSteps     = self.getInstantTimeSteps()
      XSlabelList, XSlist  = self.getAbsoluteXS(pathToXSOutput)
    if self.mrtauBoolean == True: 
      mrtauTimeSteps = self.getMrtauTimeSteps(mrtauOutput[0])
      self.getMrtauIsotopeList(mrtauOutput[0])
      
    for timeStepIndex in xrange (0,len(mrtauTimeSteps)): 
      if self.mrtauBoolean == False:
        keff, errorKeff                                        = self.getKeff(workingDir, timeStepIndex, mrtauTimeSteps)
        reactionRateInfo                                       = self.getReactionRates(workingDir, timeStepIndex, mrtauTimeSteps)
        fissionMatrixInfo                                      = self.getMatrix(workingDir, markerList[0], markerList[1], 'FissionMatrix', timeStepIndex, mrtauTimeSteps)
        fluxLabelList, fluxList, matFluxLabelList, matFluxList = self.getFluxInfo(timeStepIndex, mrtauTimeSteps)
        depLabelList, depList, timeStepList, matList           = self.getDepInfo(timeStepIndex, mrtauTimeSteps, numberOfMPI, instantOutput[1], workingDir)
        decayLabelList, decayList                              = self.getDecayHeat(timeStepIndex, mrtauTimeSteps, numberOfMPI, instantOutputFileMPI, workingDir)
        
        instantDict['keff']              = keff
        instantDict['errorKeff']         = errorKeff
        instantDict['reactionRateInfo']  = reactionRateInfo
        instantDict['fissionMatrixInfo'] = fissionMatrixInfo
        instantDict['workingDir']        = workingDir
        instantDict['fluxLabelList']     = fluxLabelList 
        instantDict['fluxList']          = fluxList
        instantDict['matFluxLabelList']  = matFluxLabelList
        instantDict['matFluxList']       = matFluxList
        instantDict['depLabelList']      = depLabelList
        instantDict['depList']           = depList
        instantDict['timeStepList']      = timeStepList
        instantDict['decayLabelList']    = decayLabelList
        instantDict['decayList']         = decayList
        instantDict['timeStepIndex']     = timeStepIndex
        instantDict['mrtauTimeSteps']    = mrtauTimeSteps
        instantDict['XSlabelList']       = XSlabelList
        instantDict['XSlist']            = XSlist
        
        self.writeCSV(instantDict, timeStepIndex, mrtauTimeSteps,jobTitle)
      
      if self.mrtauBoolean == True:
        decayHeatMrtau = self.getDecayHeatMrtau(timeStepIndex, mrtauTimeSteps)
        depList        = self.getDepInfoMrtau(timeStepIndex, mrtauTimeSteps, mrtauOutput[0])
        
        mrtauDict['workingDir']     = workingDir
        mrtauDict['decayHeatMrtau'] = decayHeatMrtau 
        mrtauDict['depList']        = depList
        mrtauDict['timeStepIndex']  = timeStepIndex
        mrtauDict['mrtauTimeSteps'] = mrtauTimeSteps
        self.writeMrtauCSV(mrtauDict)
        
  def cleanUp(self, workingDir, jobTitle):
    """
      Removes the file that RAVEN reads for postprocessing 
    """
    csvOutput = os.path.join(workingDir, jobTitle+self.perturbationNumber+'.csv')
    if os.path.isfile(csvOutput):
      os.remove(csvOutput) 
    
  def getAbsoluteXS(self,XSoutput):
    """
      Parses the absolute cross section output generated by PHISICS, according to their perturbed scaling factor 
      format is turned 
        from: <Scatter g="3 1">4.00 4.66 </Scatter>
        to: SCATTER|3|4.00 SCATTER|1|4.66 
      @ In, Xs.xml, xml file containing the absolute perturbed XS
      @ Out, XSlabelList, list of the XS labels
      @ Out, XSlist, list of the XS 
    """
    labelList = []
    valueList = []
    tree = ET.parse(XSoutput)
    root = tree.getroot()
    for materialXML in root.getiterator('library'):
      for isotopeXML in materialXML.getiterator('isotope'):
        reactionList = [j.tag for j in isotopeXML]
        for k in xrange (0, len(reactionList)):
          for groupXML in isotopeXML.getiterator(reactionList[k]):
            individualGroup = [x.strip() for x in groupXML.attrib.get('g').split(',')]
            individualGroupValues = [y.strip() for y in groupXML.text.split(',')]
            for position in xrange(0,len(individualGroup)):
              labelList.append(materialXML.attrib.get('lib_name')+'|'+isotopeXML.attrib.get('id')+'|'+reactionList[k]+'|'+individualGroup[position])
              valueList.append(individualGroupValues[position])
    #print labelList
    #print valueList
    return labelList, valueList 
    
  def removeSpaces(self, line):
    """
      removes the spaces. It makes the word splitting cleaner. 
      @ In, line, string
      @ Out, line, string 
    """
    line = re.sub(r' ',r'',line)
    return line 
    
  def getNumberOfGroups(self, workingDir):
    """
      give the number of energy groups used in the Phisics simulations  
      @ In, WorkingDir, string
      @ Out, Ngroups, integer under the form of a string 
    """
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'Number of groups', line) :
          self.Ngroups = line.split()[-1]
          break 
    #print self.Ngroups   
  
  def getPertNumber (self, workingDir):
    """
      get the current perturbation number
      @ In, workingDir, string, working directory 
      @ Out, pertNumber, string, perturbation number in a string format 
    """
    splitWorkDir = workingDir.split('/')
    pertNumber = splitWorkDir[-1]
    return pertNumber 
  
  def removeTheZeroTimeStep(self,timeSteps):
    """
      removes the first time step, t=0, to make the number of time steps match with the number of time steps in Instant
      @ In, timeSteps, list, list of the time steps in mrtau
      @ Out, timeSteps, list, time steps in mrtau withtout the first time step
    """
    timeSteps.pop(0)
    return timeSteps 
    
  def getMrtauInstantTimeSteps(self, mrtauCSVOutput):
    """
      get the time steps in the coupled mrtau instant output
      @ In, numbers-0.csv
      @ Out, timeSteps, list 
    """
    count = 0
    timeSteps = []
    with open(mrtauCSVOutput[0], 'r') as outfile:
      for line in outfile :
        if re.search(r'Material',line): 
          count = count + 1 
        if count == 1:
          stringIsFloatNumber = self.isFloatNumber(line.split(','))
          if stringIsFloatNumber is True:
            timeSteps.append(line.split(',')[0])
        if count > 1:
          break 
    timeSteps = self.removeTheZeroTimeStep(timeSteps)
    #print timeSteps 
    return timeSteps
    
  def getMrtauTimeSteps(self, mrtauCSVOutput):
    """
      get the time steps in the mrtau standalone output 
      @ In, numbers.csv
      @ Out, timeSteps, list 
    """
    timeSteps = []
    with open(mrtauCSVOutput[0], 'r') as outfile:
      for line in outfile : 
        stringIsFloatNumber = self.isFloatNumber(line.split(','))
        if stringIsFloatNumber is True:
          timeSteps.append(line.split(',')[0])
    #print timeSteps
    return timeSteps
    
  def getMrtauIsotopeList(self, mrtauCSVOutput):
    """
      get the isotope in the mrtau standalone output 
      @ In, numbers.csv
      @ Out, self.isotopeListMrtau, list 
    """
    self.isotopeListMrtau = []
    with open(mrtauCSVOutput[0], 'r') as outfile:
      for line in outfile : 
        if re.search(r'TIME',line):
          line = re.sub(r'TIME\s+\(days\)',r'',line)
          line = re.sub(r' ',r'',line)
          line = re.sub(r'\n',r'',line)
          self.isotopeListMrtau = filter(None,line.split(','))
          #print self.isotopeListMrtau
          break
    
  def getInstantTimeSteps(self):
    """
      get the time steps in the Instant output 
      @ In, Dpl_INSTANT_'jobTitle'_flux_mat.csv
      @ Out, timeSteps, list 
    """
    count = 0
    timeSteps = []
    with open(self.instantCSVOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'PHISICS',line): 
          timeSteps.append(line.split(',')[line.split(',').index('Time') + 1])
    #print timeSteps
    return timeSteps

  def reduceDigits(self, mrtauTimeSteps): 
    """
      reduce the number of digits in the mrtau time steps   
      @ In, mrtauTimeSteps, list
      @ Out, timeSteps, list 
    """
    timeSteps = []
    for i in xrange (0, len(mrtauTimeSteps)):
      timeSteps.append('%.6E' % (float(mrtauTimeSteps[i])))
    #print timeSteps
    return timeSteps  
  
  def findUnits(self, line):
    """
      return the units in which the time steps are printed in the instant output
      @ In, line, string 
      @ Out, units, string
    """ 
    units = line.split(' ')[-3]
    return units
  
  def getKeff (self, workingDir, timeStepIndex, matchedTimeSteps):
    """
      get the multiplication factor 
      @ In, workingDir, string
      @ In, timeStepIndex, integer, number of the timestep considered 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out,keff, string, floating number under string format of keff
      @ Out,errorKeff, string, floating number under string format of errorKeff
    """
    self.units = ''
    keff = []
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'Time Input Type', line):
          self.units = self.findUnits(line)
        if re.search(r'k-effective at the last', line) :
          keff = [line.split()[-1]]
        if re.search(r'error for the eigenvalue', line) :
            errorKeff = [line.split()[-1]]
    #print keff
    return keff, errorKeff

  def isNumber(self, line):
      """
        check if a string is an integer
        @ In, line, list, list of strings
        @ Out, True or False, Boolean 
      """
      try: 
        int(line[0])
        return True
      except ValueError:
        return False
        
  def isFloatNumber(self, line):
    """
      check if a string is an integer
      @ In, line, list, list of strings
      @ Out, True or False, Boolean 
    """
    try: 
      float(line[0])
      return True
    except ValueError:
      return False
   
  def convertInDays(self, instantTimeSteps): 
    """
      convert the Instant time steps (seconds) into days  
      @ In, instantTimeSteps, list
      @ Out, timeSteps, list 
    """
    timeSteps = []
    for i in xrange (0, len(instantTimeSteps)):
      timeSteps.append('%.6E' % (float(instantTimeSteps[i]) / (24 * 60 * 60)))
    return timeSteps 
   
  def declareDict(self, numbering, typeOfParameters):
    """
      declare the RR dictionary  
      @ In, numbering, dictionary, dictionary of columns (key) and column position (value). It is a matrix mapping dictionary 
      @ In, typeOfparameter, string, either 'reaction rate' or 'FissionMatrix', to select which matrix is being parsed
      @ Out, declareDict, dictionary, dictionary with keys, and empty values
    """
    declareDict = {}
    for i in xrange (1,int(self.Ngroups) + 1):
      declareDict[str(i)] = {}
      for j in xrange (1,int(self.Nregions) + 1):
        declareDict[str(i)][str(j)] = {}
        for k in xrange (0, len(numbering)):
          if numbering.keys()[k] == 'Group': pass
          else: declareDict[str(i)][str(j)][numbering.keys()[k]] = {} 
    if typeOfParameters == 'reactionRates':
      self.paramList = []
      for key in numbering.iterkeys():
        self.paramList.append(key)
    if typeOfParameters == 'FissionMatrix':
      self.matrixList = []
      for key in numbering.iterkeys():
        self.matrixList.append(key)  
    return declareDict
        
  def getReactionRates (self, workingDir, timeStepIndex, matchedTimeSteps):
    """
      get the reactions rates, power for each group in PHISICS 
      @ In, wordingDir, string, working directory.
      @ In, timeStepIndex, integer, number of the timestep considered 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, ReactionRateDict, Dict, dictionary containing the RR infos
    """
    flagStart,  count, countTimeStep = 0, 0, 0 
    self.getNumberOfGroups(workingDir)
    self.Nregions = 1 
    numbering = {}
    reactionRateDict = {}
    self.paramList = [] 
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'averaged flux\s+power', line):
          numbering = self.mapColumns(line, count, numbering)
          reactionRateDict = self.declareDict(numbering, 'reactionRates')
          #print numbering
          flagStart = 1 
          countTimeStep = countTimeStep + 1 
        if re.search(r'Fission matrices of all',line):
          flagStart = 2
        if flagStart == 1:
          if re.search(r'\w+\s+\w+',line):
            line = re.sub(r'[\|]',' ',line)
            line = line.split()
            #print line 
            stringIsNumber = self.isNumber(line)
            if stringIsNumber == True :
              for i in xrange (0,len(numbering)): 
                if self.paramList[i] == 'Group': pass 
                else: reactionRateDict[line[0]][line[1]][self.paramList[i]] = line[numbering.get(self.paramList[i])]
    #print reactionRateDict
    if reactionRateDict != {}:
      return reactionRateDict  
    
  def getMatrix (self, workingDir, startStringFlag, endStringFlag, typeOfMatrix, timeStepIndex, matchedTimeSteps):
    """
      get the reactions rates, power for each group in PHISICS 
      @ In, timeStepIndex, integer, number of the timestep considered 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, matrixDict, dictionary
    """
    flagStart, count, countTimeStep = 0, 0, 0
    numbering = {}
    matrixDict = {}
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile : 
        if re.search(startStringFlag, line):
          flagStart = 1 
          countTimeStep = countTimeStep + 1
        if re.search(endStringFlag,line):
          flagStart = 2
        if flagStart == 1 :
          if re.search(r'\s+1\s+2\s+3\s+4',line):
            line = re.sub(r'[\|]',' ',line)
            numbering = self.mapColumns(line, count, numbering)
            #print numbering 
            matrixDict = self.declareDict(numbering, typeOfMatrix)
            #print matrixDict
          if re.search(r'Region\:\s+\d',line): regionNumber = line.split()[-1]
          if re.search(r'\d+.\d+E',line):
            line = re.sub(r'[\|]',' ',line)
            line = line.split()
            for i in xrange (1,len(numbering) + 1):  
              matrixDict[line[0]][str(regionNumber)][str(i)] = line[numbering.get(str(i)) + 1]    
    #print matrixDict
    return matrixDict
  
  def mapColumns(self, line, count, numbering): 
    """
      numbers the column relative to the reaction rates 
      @ In, line, string
      @ In, count, interger, counts the column position 
    """
    line = re.sub(r'averaged',r'',line)
    line = re.sub(r'fis. ',r'',line)
    line = re.sub(r'[\|]',' ',line)
    parameterNames = line.split()
    for i in xrange(len(parameterNames)) :
      numbering[parameterNames[i]] = count
      count = count + 1
    #print numbering 
    return numbering 

  def locateXYandGroups(self, IDlist):
    """
      locates what is the position number of the x, y, z coordinates and the first energy group in the Instant 
      csv output file. 
      @ In, IDlist (list) list of all the parameter in the csv output
      @ Out, xPositionInList, yPositionInList, zPositionInList, firstGroupPositionInList, (integers), corresponding
      to the position of the parameters x, y, z and first energy group in the list. A 2D case will return 0 as z position
    """
    xPositionInList = 0
    yPositionInList = 0
    zPositionInList = 0
    firstGroupPositionInList = 0
    for i in xrange (0,len(IDlist)):
      if IDlist[i] == 'X':
        xPositionInList = i
      if IDlist[i] == 'Y':
        yPositionInList = i
      if IDlist[i] == 'Z':
        zPositionInList = i
      if IDlist[i] == '1':
        group1PositionInList = i
        break
    #print xPositionInList, yPositionInList, zPositionInList, group1PositionInList
    return xPositionInList, yPositionInList, zPositionInList, group1PositionInList
    
  def getFluxInfo(self, timeStepIndex, matchedTimeSteps):
    """
      Read the Instant CSV file to get the flux info relative to each region and each group
      The flux info are also provided for each material
      @ In, Dpl_INSTANT_HTGR_test_flux_mat.csv
      @ In, timeStepIndex, integer, number of the timestep considered 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, fluxLabelList, fluxList, matFluxLabelList, matFluxList (lists)
    """
    IDlist = []
    fluxLabelList = []
    fluxList = []
    matFluxLabelList = []
    matFluxList = []
    flagFlux, countTimeStep = 0, 0 
    with open(self.instantCSVOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'PHISICS',line): 
          flagFlux = 0 
          countTimeStep = countTimeStep + 1
        if re.search(r'FLUX BY CELLS',line):    flagFlux = 1 
        if re.search(r'FLUX BY MATERIAL',line): flagFlux = 2
        
        if flagFlux == 1:
          if re.search(r'ID\s+,\s+ID',line):
            line = re.sub(r' ',r'',line)
            IDlist = line.split(',')
            xPosition, yPosition, zPosition, group1Position = self.locateXYandGroups(IDlist)
            IDlist.remove('\n')         
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber is True:
            line = re.sub(r' ',r'',line)
            line = re.sub(r'\n',r'',line)
            #if zPosition == 0:  this is a 2D case
            for g in xrange (1,int(self.Ngroups) + 1):
              if countTimeStep == 1: # the labels are the same for each time step
                fluxLabelList.append('flux'+'|'+'cell'+line.split(',')[0]+'|'+'gr'+str(g))
              fluxList.append(line.split(',')[group1Position + g - 1])
              
        if flagFlux == 2:        
          stringIsNumber = self.isNumber(line.split(','))
          if stringIsNumber is True:
            line = re.sub(r' ',r'',line)
            line = re.sub(r'\n',r'',line)
            for g in xrange (1,int(self.Ngroups) + 1):
              if countTimeStep == 1:
                matFluxLabelList.append('flux'+'|'+'mat'+line.split(',')[0]+'|'+'gr'+str(g))
              matFluxList.append(line.split(',')[g])
    #print matFluxLabelList, matFluxList
    return fluxLabelList, fluxList, matFluxLabelList, matFluxList
 
  def getMaterialList(self, line, matList):
    """
    returns a list of all the problem materials
    @ In, matList, list
    @ Out, matLIst, list (appends additional material)
    """
    matList.append(line[1])
    return matList 
 
  def getDepInfo(self, timeStepIndex, matchedTimeSteps, numberOfMPI, mrtauCSVOutputMPI, workingDir):
    """
      Read the Instant CSV file to get the material density info relative to depletion
      @ In, numbers-0.csv
      @ Out, depLabelList, depList
    """
    materialList = []
    depLabelList = []
    depList      = []
    timeStepList = []
    matList      = []
    for mpi in xrange(0,numberOfMPI):
      with open(mrtauCSVOutputMPI[mpi], 'r') as outfile: 
        for line in outfile :
          line = self.removeSpaces(line)
          if re.search(r'TIME',line):
            line = re.sub(r'\n',r'',line)          
            self.isotopeList = line.split(',')
            #print self.isotopeList
          if re.search(r'Material',line):
            materialList = line.split(',')
            matList = self.getMaterialList(line.split(','), matList)
          stringIsFloatNumber = self.isFloatNumber(line.split(',')) 
          #print line
          if stringIsFloatNumber is True:
            line = re.sub(r'\n',r'',line)
            if (float(line.split(',')[0]) == float(matchedTimeSteps[timeStepIndex])):  
              for i in xrange (1,len(self.isotopeList)):
                timeStepList.append(line.split(',')[0])
                depLabelList.append('dep'+'|'+materialList[1]+'|'+self.isotopeList[i])
                depList.append(line.split(',')[i])
    #print depLabelList
    #print depList
    return depLabelList, depList, timeStepList, matList
  
  def getDepInfoMrtau(self, timeStepIndex, matchedTimeSteps, mrtauCSVOutput):
    """
      Read the mrtau CSV file to get the material density info relative to depletion
      @ In, numbers.csv
      @ Out, depLabelList, depList
    """
    depList = []
    with open(mrtauCSVOutput, 'r') as outfile:
      for line in outfile :
        line = self.removeSpaces(line)
        stringIsFloatNumber = self.isFloatNumber(line.split(',')) 
        if stringIsFloatNumber is True:
          if (float(line.split(',')[0]) == float(matchedTimeSteps[timeStepIndex])):
            line = re.sub(r'\n',r'',line)   
            for i in xrange (0,len(self.isotopeListMrtau)):
              depList.append(line.split(',')[i+1])
    #print depList
    return depList
    
  def findDecayHeat(self, line):
    """
      Determines if the decay heat is printed
      @ In, Dpl_INSTANT.outp-0
      @ Out, isDecayHeatPrinted, string (yes or no)
    """
    DecayHeatUnits = None 
    isDecayHeatPrinted = line.split(' ')[-2]
    if isDecayHeatPrinted == 'YES': 
      DecayHeatUnits =  line.split(' ')[-1]
      return True, DecayHeatUnits
    else: return False, DecayHeatUnits
  
  def numberOfMediaUsed(self,mpi, workingDir, instantOutputFileMPI):
    """
      finds the number of media used in a given instant output. 
      @ In, numberOfMPI, integer, number of MPI user-selected
      @ In, workingDir, string, working directory 
      @ in, instantOutputFileMPI, string, instant output file with the MPI postpend
      @ Out, mediaUsed, integer, number of media treated in one Instant MPI output 
    """
    count = 0 
    with open(os.path.join(workingDir, instantOutputFileMPI[mpi])) as outfile:
      for line in outfile:
        if re.search(r'Medium\s+\d+\s+used',line):
          count = count + 1 
    return count
    
  def locateMaterialInFile(self, numberOfMPI, workingDir, instantOutputFileMPI):
    """
      finds the material names in a given Instant output file. 
      @ In, numberOfMPI, integer, number of MPI user-selected
      @ In, workingDir, string, working directory 
      @ in, instantOutputFileMPI, string, instant output file with the MPI postpend
      @ Out, materialDict, dictionary, dictionary listing the media treated in a given mpi output. 
              format: {MPI-0:{1:fuel1_1, 2:fuel1_5, 3:{fuel1_7}}, MPI-2:{1:fuel1_2, 2:fuel1_3, 3:fuel1_4, 4:fuel1_6}}
    """
    materialsDict = {}
    for mpi in xrange(0,numberOfMPI):
      count = 0 
      materialsDict['MPI-'+str(mpi)] = {}
      mediaUsed = self.numberOfMediaUsed(mpi, workingDir, instantOutputFileMPI)
      with open(os.path.join(workingDir, instantOutputFileMPI[mpi])) as outfile:
        for line in outfile:
          if re.search(r'Density spatial moment',line):
            count = count + 1 
            matLine = filter(None, line.split(' '))
            materialsDict['MPI-'+str(mpi)][count] = matLine[matLine.index('Material') + 1]
            if count == mediaUsed:
              break 
    #print materialsDict
    return materialsDict
   
  def getDecayHeat(self, timeStepIndex, matchedTimeSteps, numberOfMPI, instantOutputFileMPI, workingDir):
    """
      Read the main output file to get the decay heat
      @ In, timeStepIndex, integer, number of the timestep considered 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ In, numberOfMPI, integer, number of MPI used
      @ In, instantOutputFileMPI, string, instant output file name 
      @ In, workingDir, string, working directory of the output files 
      @ Out, decayLabelList, list, list of the decay labels (under the format fuel|isotope)
      @ Out, decayList, list, list of the decay values 
    """
    materialList = []
    decayLabelList = []
    decayList = []
    timeStepList = []
    for mpi in xrange(0,numberOfMPI):
      decayFlag, breakFlag, matLineFlag, materialCounter = 0, 0, 0, 0
      with open(os.path.join(workingDir, instantOutputFileMPI[mpi])) as outfile:
        for line in outfile:
          if re.search(r'Decay Heat computed',line) and mpi == 0 :
            self.isDecayHeatPrinted, self.decayHeatUnits = self.findDecayHeat(line)
            if self.isDecayHeatPrinted is False: 
              decayLabelList = ['decayHeat']
              decayList = 0
              break 
            else: pass 
          if re.search(r'INDIVIDUAL DECAY HEAT BLOCK',line):  
            decayFlag = 1
            materialCounter = materialCounter + 1  
          if re.search(r'CUMULATIVE DECAY HEAT BLOCK',line):  decayFlag = 0
          if re.search(r'BURNUP OUTPUT',line):  breakFlag = 1
          if decayFlag == 1 and breakFlag == 0 :
            line = re.sub(r'\n',r'',line)
            decayLine = filter(None, line.split(' '))
            if decayLine != []: stringIsFloatNumber = self.isFloatNumber(decayLine)
            if stringIsFloatNumber is True and decayLine != []:
              if (float(decayLine[0]) == float(matchedTimeSteps[timeStepIndex])):  
                for i in xrange (1,len(self.isotopeList)):
                  decayLabelList.append('decay'+'|'+self.materialsDict['MPI-'+str(mpi)][materialCounter]+'|'+self.isotopeList[i])
                  decayList.append(decayLine[i])
          if breakFlag == 1: break
    #print decayLabelList
    #print decayList
    return decayLabelList, decayList
      
  def getDecayHeatMrtau(self, timeStepIndex, matchedTimeSteps):
    """
      get the decay heat from the standalone mrtau output
      @ In, timeStepIndex, integer, number of the timestep considered 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ Out, decayListMrtau, list, list of the decay values from mrtau
    """
    decayFlag = 0
    breakFlag = 0
    self.decayLabelListMrtau = []
    self.numDensityLabelListMrtau = []
    decayListMrtau = []
    with open(self.decayHeatMrtauOutput, 'r') as outfile:
      for line in outfile:
        if re.search(r'INDIVIDUAL DECAY HEAT BLOCK',line):
          decayFlag = 1
        if re.search(r'CUMULATIVE DECAY HEAT BLOCK',line):
          breakFlag = 1
        if decayFlag == 1 and breakFlag == 0:
          line = re.sub(r'\n',r'',line)
          #print line 
          decayLine = filter(None, line.split(' '))
          #print decayLine 
          if decayLine != []: stringIsFloatNumber = self.isFloatNumber(decayLine) 
          if stringIsFloatNumber is True and decayLine != []:
            if (float(decayLine[0]) == float(matchedTimeSteps[timeStepIndex])):
              for i in xrange (0,len(self.isotopeListMrtau)):
                if int(self.perturbationNumber) == 1:
                  self.numDensityLabelListMrtau.append('numDensity'+'|'+self.isotopeListMrtau[i]) 
                  self.decayLabelListMrtau.append('decay'+'|'+self.isotopeListMrtau[i])
                decayListMrtau.append(decayLine[i+1])
          if breakFlag == 1:
            break 
    #print self.numDensityLabelListMrtau
    return decayListMrtau
  
  def getFissionMatrixListAndFissionValuesList(self, instantDict):
    """
      put the fission matrx labels and the fission matrix values in a list 
      @ In, None  
      @ Out,  fissionMatrixNames, list, list of parameter labels that are in the fission matrix 
      @ Out,  fissValues, list, values relative to the labels, within the fission matrix 
      
    """
    fissionMatrixNames = []
    fissValues = [] 
    for i in xrange(0,len(self.paramList)):
      for j in xrange(1,int(self.Ngroups) + 1):
        for k in xrange(1, int(self.Nregions) + 1):
          if self.paramList[i] == 'Group': pass
          else:
            fissionMatrixNames.append(self.paramList[i]+'|gr'+str(j)+'|reg'+str(k))
            fissValues.append(instantDict.get('reactionRateInfo').get(str(j)).get(str(k)).get(self.paramList[i]))
    if 'Group' in fissionMatrixNames: fissionMatrixNames.remove('Group')
    print fissionMatrixNames
    return fissionMatrixNames, fissValues
    
    
  def writeCSV(self,instantDict, timeStepIndex, matchedTimeSteps,jobTitle):
    """
      print the instant coupled to mrtau data in csv files 
      @ In, InstantDict, dictionary, contains all the values collected from instant output 
      @ In, timeStepIndex, integer, number of the timestep considered 
      @ In, matchedTimeSteps, list, list of time steps considered
      @ In, jobTitle, string, job title parsed from instant input 
      @ Out, None 
      
    """
    if self.paramList != []:
      fissionMatrixNames, fissValues = self.getFissionMatrixListAndFissionValuesList(instantDict) 

      csvOutput = os.path.join(instantDict.get('workingDir'),jobTitle+self.perturbationNumber+'.csv')
      with open(csvOutput, 'a+') as f:
        instantWriter = csv.writer(f, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
        if timeStepIndex == 0:
          instantWriter.writerow(['time'] + ['keff'] + ['errorKeff'] + fissionMatrixNames + instantDict.get('fluxLabelList') + instantDict.get('matFluxLabelList') + instantDict.get('depLabelList') + instantDict.get('decayLabelList') + instantDict.get('XSlabelList')) 
        instantWriter.writerow([str(matchedTimeSteps[timeStepIndex])] + instantDict.get('keff') + instantDict.get('errorKeff') + fissValues + instantDict.get('fluxList') + instantDict.get('matFluxList') + instantDict.get('depList') + instantDict.get('decayList') + instantDict.get('XSlist'))
      
  def writeMrtauCSV(self, mrtauDict):
    """
      print the mrtau standalone data in a csv file  
      @ In, mrtauDict, dictionary, contains all the values collected from instant output  
      @ Out, None 
    """ 
    csvOutput = os.path.join(mrtauDict.get('workingDir'),'mrtau'+self.perturbationNumber+'.csv')
    with open(csvOutput, 'a+') as f:
      mrtauWriter = csv.writer(f, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
      if mrtauDict.get('timeStepIndex') == 0:
        mrtauWriter.writerow(['time'] + self.numDensityLabelListMrtau + self.decayLabelListMrtau) 
      mrtauWriter.writerow( [str(mrtauDict.get('matchedTimeSteps')[mrtauDict.get('timeStepIndex')])] + mrtauDict.get('depList') + mrtauDict.get('decayHeatMrtau'))
    
    
