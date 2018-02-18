"""
Created on July 25th, 2017
@author: rouxpn
"""

import os
import sys
import re
import csv

class phisicsdata():
  """
    This class parses the phisics output of interest. The output of interest are placed in the a csv file.
  """

  def __init__(self, output, workingDir, mrtauBoolean, jobTitle):
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
    #print output
    self.mrtauBoolean = mrtauBoolean
    instantOutput = [output, 'numbers-0.csv', 'Dpl_INSTANT_'+jobTitle+'_flux_mat.csv']
    mrtauOutput = ['numbers.csv', 'DecayHeat.out']
    self.instantCSVOutput = workingDir+'/'+instantOutput[2]
    if mrtauBoolean == False: self.mrtauCSVOutput = workingDir+'/'+instantOutput[1]
    if mrtauBoolean == True: self.mrtauCSVOutput = workingDir+'/'+mrtauOutput[0]
    self.decayHeatMrtauOutput = workingDir+'/'+mrtauOutput[1]
    markerList = ['Fission matrices of','Scattering matrices of','Multigroup solver ended!']
    self.pathToPhisicsOutput    = workingDir+'/'+instantOutput[0]
    self.perturbationNumber     = self.getPertNumber(workingDir)

    if self.mrtauBoolean == False:
      mrtauTimeSteps            = self.getMrtauInstantTimeSteps()
      instantTimeSteps          = self.getInstantTimeSteps()
      matchedTimeSteps          = self.commonInstantMrtauTimeStep(instantTimeSteps, mrtauTimeSteps)
    if self.mrtauBoolean == True:
      matchedTimeSteps = self.getMrtauTimeSteps()
      self.getMrtauIsotopeList()
    for timeStepIndex in xrange (0,len(matchedTimeSteps)):
      if self.mrtauBoolean == False:
        keff, errorKeff           = self.getKeff(workingDir, timeStepIndex, matchedTimeSteps)
        reactionRateInfo          = self.getReactionRates(workingDir, timeStepIndex, matchedTimeSteps)
        fissionMatrixInfo         = self.getMatrix(workingDir, markerList[0], markerList[1], 'FissionMatrix', timeStepIndex, matchedTimeSteps)
        fluxLabelList, fluxList, matFluxLabelList, matFluxList     = self.getFluxInfo(timeStepIndex, matchedTimeSteps)
        depLabelList, depList, timeStepList, matList     = self.getDepInfo(timeStepIndex, matchedTimeSteps)
        decayLabelList, decayList  = self.getDecayHeat(timeStepIndex, matchedTimeSteps)
        self.writeCSV(keff, errorKeff, reactionRateInfo, fissionMatrixInfo, workingDir, fluxLabelList, fluxList, matFluxLabelList, matFluxList, depLabelList, depList, timeStepList, decayLabelList, decayList, timeStepIndex, matchedTimeSteps)

      if self.mrtauBoolean == True:
        decayHeatLabelMrtau, decayHeatMrtau, numberDensityLabelMrtau = self.getDecayHeatMrtau(timeStepIndex, matchedTimeSteps)
        depList = self.getDepInfoMrtau(timeStepIndex, matchedTimeSteps)
        self.writeMrtauCSV(workingDir, numberDensityLabelMrtau, depList, timeStepIndex, matchedTimeSteps, decayHeatLabelMrtau, decayHeatMrtau, numberDensityLabelMrtau)

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
    """
    splitWorkDir = workingDir.split('/')
    pertNumber = splitWorkDir[-1]
    return pertNumber

  def getMrtauInstantTimeSteps(self):
    """
      get the time steps in the coupled mrtau instant output
      IN: numbers-0.csv
      OUT: timeSteps, list
    """
    count = 0
    timeSteps = []
    with open(self.mrtauCSVOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'Material',line):
          count = count + 1
        if count == 1:
          stringIsFloatNumber = self.isFloatNumber(line.split(','))
          if stringIsFloatNumber is True:
            timeSteps.append(line.split(',')[0])
        if count > 1:
          break
    #print timeSteps
    return timeSteps

  def getMrtauTimeSteps(self):
    """
      get the time steps in the mrtau standalone output
      IN: numbers.csv
      OUT: timeSteps, list
    """
    timeSteps = []
    with open(self.mrtauCSVOutput, 'r') as outfile:
      for line in outfile :
        stringIsFloatNumber = self.isFloatNumber(line.split(','))
        if stringIsFloatNumber is True:
          timeSteps.append(line.split(',')[0])
    #print timeSteps
    return timeSteps

  def getMrtauIsotopeList(self):
    """
      get the isotope in the mrtau standalone output
      IN: numbers.csv
      OUT: self.isotopeListMrtau, list
    """
    self.isotopeListMrtau = []
    with open(self.mrtauCSVOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'TIME',line):
          line = re.sub(r'\(days\)',r'',line)
          self.isotopeListMrtau = filter(None,line.split(' '))
          break

  def getInstantTimeSteps(self):
    """
      get the time steps in the Instant output
      IN: Dpl_INSTANT_HTGR_test_flux_mat.csv
      OUT: timeSteps, list
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
      IN: mrtauTimeSteps, list
      OUT: timeSteps, list
    """
    timeSteps = []
    for i in xrange (0, len(mrtauTimeSteps)):
      timeSteps.append('%.6E' % (float(mrtauTimeSteps[i])))
    #print timeSteps
    return timeSteps

  def convertInDays(self, instantTimeSteps):
    """
      convert the Instant time steps (seconds) into days
      IN: instantTimeSteps, list
      OUT: timeSteps, list
    """
    timeSteps = []
    #print instantTimeSteps
    for i in xrange (0, len(instantTimeSteps)):
      timeSteps.append('%.6E' % (float(instantTimeSteps[i]) / (24 * 60 * 60)))
    #print timeSteps
    return timeSteps

  def commonInstantMrtauTimeStep(self, instantTimeSteps, mrtauTimeSteps):
    """
      get the common time steps in the Instant output and mrtau input
      IN: instantTimeSteps, mrtauTimeSteps, lists
      OUT: timeSteps, list
    """
    #print mrtauTimeSteps
    #print instantTimeSteps
    instantTimeSteps = self.convertInDays(instantTimeSteps)
    mrtauTimeSteps = self.reduceDigits(mrtauTimeSteps)
    commonTimeSteps = list(set(instantTimeSteps) & set(mrtauTimeSteps))
    return commonTimeSteps

  def findUnits(self, line):
    """
      return the units in which the time steps are printed in the instant output
      IN: line, string
      OUT: units, string
    """
    units = line.split(' ')[-3]
    return units

  def getKeff (self, workingDir, timeStepIndex, matchedTimeSteps):
    """
      get the multiplication factor
    """
    count = 0
    self.units = ''
    keff = []
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'Time Input Type', line):
          self.units = self.findUnits(line)
        if re.search(r'k-effective at the last', line) :
          keff = [line.split()[-1]]
          count = count + 1
        if re.search(r'error for the eigenvalue', line) :
            errorKeff = [line.split()[-1]]
        if count  == 1:
          if re.search(r'Time\('+self.units+'\)', line):
            timeStep = self.convertInDays([line.split(' ')[-1]])
            #print timeStep
            if float(timeStep[0]) - float(matchedTimeSteps[timeStepIndex]) < 1E-03: break
    #print keff
    return keff, errorKeff

  def isNumber(self, line):
      """
        check if a string is an integer
      """
      try:
        int(line[0])
        return True
      except ValueError:
        return False

  def isFloatNumber(self, line):
    """
      check if a string is an integer
    """
    try:
      float(line[0])
      return True
    except ValueError:
      return False

  def declareDict(self, numbering, typeOfParameters):
    """
      declare the RR dictionary
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
      IN: wordingDir, string, working directory.
      IN: timeStepIndex, integer, number of the timestep considered
      IN: matchedTimeSteps, list, list of time steps considered
      OUT: ReactionRateDict, Dict, dictionary containing the RR infos
    """
    flagStart,  count, countTimeStep = 0, 0, 0
    self.getNumberOfGroups(workingDir)
    self.Nregions = 1
    numbering = {}
    reactionRateDict = {}
    self.paramList = []
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        #print flagStart
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
              #print line
              for i in xrange (0,len(numbering)):
                #print i
                if self.paramList[i] == 'Group': pass
                else: reactionRateDict[line[0]][line[1]][self.paramList[i]] = line[numbering.get(self.paramList[i])]
        if countTimeStep  == 1:
          if 'Time(seconds)' in line:
            timeStep = self.convertInDays([line.split(' ')[-1]])
            #print timeStep
            if float(timeStep[0]) - float(matchedTimeSteps[timeStepIndex]) < 1E-03: break
    if reactionRateDict != {}:
      return reactionRateDict

  def getMatrix (self, workingDir, startStringFlag, endStringFlag, typeOfMatrix, timeStepIndex, matchedTimeSteps):
    """
      get the reactions rates, power for each group in PHISICS
      IN: timeStepIndex, integer, number of the timestep considered
      IN: matchedTimeSteps, list, list of time steps considered
      OUT: matrixDict, dictionary
    """
    flagStart, count, countTimeStep = 0, 0, 0
    numbering = {}
    matrixDict = {}
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        #print flagStart
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
            #print line
            for i in xrange (1,len(numbering) + 1):
              #print i
              matrixDict[line[0]][str(regionNumber)][str(i)] = line[numbering.get(str(i)) + 1]
        if countTimeStep  == 1:
          if 'Time(seconds)' in line:
            timeStep = self.convertInDays([line.split(' ')[-1]])
            if float(timeStep[0]) - float(matchedTimeSteps[timeStepIndex]) < 1E-03: break
    #print matrixDict
    return matrixDict

  def mapColumns(self, line, count, numbering):
    """
      numbers the column relative to the reaction rates
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
      IN: IDlist (list) list of all the parameter in the csv output
      OUT: xPositionInList, yPositionInList, zPositionInList, firstGroupPositionInList, (integers), corresponding
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
      IN: Dpl_INSTANT_HTGR_test_flux_mat.csv
      IN: timeStepIndex, integer, number of the timestep considered
      IN: matchedTimeSteps, list, list of time steps considered
      OUT: fluxLabelList, fluxList, matFluxLabelList, matFluxList (lists)
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
          timeStep = line.split(',')[line.split(',').index('Time') + 1]
          timeStep = self.convertInDays([timeStep])
          if float(timeStep[0]) - float(matchedTimeSteps[timeStepIndex]) < 1E-03:  countTimeStep = countTimeStep + 1
        if countTimeStep == 1:
          if re.search(r'FLUX BY CELLS',line):
            flagFlux = 1
          if re.search(r'FLUX BY MATERIAL',line):
            flagFlux = 2
          #print line
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
              #if zPosition == 0: ## it means this is a 2D case
              for g in xrange (1,int(self.Ngroups) + 1):
                fluxLabelList.append('flux'+'|'+'cell'+line.split(',')[0]+'|'+'gr'+str(g))
                fluxList.append(line.split(',')[group1Position + g - 1])
          if flagFlux == 2:
            stringIsNumber = self.isNumber(line.split(','))
            if stringIsNumber is True:
              line = re.sub(r' ',r'',line)
              line = re.sub(r'\n',r'',line)
              for g in xrange (1,int(self.Ngroups) + 1):
                matFluxLabelList.append('flux'+'|'+'mat'+line.split(',')[0]+'|'+'gr'+str(g))
                matFluxList.append(line.split(',')[g])
    #print matFluxLabelList, matFluxList
    return fluxLabelList, fluxList, matFluxLabelList, matFluxList

  def getMaterialList(self, line, matList):
    """
    returns a list of all the problem materials
    IN: matList, list
    OUT: matLIst, list (appends additional material)
    """
    matList.append(line[1])
    return matList

  def getDepInfo(self, timeStepIndex, matchedTimeSteps):
    """
      Read the Instant CSV file to get the material density info relative to depletion
      IN: numbers-0.csv
      OUT: depLabelList, depList
    """
    materialList = []
    depLabelList = []
    depList = []
    timeStepList = []
    matList = []
    with open(self.mrtauCSVOutput, 'r') as outfile:
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
          if (float(line.split(',')[0]) - float(matchedTimeSteps[timeStepIndex])) < 1E-3:
            for i in xrange (1,len(self.isotopeList)):
              timeStepList.append(line.split(',')[0])
              depLabelList.append('dep'+'|'+materialList[1]+'|'+self.isotopeList[i])
              depList.append(line.split(',')[i])
    #print depLabelList
    #print depList
    return depLabelList, depList, timeStepList, matList

  def getDepInfoMrtau(self, timeStepIndex, matchedTimeSteps):
    """
      Read the mrtau CSV file to get the material density info relative to depletion
      IN: numbers.csv
      OUT: depLabelList, depList
    """
    materialList = []
    depLabelList = []
    depList = []
    timeStepList = []
    with open(self.mrtauCSVOutput, 'r') as outfile:
      for line in outfile :
        line = self.removeSpaces(line)
        stringIsFloatNumber = self.isFloatNumber(line.split(','))
        #print line
        if stringIsFloatNumber is True:
          line = re.sub(r'\n',r'',line)
          for i in xrange (1,len(self.isotopeListMrtau)):
            depList.append(line.split(',')[i-1])
    #print depList
    return depList

  def findDecayHeat(self, line):
    """
      Determines if the decay heat is printed
      IN: Dpl_INSTANT.outp-0
      OUT: isDecayHeatPrinted, string (yes or no)
    """
    DecayHeatUnits = None
    isDecayHeatPrinted = line.split(' ')[-2]
    if isDecayHeatPrinted == 'YES':
      DecayHeatUnits =  line.split(' ')[-1]
      return True, DecayHeatUnits
    else: return False, DecayHeatUnits


  def getDecayHeat(self, timeStepIndex, matchedTimeSteps):
    """
      Read the main output file to get the decay heat
      IN: Dpl_INSTANT.outp-0
      OUT: decayLabelList, decayList
    """
    isotopeList = []
    materialList = []
    decayLabelList = []
    decayList = []
    timeStepList = []
    decayFlag, breakFlag = 0, 0
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile:
        if re.search(r'Decay Heat computed',line):
          self.isDecayHeatPrinted, self.decayHeatUnits = self.findDecayHeat(line)
          if self.isDecayHeatPrinted is False:
            decayLabelList = ['decayHeat']
            decayList = 0
            break
          else: pass
        if re.search(r'Density spatial moment',line):
          matLine = filter(None, line.split(' '))
          matDecay = matLine[matLine.index('Material') + 1]
          #print matDecay
        if re.search(r'INDIVIDUAL DECAY HEAT BLOCK',line):
          decayFlag = 1
        if re.search(r'CUMULATIVE DECAY HEAT BLOCK',line):
          decayFlag = 2
        if re.search(r'BURNUP OUTPUT',line):
          breakFlag = 1
        if decayFlag == 1 and breakFlag == 0 :
          line = re.sub(r'\n',r'',line)
          decayLine = filter(None, line.split(' '))
          #print decayLine
          if decayLine != []: stringIsFloatNumber = self.isFloatNumber(decayLine)
          #print stringIsFloatNumber
          if stringIsFloatNumber is True and decayLine != []:
            #print decayLine[0]
            if (float(decayLine[0]) - float(matchedTimeSteps[timeStepIndex])) < 1E-3:
              for i in xrange (1,len(self.isotopeList)):
                timeStepList.append(decayLine[0])
                decayLabelList.append('decay'+'|'+matDecay+'|'+self.isotopeList[i])
                decayList.append(decayLine[i])
        if breakFlag == 1:
          break
    #print decayLabelList
    #print decayList
    return decayLabelList, decayList

  def getDecayHeatMrtau(self, timeStepIndex, matchedTimeSteps):
    """
      get the decay heat from the standalone mrtau output
    """
    decayFlag = 0
    breakFlag = 0
    decayLabelListMrtau = []
    numDensityLabelListMrtau = []
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
            for i in xrange (1,len(self.isotopeListMrtau)):
              numDensityLabelListMrtau.append('numDensity'+'|'+self.isotopeListMrtau[i-1])
              decayLabelListMrtau.append('decay'+'|'+self.isotopeListMrtau[i-1])
              decayListMrtau.append(decayLine[i-1])
    #print decayLabelListMrtau
    return decayLabelListMrtau, decayListMrtau, numDensityLabelListMrtau

  def writeCSV(self,keff, errorKeff, reactionRateDict, fissionMatrixDict, workingDir, fluxLabelList, fluxList, matFluxLabelList, matFluxList, depLabelList, depList, timeStepList, decayLabelList, decayList, timeStepIndex, matchedTimeSteps):
    """
      print the instant coupled to mrtau data in csv files
    """
    fissionMatrixNames = []
    keffCSV = workingDir+'/'+'keff'+self.perturbationNumber+'.csv'
    fissvalues = []
    if self.paramList != []:
      for i in xrange(0,len(self.paramList)):
        for j in xrange(1,int(self.Ngroups) + 1):
          for k in xrange(1, int(self.Nregions) + 1):
            if self.paramList[i] == 'Group': pass
            else:
              fissionMatrixNames.append(self.paramList[i]+'|gr'+str(j)+'|reg'+str(k))
              fissvalues.append(reactionRateDict.get(str(j)).get(str(k)).get(self.paramList[i]))
      #print fissionMatrixNames
      if 'Group' in fissionMatrixNames: fissionMatrixNames.remove('Group')
      with open(keffCSV, 'wb') as csvfile:
        if timeStepIndex == 0:
          keffwriter = csv.writer(csvfile, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
          keffwriter.writerow(['time'] + ['keff'] + ['errorKeff'] + fissionMatrixNames + fluxLabelList + matFluxLabelList + depLabelList + decayLabelList)
        if timeStepIndex > 0:
          keffwriter.writerow([str(matchedTimeSteps[timeStepIndex])] + keff + errorKeff + fissvalues + fluxList + matFluxList + depList + decayList)

  def writeMrtauCSV(self, workingDir, depLabelList, depList, timeStepIndex, matchedTimeSteps, decayHeatLabelMrtau, decayHeatMrtau, numberDensityLabelMrtau):
    """
      print the mrtau standalone data in csv files
    """
    keffCSV = workingDir+'/'+'keff'+self.perturbationNumber+'.csv'
    with open(keffCSV, 'wb') as f:
      print timeStepIndex
      print "\n\n\n\n\n"
      if timeStepIndex == 0:
        self.test = csv.writer(f, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
        self.test.writerow(['time'] + ['keff'] + ['errorKeff'])
      if timeStepIndex > 0:
        self.test.writerow([str(matchedTimeSteps[timeStepIndex])])


