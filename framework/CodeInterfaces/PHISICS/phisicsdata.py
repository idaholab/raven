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

  def __init__(self, output, workingDir):
    """
      read the phisics output
    """
    #print output
    markerList = ['Fission matrices of','Scattering matrices of','Multigroup solver ended!']      
    self.pathToPhisicsOutput    = workingDir+'/'+output 
    self.perturbationNumber     = self.getPertNumber(workingDir)
    keffInfo                    = self.getKeff(workingDir)
    reactionRateInfo            = self.getReactionRates(workingDir)
    fissionMatrixInfo           = self.getMatrix(workingDir, markerList[0], markerList[1], 'FissionMatrix')
    self.writeCSV(keffInfo, reactionRateInfo, fissionMatrixInfo, workingDir)
  
  def getNumberOfGroups(self, workingDir):
    """
      give the number of energy groups used in the Phisics simulations  
    """
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'Number of groups', line) :
          self.Ngroups = line.split()[-1]
          break 
    #print self.Ngroups   
  
  def getPertNumber (self, workingDir):
    """
      get the multiplication factor 
    """
    splitWorkDir = workingDir.split('/')
    pertNumber = splitWorkDir[-1]
    return pertNumber 
     
  def getKeff (self, workingDir):
    """
      get the multiplication factor 
    """
    keffDict = {}
    keffReportedInfo = ['keff','errorKeff']
    for i in xrange (0,len(keffReportedInfo)):
      keffDict[keffReportedInfo[i]] = {} 
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        if re.search(r'k-effective at the last', line) :
          keff = line.split()[-1]
          keffDict['keff'][self.perturbationNumber] = keff 
        if re.search(r'error for the eigenvalue', line) :
          errorKeff = line.split()[-1]
          keffDict['errorKeff'][self.perturbationNumber] = errorKeff 
          break 
    #print keffDict
    return keffDict

  def isNumber(self, line):
      """
        check if a string is an integer
      """
      try: 
        int(line[0])
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
          declareDict[str(i)][str(j)][numbering.keys()[k]] = {}
    if typeOfParameters == 'reactionRates':
      self.paramList = []
      for key in numbering.iterkeys():
        self.paramList.append(key)
    if typeOfParameters == 'FissionMatrix':
      self.matrixList = []
      for key in numbering.iterkeys():
        self.matrixList.append(key)  
    return declareDict
        
  def getReactionRates (self, workingDir):
    """
      get the reactions rates, power for each group in PHISICS 
    """
    flagStart,  count = 0, 0 
    self.getNumberOfGroups(workingDir)
    self.Nregions = 1 
    numbering = {}
    reactionRateDict = {}
    self.paramList = [] 
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        #print flagStart
        if re.search(r'averaged flux\s+power', line) :
          numbering = self.mapColumns(line, count, numbering)
          reactionRateDict = self.declareDict(numbering, 'reactionRates')
          #print numbering
          flagStart = 1 
        if re.search(r'Fission matrices of all',line):
          flagStart = 2
        if flagStart == 1 :
          if re.search(r'\w+\s+\w+',line):
            line = re.sub(r'[\|]',' ',line)
            line = line.split()
            #print line 
            stringIsNumber = self.isNumber(line)
            if stringIsNumber == True :
              #print line
              for i in xrange (0,len(numbering)):   ## starts from 2 to exclude the first two elements (groups # and region #)
                #print i 
                reactionRateDict[line[0]][line[1]][self.paramList[i]] = line[numbering.get(self.paramList[i])]
    if reactionRateDict != {}:
      return reactionRateDict  
    
  def getMatrix (self, workingDir, startStringFlag, endStringFlag, typeOfMatrix):
    """
      get the reactions rates, power for each group in PHISICS 
    """
    flagStart, count = 0, 0 
    numbering = {}
    matrixDict = {}
    with open(self.pathToPhisicsOutput, 'r') as outfile:
      for line in outfile :
        #print flagStart
        if re.search(startStringFlag, line):
          flagStart = 1 
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
   
  def writeCSV(self,keffDict, reactionRateDict, fissionMatrixDict, workingDir):
    """
      print the data in csv files 
    """ 
    scatteringMatrixNames = []
    fissionMatrixNames = []
    keffCSV = workingDir+'/'+'keff'+self.perturbationNumber+'.csv'
    print keffCSV
    print "\n\n\n\n\n\n\n\n\n"
    matrixCSV = workingDir+'fmatrix.csv'
    pertFile = 'pert'+self.perturbationNumber+'.csv'
    with open(keffCSV, 'wb') as csvfile:
      keffwriter = csv.writer(csvfile, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
      keffwriter.writerow(['PertNumber']  + ['keff'] + ['errorKeff'] ) 
    #  keffwriter.writerow([keffDict.get('keff').keys()[0], keffDict.get('keff').itervalues().next(), keffDict.get('errorKeff').itervalues().next(), reactionRateDict.get(str(1:3)).get(str('1')).get(self.paramList[2:6])])
    fissvalues = []
    if self.paramList != []:
      for i in xrange(0,len(self.paramList)):
        for j in xrange(1,int(self.Ngroups) + 1):
          for k in xrange(1, int(self.Nregions) + 1):
            fissionMatrixNames.append(self.paramList[i]+'|gr'+str(j)+'|reg'+str(k))
            fissvalues.append(reactionRateDict.get(str(j)).get(str(k)).get(self.paramList[i]))
      #print fissionMatrixNames
      if 'Group' in fissionMatrixNames: fissionMatrixNames.remove('Group')
      with open(keffCSV, 'wb') as csvfile:
        keffwriter = csv.writer(csvfile, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
        keffwriter.writerow(['keff'] + ['errorKeff'] + fissionMatrixNames) 
        keffwriter.writerow([keffDict.get('keff').itervalues().next()] + [keffDict.get('errorKeff').itervalues().next()] + fissvalues)
    

