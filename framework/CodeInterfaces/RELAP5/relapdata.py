import re
import xml.etree.ElementTree as ET
"""
Created on May 5, 2016

@author: alfoa
"""

class relapdata:
  """
    Class that parses output of relap5 output file and reads in trip, minor block and write a csv file
  """
  def __init__(self,filen):
    """
      Constructor
      @ In, filen, string, file name to be parsed
      @ Out, None
    """
    self.lines=open(filen,"r").readlines()
    self.trips=self.returntrip(self.lines)
    self.minordata=self.getminor(self.lines)
    self.EndTime=self.gettime(self.lines)
    self.readraven()

  def hasAtLeastMinorData(self):
    """
      Method to check if at least the minor edits are present
      @ In, None
      @ Out, hasMinor, bool, True if it has minor data
    """
    hasMinor = self.minordata != None
    return hasMinor

  def gettime(self,lines):
    """
      Method to check ended time of the simulation
      @ In, lines, list, list of lines of the output file
      @ Out, time, float, Final time
    """
    return self.gettimeDeck(lines)[-1]

  def gettimeDeck(self,lines):
    """
      Method to check ended time of the simulation (multi-deck compatible)
      @ In, lines, list, list of lines of the output file
      @ Out, times, list, list of floats. Final times
    """
    times = []
    for i in lines:
      if re.match('^\s*Final time=',i):
        times.append(i.split()[2])
    return times

  def returntrip(self,lines):
    """
      Method to return the trip information
      @ In, lines, list, list of lines of the output file
      @ Out, tripArray, list, list of dictionaries containing the trip info
    """
    tripArray=[]
    for i in range(len(lines)):
      if re.match('^\s*0Trip\s*number',lines[i]):
        tripArray=[]
        i=i+1
        while not (re.match('^0System|^0\s*Total',lines[i])):
          temp1 = lines[i].split();
          for j in range(len(temp1)/2):
            if (float(temp1[2*j+1])>-1.000):
              tripArray.append({temp1[2*j]:temp1[2*j+1]});
          i=i+1;
    return tripArray;

  def readminorblock(self,lines,i):
    """
      Method that reads in a block of minor edit data and returns a dictionary of lists
      @ In, lines, list, list of lines of the output file
      @ In, i, int, line number where to start the reading
      @ Out, minorDict, dict, dictionary containing the minor edit info
    """
    minorDict={}
    edit_keys=[]
    flagg1 = 0
    flagg2 = 0
    block_count=0
    while(flagg1==0 & flagg2==0):
      if flagg1==0:
        tempkeys=[]
        temp1 = re.split('\s{2,}|\n',lines[i])
        temp2 = re.split('\s{2,}|\n',lines[i+1])
        temp1.pop()
        temp2.pop()
        temp2.pop(0)
        temparray=[]
        for j in range(len(temp1)):
          tempkeys.append(temp1[j]+'_'+temp2[j])
          edit_keys.append(temp1[j]+'_'+temp2[j])
          temparray.append([]);     #   allocates array for data block
        i=i+4
        while not re.match('^\s*1 time|^1RELAP5|^\s*\n|^\s*1RELAP5|^\s*MINOR EDIT',lines[i]):
          tempdata=lines[i].split()
          if ('Reducing' not in tempdata):
            for k in range(len(temparray)): temparray[k].append(tempdata[k])
          i=i+1
          if re.match('^\s*1 time|^\s*1\s*R5|^\s*\n|^1RELAP5',lines[i]): break
        for l in range(len(tempkeys)): minorDict.update({tempkeys[l]:temparray[l]})
        if re.match('^\s*1\s*R5|^\s*\n|^\s*1RELAP5|^\s*MINOR EDIT',lines[i]):
          flagg2=1
          flagg1=1
        elif re.match('^\s*1 time',lines[i]):
          block_count=block_count+1
          flagg=1
    return minorDict

  def getminor(self,lines):
    """
      Method that looks for key word MINOR EDIT for reading minor edit block
      and calls readminor block to read in the block of minor edit data
      @ In, lines, list, list of lines of the output file
      @ Out, minorDict, dict, dictionary containing the minor edit info
    """
    count  = 0
    minorDict = None
    for i in range(len(lines)):
      if re.match('^MINOR EDIT',lines[i]):
        j=i+1
        count=count+1
        tempdict=self.readminorblock(self.lines,j)
        if (count==1): minorDict=tempdict;
        else:
          for k in minorDict.keys():
            for l in tempdict.get(k):
              minorDict[k].append(l)
    return minorDict

  def readraven(self):
    """
      Method that looks for the RAVEN keyword where the sampled vars are stored
      @ In, None
      @ Out, None
    """
    flagg=0
    self.ravenData={}
    for i in range(len(self.lines)):
      if re.search('RAVEN',self.lines[i]):
        i=i+1
        while flagg==0:
          if re.search('RAVEN',self.lines[i]): flagg=1
          else: self.ravenData[self.lines[i].split()[1].replace("*","")]=self.lines[i].split()[3]
          i=i+1
    return

  def write_csv(self,filen):
    """
      Method that writes the csv file from minor edit data
      @ In, filen, string, input file name
      @ Out, None
    """
    IOcsvfile=open(filen,'w')
    if self.minordata != None:
      for i in range(len(self.minordata.keys())): IOcsvfile.write('%s,' %(self.minordata.keys()[i].strip().replace("1 time_(sec)","time").replace(' ', '_')))
    for j in range(len(self.ravenData.keys())):
      IOcsvfile.write('%s' %(self.ravenData.keys()[j]))
      if j+1<len(self.ravenData.keys()): IOcsvfile.write(',')
    IOcsvfile.write('\n')
    for i in range(len(self.minordata.get(self.minordata.keys()[0]))):
      for j in range(len(self.minordata.keys())): IOcsvfile.write('%s,' %(self.minordata.get(self.minordata.keys()[j])[i]))
      for k in range(len(self.ravenData.keys())):
        IOcsvfile.write('%s' %(self.ravenData[self.ravenData.keys()[k]]))
        if k+1<len(self.ravenData.keys()): IOcsvfile.write(',')
      IOcsvfile.write('\n')

