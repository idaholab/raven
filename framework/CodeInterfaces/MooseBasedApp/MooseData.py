import os
import re
import csv
class mooseData:
  """   class that parses output of Moose Vector PP output files and reads in trip, minor block and write a csv file """
  def __init__(self,filen,workingDir,outputFile):
    self.vppFiles = []
    csvfiles = []
    csvreaders=[]
    for i in range(len(filen)):
      csvfiles.append(open(filen[i],"r"))
      csvreaders.append(csv.DictReader(csvfiles[-1]))
    tempDict = self.__read(csvreaders)
    writeDict = self.__sortDict(tempDict)
    self.__write_csv(writeDict,workingDir,outputFile)

  def __read(self,csvreaders):
    """ This method reads the VectorPostProcessor outputs send in as a list of csv.DictReader objects
    @csvreaders, input, list of csv.DictReader objects
    @tempDict, output, temporary dictionary of the data in the outputs (not sorted)
    """
    tempDict = {}
    for icsv, csvdictread in enumerate(csvreaders):
      tempDict[icsv] = {}
      for row in csvdictread:
        tempDict[icsv][row['id']] = row
    return tempDict

  def __sortDict(self, tempDict):
    """ This method returns the sorted Dictionary
    @tempDict, input, temporary dictionary of the data in the outputs (not sorted)
    @sortedDict, output, sorted Dictionary """
    sortedDict ={}
    time = tempDict.keys()
    for location in tempDict[time[0]].keys():
      sortedDict[location] = {}
    for i in range(len(time)):
      for location in tempDict[time[i]].keys():
        for key in tempDict[time[i]][location].keys():
          if 'J' in key:
            sortedDict[location][time[i]] = tempDict[time[i]][location][key]
    # Below assumes that the location coordinates does not change in time.
    for location in sortedDict.keys():
      sortedDict[location]['x'] = tempDict[time[0]][location]['x']
      sortedDict[location]['y'] = tempDict[time[0]][location]['y']
      sortedDict[location]['z'] = tempDict[time[0]][location]['z']
    return sortedDict

  def __write_csv(self,writeDict,workingDir,baseName):
    """   writes the csv file using the input Dictionary
    @writeDict, input, data to write
    @filen, input filename
    """
    self.vppFiles = os.path.join(workingDir,str(baseName+'_VPP'))
    IOcsvfile=open(os.path.join(workingDir,str(baseName+'_VPP.csv')),'w')
    location = {}
    timeStep = {}
    for key in writeDict.keys():
      location[key] = {}
      timeStep[key] = {}
      for coordinate in writeDict[key].keys():
        if coordinate is 'x' or coordinate is 'y' or coordinate is 'z':
          location[key][coordinate] = writeDict[key][coordinate]
        else:
          timeStep[key][coordinate] = writeDict[key][coordinate]
    locationNo = len(location.keys())
    j = 0
    IOcsvfile.write('timeStep,')
    tempString = []
    while(j < locationNo):
      key = location.keys()[j]
      tempString = ('ID%s,x%s,y%s,z%s,value%s,' %(j+1,j+1,j+1,j+1,j+1))
      j = j + 1
      IOcsvfile.write('%s' %(tempString))
    IOcsvfile.write('\n')
    for time in timeStep[timeStep.keys()[0]].keys():
      j = 0
      IOcsvfile.write('%s,' %(time))
      while(j < locationNo):
        key = location.keys()[j]
        if j == (locationNo-1): tempString = ('%s,%s,%s,%s,%s'  %(key,location[key]['x'],location[key]['y'],location[key]['z'],timeStep[key][time]))
        else:                   tempString = ('%s,%s,%s,%s,%s,' %(key,location[key]['x'],location[key]['y'],location[key]['z'],timeStep[key][time]))
        j = j + 1
        IOcsvfile.write('%s' %(tempString))
      IOcsvfile.write('\n')


