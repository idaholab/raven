from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
import numpy as np
import csv
from pathlib import Path

def parseLine(line):
  """
  Parse composition line by deleting whitespace and separating the isotope and atomic density.
    @ In, line, string, line of isotope and composition
    @ Out, result, tuple, (isotope, atomic density)
  """
  line = line.lstrip()
  isotope, atomDensity = line.split("  ")
  result = (isotope, float(atomDensity))
  return result

def filterTrace(compDict, percentCutoff):
  """
  Filters isotopes with less than percentCutoff for easier calculation.
    @ In, compDict, dictionary, key=isotope
                                value=atomic density
    @ In, percentCutoff, float, cutoff threshold for ignoring isotopes (0 -1)
    @ Out, comptDict, dictionary, key=isotope
                                  value=compDict
  """
  # check if percentCutoff value is valid
  if percentCutoff < 0 or percentCutoff > 1:
    raise ValueError('Percent has to be between 0 and 1')

  # calculate atomicDensityCutoff
  totalAtomicDensity = sum(compDict.values())
  atomicDensityCutoff = percentCutoff * totalAtomicDensity

  # delete list since cannot change dictionary during iteration
  deleteList = []
  for key, atomDensity in compDict.items():
    if atomDensity < atomicDensityCutoff:
      deleteList.append(key)

  # delete the isotopes with less than percentCutoff
  for isotope in deleteList:
    del compDict[isotope]

  return compDict



def bumatRead(bumatFile, percentCutoff):
  """
  Reads serpent .bumat output file and stores the composition in a dictionary.
    @ In, bumatFile, string, bumat file path
    @ In, percentCutoff, float, cutoff threshold for ignoring isotopes (0 -1)
    @ Out, compDict, dictionary, key=isotope
                                value=atomic density
  """
  with open(bumatFile) as f:
    compLines = f.readlines()[5:]

  compDict = {}
  header = compLines[0]
  for i in range(1, len(compLines)):
    parsed = parseLine(compLines[i])
    # isotope as key, atomic density as value
    compDict[parsed[0]] = parsed[1]

  compDict = filterTrace(compDict, percentCutoff)
  return compDict


def searchKeff(resFile):
  """
  Searches and returns the mean keff value in the .res file.
    @ In, resFile, string, path to .res file
    @ Out, keffDict, dictionary, key = keff or sd
                                 value = list of keff or sd
  """
  with open(resFile) as f:
    lines = f.readlines()

  keffList = []
  sdList = []

  for i in range(0, len(lines)):
    if 'IMP_KEFF' in lines[i]:
      keffList.append(keffLineParse(lines[i])[0])
      sdList.append(keffLineParse(lines[i])[1])

  keffDict = {}
  keffDict['keff'] = keffList 
  keffDict['sd'] = sdList
  return keffDict


def keffLineParse(keffLine):
  """
  Parses through the anaKeff line in .res file.
    @ In, keffLine, string, string from .res file listing IMPKEFF
    @ Out, keffTuple, tuple, (mean IMPKEFF, sd of IMPKEFF)
  """
  start = keffLine.find('=')
  newKeffLine = keffLine[start:]
  start = newKeffLine.find('[')
  end = newKeffLine.find(']')
  
  # +3 and -1 is to get rid of leading and trailing whitespace
  keffSd = newKeffLine[start + 3:end - 1]
  (keff, sd) = keffSd.split(' ')
  keffTuple = (keff, sd)
  return keffTuple


def csvRenderDict(csvFilename, dictionary, header):
  """
  Renders csv given the dictionary column 1 = key, column 2 = value.
    @ In, csvFilename, string, path of csv file to be created
    @ In, dictionary, dictionary, dictionary to be rendered into csv file
    @ In, header, list, list of length 2 of header strings
    @ Out, bool, bool, True if successful
  """
  with open(csvFilename, 'w') as csvFile:
    writer = csv.writer(csvFile)
    # write header
    writer.writerow(header)
    for key, value in dictionary.items():
      writer.writerow([key, value])
  return True

def readFileIntoList(file):
  """ 
  Reads file into list, every line as element.
    @ In, file, string, name of file
    @ Out, listFromFile, list, contents in the file as list
  """
  read = open(file, 'r')
  lines = read.readlines()
  listFromFile = []
  for line in lines:
    listFromFile.append(line.strip())
  read.close()
  return listFromFile

def findDeptime(inputFile):
  """
  Finds the deptime from the input file.
    @ In, inputFile, string, input file path
    @ Out, deptime, string, depletion time in days
  """
  hit = False
  with open(inputFile, 'r') as file:
    for line in file:
      if line.split(' ')[0] == 'dep':
        if line.split(' ')[1] != 'daystep':
          print('Currently can only take daystep')
          raise ValueError()
        else:
          hit = True
          continue
      if hit:
        deptime = line.split(' ')[0]
        break

  return deptime



def makeCsv(csvFilename, inBumatDict, outBumatDict,
       keffDict, isoList, inputFile):
  """ 
  Renders the  csv as filename with the given bumat dict and keff dict.
    @ In, csvFilename, string, filename of csv output
    @ In, inBumatDict, dictionary, key=isotope
                                   value=atomic density
    @ In, outBumatDict, dictionary, key=isotope
                                    value=atomic density
    @ In, keffDict, dictionary, key='keff''sd'
                                value=keff and sd
    @ In, isoList, list, list of isotopes to track
    @ In, inputFile, string, path to input file
    @ Out, None
  """

  # parse through, get keff value
  bocKeff = keffDict['keff'][0]
  eocKeff = keffDict['keff'][1]
  deptime = findDeptime(inputFile)
  
  with open(csvFilename, 'w') as csvFile:
    writer = csv.writer(csvFile)
    # fresh isoList
    headerList = (['f'+iso for iso in isoList] +
            ['bocKeff', 'eocKeff', 'deptime'] +
            ['d'+iso for iso in isoList])
    writer.writerow(headerList)
    # initialize as zero
    freshAdensList = [0] * len(isoList)
    depAdensList = [0] * len(isoList)
    for key in inBumatDict:
      if key in isoList:
        index = isoList.index(key)
        freshAdensList[index] = inBumatDict[key]
    for key in outBumatDict:
      if key in isoList:
        index = isoList.index(key)
        depAdensList[index] = outBumatDict[key] 

    row = freshAdensList + [bocKeff, eocKeff, deptime] + depAdensList
    # add keff value to adens list, like header
    writer.writerow(row)
