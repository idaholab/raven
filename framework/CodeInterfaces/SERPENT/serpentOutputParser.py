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
Created May 9th, 2019

@author: alfoa
"""
#External Modules--------------------begin
import csv
#External Modules--------------------end

#Internal Modules--------------------begin
#Internal Modules--------------------end

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
  compLines = open(bumatFile,"r").readlines()[5:]
  compDict = {}
  header = compLines.pop(0)
  for line in compLines:
    parsed = parseLine(line)
    # isotope as key, atomic density as value
    compDict[parsed[0]] = parsed[1]

  compDict = filterTrace(compDict, percentCutoff)
  return compDict

def searchKeff(resFile):
  """
    Searches and returns the mean keff value in the .res file.
    @ In, resFile, string, path to .res file
    @ Out, keffDict, dict, key = keff or sd
                                 value = list of keff or sd
  """
  lines = open(resFile,"r").readlines()

  keffList = []
  sdList = []
  keffDict = {}

  for line in lines:
    if 'IMP_KEFF' in line:
      keffList.append(keffLineParse(line)[0])
      sdList.append(keffLineParse(line)[1])

  keffDict['keff'] = keffList
  keffDict['sd'] = sdList
  return keffDict

def keffLineParse(keffLine):
  """
    Parses through the anaKeff line in .res file.
    @ In, keffLine, string, string from .res file listing IMPKEFF
    @ Out, keffTuple, list, (mean IMPKEFF, sd of IMPKEFF)
  """
  newKeffLine = keffLine[keffLine.find('='):]
  start = newKeffLine.find('[')
  end = newKeffLine.find(']')
  keffSd = newKeffLine[start + 1:end].strip()
  keffTuple = keffSd.split()
  return keffTuple

def findDeptime(inputFile):
  """
    Finds the deptime from the input file.
    @ In, inputFile, string, input file path
    @ Out, deptime, string, depletion time in days
  """
  deptime = -1
  hit = False
  with open(inputFile, 'r') as file:
    for line in file:
      if hit:
        deptime = line.split(' ')[0]
        break
      if line.split()[0] == 'dep' and line.split()[1] != 'daystep':
        raise ValueError('Currently can only take daystep')
      else:
        hit = True
  return deptime


def makeCsv(csvFilename, inBumatDict, outBumatDict, keffDict, isoList, inputFile):
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
