"""
  Created on March 28, 2018
  @author: JYoo
"""
import os
class CTFparser():
  """
    Import the CobraTF input as list of lines, provide methods to add/change entries and print it back
  """
  def __init__(self,inputFile):
    """
      Constructor
      @ In, inputFile, string, input file name
      @ Out, None
    """
    IOfile = open(inputFile, 'r')
    self.currentInputFile = inputFile
    self.modifiedDictionary = {}    
    self.lines = IOfile.readlines()

  def printInput(self, outFile=None):
    """
      Method to print out the new input
      @ In, outFile, string, optional, output file root
      @ Out, None
    """
    if outFile is None:
        outFile = self.currentInputFile
    newInput = open(outFile, 'w')
    newInput.writelines(self.lines)
    newInput.close()

  def modifDictionaryList(self, modifDictList):
    """
      Method to update self.modifiedDictionary
      @ In, modifDictList, dictionary, modified list of variables
      @ Out, None
    """
    self.modifiedDictionary.update(modifDictList)
    return

  def changeVariable(self, modifDict):
    """
      Method to modify the CTF input variables
      CTF input is modified based on the request from RAVEN with a form (lineNumber|position)
      @ In, modifDict, dictionary
      @ Out, None
    """
    for count in range(len(modifDict)):
      # print(lineNumberPosition, value)
      lineNumber, position = list(modifDict.keys())[count].split("|")
      value = list(modifDict.values())[count]
      # type conversion from str to int
      lineNumber = int(lineNumber)
      position = int(position)
      # error for the commented line
      if self.lines[lineNumber - 1].startswith("*"):
          raise IOError("Error: the line number requested by the user (line " +
                          str(lineNumber) + ") is the commented line")
      # error for the blank line
      if len(self.lines[lineNumber - 1].split()) == 0:
          raise IOError("Error: the line number requested by the user (line " +
                        str(lineNumber) + ") is the blank line")
      if lineNumber > len(self.lines):
          raise IOError("Error: the line number requested by the user (line " + str(lineNumber) +
                        ") is larger than the total number of lines of the original input file (=" + str(len(self.lines)) + ")")
      lineToModify = self.lines[lineNumber - 1]
      splittedLine = lineToModify.split()
      if position > len(splittedLine):
          raise IOError("Error: position number requested by the user (" + str(position) + ") is larger than the number of positions allowed in the line " +
                        str(lineNumber) + " (" + str(len(self.lines[lineNumber - 1].split())) + ")")
      # assign the new value
      splittedLine[position - 1] = str(value)
      # replace the line with the new value
      self.lines[lineNumber - 1] = " ".join(splittedLine)
      self.lines[lineNumber - 1] += '\n'
      lineNumPosition = str(lineNumber) + '|' + str(position)
      # ModiList <= {'LineNumber|Position': value}
      modifDictList = {lineNumPosition: []}
      modifDictList[lineNumPosition].append(value)
      # create the modifDictionaryList (self.modifiedDictionary)
      self.modifDictionaryList(modifDictList)
      
