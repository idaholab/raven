"""
Created on March 28, 2018
@author: JYoo
"""
import os
import fileinput
import re

class CTFparser():
   """
    Import the CobraTF input as list of lines, provide methods to add/change entries and print it back
   """
   def __init__(self,inputFile):
	"""
	# Constructor
        # @ In, inputFile, string, input file name
        # @ Out, None
	"""
        self.printTag = 'CTF PARSER'
        if not os.path.exists(inputFile):
            raise IOError(self.printTag + ' ERROR: not found CTF input file')

        IOfile = open(inputFile, 'r')
        self.inputfile = inputFile
        """# modified list in a form of Dictionary"""
        self.Modif_Dict = {}
        """# read original input file"""
        self.lines = IOfile.readlines()

   def printInput(self, outfile=None):
	"""
        # Method to print out the new input
        # @ In, outfile, string, optional, output file root
        # @ Out, None
	"""
        if outfile == None:
            outfile = self.inputfile
        newInput = open(outfile, 'w')

        for i in self.lines:
            newInput.write('%s' % (i))
        newInput.close()

   def modifDictionaryList(self, ModifDictList):
        """# update self.Modif_Dict dictionary"""
        self.Modif_Dict.update(ModifDictList)
        return
   def changeVariable(self, modifDict):
        """
	# Method to modify the CTF input variables
	# CTF input is modified based on the request from RAVEN with a form (lineNumber|position)
        """
        i = 0
        for lineNumberPosition, value in modifDict.items():
            """#print(lineNumberPosition, value)"""
            lineNumber, position = list(modifDict.keys())[i].split("|")
            value = list(modifDict.values())[i]

            """# type conversion from str to int"""
            lineNumber = int(lineNumber)
            position = int(position)

            """# error for the commented line"""
            if self.lines[lineNumber - 1].startswith("*"):
                raise IOError("Error: the line number requested by the user (line " +
                              str(lineNumber) + ") is the commented line")
            """# error for the blank line"""
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

            """# assign the new value"""
            splittedLine[position - 1] = str(value)
            """# replace the line with the new value"""
            self.lines[lineNumber - 1] = " ".join(splittedLine)
            self.lines[lineNumber - 1] += '\n'

            LineNumPosition = str(lineNumber) + '|' + str(position)
            """# ModiList <= {'LineNumber|Position': value}"""
            ModifDictList = {LineNumPosition: []}
            ModifDictList[LineNumPosition].append(value)

            """# create the modifDictionaryList (self.Modif_Dict)"""
            self.modifDictionaryList(ModifDictList)

            i = i + 1
        return
