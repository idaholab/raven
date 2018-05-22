#!/usr/bin/env python
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
  Created on May 22, 2018
  @author: alfoa
  This a utility tool to fix the docstrings in RAVEN
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3

#External Modules--------------------
import sys
import os
#External Modules--------------------

def trimQuoteDocstring(docstring, quoteIndent, maxColumns=120):
  """
    This method is aimed to trim to the quote docstrings to the
    be limited in the maxColumns number
    @ In, docstring, str, the docstring that need to be modified
    @ In, quoteIndent, int, the indentation number of columns of the docstring
    @ In, maxColumns, int, optional, the maximum number of columns allowed (default 120)
    @ Out, outputLines, list, the list of new lines that need to be inserted
  """
  lines = [elm.expandtabs() for elm in docstring]
  outputLines = [" "*quoteIndent+'"""'+"\n"]
  # find if "@" are present
  inIndex = []
  outIndex = []
  for cnt, line in enumerate(lines):
    if line.strip().startswith("@"):
      if "@ in" in line.lower() or '@in' in line.lower():
        inIndex.append(cnt)
      elif "@ out" in line.lower() or '@out' in line.lower():
        outIndex.append(cnt)
  for cnt, line in enumerate(lines):
    indentedLine = " "*(quoteIndent+2) + line.lstrip()
    if cnt in inIndex+outIndex:
      indentedLine = indentedLine.replace("@Out","@ Out").replace("@In","@ In")
      breakIndex = [i for i, ltr in enumerate(indentedLine) if ltr == ',']
      breakIndex = quoteIndent+8
    else:
      breakIndex = quoteIndent+1

    if len(indentedLine) > maxColumns:
      # we need to split it
      newLines = []
      words = indentedLine.strip().split()
      words.insert(0," "*(quoteIndent))
      lineToAdd = ""
      for i, word in enumerate(words):
        if len(lineToAdd+word)+3  < maxColumns:
          lineToAdd = lineToAdd +" "+word
        else:
          newLines.append(lineToAdd+"\n")
          lineToAdd = " "*breakIndex+" "+word
        if i == len(words)-1:
          newLines.append(lineToAdd+"\n")
      outputLines+=newLines
    else:
      outputLines.append(line)
  outputLines.append(" "*quoteIndent+'"""'+"\n")
  # Return a single string:
  return outputLines

def trimPoundDocstring(docstring, quoteIndent, maxColumns=120):
  """
    This method is aimed to trim to the pound (#) docstrings to the
    be limited in the maxColumns number
    @ In, docstring, str, the docstring that need to be modified
    @ In, quoteIndent, int, the indentation number of columns of the docstring
    @ In, maxColumns, int, optional, the maximum number of columns allowed (default 120)
    @ Out, outputLines, list, the list of new lines that need to be inserted
  """
  # Convert tabs to spaces (following the normal Python rules)
  # and split into a list of lines:
  if docstring.count("#")>1:
    lines = [elm.expandtabs() for elm in docstring.split("#")]
    if len(lines[0].strip()) == 0:
      lines.pop(0)
  else:
    lines = [docstring]
  outputLines = []
  # find if "@" are present
  inIndex = []
  outIndex = []

  for cnt, line in enumerate(lines):
    indentedLine = " "*(quoteIndent) + line.lstrip()
    if indentedLine.strip().endswith("---------"):
      while len(indentedLine.strip()) > maxColumns-3:
        if indentedLine.strip()[-1] == '-':
          indentedLine = indentedLine.strip()
          indentedLine = indentedLine[:-1]
        else:
          break

    breakIndex = quoteIndent

    if len(indentedLine) > maxColumns:
      # we need to split it
      newLines = []
      words = indentedLine.strip().replace("#","").split()
      words.insert(0," "*(breakIndex-1) + "#")
      lineToAdd = ""
      for i, word in enumerate(words):
        if len(lineToAdd+word)+3  < maxColumns:
          lineToAdd = lineToAdd +" "+word
        else:
          newLines.append(lineToAdd+"\n")
          lineToAdd = " "*breakIndex + "# "+ word
        if i == len(words)-1:
          newLines.append(lineToAdd+"\n")
      outputLines+=newLines
    else:
      if len(line.strip()) > 0:
        add = '\n' if '\n' not in indentedLine.lstrip() else ''
        if '#' not in indentedLine.strip():
          outputLines.append(" "*(breakIndex) + "# "+indentedLine.lstrip()+add)
        else:
          outputLines.append(indentedLine+add)

  # Return a single string:
  return outputLines


if __name__ == '__main__':
  # main portion of this tool
  # the script must be run as follows
  # python fix_raven_docstrings.py fileNameToBeConverted.py
  # In argouments:
  # fileName, required, the file name to be converted
  # -i, optional, if "-i" is passed, the conversion is going to be done in place
  #               otherwise a new file with the suffix _converted will be generated
  # -c_max $int$, optional, the maximum number of columns allowed. Default is 119 (120 considering \n)
  fileName = os.path.abspath(sys.argv[1])
  outFileName = None
  maxColumns = 119
  if '-i' in sys.argv:
    outFileName = fileName
  else:
    outFileName = fileName.split(".")[0]+"_converted."+fileName.split(".")[1]

  if '-c_max' in sys.argv:
    maxColumns = int(sys.argv[sys.argv.index("-c_max")+1])
  with open(fileName, mode='r') as fobj:
    lines = fobj.readlines()
  lineNumber = 0
  endLineNumber = 0
  outLines = []
  while lineNumber != len(lines):
    if lines[lineNumber].strip().startswith('"""') and lines[lineNumber].strip().count('"') == 3:
      endLineNumber = lineNumber+1
      while not lines[endLineNumber].strip().startswith('"""'):
        endLineNumber+=1
      docstringList =  lines[lineNumber+1:endLineNumber]
      trailingSpaces = len( lines[lineNumber]) - len( lines[lineNumber].lstrip())
      newDocString = trimQuoteDocstring(docstringList,trailingSpaces,maxColumns)
      outLines += newDocString
      lineNumber = endLineNumber
    else:
      outLines.append(lines[lineNumber])
    lineNumber+=1

  lineNumber = 0
  endLineNumber = 0
  newOutLines = []
  while lineNumber != len(outLines):
    skip = '("#1\n")' not in outLines[lineNumber]
    maxTrue = "#" in outLines[lineNumber].strip() and len(outLines[lineNumber]) > maxColumns
    innerPound = outLines[lineNumber].strip().startswith("#") and  "#" in outLines[lineNumber].strip()[1:]
    begin = " "
    end = " "
    if len(outLines[lineNumber].strip()) > 10:
      begin = outLines[lineNumber].strip()[0:10]
    if len(outLines[lineNumber].strip()) > 0:
      end =  outLines[lineNumber].strip()[-1]
    poundInPrinting = not outLines[lineNumber].strip().startswith("#") and outLines[lineNumber].strip().count("#") > 0 \
                      and (begin in ["self.raise","Fvect.writ"] or end in [")","]",",",":"] or outLines[lineNumber].strip().endswith("comment"))
    if (maxTrue or not innerPound or not skip) and not poundInPrinting:
      endLineNumber = lineNumber
      trailingSpaces = len( outLines[lineNumber]) - len( outLines[lineNumber].lstrip())
      if outLines[lineNumber].strip().startswith("#"):
        newDocString = trimPoundDocstring(outLines[lineNumber],trailingSpaces,maxColumns)
        newOutLines += newDocString
      elif not outLines[lineNumber].strip().startswith("#") and "#" in outLines[lineNumber].strip()[1:]:
        splitted = outLines[lineNumber].split("#")
        newDocString = trimPoundDocstring(splitted[-1],trailingSpaces,maxColumns)
        newOutLines += newDocString
        add = '\n' if '\n' not in splitted[0].lstrip() else ''
        newOutLines.append(splitted[0]+add)
      else:
        newOutLines.append(outLines[lineNumber])
      lineNumber = endLineNumber
    else:
      newOutLines.append(outLines[lineNumber])
    lineNumber+=1
  outLines = newOutLines

  with open(outFileName,"w") as outObj:
    outObj.writelines(outLines)







