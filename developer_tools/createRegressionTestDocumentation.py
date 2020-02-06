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
 Created on Jan 20, 2017

 @author: alfoa
"""
import os
from glob import glob
import inspect
import xml.etree.ElementTree as ET
import copy
import os
from collections import OrderedDict


class testDescription(object):
  """
    Class that handles the checks on the description of the tests
  """
  def __init__(self):
    """
      Constructor
    """
    self.__undescribedFiles ,self.__describedFiles = self.noDescriptionTestsAndInformationOnTheOther()
    self.__totTestFiles = len(self.__undescribedFiles) + len(self.__describedFiles.keys())
    self.__allDescribed = len(self.__undescribedFiles) == 0
    self.__ravenDir     = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    self.__userPath     = os.path.abspath(os.path.join(self.__ravenDir,".."))

  def areAllTestDescribed(self):
    """
      Method to check if all the tests are described
      @ In, None
      @ Out, __allDescribed, bool, all described?
    """
    return self.__allDescribed

  def getFoldersOfUndocumentedTests(self):
    """
      Method to get all the folders of tests that contain
      undocumented tests
      @ In, None
      @ Out, undocumentedFolders, list, list containing folders with undocumented tests
    """
    undocumentedFolders = []
    for testName in self.__undescribedFiles:
      dirName = os.path.dirname(testName)
      if dirName not in undocumentedFolders: undocumentedFolders.append(dirName)
    return undocumentedFolders

  def getTotalNumberOfTests(self):
    """
      Method to get the number of tests
      @ In, None
      @ Out, __totTestFiles, int, number of tests
    """
    return self.__totTestFiles

  def getDescriptionCoverage(self):
    """
      Method to get the description coverage in %
      @ In, None
      @ Out, getDescriptionCoverage, float, percent of description coverage
    """
    if self.areAllTestDescribed(): return 100.
    else                         : return (float(len(self.__describedFiles.keys()))/float(self.__totTestFiles))*100

  def getUndescribedFileNames(self):
    """
      Method to get the list of un-described files
      @ In, None
      @ Out, __undescribedFiles, list, list of un-described files
    """
    return self.__undescribedFiles

  def noDescriptionTestsAndInformationOnTheOther(self):
    """
      This method returns a dictionary of framework tests (i.e. the ones with an XML
      extension and listed in the "tests" files) that have a TestDescription (with
      the info contained) and a list of test file names that do not report any
      description
      @ In, None
      @ Out, outputTuple, tuple, tuple (list(file names without a description),
                                        dictionary({'fileName':'xmlNode with the description'}))
    """
    __testInfoList = []
    __testList = []
    filesWithDescription = OrderedDict()
    noDescriptionFiles = []
    startDir = os.path.join(os.path.dirname(__file__),'../')
    for dirr,_,_ in os.walk(startDir):
      __testInfoList.extend(glob(os.path.join(dirr,"tests")))
    for testInfoFile in __testInfoList:
      if 'moose' in testInfoFile.split(os.sep) or not os.path.isfile(testInfoFile):
        continue
      fileObject = open(testInfoFile,"r+")
      fileLines = fileObject.readlines()
      dirName = os.path.dirname(testInfoFile)
      # I do not want to use getpot!
      for line in fileLines:
        if line.strip().startswith("input"):
          fileName = line.split("=")[-1].replace("'", "").replace('"', '').rstrip().strip()
          fileName = os.path.join(dirName,fileName)
          if os.path.split(fileName)[-1].lower().endswith('xml'):
            __testList.append(os.path.abspath(fileName))
          if os.path.split(fileName)[-1].lower().endswith('py'):
            __testList.append(os.path.abspath(fileName))
      fileObject.close()
    for testFile in __testList:
      if testFile.endswith('xml'):
        try: root = ET.parse(testFile).getroot()
        except Exception as e: print('file :'+testFile+'\nXML Parsing error!',e,'\n')
        if root.tag != 'Simulation': print('\nThe root node is not Simulation for file '+testFile+'\n')
        testInfoNode = root.find("TestInfo")
        if testInfoNode is None and root.tag == 'Simulation': noDescriptionFiles.append(testFile)
        else: filesWithDescription[testFile] = copy.deepcopy(testInfoNode)
      else:
        fileLines = open(testFile,"r+").readlines()
        xmlPortion = []
        startReading = False
        for line in fileLines:
          if startReading:
            xmlPortion.append(line)
          if '<TestInfo' in line:
            startReading = True
            xmlPortion.append("<TestInfo>")
          if '</TestInfo' in line:
            startReading = False
        if len(xmlPortion) >0:
          try: testInfoNode = ET.fromstringlist(xmlPortion)
          except ET.ParseError as e: print('file :'+testFile+'\nXML Parsing error!',e,'\n')
        else                 : testInfoNode = None
        if testInfoNode is None: noDescriptionFiles.append(testFile)
        else: filesWithDescription[testFile] = copy.deepcopy(testInfoNode)
    outputTuple = noDescriptionFiles, filesWithDescription
    return outputTuple

  def _fromXmlToLatexDocument(self,xmlNode, fileName):
    """
      Template method to construct a latex documentation from a <TestInfo> xml block
      @ In, xmlNode, xml.etree.ElementTree, xml node containing the information
      @ In, fileName, string, file name of the test
      @ Out, output, tuple, tuple(latexString = string representing the latex documentation for this test,
                                  chapterName = the name should be given to the chapter)
    """

    descriptionNode  = xmlNode.find("description")
    authorNode       = xmlNode.find("author")
    nameNode         = xmlNode.find("name")
    createdDateNode  = xmlNode.find("created")
    classTestedNode  = xmlNode.find("classesTested")
    requirementsNode = xmlNode.find("requirements")
    analyticNode     = xmlNode.find("analytic")
    revisionsNode    = xmlNode.find("revisions")
    # check
    if descriptionNode is not None: description = descriptionNode.text
    else                          : raise IOError("XML node <description> not found for test "+ fileName)
    if authorNode is not None     : author = authorNode.text
    else                          : raise IOError("XML node <author> not found for test "+ fileName)
    if nameNode is not None       : name = nameNode.text
    else                          : raise IOError("XML node <name> not found for test "+ fileName)
    if createdDateNode is not None: createdDate = createdDateNode.text
    else                          : raise IOError("XML node <created> not found for test "+ fileName)
    if classTestedNode is not None: classTested = classTestedNode.text
    else                          : raise IOError("XML node <classesTested> not found for test "+ fileName)

    nameChapter = name.replace("/", " ").replace("_", " ").upper()
    fileLocation = '.'+fileName.replace(self.__userPath,"")
    latexString =  "This test can be found at ``\path{"+fileLocation+"}''.\n"
    latexString += " This test can be called executing the following command:"
    latexString += " \\begin{lstlisting}[language=bash]\n"
    latexString += " ./run_tests --re="+name+"\n"
    latexString += " \\end{lstlisting}"
    latexString += " or \n"
    latexString += " \\begin{lstlisting}[language=bash]\n"
    latexString += " ./run_framework_tests --re="+name+"\n"
    latexString += " \\end{lstlisting}"
    latexString += ' \\begin{itemize} \n'
    # Test description
    latexString += '   \\item Test Description:\n'
    latexString += '   \\begin{itemize} \n'
    latexString += '     \\item ' +description.strip().replace("_", "\_").replace("#","\#")+'\n'
    latexString += '   \\end{itemize} \n'
    # is analytical?
    if analyticNode is not None:
      analyticalDescription = analyticNode.text.replace("_", "\_")
      latexString += '   \\item This test is analytic:\n'
      latexString += '   \\begin{itemize} \n'
      latexString += '     \\item ' +str(analyticalDescription).strip().replace("#","\#")+'\n'
      latexString += '   \\end{itemize} \n'
    # author
    latexString += '   \\item Original Author:\n'
    latexString += '   \\begin{itemize} \n'
    latexString += '     \\item ' +str(author).strip()+'\n'
    latexString += '   \\end{itemize} \n'
    # createdDate
    latexString += '   \\item Creation date:\n'
    latexString += '   \\begin{itemize} \n'
    latexString += '     \\item ' +str(createdDate).strip()+'\n'
    latexString += '   \\end{itemize} \n'
    # classTested
    latexString += '   \\item The classes tested in this test are:\n'
    latexString += '   \\begin{itemize} \n'
    latexString += '     \\item ' +str(classTested).strip()+'\n'
    latexString += '   \\end{itemize} \n'
    # is requirement?
    if requirementsNode is not None:
      requirementDescription = requirementsNode.text.split() if "," not in requirementsNode.text else requirementsNode.text.split(",")
      latexString += '   \\item This test fulfills the following requirement:\n'
      latexString += '   \\begin{itemize} \n'
      for req in requirementDescription:
        latexString += '     \\item ' +req.strip().replace("#","\#")+'\n'
      latexString += '   \\end{itemize} \n'
    if revisionsNode is not None and len(revisionsNode) > 0:
      latexString += '   \\item Since the creation of this test, the following main revisions have been performed:\n'
      latexString += '   \\begin{enumerate} \n'
      for child in revisionsNode:
        revisionText   = str(child.text).strip().replace("_", "\_").replace("#","\#")
        revisionAuthor = child.attrib.get('author',"None").strip()
        revisionDate   = child.attrib.get('date',"None").strip()
        latexString += '     \\item revision info:\n'
        latexString += '       \\begin{itemize} \n'
        latexString += '         \\item author     : ' +revisionAuthor+'\n'
        latexString += '         \\item date       : ' +revisionDate+'\n'
        latexString += '         \\item description: ' +revisionText+'\n'
        latexString += '       \\end{itemize} \n'
      latexString += '   \\end{enumerate} \n'
    latexString += ' \\end{itemize} \n'
    output = latexString, nameChapter
    return output

  def splitTestDescription(self):
    """
      This method is aimed to create 3 dictionaries of test information:
      1) verification tests
      2) analytical tests
      3) requirement tests
      @ In, None
      @ Out, tupleOut, tuple, tuple of the 3 dictionaries ( tuple(verificationDict,analyticalDict,requirementDict) )
    """
    verificationDict = OrderedDict()
    requirementDict  = OrderedDict()
    analyticalDict   = OrderedDict()
    for testFileName, xmlNode in self.__describedFiles.items():
      if xmlNode is not None:
        if xmlNode.find("requirements") is not None:
          # requirement
          requirementDict[testFileName] = xmlNode
        if xmlNode.find("analytic") is not None:
          # analytic
          analyticalDict[testFileName] = xmlNode
        if xmlNode.find("analytic") is None and xmlNode.find("requirements") is None:
          # verification
          verificationDict[testFileName] = xmlNode
    tupleOut = verificationDict, analyticalDict, requirementDict
    return tupleOut

  def createLatexFile(self, fileName, documentClass = "article", latexPackages=[''], bodyOnly=False):
    """
      This method is aimed to create a latex file containing all the information
      found in the described tests
      @ In, fileName, string, filename (absolute path)
      @ In, documentClass, string, latex class document
      @ In, latexPackages, list, list of latex packages
      @ In, bodyOnly, bool, create a full document or just the document body (\begin{document} to \end{document})
      @ Out, None
    """
    fileObject = open(fileName,"w+")
    if not bodyOnly:
      fileObject.write(" \\documentclass{"+documentClass+"}\n")
      for packageLatex in latexPackages: fileObject.write(" \\usepackage{"+packageLatex.strip()+"} \n")
      fileObject.write(" \\usepackage{hyperref} \n \\usepackage[automark,nouppercase]{scrpage2} \n")
      fileObject.write(" \\usepackage[obeyspaces,dvipsnames,svgnames,x11names,table,hyperref]{xcolor} \n")
      fileObject.write(" \\usepackage{times} \n \\usepackage[FIGBOTCAP,normal,bf,tight]{subfigure} \n")
      fileObject.write(" \\usepackage{amsmath} \n \\usepackage{amssymb} \n")
      fileObject.write(" \\usepackage{soul} \n \\usepackage{pifont} \n \\usepackage{enumerate} \n")
      fileObject.write(" \\usepackage{listings}  \n \\usepackage{fullpage} \n \\usepackage{xcolor} \n")
      fileObject.write(" \\usepackage{ifthen}  \n \\usepackage{textcomp}  \n  \\usepackage{mathtools} \n")
      fileObject.write(" \\usepackage{relsize}  \n \\usepackage{lscape}  \n \\usepackage[toc,page]{appendix} \n")
      fileObject.write("\n")
      fileObject.write(' \\lstdefinestyle{XML} {\n language=XML, \n extendedchars=true, \n breaklines=true, \n breakatwhitespace=true, \n')
      fileObject.write(' emphstyle=\color{red}, \n basicstyle=\\ttfamily, \n commentstyle=\\color{gray}\\upshape, \n ')
      fileObject.write(' morestring=[b]", \n morecomment=[s]{<?}{?>}, \n morecomment=[s][\color{forestgreen}]{<!--}{-->},')
      fileObject.write(' keywordstyle=\\color{cyan}, \n stringstyle=\\ttfamily\color{black}, tagstyle=\color{blue}\\bf \\ttfamily \n }')
      fileObject.write(" \\title{RAVEN regression tests' description}\n")
    fileObject.write(" \\begin{document} \n \\maketitle \n")
    # Introduction
    fileObject.write(" \\section{Introduction} \n")
    fileObject.write(" This document has been automatically \n")
    fileObject.write(" generated by the script ``\\path{raven\developer_tools\createRegressionTestDocumentation.py}''\n")
    fileObject.write("Currently there are " + str(descriptionClass.getTotalNumberOfTests()) + "\n")
    fileObject.write(" regression tests in the RAVEN framework. The \%  of tests that are commented is currently equal to \n"+ str(descriptionClass.getDescriptionCoverage())+" \%.\n")
    # Documented tests
    fileObject.write("\section{Documented Tests}\n")
    fileObject.write("Regression tests for the $Python$ RAVEN framework are found in \path{raven/tests/framework}.\n")
    fileObject.write("There is a hierarchy of folders with tests collected by similar testing.\n")
    fileObject.write("Every test is described in a special XML node ($<TestInfo>$) within the $<Simulation>$ block.\n")
    fileObject.write("An example is reported below:\n")
    fileObject.write("\\begin{lstlisting}[style=XML]\n")
    fileObject.write("<Simulation>\n")
    fileObject.write("  ...\n")
    fileObject.write("  <TestInfo>\n")
    fileObject.write("    <name>framework/path/to/test/label</name>\n")
    fileObject.write("    <author>AuthorGitLabTag</author>\n")
    fileObject.write("    <created>YYYY-MM-DD</created>\n")
    fileObject.write("    <classesTested>Module.Class, Module.Class</classesTested>\n")
    fileObject.write("    <description>\n")
    fileObject.write("        Paragraph describing work-flows, modules, classes, entities, etc.,\n")
    fileObject.write("        how they are tested, and any other notes\n")
    fileObject.write("    </description>\n")
    fileObject.write("    <requirements>RequirementsLabel</requirements>\n")
    fileObject.write("    <analytic>paragraph description of analytic test</analytic>\n")
    fileObject.write("    ...\n")
    fileObject.write("  </TestInfo>\n")
    fileObject.write("  ...\n")
    fileObject.write("</Simulation>\n")
    fileObject.write("\\end{lstlisting}\n")
    fileObject.write("The $<requirements>$ and $<analytic>$ nodes are optional, for those tests who satisfy an NQA design requirement \n")
    fileObject.write("and or have an analytic solution documented in the analytic tests document. Other notes on block contents:\n")
    fileObject.write("\\begin{itemize} \n")
    fileObject.write("  \\item \\textbf{$<name>$}: this is the test framework path, as well as the name (label) assigned in the tests file block.")
    fileObject.write(        "This is the path and name that show up when running the tests using the testing harness (\\path{run_tests})\n")
    fileObject.write("  \\item \\textbf{$<author>$}: this is the GitLab tag of the author who constructed this test originally, i.e. \\textit{alfoa for @alfoa} \n")
    fileObject.write("  \\item \\textbf{$<created>$}: this is the date on which the test was originally created, in year-month-day \\textit{YYYY-MM-DD} XSD date format \n")
    fileObject.write("  \\item \\textbf{$<classesTested>$}: a list of the classes tested in the python framework, listed as Entity.Class, i.e. \\textit{Samplers.MonteCarlo} \n")
    fileObject.write("  \\item \\textbf{$<description>$}: general notes about what work-flows or other methods are tested \n")
    fileObject.write("  \\item \\textbf{$<requirements>$} (optional): lists the NQA requirement that this test satisfies \n")
    fileObject.write("  \\item \\textbf{$<analytic>$} (optional): describes the analytic nature of this test and how it is documented in the analytic tests documentation \n")
    fileObject.write("\\end{itemize} \n")
    fileObject.write("An additional node is optionally available to demonstrate significant revisions to a test: \n")
    fileObject.write("\\begin{lstlisting}[style=XML,morekeywords={author,date}]\n")
    fileObject.write("<Simulation>\n")
    fileObject.write("  ...\n")
    fileObject.write("  <TestInfo>\n")
    fileObject.write("    ...\n")
    fileObject.write("    <revisions>\n")
    fileObject.write("      <revision author='AuthorGitLabTag' date='YYYY-MM-DD'>paragraph description of revision</revision>\n")
    fileObject.write("      <revision author='AuthorGitLabTag' date='YYYY-MM-DD'>paragraph description of revision</revision>\n")
    fileObject.write("    <revisions>\n")
    fileObject.write("    ...\n")
    fileObject.write("  </TestInfo>\n")
    fileObject.write("  ...\n")
    fileObject.write("</Simulation>\n")
    fileObject.write("\\end{lstlisting}\n")
    fileObject.write("The following sub-sections collect all the documented tests. \n")

    verificationDict, analyticalDict, requirementDict = self.splitTestDescription()
    # list of tests documented

    if len(requirementDict.keys()) > 0:
      # list the requirement tests
      fileObject.write("\subsection{Requirement tests' description}\n")
      fileObject.write("\n This section contains the description of all the requirement tests. \n")
      for testFileName, xmlNode in requirementDict.items():
        latexString, chapterName = self._fromXmlToLatexDocument(xmlNode,testFileName)
        fileObject.write("\subsubsection{"+chapterName.strip()+"}\n")
        fileObject.write("\n")
        fileObject.write(latexString)
    if len(analyticalDict.keys()) > 0:
      # list the analytical tests
      fileObject.write("\subsection{Analytical tests' description}\n")
      fileObject.write("\n This section contains the description of all the analytical tests. \n")
      for testFileName, xmlNode in analyticalDict.items():
        latexString, chapterName = self._fromXmlToLatexDocument(xmlNode,testFileName)
        fileObject.write("\subsubsection{"+chapterName.strip()+"}\n")
        fileObject.write("\n")
        fileObject.write(latexString)
    if len(verificationDict.keys()) > 0:
      # list the analytical tests
      fileObject.write("\subsection{Verification tests' description}\n")
      fileObject.write("\n This section contains the description of all the verification tests. \n")
      for testFileName, xmlNode in verificationDict.items():
        latexString, chapterName = self._fromXmlToLatexDocument(xmlNode,testFileName)
        fileObject.write("\subsubsection{"+chapterName.strip()+"}\n")
        fileObject.write("\n")
        fileObject.write(latexString)
    # section regarding undocumented tests
    if not self.areAllTestDescribed():
      undocumentedFolders = self.getFoldersOfUndocumentedTests()
      fileObject.write("\section{Undocumented tests}\n")
      fileObject.write("Currently, There are "+str(len(self.__undescribedFiles))+" undocumented tests:\n")
      fileObject.write("\\begin{enumerate}\n")
      for folderName in undocumentedFolders:
        fileObject.write("  \\item Folder: \\path{"+folderName+"}. Tests: \n")
        fileObject.write("  \\begin{itemize}\n")
        fileNameWithFolderRoot = [fileName for fileName in self.__undescribedFiles if folderName.strip() == os.path.dirname(fileName)]
        for fileName in fileNameWithFolderRoot:
          fileLocation = '.'+fileName.replace(self.__userPath,"")
          fileObject.write("  \\item \\path{"+fileLocation+"} \n")
        fileObject.write("   \\end{itemize}\n")
      fileObject.write("\\end{enumerate}\n")
    fileObject.write("\end{document}")
    fileObject.close()


if __name__ == '__main__':
  descriptionClass = testDescription()
  noDescriptionFiles, filesWithDescription = descriptionClass.noDescriptionTestsAndInformationOnTheOther()
  if not descriptionClass.areAllTestDescribed():
    print("There are "+str(len(noDescriptionFiles))+" test files without a test description.")
    print("Files without test description are:")
    for fileName in noDescriptionFiles: print(fileName)
  totFile                = descriptionClass.getTotalNumberOfTests()
  totFileWithDescription = len(filesWithDescription.keys())
  print("\nTotal framework test files are   : "+str(totFile))
  print("\n% of tests that got commented is : "+str(descriptionClass.getDescriptionCoverage())+" %")
  print("\nFolders that contain undocumented tests are:\n")
  for folderName in descriptionClass.getFoldersOfUndocumentedTests(): print(folderName)
  descriptionClass.createLatexFile("regression_tests_documentation_body.tex",bodyOnly=True)










