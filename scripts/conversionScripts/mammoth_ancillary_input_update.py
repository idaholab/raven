import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os

def convert(tree,fileName=None):
  """
    Converts the file type of miscellaneous inputs in MAMMOTH interface MultiRun to 
    ancillary input as of merge request 728.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @ Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  # Check if this is RAVEN input
  if simulation.tag!='Simulation' and simulation.tag!='ExternalModel': return tree
  # Check if this is MAMMOTH interface input
  mammothInput = False
  mammothMultiRunNames = []
  for code in simulation.iter('Code'):
    if code.attrib['subType'].lower == 'mammoth':
      mammothInput = True
      mammothMultiRunNames.append(code.attrib['name'])
    for codeChild in code:
      if codeChild.tag == 'alias'
      checkAliases = True
      break
  if not mammothInput:
    return tree
  # Find all Inputs with class=Files from Mammoth MultiRun
  inputFileNames = []
  for mammothMultiRun in mammothMultiRunNames:
    for iput in simulation.findall("./Steps/MultiRun/[Model='"+mammothMultiRun+"']/Input"):
      if iput.attrib[class] == 'Files':
        inputFileNames.append(iput.text)
  # Change any found Mammoth MultiRun Input Files' with no type to ancillaryInput
  for inputFileName in inputFileNames:
    for blankInputFileType in simulation.findall("./Files/Input/[@name='"+inputFileName+"']/[@type='']"):
      blankInputFileType.set('type', 'ancillaryInput')

  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
