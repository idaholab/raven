import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os

def convert(tree,fileName=None):
  """
    Converts variables and aliases in MAMMOTH interface input files to syntax
    established in merge request 643.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @ Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  # Check if this is RAVEN input
  if simulation.tag!='Simulation' and simulation.tag!='ExternalModel': return tree
  # Check if this is MAMMOTH interface input
  mammothInput = False
  for code in simulation.iter('Code'):
    if code.attrib['subType'].lower == 'mammoth':
      mammothInput = True
    for codeChild in code:
      if codeChild.tag == 'alias'
      checkAliases = True
      break
  if not mammothInput:
    return tree
  # Collect alias variable names and update the full variable syntax
  aliasStorage = []
  if checkAliases:
    for alias in simulation.iter('alias'):
      aliasStorage.append(alias.attrib['variable'])
      splitVarName = alias.text.split(':')
      # Rattlesnake variable case
      if len(splitVarName) == 1:
        alias.text = 'Rattlesnake@' + alias.text
      # Named Application Case
      else:
        alias.text = splitVarName[1] + '@' + splitVarName[0]
  # Check if variables are listed as aliases, if not, update variable syntax
  for variable in simulation.iter('variable'):
    if variable.attrib['name'] not in aliasStorage:
      splitVarName = variable.attrib['name'].split(':')
      # Rattlesnake variable case
      if len(splitVarName) == 1:
        variable.attrib['name'] = 'Rattlesnake@' + variable.attrib['name']
      # Named Application case
      else:
        variable.attrib['name'] = splitVarName[1] + '@' + splitVarName[0]
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
