import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os

def convert(tree,fileName=None):
  """
    Converts input files to be compatible with merge request ???.
    Changes list of <variable> nodes to <variables> nodes.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation': return tree #this isn't an input file
  models = simulation.find('Models')
  if models is not None:
    extmod = models.find('ExternalModel')
    if extmod is not None:
      vars = []
      toRemove = []
      for child in extmod:
        if child.tag=='variable':
          vars.append(child.text)
          toRemove.append(child)
      for child in toRemove:
        extmod.remove(child)
      if len(vars)>0:
        variables = ET.Element('variables')
        extmod.append(variables)
        variables.text = ','.join(vars)
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
