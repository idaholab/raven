import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os

def convert(tree,fileName=None):
  """
    Converts input files to be compatible with merge request !462
    change the attribute of node <variablesTransformation> from 'model' to 'distribution'
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation': return tree #this isn't an input file
  for distNode in simulation.iter('MultivariateNormal'):
    distName = distNode.get('name')
    break
  for vtNode in simulation.iter('variablesTransformation'):
    vtNode.set('distribution', distName)
    modelName = vtNode.get('model')
    if modelName != None:
      del vtNode.attrib['model']
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
