import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os

def convert(tree):
  """
    Converts input files to be compatible with merge request 255 (Talbpaul/redundant input).  Removes the <Files> node
    from the RunInfo block and makes it into its own node.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation': return tree #this isn't an input file
  runinfo = simulation.find('RunInfo')
  oldFilesNode = runinfo.find('Files')
  if oldFilesNode is not None:
    fileNameList = oldFilesNode.text.split(',')
    runinfo.remove(oldFilesNode)
    newFiles = ET.Element('Files')
    for name in fileNameList:
      newfile = ET.Element('Input')
      newfile.set('name',name)
      newfile.set('type','')
      newfile.text = name
      newFiles.append(newfile)
    simulation.insert(1,newFiles)
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
