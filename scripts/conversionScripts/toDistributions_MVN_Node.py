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
  for child in simulation:
    if child.tag == 'Distributions':
      MVNNode = child.find('MultivariateNormal')    
      if MVNNode is not None:
        dataFileNameNode = MVNNode.find('data_filename')
        covFileName = dataFileNameNode.text
        dataWorkingDirNode = MVNNode.find('working_dir')
        covFileDir = dataWorkingDirNode.text
        if '~' in covFileDir: covFileDir = os.path.expanduser(covFileDir)
        if os.path.isabs(covFileDir): covFileDir = covFileDir
        elif "runRelative" in dataWorkingDirNode.attrib:
          covFileDir = os.path.abspath(covFileName)
        else:
          if covFileDir == None: raise IOError('Relative working directory is requested but the given name is None' )
          covFileDir = os.path.join(os.getcwd(),covFileDir.strip())
        covFileName = os.path.join(covFileDir,covFileName.strip())
        MVNNode.remove(dataFileNameNode)
        MVNNode.remove(dataWorkingDirNode)
        covData = ''
        if os.path.isfile(covFileName):
          for line in file(covFileName,'r'):
            covData += line.rstrip() + ' '
        else:
          print 'Error! The following file is not exist: ', covFileName
        covNode = ET.Element('covariance')
        covNode.text = covData
        MVNNode.append(covNode)
      else:
        print 'No conversion needed'

  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
