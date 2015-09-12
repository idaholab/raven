import xml.etree.ElementTree as ET
import copy
import string


def checkInAttributes(element,convertDict):
  """
    Check if the there is something that needs to be converted in the attrib dictionary of xmlnode "element"
    @ In, element, the element to check in
    @ In, convertDict, the dictionary where the keywords to be converted are contained
  """
  for key in convertDict.keys():
    if key in element.attrib.keys():
      newValue = element.attrib.pop(key)
      element.set(convertDict[key],newValue)
  return element

def XMLpreprocess(xmlNode,xmlFileName=None):
  """
    Preprocess the xml input file, load external xml files into the main ET
    @ In, xmlNode, xml.etree.ElementTree object, root element of RAVEN input file
    @ In, xmlFileName, the raven input file name
  """
  convertDict = {'global_grid':'globalGrid',
                 'sampler_init':'samplerInit',
                 'initial_seed':'initialSeed',
                 'reseed_at_each_iteration':'reseedEachIteration',
                 'dist_init':'distInit',
                 'initial_grid_disc':'initialGridDisc',
                 'print_end_xml':'printEndXmlSummary',
                 'algorithm_type':'algorithmType'}

  for element in xmlNode.iter():
    if element.tag in convertDict.keys(): element.tag = convertDict[element.tag]
    element = checkInAttributes(element,convertDict)
    for subelement in element:
      if subelement.tag in convertDict.keys(): subelement.tag = convertDict[subelement.tag]
      subelement = checkInAttributes(subelement,convertDict)
      for subsubelement in subelement:
        if subsubelement.tag in convertDict.keys(): subsubelement.tag = convertDict[subsubelement.tag]
        subsubelement = checkInAttributes(subsubelement,convertDict)
        for subsubsubelement in subsubelement:
          if subsubsubelement.tag in convertDict.keys(): subsubsubelement.tag = convertDict[subsubsubelement.tag]
          subsubsubelement = checkInAttributes(subsubsubelement,convertDict)
          for subsubsubsubelement in subsubsubelement:
            if subsubsubsubelement.tag in convertDict.keys(): subsubsubsubelement.tag = convertDict[subsubsubsubelement.tag]
            subsubsubsubelement = checkInAttributes(subsubsubsubelement,convertDict)

def convert(tree,fileName=None):
    """
        Converts input files to be compatible with merge request  (alfoa/camelBackSyntaxSamplers).  Replace the nodes/attributes contained in this dictionary
        {stringToReplace:NewString}

        {'global_grid':'globalGrid',
        'sampler_init':'samplerInit',
        'initial_seed':'initialSeed',
        'reseed_at_each_iteration':'reseedEachIteration',
        'dist_init':'distInit',
        'initial_grid_disc':'initialGridDisc',
        'print_end_xml':'printEndXmlSummary',
        'algorithm_type':'algorithmType',
        }
        @ In, tree, xml.etree.ElementTree object, the contents of a RAVEN input file
        @ In, fileName, the name for the raven input file
        @Out, tree, xml.etree.ElementTree object, the modified RAVEN input file
    """
    simulation = tree.getroot()
    if simulation.tag!='Simulation': return tree #this isn't an input file
    XMLpreprocess(simulation,fileName)
    return tree

if __name__=='__main__':
    import convert_utils
    import sys
    convert_utils.standardMain(sys.argv,convert)
