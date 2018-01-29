from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#Internal Modules---------------------------------------------------------------
import MessageHandler
from utils import utils
from utils import xmlUtils as xmlU
#Internal Modules End-----------------------------------------------------------

#External Modules---------------------------------------------------------------
import numpy as np
import xml.etree.ElementTree as ET
import copy
import itertools
from collections import OrderedDict
#External Modules End-----------------------------------------------------------

#class FTgate(MessageHandler.MessageUser):
class FTgate():

    #def __init__(self, xmlNode, messageHandler):
    def __init__(self, xmlNode):
        """
          Method that initializes the gate
          @ In, xmlNode, xml.etree.Element, Xml element node
          @ Out, None
        """
        #self.messageHandler = messageHandler
        self.name         = None
        self.gate         = None
        self.arguments    = []
        self.negations    = []
        self.params       = {}
        self.allowedGates = {'not':1,'and':'inf','or':'inf','xor':'inf','iff':2,'nand':'inf','nor':'inf','atleast':'inf','cardinality':'inf','imply':2}

        self.name = xmlNode.get('name')

        for child in xmlNode:
            if child.attrib:
                self.params = child.attrib
            if child.tag in self.allowedGates.keys():
                self.gate = child.tag
            else:
                raise IOError('FTImporterPostProcessor Post-Processor; gate ' + str(child.tag) + ' : is not recognized. Allowed gates are: '+ str(self.allowedGates.keys()))

        for node in findAllRecursive(xmlNode, 'gate'):
            self.arguments.append(node.get('name'))

        for node in findAllRecursive(xmlNode, 'basic-event'):
            self.arguments.append(node.get('name'))

        for node in findAllRecursive(xmlNode, 'house-event'):
            self.arguments.append(node.get('name'))

        for child in xmlNode:
            for childChild in child:
                if childChild.tag == 'not':
                    event = list(childChild.iter())
                    if len(event)>2:
                        raise IOError('FTImporterPostProcessor Post-Processor; gate ' + str(self.name) + ' contains a negations of multiple basic events')
                    elif event[1].tag in ['gate','basic-event','house-event']:
                        self.negations.append(event[1].get('name'))

        if self.gate in ['iff'] and len(self.arguments)>self.allowedGates['iff']:
            raise IOError('FTImporterPostProcessor Post-Processor; iff gate ' + str(self.name) + ' has more than 2 events')
        if self.gate in ['imply'] and len(self.arguments)>self.allowedGates['imply']:
            raise IOError('FTImporterPostProcessor Post-Processor; imply gate ' + str(self.name) + ' has more than 2 events')

    def returnArguments(self):
        """
          Method that returns the arguments of the gate
          @ In, None
          @ Out, self.arguments, list, list that contains the arguments of the gate
        """
        return self.arguments

    def evaluate(self,argValues):
        """
          Method that evaluates the gate
          @ In, argValues, dict, dictionary containing all available variables
          @ Out, outcome, float, calculated outcome of the gate
        """
        argumentValues = copy.deepcopy(argValues)
        for key in self.negations:
            if argumentValues[key]==1:
                argumentValues[key]=0
            else:
                argumentValues[key]=1
        if set(self.arguments) <= set(argumentValues.keys()):
            argumentsToPass = OrderedDict()
            for arg in self.arguments:
                argumentsToPass[arg] = argumentValues[arg]
            outcome = self.evaluateGate(argumentsToPass)
            return outcome
        else:
            raise IOError('FTImporterPostProcessor Post-Processor; gate ' + str(self.name) + ' can receive these arguments ' + str(self.arguments) + ' but the following were passed ' + str(argumentValues.keys()) )

    def evaluateGate(self,argumentValues):
        """
          Method that evaluates the gate.
          Note that argumentValues is passed directly (instead of argumentValues.values()) in case of the imply gate since the events IDs are important
          @ In, argumentValues, OrderedDict, dictionary containing only the variables of interest to the gate
          @ Out, outcome, float, calculated outcome of the gate
        """
        if self.gate == 'and':
            outcome = ANDgate(argumentValues.values())
        elif self.gate == 'or':
            outcome = ORgate(argumentValues.values())
        elif self.gate == 'nor':
            outcome = NORgate(argumentValues.values())
        elif self.gate == 'nand':
            outcome = NANDgate(argumentValues.values())
        elif self.gate == 'xor':
            outcome = XORgate(argumentValues.values())
        elif self.gate == 'iff':
            outcome = IFFgate(argumentValues.values())
        elif self.gate == 'atleast':
            outcome = ATLEASTgate(argumentValues.values(),float(self.params['min']))
        elif self.gate == 'cardinality':
            outcome = CARDINALITYgate(argumentValues.values(),float(self.params['min']),float(self.params['max']))
        elif self.gate == 'imply':
            outcome = IMPLYgate(argumentValues)
        elif self.gate == 'not':
            outcome = NOTgate(argumentValues.values())
        return outcome

def NOTgate(value):
    """
      Method that evaluates the NOT gate
      @ In, value, list, list of values
      @ Out, outcome, float, calculated outcome of the gate
    """
    if len(value)>1:
        raise IOError('NOT gate has received in input ' + str(len(value)) + ' values instead of 1.')
    if value[0]==0:
        outcome = 1
    else:
        outcome = 0
    return outcome

def ANDgate(argumentValues):
    """
      Method that evaluates the AND gate
      @ In, argumentValues, list, list of values
      @ Out, outcome, float, calculated outcome of the gate
    """
    if 0 in argumentValues:
        outcome = 0
    else:
        outcome = 1
    return outcome

def ORgate(argumentValues):
    """
      Method that evaluates the OR gate
      @ In, argumentValues, list, list of values
      @ Out, outcome, float, calculated outcome of the gate
    """
    if 1 in argumentValues:
        outcome = 1
    else:
        outcome = 0
    return outcome

def NANDgate(argumentValues):
    """
      Method that evaluates the NAND gate
      @ In, argumentValues, list, list of values
      @ Out, outcome, float, calculated outcome of the gate
    """
    out = []
    out.append(ANDgate(argumentValues))
    outcome = NOTgate(out)
    return outcome

def NORgate(argumentValues):
    """
      Method that evaluates the NOR gate
      @ In, argumentValues, list, list of values
      @ Out, outcome, float, calculated outcome of the gate
    """
    out = []
    out.append(ORgate(argumentValues))
    outcome = NOTgate(out)
    return outcome

def XORgate(argumentValues):
    """
      Method that evaluates the XOR gate
      The XOR gate gives a true (1 or HIGH) output when the number of true inputs is odd.
      https://electronics.stackexchange.com/questions/93713/how-is-an-xor-with-more-than-2-inputs-supposed-to-work
      @ In, argumentValues, list, list of values
      @ Out, outcome, float, calculated outcome of the gate
    """
    if argumentValues.count(1.) % 2 != 0:
        outcome = 1
    else:
        outcome = 0
    return outcome

def IFFgate(argumentValues):
    """
      Method that evaluates the IFF gate
      @ In, argumentValues, list, list of values
      @ Out, outcome, float, calculated outcome of the gate
    """
    out = []
    out.append(XORgate(argumentValues))
    outcome = NOTgate(out)
    return outcome

def IMPLYgate(argumentValues):
    """
      Method that evaluates the IMPLY gate
      Note that this gate requires a specific definition of the two inputs. This definition is specifed in the order of the events provided in the input file
      As an example, BE1->BE2 is translated as:
        <define-gate name="TOP">
            <imply>
                <basic-event name="BE1"/>
                <basic-event name="BE2"/>
            </imply>
        </define-gate>
      @ In, argumentValues, list, list of values
      @ Out, outcome, float, calculated outcome of the gate
    """
    keys = argumentValues.keys()
    if argumentValues[keys[0]]==1 and argumentValues[keys[1]]==0:
        outcome = 0
    else:
        outcome = 1
    return outcome

def ATLEASTgate(argumentValues,k):
    """
      Method that evaluates the ATLEAST gate
      @ In, argumentValues, list, list of values
      @ In, k, float, max number of allowed events
      @ Out, outcome, float, calculated outcome of the gate
    """
    if argumentValues.count(1) >= k:
        outcome = 1
    else:
        outcome = 0
    return outcome

def CARDINALITYgate(argumentValues,l,h):
    """
      Method that evaluates the CARDINALITY gate
      @ In, argumentValues, list, list of values
      @ In, l, float, min number of allowed events
      @ In, h, float, max number of allowed events
      @ Out, outcome, float, calculated outcome of the gate
    """
    nOcc = argumentValues.count(1)
    if nOcc >= l and nOcc <= h:
        outcome = 1
    else:
        outcome = 0
    return outcome

def findAllRecursive(node, element):
    """
      A function for recursively traversing a node in an elementTree to find
      all instances of a tag.
      Note that this method differs from findall() since it goes for all nodes,
      subnodes, subsubnodes etc. recursively
      @ In, node, ET.Element, the current node to search under
      @ In, element, str, the string name of the tags to locate
      @ InOut, result, list, a list of the currently recovered results
    """
    result=[]
    for elem in node.iter(tag=element):
        result.append(elem)
    return result
