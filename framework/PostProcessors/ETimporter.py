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
Created on Nov 1, 2017

@author: maljdan, mandd

"""

from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import InputData
import Files
import Runners
#Internal Modules End-----------------------------------------------------------


class ETimporter(PostProcessor):
  """
    This is the base class for postprocessors
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """

    self.ETformat = None
    self.allowedFormats = ['OpenPSA']


  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inputSpecification = super(RavenOutput, cls).getInputSpecification()

    ## TODO: Fill this in with the appropriate tags

    return inputSpecification


  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    # if 'externalFunction' in initDict.keys(): self.externalFunction = initDict['externalFunction']
    self.inputs = inputs
    self._workingDir = runInfo['WorkingDir']

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'fileFormat':
        self.ETformat = child.text
        if self.ETformat not in self.ETformat:
          self.raiseAnError(IOError, 'ETimporterPostProcessor Post-Processor ' + self.name + ', format ' + child.text + ' : is not supported')
      else:
        self.raiseAnError(IOError, 'ETimporterPostProcessor Post-Processor ' + self.name + ', node ' + child.tag + ' : is not recognized')

  def inputToInternal(self, currentInput):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputToInternal, list, list of current inputs
    """
    pass

  def run(self, input):
    """
      This method executes the postprocessor action.
      @ In,  input, object, object containing the data to process. (inputToInternal output)
      @ Out, None
    """
    root = ET.parse('eventTree.xml').getroot()
    eventTree = root.findall('initial-state')

    if len(eventTree) > 1:
      sys.stderr.write('More than one initial-state identified. I do not know what to do now.\n')
      sys.exit(1)

    eventTree = eventTree[0]

    ## These outcomes will be encoded as integers starting at 0
    outcomes = []

    ## These variables will be mapped into an array where there index
    variables = []
    values = {}

    for node in root.findall('define-functional-event'):
      event = node.get('name')

      ## First, map the variable to an index by placing it in a list
      variables.append(event)

      ## Also, initialize the dictionary of values for this variable so we can
      ## encode them as integers as well
      values[event] = []

      ## Iterate through the forks that use this event and gather all of the
      ## possible states
      for fork in findAllRecursive(eventTree, 'fork'):
        if fork.get('functional-event') == event:
          for path in fork.findall('path'):
            state = path.get('state')
            if state not in values[event]:
              values[event].append(state)

    ## Iterate through the sequences and gather all of the possible outcomes
    ## so we can numerically encode them latter
    for node in root.findall('define-sequence'):
      outcome = node.get('name')
      if outcome not in outcomes:
        outcomes.append(outcome)

    print('*' * 80)
    print('Inputs: {}'.format(variables))
    print('Value Map: {}'.format(values))
    print('Outputs: {}'.format(outcomes))
    print('*' * 80)
    print('\n')

    d = len(variables)
    n = len(findAllRecursive(eventTree, 'sequence'))

    pointSet = -1 * np.ones((n, d + 1))

    rowCounter = 0
    for node in eventTree:
      newRows = ConstructPointDFS(node, variables, values, outcomes, pointSet, rowCounter)
      rowCounter += newRows

    print(pointSet)


  def findAllRecursive(node, element, result = None):
	"""
      A function for recursively traversing a node in an elementTree to find
      all instances of a tag
      @ In, node, ET.Element, the current node to search under
      @ In, element, str, the string name of the tags to locate
      @ InOut, result, list, a list of the currently recovered results
	"""
	if result is None:
		result = []
	for item in node.getchildren():
		if item.tag == element:
			result.append(item)
		findAllRecursive(item, element, result)
	return result

  def ConstructPointDFS(node, inputMap, stateMap, outputMap, X, rowCounter):
    """
      Construct a "sequence" using a depth-first search on a node, each call
      will be on a fork except in the base case which will be called on a
      sequence node. The recursive case will traverse into a path node, thus
      path nodes will be "skipped" in the call stack as one level of paths
      will be processed per recursive call in order to set one of the columns
      of X for the row identified by rowCounter.
      @ In, node, ET.Element, the current node to process
      @ In, inputMap, list, a map for converting string variable names into
            sequential non-negative integers that can be used to index X
      @ In, stateMap, dict, a map similar to above, but instead converts the
            possible states for each event (variable) into non-negative
            integers
      @ In, outputMap, list, a map for converting string outcome values into
            non-negative integers
      @ In, X, np.array, data object to populate with values
      @ In, rowCounter, int, the row we are currently editing in X
      @ Out, offset, int, the number of rows of X this call has populated
    """
    if node.tag == 'sequence':
      col = X.shape[1]-1
      outcome = node.get('name')
      val = outputMap.index(outcome)

      print(ET.tostring(node, method='xml'))
      try:
          input('X[{},{}] = {} ({}), continue:'.format(rowCounter, col, val, outcome))
      except SyntaxError as se:
          pass

      X[rowCounter, col] = val
      rowCounter += 1
    elif node.tag == 'fork':
      event = node.get('functional-event')
      col = inputMap.index(event)

      for path in node.findall('path'):
          state = path.get('state')
          val = stateMap[event].index(state)

          print(' ===== path ===== ')
          print(ET.tostring(node, method='xml'))
          try:
              input('X[{},{}] = {} ({}), continue:'.format(rowCounter, col, val, state))
          except SyntaxError as se:
              pass

          ## Fill in the rest of the data as the recursive nature will only
          ## fill in the details under this branch, later iterations will
          ## correct lower rows if a path does change
          X[rowCounter, col] = val

          for fork in path.getchildren():
              newCounter = ConstructPointDFS(fork, inputMap, stateMap, outputMap, X, rowCounter)
              for i in range(newCounter-rowCounter):
                  X[rowCounter+i, col] = val
              rowCounter = newCounter

    return rowCounter