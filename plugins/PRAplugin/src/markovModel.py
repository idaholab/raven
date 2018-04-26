from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import math
import sys
import random
import copy
from operator import mul
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
from utils.randomUtils import random
#Internal Modules End-----------------------------------------------------------


class markovModel(ExternalModelPluginBase):
  """
    This class is designed to create a Markov model 
  """

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to the Markov Model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.initState = None
    container.finState  = None
    container.states = {}

    for child in xmlNode:
      if   child.tag == 'initState':
        container.initState  = child.text.strip()
      if   child.tag == 'finState':
        container.finState  = child.text.strip()
      elif child.tag == 'endTime':
        container.endTime = float(child.text.strip())
      elif child.tag == 'state':
        container.states[child.get('name')] = {}
        for childChild in child:
          if childChild.tag == 'transition':
            if childChild.get('type') == 'lambda':
              value = float(childChild.get('value'))
            elif childChild.get('type') == 'tau':
              value = 1. / float(childChild.get('value'))
            elif childChild.get('type') == 'instant':
              value = [float(childChild.get('value'))]
            elif childChild.get('type') == 'unif':
              value = [float(var.strip()) for var in childChild.get('value').split(",")]
            else:
              raise IOError("markovModel: transition " + str (childChild.get('type')) + " is not allowed")
            container.states[child.get('name')][childChild.text.strip()] = value
          else:
            raise IOError("markovModel: xml node " + str (childChild.tag) + " is not allowed")

    statesIDs = container.states.keys()
    for state in container.states:
      transitions = container.states[state].keys()
      if not set(transitions).issubset(set(statesIDs)):
        raise IOError("markovModel: the set of transtions " + str (set(transitions)) + " out of state " + str(state) + " lead to not defined states")
    if container.initState is None:
      raise IOError("markovModel: <initState> XML block is not specified")
    if container.finState is None:
      raise IOError("markovModel: <finState> XML block is not specified")

  def initialize(self, container,runInfoDict,inputFiles):
    """
      Method to initialize this Markov model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    np.random.seed(250678)

  def run(self, container, Inputs):
    """
      This method computes all the final state at the end of the specified time
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    time = 0.
    actualState = str(int(Inputs[container.initState][0]))
    while True:
      transitionDict = copy.deepcopy(container.states[actualState])
      transitionTime , newState = self.newState(transitionDict)
      time += transitionTime
      if time >= container.endTime:
        break
      else:
        actualState = newState

    container.__dict__[container.finState] = np.asarray(actualState)

  def newState(self,dictIn):
    """
      Method which calculate the next transtion out of a state
      @ In, dictIn, dict, dictionary containing all possible transtions out of a state
      @ Out, detTransitionTime, float, time of the next transtion
      @ Out, detState, float, arrival state for the next transtion
    """
    detTrans   = {}
    stochTrans = {}
    detTransitionTime   = sys.float_info.max
    stochTransitionTime = sys.float_info.max

    for key in dictIn.keys():
      if type(dictIn[key]) is list:
        detTrans[key]  = copy.deepcopy(dictIn[key])
      else:
        stochTrans[key] = copy.deepcopy(dictIn[key])
    if detTrans:
      detTransitionTime, detState     = self.detNewState(detTrans)
    if stochTrans:
      stochTransitionTime, stochState = self.stochNewState(stochTrans)
    if stochTransitionTime < detTransitionTime:
      return stochTransitionTime, stochState
    else:
      return detTransitionTime, detState

  def detNewState(self,detTrans):
    """
      Method which calculate the next transtion out of a state for a determinisct transtion
      @ In, dictIn, dict, dictionary containing all possible transtions out of a state
      @ Out, detTransitionTime, float, time of the next transtion
      @ Out, detTransitionState, float, arrival state for the next transtion
    """
    detTransitionTime  = sys.float_info.max
    detTransitionState = None
    for key in detTrans.keys():
      if len(detTrans[key]) == 1:
        time = detTrans[key][0]
        if time<detTransitionTime:
          detTransitionTime = time
          detTransitionState = key
      elif len(detTrans[key]) == 2:
        lowVal  = min(detTrans[key])
        highVal = max(detTrans[key])
        time = np.random.uniform(low=lowVal, high=highVal)
        if time<detTransitionTime:
          detTransitionTime = time
          detTransitionState = key

    return detTransitionTime, detTransitionState

  def stochNewState(self,stochTrans):
    """
      Method which calculate the next transtion out of a state for a stochastic transtion
      @ In, dictIn, dict, dictionary containing all possible transtions out of a state
      @ Out, transitionTime, float, time of the next transtion
      @ Out, state, float, arrival state for the next transtion
    """
    totLambda = sum(stochTrans.values())
    transitionTime = np.random.exponential(1./totLambda)
    for transition in stochTrans.keys():
      stochTrans[transition] = stochTrans[transition]/totLambda
    state = np.random.choice(stochTrans.keys(), size = 1, p=stochTrans.values())[0]
    return transitionTime, state



