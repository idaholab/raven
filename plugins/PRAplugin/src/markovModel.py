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

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.initState = None
    container.finState  = None
    container.states = {}

    np.random.seed(250678)

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
            container.states[child.get('name')][childChild.text.strip()] = value
          else:
            print('error')
    statesIDs = container.states.keys()
    for state in container.states:
      if state not in statesIDs:
        print('state error')
      transitions = container.states[state].keys()
      if not set(transitions).issubset(set(statesIDs)):
        print('transition error')
    if container.initState is None:
      print('initState Error')
    if container.finState is None:
      print('finState Error')

  def initialize(self, container,runInfoDict,inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    pass

  def run(self, container, Inputs):
    """
      This is a simple example of the run method in a plugin.
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
    totLambda = sum(stochTrans.values())
    transitionTime = np.random.exponential(1./totLambda)
    for transition in stochTrans.keys():
      stochTrans[transition] = stochTrans[transition]/totLambda
    state = np.random.choice(stochTrans.keys(), size = 1, p=stochTrans.values())[0]
    return transitionTime, state

  def newStateOLD(self,dictIn):
    if type(dictIn.values()[0]) is list and len(dictIn.values())==1:
      if len(dictIn.values()[0]) == 2:
        lowVal  = min(dictIn.values()[0])
        highVal = max(dictIn.values()[0])
        state = dictIn.keys()[0]
        transitionTime = np.random.uniform(low=lowVal, high=highVal)
        return transitionTime, state
      else:
        print('error 234')
    product = reduce(mul, dictIn.values(), 1)
    if   product < 0. and len(dictIn.values())>=2:
      print(error)
    elif product < 0. and len(dictIn.values())==1:
      transitionTime = -dictIn.values()[0]
      state          = dictIn.keys()[0]
    elif product >0.:
      totLambda = sum(dictIn.values())
      transitionTime = np.random.exponential(1./totLambda)
      for transition in dictIn.keys():
        dictIn[transition] = dictIn[transition]/totLambda
      state = np.random.choice(dictIn.keys(), size = 1, p=dictIn.values())[0]
    return transitionTime, state





