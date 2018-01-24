from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import math
import sys
import random
import copy
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
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
    allowedCalcMode = ['steadyState', 'simulated']
    container.states = {}

    for child in xmlNode:
      if   child.tag == 'initState':
        container.initState  = child.text.strip()
      if   child.tag == 'finState':
        container.finState  = child.text.strip()
      elif child.tag == 'calcMode':
        container.calcMode = child.text.strip()
      elif child.tag == 'endTime':
        container.endTime = float(child.text.strip())
      elif child.tag == 'deltaT':
        container.deltaT = float(child.text.strip())
      elif child.tag == 'state':
        container.states[child.get('name')] = {}
        for childChild in child:
          if childChild.tag == 'transition':
            if childChild.get('type') == 'lambda':
              value = float(childChild.get('value'))
            elif childChild.get('type') == 'tau':
              value = 1. / float(childChild.get('value'))
            container.states[child.get('name')][childChild.text.strip()] = value
          else:
            print('error')
    statesIDs = container.states.keys()
    print(container.states)
    for state in container.states:
      if state not in statesIDs:
        print('state error')
      transitions = container.states[state].keys()
      if not set(transitions).issubset(set(statesIDs)):
        print(container.states[state].keys(),statesIDs)
        print('transition error')
    if container.calcMode not in allowedCalcMode:
      print('calcMode error')
    if container.initState is None:
      print('initState Error')
    if container.finState is None:
      print('finState Error')
    if container.deltaT is None:
      container.timeArray = np.arange(0.,container.endTime,container.deltaT)

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
    while time < container.endTime:
      #totLambda = sum(container.states[actualState].values())
      #transitionTime = np.random.exponential(1./totLambda)
      transitionTime , actualState = self.detNewState(copy.deepcopy(container.states[actualState]))     
      time += transitionTime
      '''
      rng = random.uniform(0, 1)
      probability = 0. 
      for transition in container.states[actualState].keys():
        probabilityOLD = copy.deepcopy(probability)
        probabilityNEW = copy.deepcopy(probability) + container.states[actualState][transition]/totLambda
        if  probabilityOLD < rng < probabilityNEW:
          time += transitionTime
          actualState = copy.deepcopy(transition)
          break
        else:
          probability += container.states[actualState][transition]/totLambda
      '''

    container.finalState = int(actualState)

  def detNewState(self,dict):
    totLambda = sum(dict.values())
    transitionTime = np.random.exponential(1./totLambda)
    rng = random.uniform(0, 1)
    for transition in dict.keys():
      dict[transition] = dict[transition]/totLambda
    state = np.random.choice(dict.keys(), size = 1, p=dict.values())[0]
    return transitionTime, state


  def runOLD(self, container, Inputs):
    """
      This is a simple example of the run method in a plugin.
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    time = 0.
    actualState = str(int(Inputs[container.initState][0]))
    nextState= None
    while time < container.endTime:
      transitionTime = sys.float_info.max
      for transition in container.states[actualState].keys():
        value = np.random.exponential(1./container.states[actualState][transition])
        if value<transitionTime:
          transitionTime = value
          nextState = transition
      time = time + transitionTime
      actualState = nextState

    container.finalState = int(actualState)







