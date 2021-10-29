"""
  Author:  A. Alfonsi
  Date  :  11/17/2017
"""
import numpy as np
import math

from PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase


class SumOfExponential(ExternalModelPluginBase):
  # Example External Model plugin class
  #################################
  #### RAVEN API methods BEGIN ####
  #################################
  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.coefficients = {}
    container.startValue   = None
    container.endValue     = None
    container.numberPoints = 10
    outputVarNode    = xmlNode.find("outputVariable")
    if outputVarNode is None:
      raise IOError("ExamplePlugin: <outputVariable> XML block must be inputted!")
    container.outputVariable = outputVarNode.text.strip()
    monotonicVarNode = xmlNode.find("monotonicVariable")
    if monotonicVarNode is None:
      raise IOError("ExamplePlugin: <monotonicVariable> XML block must be inputted!")
    container.monotonicVariableName = monotonicVarNode.text.strip()

    for child in xmlNode:
      if child.tag.strip() == "variables":
        # get verbosity if it exists
        container.variables = [var.strip() for var in child.text.split(",")]
        if container.outputVariable not in container.variables:
          raise IOError("ExamplePlugin: "+container.outputVariable+" variable MUST be present in the <variables> definition!")
        if container.monotonicVariableName not in container.variables:
          raise IOError("ExamplePlugin: "+container.monotonicVariableName+" variable MUST be present in the <variables> definition!")
        if len(container.variables) < 2:
          raise IOError("ExamplePlugin: at least 1 input and 1 output variable ("+container.outputVariable+") must be listed in the <variables> definition!!")
      if child.tag.strip() == "coefficient":
        if "varName" not in child.attrib:
          raise IOError("ExamplePlugin: attribute varName must be present in <coefficient> XML node!")
        container.coefficients[child.attrib['varName']] = float(child.text)
      if child.tag.strip() == "startMonotonicVariableValue":
        container.startValue = float(child.text)
      if child.tag.strip() == "endMonotonicVariableValue":
        container.endValue = float(child.text)
      if child.tag.strip() == "numberCalculationPoints":
        container.numberPoints = int(child.text)
    if container.startValue is None:
      raise IOError("ExamplePlugin: <startMonotonicVariableValue> XML has not been inputted!")
    if container.endValue is None:
      raise IOError("ExamplePlugin: <endMonotonicVariableValue> XML has not been inputted!")
    container.variables.pop(container.variables.index("Xi"))
    container.variables.pop(container.variables.index("monotonicVariable"))

  def initialize(self, container,runInfoDict,inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    for var in container.variables:
      if var not in container.coefficients:
        container.coefficients[var] = 1.0
        print("ExamplePlugin: not found coefficient for variable "+var+". Default value is 1.0!")
    container.stepSize = (container.endValue - container.startValue)/float(container.numberPoints)

  def run(self, container, Inputs):
    """
      This is a simple example of the run method in a plugin.
      This method takes the variables in input and computes
      oneOutputOfThisPlugin(t) = var1Coefficient*exp(var1*t)+var2Coefficient*exp(var2*t) ...
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    Xi                = np.zeros(container.numberPoints+1)
    monotonicVariable = np.zeros(container.numberPoints+1)
    monoVarVal = container.startValue
    monotonicVariable[0] = container.startValue
    varCoeff = np.asarray([container.coefficients[var] for var in container.variables])
    varExponents = np.asarray([Inputs[var]*monoVarVal for var in container.variables])
    Xi[0] = np.sum(varCoeff*np.exp(varExponents))
    for step in range(container.numberPoints):
      monoVarVal+=container.stepSize
      monotonicVariable[step+1] = monoVarVal
      varExponents = np.asarray([Inputs[var]*(monoVarVal-monotonicVariable[step]) for var in container.variables])
      if np.max(varExponents) >= np.finfo(varExponents.dtype).maxexp:
        print("ExamplePlugin: the exponents of the exponential cause overflow. Increase the number of <numberCalculationPoints>!")
      Xi[step+1] = np.sum(varCoeff*np.exp(varExponents))
      Xi[step+1]+=Xi[step]
    container.__dict__[container.outputVariable] = Xi
    container.__dict__[container.monotonicVariableName] = monotonicVariable
  ###############################
  #### RAVEN API methods END ####
  ###############################

