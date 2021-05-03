"""
  Author:  talbpaul
  Date  :  2021-04-02
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from PluginBaseClasses.OutStreamPlotPlugin import PlotPlugin, InputTypes, InputData


class Correlation(PlotPlugin):
  # Example Plot plugin class
  @classmethod
  def getInputSpecification(cls):
    """
      Define the acceptable user inputs for this class.
      @ In, None
      @ Out, specs, InputData.ParameterInput,
    """
    specs = super().getInputSpecification()
    specs.addSub(InputData.parameterInputFactory('bins', contentType=InputTypes.IntegerType))
    specs.addSub(InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType))
    specs.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType))
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'ExamplePlugin.Correlation'
    self._numBins = 10      # number of bins to use; np default is 10 currently
    self._vars = None       # list of variables to plot correlations for
    self._sourceName = None # name of source data object
    self._source = None     # actual source data object

  def handleInput(self, spec):
    """
      Reads in data from the input file
      @ In, spec, InputData.ParameterInput, input information
      @ Out, None
    """
    super().handleInput(spec)
    for node in spec.subparts:
      if node.getName() == 'bins':
        self._numBins = node.value
      elif node.getName() == 'variables':
        self._vars = node.value
      elif node.getName() == 'source':
        self._sourceName = node.value
    # input checking
    if self._vars is None:
      self.raiseAnError(IOError, 'Input missing the <variables> node!')
    if self._sourceName is None:
      self.raiseAnError(IOError, 'Input missing the <source> node!')

  def initialize(self, stepEntities):
    """
      Set up plotter for each run
      @ In, stepEntities, dict, entities from the Step
      @ Out, None
    """
    super().initialize(stepEntities)
    src = self.findSource(self._sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'Source DataObject {self._sourceName} was not found in the Step!')
    self._source = src

  def run(self):
    """
      Generate the plot
      @ In, None
      @ Out, None
    """
    n = len(self._vars)
    fig, axes = plt.subplots(n, n, tight_layout=True)
    data = self._source.asDataset()
    for v1, var1 in enumerate(self._vars):
      var1Data = data[var1].values
      for v2, var2 in enumerate(self._vars):
        ax = axes[v2, v1] # TODO wasn't this a flattened array for some matplotlibs?
        if var1 == var2:
          counts, edges = np.histogram(var1Data, bins=self._numBins)
          ax.step(0.5 * (edges[:-1] + edges[1:]), counts, '.-', where='mid')
          ax.set_xlabel(var1)
          ax.set_ylabel(var1)
        else:
          var2Data = data[var2].values
          ax.scatter(var1Data, var2Data, marker='.')
          ax.set_xlabel(var1)
          ax.set_ylabel(var2)
        if v1 == 0:
          ax.set_ylabel(var2)
        else:
          ax.set_ylabel('')
        if v2 == n - 1:
          ax.set_xlabel(var1)
        else:
          ax.set_xlabel('')
    fName = os.path.abspath(f'{self.name}.png')
    plt.savefig(fName)
    self.raiseAMessage(f'Saved figure to "{fName}"')



