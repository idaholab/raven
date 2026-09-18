# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Radial visualisation (Radviz) embedding for multi-dimensional samples.

This plot projects optimisation decision variables onto a 2-D spring layout so
clusters, antagonistic anchors, and latent trade-offs become easy to spot.
Typical what-if scenario: "How do rank-one samples redistribute if I focus on
the final generation only?" — filter with `<index>`/`<generation>` to watch
the cloud pivot around competing anchors after constraints or preferences
change.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import radviz

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class RadvizEmbeddingPlot(PlotInterface):
  """
  Project multiple variables onto a Radviz embedding to inspect high-dimensional
  trade-offs. Supports what-if exploration by switching generations or colour
  labels to compare how candidate sets re-balance when objectives or
  constraints shift.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject providing the samples to visualise."""))
    variables = InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""List of variables (at least three) used for the Radviz embedding.""")
    spec.addSub(variables)
    spec.addSub(InputData.parameterInputFactory('label', contentType=InputTypes.StringType,
        descr=r"""Optional column used to colour samples (categorical or numeric)."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> is provided, limits the plot to a specific generation."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'RadvizEmbeddingPlot'
    self.source = None
    self.sourceName = None
    self.variables = []
    self.labelVar = None
    self.index = None
    self.generation = None

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for RadvizEmbeddingPlot "{self.name}".')
    self.sourceName = src.value

    varNode = spec.findFirst('variables')
    if varNode is None or len(varNode.value) < 3:
      self.raiseAnError(IOError, f'RadvizEmbeddingPlot "{self.name}" requires at least three <variables>.')
    self.variables = [entry for entry in varNode.value if entry]

    labelNode = spec.findFirst('label')
    if labelNode is not None and labelNode.value:
      self.labelVar = labelNode.value

    indexNode = spec.findFirst('index')
    if indexNode is not None and indexNode.value:
      self.index = indexNode.value
    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for RadvizEmbeddingPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.variables)
    if self.labelVar:
      needed.append(self.labelVar)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" missing variable(s) {missing} required by RadvizEmbeddingPlot "{self.name}".')
    self.source = src

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source DataObject "{self.source.name}" is empty; RadvizEmbeddingPlot "{self.name}" skipped.')
      return
    subset = df.copy()
    if self.index:
      subset[self.index] = subset[self.index].astype(float)
      if self.generation is not None:
        mask = np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)
        subset = subset[mask]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if subset.empty:
      self.raiseAWarning(f'RadvizEmbeddingPlot "{self.name}" had no samples after filtering.')
      return

    data = subset[self.variables].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
      self.raiseAWarning(f'RadvizEmbeddingPlot "{self.name}" found no finite rows for variables {self.variables}.')
      return

    if self.labelVar and self.labelVar in subset.columns:
      labels = subset.loc[data.index, self.labelVar]
      if np.issubdtype(labels.dtype, np.number):
        # Bin numeric labels into deciles for colouring
        labels = pd.qcut(labels, q=min(5, len(labels.unique())), duplicates='drop').astype(str)
      else:
        labels = labels.astype(str)
    else:
      labels = pd.Series(['samples'] * len(data), index=data.index)

    plot_df = data.copy()
    label_col = '_label'
    plot_df[label_col] = labels

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    try:
      radviz(plot_df, label_col, ax=ax, color=None, alpha=0.85)
    except Exception as err:
      self.raiseAWarning(f'Radviz embedding failed for "{self.name}": {err}')
      return
    ax.set_title('Radviz embedding')
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
