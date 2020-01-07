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
  A view widget for visualizing scatterplots of data utilizing matplotlib.
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3

import matplotlib

try:
  from PySide import QtCore as qtc
  from PySide import QtGui as qtw
except ImportError as e:
  from PySide2 import QtCore as qtc
  from PySide2 import QtWidgets as qtw

from .BaseHierarchicalView import BaseHierarchicalView

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mpl_toolkits
import matplotlib.pyplot
import matplotlib.ticker


import numpy as np
from . import colors

class ScatterView(BaseHierarchicalView):
  """
    A view widget for visualizing scatterplots of data utilizing matplotlib.
  """
  def __init__(self, mainWindow=None):
    """
      Constructor for the Scatter plot view
      @ In, mainWindow, MainWindow, the main window associated to this dependent
        view
    """
    BaseHierarchicalView.__init__(self, mainWindow)

    self.setLayout(qtw.QVBoxLayout())
    layout = self.layout()
    self.clearLayout(layout)

    mySplitter = qtw.QSplitter()
    mySplitter.setOrientation(qtc.Qt.Vertical)
    layout.addWidget(mySplitter)

    self.fig = Figure(facecolor='white')
    self.mplCanvas = FigureCanvas(self.fig)
    self.mplCanvas.axes = self.fig.add_subplot(111)

    self.colorbar = None

    mySplitter.addWidget(self.mplCanvas)

    controls = qtw.QGroupBox()
    controls.setLayout(qtw.QGridLayout())
    subLayout = controls.layout()
    row = 0
    col = 0

    self.rightClickMenu = qtw.QMenu()
    self.axesLabelAction = self.rightClickMenu.addAction('Show Axis Labels')
    self.axesLabelAction.setCheckable(True)
    self.axesLabelAction.setChecked(True)
    self.axesLabelAction.triggered.connect(self.updateScene)

    self.cmbVars = {}

    for i,name in enumerate(['X','Y','Z','Color']):
      varLabel = name + ' variable:'
      self.cmbVars[name] = qtw.QComboBox()

      if name == 'Z':
        self.cmbVars[name].addItem('Off')
      elif name == 'Color':
        self.cmbVars[name].addItem('Cluster')

      dimNames = self.mainWindow.getDimensions()
      self.cmbVars[name].addItems(dimNames)

      if i < len(dimNames):
        self.cmbVars[name].setCurrentIndex(i)
      else:
        self.cmbVars[name].setCurrentIndex(len(dimNames)-1)

      self.cmbVars[name].currentIndexChanged.connect(self.updateScene)

      subLayout.addWidget(qtw.QLabel(varLabel),row,col)
      col += 1
      subLayout.addWidget(self.cmbVars[name],row,col)
      row += 1
      col = 0

    self.lblColorMaps = qtw.QLabel('Colormap')
    self.cmbColorMaps = qtw.QComboBox()
    self.cmbColorMaps.addItems(matplotlib.pyplot.colormaps())
    self.cmbColorMaps.setCurrentIndex(self.cmbColorMaps.findText('coolwarm'))
    self.cmbColorMaps.currentIndexChanged.connect(self.updateScene)
    subLayout.addWidget(self.lblColorMaps,row,col)
    col += 1
    subLayout.addWidget(self.cmbColorMaps,row,col)
    mySplitter.addWidget(controls)

    self.cmbVars['Z'].setCurrentIndex(0)
    self.updateScene()

  def sizeHint(self):
    """
      This property holds the recommended size for the widget. If the value of
      this property is an invalid size, no size is recommended. The default
      implementation of PySide.QtGui.QWidget.sizeHint() returns an invalid
      size if there is no layout for this widget, and returns the layout's
      preferred size otherwise. (Copied from base class text)
      @ In, None
      @ Out, QSize, the recommended size of this widget
    """
    return qtc.QSize(300,600)

  def selectionChanged(self):
    """
      An event handler triggered when the user changes the selection of the
      data.
      @ In, None
      @ Out, None
    """
    self.updateScene()

  def updateScene(self):
    """
      A method for drawing the scene of this view.
      @ In, None
      @ Out, None
    """
    fontSize=16
    smallFontSize=12
    rows = self.mainWindow.getSelectedIndices()
    names = self.mainWindow.getDimensions()
    data = self.mainWindow.getData()

    # self.fig = Figure(facecolor='white')
    # self.mplCanvas = FigureCanvas(self.fig)

    self.fig.clear()

    if self.cmbVars['Z'].currentIndex() == 0:
      dimensionality = 2
      self.mplCanvas.axes = self.fig.add_subplot(111)
    else:
      dimensionality = 3
      self.mplCanvas.axes = self.fig.add_subplot(111, projection='3d')

    myColormap = colors.cm.get_cmap(self.cmbColorMaps.currentText())

    if len(rows) == 0:
      rows = list(range(data.shape[0]))

    allValues = {}
    values = {}
    mins = {}
    maxs = {}

    specialColorKeywords = ['Cluster']

    string_type = '|U7' #If python 2 compatibility is needed, use '|S7'
    for key,cmb in self.cmbVars.items():
      if dimensionality == 2 and key == 'Z':
        continue
      if cmb.currentText() == 'Cluster':
        labels = self.mainWindow.getLabels()
        allValues[key] = np.array([self.mainWindow.getColor(label).name() for label in labels], dtype=string_type)
        values[key] = allValues[key][rows]
        self.lblColorMaps.setEnabled(False)
        self.cmbColorMaps.setEnabled(False)
        self.lblColorMaps.setVisible(False)
        self.cmbColorMaps.setVisible(False)
      else:
        col = names.index(cmb.currentText())
        allValues[key] = data[:,col]
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        values[key] = allValues[key][rows]
        self.lblColorMaps.setEnabled(True)
        self.cmbColorMaps.setEnabled(True)
        self.lblColorMaps.setVisible(True)
        self.cmbColorMaps.setVisible(True)

    kwargs = {'edgecolors': 'none', 'c': values['Color']}

    if dimensionality == 2:
      kwargs['x'] = values['X']
      kwargs['y'] = values['Y']
    else:
      kwargs['xs'] = values['X']
      kwargs['ys'] = values['Y']
      kwargs['zs'] = values['Z']

    if self.cmbVars['Color'].currentText() not in specialColorKeywords:
      kwargs['c'] = values['Color']
      kwargs['cmap'] = myColormap
      kwargs['vmin'] = mins['Color']
      kwargs['vmax'] = maxs['Color']

    myPlot = self.mplCanvas.axes.scatter(**kwargs)

    if self.axesLabelAction.isChecked():
      self.mplCanvas.axes.set_xlabel(self.cmbVars['X'].currentText(),size=fontSize,labelpad=10)
      self.mplCanvas.axes.set_ylabel(self.cmbVars['Y'].currentText(),size=fontSize,labelpad=10)
      if dimensionality == 3:
        self.mplCanvas.axes.set_zlabel(self.cmbVars['Z'].currentText(),size=fontSize,labelpad=10)

    ticks = np.linspace(mins['X'],maxs['X'],5)
    self.mplCanvas.axes.set_xticks(ticks)
    self.mplCanvas.axes.set_xlim([ticks[0],ticks[-1]])
    self.mplCanvas.axes.xaxis.set_ticklabels([])
    self.mplCanvas.axes.yaxis.set_ticklabels([])
    self.mplCanvas.axes.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))

    ticks = np.linspace(mins['Y'],maxs['Y'],5)
    self.mplCanvas.axes.set_yticks(ticks)
    self.mplCanvas.axes.set_ylim([ticks[0],ticks[-1]])
    self.mplCanvas.axes.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))

    if dimensionality == 3:
      ticks = np.linspace(mins['Z'],maxs['Z'],3)
      self.mplCanvas.axes.set_zticks(ticks)
      self.mplCanvas.axes.set_zlim([ticks[0],ticks[-1]])
      self.mplCanvas.axes.zaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))

    for label in  (self.mplCanvas.axes.get_xticklabels()+self.mplCanvas.axes.get_yticklabels()):
      label.set_fontsize(smallFontSize)

    self.mplCanvas.draw()

  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations.  For this class in particular, we will test:
        - Switching from 2D to a 3D projection
        - Changing from a color map to the cluster colors.
        - Toggling the axes labels on and off and displaying both.
        @ In, None
        @ Out, None
    """
    self.cmbVars['Z'].setCurrentIndex(self.cmbVars['Z'].count()-1)
    self.cmbVars['Color'].setCurrentIndex(0)
    self.updateScene()
    self.axesLabelAction.setChecked(True)
    self.updateScene()
    self.axesLabelAction.setChecked(False)
    self.updateScene()


    super(ScatterView, self).test()
    BaseHierarchicalView.test(self)
