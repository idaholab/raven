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
  A view widget for visualizing 3D scatterplots of data utilizing matplotlib.
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3

try:
  from PySide import QtCore as qtc
  from PySide import QtGui as qtg
  from PySide import QtGui as qtw
except ImportError as e:
  from PySide2 import QtCore as qtc
  from PySide2 import QtGui as qtg
  from PySide2 import QtWidgets as qtw


from .BaseTopologicalView import BaseTopologicalView

import matplotlib
#Is the below needed?
#matplotlib.use('Qt5Agg')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mpl_toolkits
import matplotlib.pyplot
import matplotlib.ticker

import numpy as np
from . import colors

class ScatterView3D(BaseTopologicalView):
  """
     A view widget for visualizing 3D scatterplots of data utilizing matplotlib.
  """
  def __init__(self, parent=None, amsc=None, title="3D Projection"):
    """ Initialization method that can optionally specify the parent widget,
        an AMSC object to reference, and a title for this widget.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, amsc, an optional AMSC_Object specifying the underlying data
          object for this widget to use.
        @ In, title, an optional string specifying the title of this widget.
    """
    super(ScatterView3D, self).__init__(parent,amsc,title)

  def Reinitialize(self, parent=None, amsc=None, title="3D Projection"):
    """ Reinitialization method that resets this widget and can optionally
        specify the parent widget, an AMSC object to reference, and a title for
        this widget.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, amsc, an optional AMSC_Object specifying the underlying data
          object for this widget to use.
        @ In, title, an optional string specifying the title of this widget.
    """
    # Try to apply a new layout, if one already exists then make sure to grab
    # it for updating
    if self.layout() is None:
      self.setLayout(qtw.QVBoxLayout())
    layout = self.layout()
    self.clearLayout(layout)

    mySplitter = qtw.QSplitter()
    mySplitter.setOrientation(qtc.Qt.Vertical)
    layout.addWidget(mySplitter)

    self.fig = Figure(facecolor='white')
    self.mplCanvas = FigureCanvas(self.fig)
    self.mplCanvas.axes = self.fig.add_subplot(111, projection='3d')
    # We want the axes cleared every time plot() is called,
    # so axes.hold used to be called, but now that has been removed.
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

    self.chkExts = qtw.QCheckBox('Show Extrema')
    self.chkExts.setTristate(True)
    self.chkExts.setCheckState(qtc.Qt.PartiallyChecked)
    self.chkExts.stateChanged.connect(self.updateScene)
    subLayout.addWidget(self.chkExts,row,col)
    row += 1
    col = 0

    self.chkEdges = qtw.QCheckBox('Show Edges')
    self.chkEdges.setChecked(False)
    self.chkEdges.stateChanged.connect(self.updateScene)
    subLayout.addWidget(self.chkEdges,row,col)
    row += 1
    col = 0

    self.cmbVars = {}
    for i,name in enumerate(['X','Y','Z','Color']):
      varLabel = name + ' variable:'
      self.cmbVars[name] = qtw.QComboBox()
      dimNames = self.amsc.GetNames()
      self.cmbVars[name].addItems(dimNames)
      if name == 'Color':
        self.cmbVars[name].addItems(['Segment'])
        self.cmbVars[name].addItems(['Minimum Flow'])
        self.cmbVars[name].addItems(['Maximum Flow'])
      self.cmbVars[name].addItem('Predicted from Linear Fit')
      self.cmbVars[name].addItem('Residual from Linear Fit')

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

    self.cmbColorMaps = qtw.QComboBox()
    self.cmbColorMaps.addItems(matplotlib.pyplot.colormaps())
    self.cmbColorMaps.setCurrentIndex(self.cmbColorMaps.findText('coolwarm'))
    self.cmbColorMaps.currentIndexChanged.connect(self.updateScene)
    subLayout.addWidget(qtw.QLabel('Colormap'),row,col)
    col += 1
    subLayout.addWidget(self.cmbColorMaps,row,col)
    mySplitter.addWidget(controls)

    self.modelsChanged()
    self.updateScene()

  def sizeHint(self):
    """ This property holds the recommended size for the widget. If the value of
        this property is an invalid size, no size is recommended. The default
        implementation of PySide.QtGui.QWidget.sizeHint() returns an invalid
        size if there is no layout for this widget, and returns the layout's
        preferred size otherwise. (Copied from base class text)
    """
    return qtc.QSize(300,600)

  def selectionChanged(self):
    """ An event handler triggered when the user changes the selection of the
        data.
    """
    self.updateScene()

  def persistenceChanged(self):
    """ An event handler triggered when the user changes the persistence setting
        of the data.
    """
    self.modelsChanged()

  def modelsChanged(self):
    """ An event handler triggered when the user requests a new set of local
        models.
    """
    enabled = self.amsc.FitsSynced()
    for cmb in self.cmbVars.values():
      for i in range(cmb.count()):
        if 'Predicted' in cmb.itemText(i) or 'Residual' in cmb.itemText(i):
          item = cmb.model().item(i,0)
          if enabled:
            item.setFlags(qtc.Qt.ItemIsSelectable | qtc.Qt.ItemIsEnabled)
          else:
            item.setFlags(qtc.Qt.NoItemFlags)
        ## If this cmb is currently displaying fit information, then change it
        ## to display the output dimension
        if not enabled and (   'Predicted' in cmb.currentText() \
                            or 'Residual' in cmb.currentText()):
          cmb.setCurrentIndex(self.amsc.GetDimensionality())
    self.updateScene()

  def updateScene(self):
    """ A method for drawing the scene of this view.
    """
    fontSize=24
    smallFontSize=20
    rows = self.amsc.GetSelectedIndices()
    names = self.amsc.GetNames()
    self.mplCanvas.axes.clear()

    myColormap = colors.cm.get_cmap(self.cmbColorMaps.currentText())

    if len(rows) == 0:
      rows = list(range(self.amsc.GetSampleSize()))

    allValues = {}
    values = {}
    mins = {}
    maxs = {}
    minValues = {}
    maxValues = {}

    minIdxs = []
    maxIdxs = []

    minDrawParams = {'c':colors.minBrushColor.name(), 'marker':'v', 's':160,
                     'zorder':3, 'edgecolors':colors.minPenColor.name()}
    maxDrawParams = {'c':colors.maxBrushColor.name(), 'marker':'^', 's':160,
                     'zorder':3, 'edgecolors':colors.maxPenColor.name()}

    if self.chkExts.checkState() == qtc.Qt.Checked \
    or self.chkExts.checkState() == qtc.Qt.PartiallyChecked:
      minMaxPairs = self.amsc.GetSelectedSegments()
      for extPair in minMaxPairs:
        minIdxs.append(extPair[0])
        maxIdxs.append(extPair[1])

      extIdxs = self.amsc.GetSelectedExtrema()
      for extIdx in extIdxs:
        if self.amsc.GetClassification(extIdx) == 'maximum':
          maxIdxs.append(extIdx)
        elif self.amsc.GetClassification(extIdx) == 'minimum':
          minIdxs.append(extIdx)

      ## Remove any duplicates
      minIdxs = list(set(minIdxs))
      maxIdxs = list(set(maxIdxs))

      if len(minIdxs) == 0 and len(maxIdxs) == 0:
        minMaxPairs = self.amsc.GetCurrentLabels()
        for extPair in minMaxPairs:
          minIdxs.append(extPair[0])
          maxIdxs.append(extPair[1])

      ## Remove the extrema from the list of regular points that will be
      ## rendered
      for extIdx in minIdxs + maxIdxs:
        if extIdx in rows:
          rows.remove(extIdx)

    specialColorKeywords = ['Segment','Minimum Flow', 'Maximum Flow']

    string_type = '|U7' #If python 2 compatibility is needed, use '|S7'
    for key,cmb in self.cmbVars.items():
      if cmb.currentText() == 'Predicted from Linear Fit':
        allValues[key] = self.amsc.PredictY(None)
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        minValues[key] = allValues[key][minIdxs]
        maxValues[key] = allValues[key][maxIdxs]
        values[key] = self.amsc.PredictY(rows)
      elif cmb.currentText() == 'Residual from Linear Fit':
        allValues[key] = self.amsc.Residuals(None)
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        minValues[key] = allValues[key][minIdxs]
        maxValues[key] = allValues[key][maxIdxs]
        values[key] = self.amsc.Residuals(rows)
      elif cmb.currentText() == 'Segment':
        colorMap = self.amsc.GetColors()
        partitions = self.amsc.Partitions()
        allValues[key] = np.zeros(self.amsc.GetSampleSize(),dtype=string_type)
        for extPair,items in partitions.items():
          for item in items:
            allValues[key][item] = colorMap[extPair]
        values[key] = allValues[key][rows]
        minValues[key] = [colorMap[minIdx] for minIdx in minIdxs]
        maxValues[key] = [colorMap[maxIdx] for maxIdx in maxIdxs]
      elif cmb.currentText() == 'Maximum Flow':
        colorMap = self.amsc.GetColors()
        partitions = self.amsc.Partitions()
        allValues[key] = np.zeros(self.amsc.GetSampleSize(),dtype=string_type)
        for extPair,items in partitions.items():
          for item in items:
            allValues[key][item] = colorMap[extPair[1]]
        values[key] = allValues[key][rows]
        minValues[key] = [colorMap[minIdx] for minIdx in minIdxs]
        maxValues[key] = [colorMap[maxIdx] for maxIdx in maxIdxs]
      elif cmb.currentText() == 'Minimum Flow':
        colorMap = self.amsc.GetColors()
        partitions = self.amsc.Partitions()
        allValues[key] = np.zeros(self.amsc.GetSampleSize(),dtype=string_type)
        for extPair,items in partitions.items():
          for item in items:
            allValues[key][item] = colorMap[extPair[0]]
        values[key] = allValues[key][rows]
        minValues[key] = [colorMap[minIdx] for minIdx in minIdxs]
        maxValues[key] = [colorMap[maxIdx] for maxIdx in maxIdxs]
      else:
        col = names.index(cmb.currentText())
        if col == len(names)-1:
          allValues[key] = self.amsc.GetY(None)
          mins[key] = min(allValues[key])
          maxs[key] = max(allValues[key])
          minValues[key] = allValues[key][minIdxs]
          maxValues[key] = allValues[key][maxIdxs]
          values[key] = self.amsc.GetY(rows)
        else:
          allValues[key] = self.amsc.GetX(None,col)
          mins[key] = min(allValues[key])
          maxs[key] = max(allValues[key])
          minValues[key] = allValues[key][minIdxs]
          maxValues[key] = allValues[key][maxIdxs]
          values[key] = self.amsc.GetX(rows,col)

    if self.chkEdges.isChecked():
      lines  = []
      lineColors = []
      lines2 = []
      lineIdxs = []
      for row in rows + minIdxs + maxIdxs:
        cols = self.amsc.GetNeighbors(int(row))
        for col in cols:
          if col in rows + minIdxs + maxIdxs:
            if row < col:
              A = row
              B = col
            elif col > row:
              B = row
              A = col
            lineIdxs.append((A,B))
            lines.append([(allValues['X'][row],
                           allValues['Y'][row],
                           allValues['Z'][row]),
                          (allValues['X'][col],
                           allValues['Y'][col],
                           allValues['Z'][col])])
            if self.cmbVars['Color'].currentText() not in specialColorKeywords:
              lineColors.append(myColormap(((allValues['Color'][row]+allValues['Color'][col])/2.-mins['Color'])/(maxs['Color']-mins['Color'])))
            elif allValues['Color'][row] == allValues['Color'][col]:
              lineColors.append(allValues['Color'][row])
            else:
              lineColors.append('#CCCCCC')

      lc = mpl_toolkits.mplot3d.art3d.Line3DCollection(lines,colors=lineColors,linewidths=1)
      self.mplCanvas.axes.add_collection(lc)

    if self.cmbVars['Color'].currentText() not in specialColorKeywords:
      myPlot = self.mplCanvas.axes.scatter(values['X'], values['Y'],
                                           values['Z'], c=values['Color'],
                                           cmap=myColormap,
                                           vmin=mins['Color'],
                                           vmax=maxs['Color'],
                                           edgecolors='none')

      if self.colorbar is None:
        self.colorbar = self.fig.colorbar(myPlot)
      else:
        # This is intended to be a deprecated feature, but how else can we
        # force matplotlib to rescale the axis on the colorbar?
        self.colorbar.update_bruteforce(myPlot)
        ## Here is its replacement, but this guy will not rescale the colorbar
        #self.colorbar.update_normal(myPlot)
      self.colorbar.set_label(self.cmbVars['Color'].currentText(),size=fontSize,labelpad=10)
      self.colorbar.set_ticks(np.linspace(mins['Color'],maxs['Color'],5))
      self.colorbar.ax.tick_params(labelsize=smallFontSize)
      if self.chkExts.checkState() == qtc.Qt.PartiallyChecked:
        maxValues['Color'] = colors.maxBrushColor.name()
        minValues['Color'] = colors.minBrushColor.name()
      self.mplCanvas.axes.scatter(maxValues['X'], maxValues['Y'],
                                  maxValues['Z'], c=maxValues['Color'],
                                  cmap=myColormap,
                                  marker=maxDrawParams['marker'],
                                  s=maxDrawParams['s'],
                                  zorder=maxDrawParams['zorder'],
                                  vmin=mins['Color'], vmax=maxs['Color'],
                                  edgecolors=maxDrawParams['edgecolors'])
      self.mplCanvas.axes.scatter(minValues['X'], minValues['Y'],
                                  minValues['Z'], c=minValues['Color'],
                                  cmap=myColormap,
                                  marker=minDrawParams['marker'],
                                  s=minDrawParams['s'],
                                  zorder=minDrawParams['zorder'],
                                  vmin=mins['Color'], vmax=maxs['Color'],
                                  edgecolors=minDrawParams['edgecolors'])
    else:
      myPlot = self.mplCanvas.axes.scatter(values['X'], values['Y'],
                                           values['Z'], c=values['Color'],
                                           edgecolors='none')

      if self.chkExts.checkState() == qtc.Qt.PartiallyChecked:
        maxValues['Color'] = colors.maxBrushColor.name()
        minValues['Color'] = colors.minBrushColor.name()
      self.mplCanvas.axes.scatter(maxValues['X'], maxValues['Y'],
                                  maxValues['Z'], c=maxValues['Color'],
                                  marker=maxDrawParams['marker'],
                                  s=maxDrawParams['s'],
                                  zorder=maxDrawParams['zorder'],
                                  edgecolors=maxDrawParams['edgecolors'])
      self.mplCanvas.axes.scatter(minValues['X'], minValues['Y'],
                                  minValues['Z'], c=minValues['Color'],
                                  marker=minDrawParams['marker'],
                                  s=minDrawParams['s'],
                                  zorder=minDrawParams['zorder'],
                                  edgecolors=minDrawParams['edgecolors'])

    if self.axesLabelAction.isChecked():
      self.mplCanvas.axes.set_xlabel(self.cmbVars['X'].currentText(),size=fontSize,labelpad=20)
      self.mplCanvas.axes.set_ylabel(self.cmbVars['Y'].currentText(),size=fontSize,labelpad=20)
      self.mplCanvas.axes.set_zlabel(self.cmbVars['Z'].currentText(),size=fontSize,labelpad=20)

    #Doesn't do anything
    self.mplCanvas.axes.set_axisbelow(True)

    ticks = np.linspace(mins['X'],maxs['X'],3)
    self.mplCanvas.axes.set_xticks(ticks)
    self.mplCanvas.axes.set_xlim([ticks[0],ticks[-1]])
    ticks = np.linspace(mins['Y'],maxs['Y'],3)
    self.mplCanvas.axes.set_yticks(ticks)
    self.mplCanvas.axes.set_ylim([ticks[0],ticks[-1]])
    ticks = np.linspace(mins['Z'],maxs['Z'],3)
    # ticks = np.linspace(0, 2, 5)
    self.mplCanvas.axes.set_zticks(ticks)
    self.mplCanvas.axes.set_zlim([ticks[0],ticks[-1]])

    self.mplCanvas.axes.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
    self.mplCanvas.axes.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
    self.mplCanvas.axes.zaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))

    for label in  (self.mplCanvas.axes.get_xticklabels()+self.mplCanvas.axes.get_yticklabels()+self.mplCanvas.axes.get_zticklabels()):
      label.set_fontsize(smallFontSize)

    self.mplCanvas.draw()

  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations.  For this class in particular, we will test:
        - Toggling the edges on and off and updating the display in both cases.
        - Toggling all three states of the extrema display.
        - Changing the color attribute to cycle through each of the labeled
          variables: Segment, Minimum Flow, Maximum Flow, Local fit value, and
          local fit error/residual.
        - Resizing the display
        - Subselecting the data that is displayed.
        @ In, None
        @ Out, None
    """
    self.amsc.ClearSelection()

    self.axesLabelAction.setChecked(True)
    self.chkExts.setCheckState(qtc.Qt.Checked)
    self.chkEdges.setChecked(True)
    self.cmbVars['Color'].setCurrentIndex(self.cmbVars['Color'].count()-5)
    self.updateScene()

    self.axesLabelAction.setChecked(False)
    self.chkExts.setCheckState(qtc.Qt.Unchecked)
    self.chkEdges.setChecked(True)
    self.cmbVars['Color'].setCurrentIndex(self.cmbVars['Color'].count()-4)
    self.updateScene()

    self.chkExts.setCheckState(qtc.Qt.PartiallyChecked)
    self.cmbVars['Color'].setCurrentIndex(self.cmbVars['Color'].count()-3)
    self.updateScene()

    self.cmbVars['Color'].setCurrentIndex(self.cmbVars['Color'].count()-2)
    self.updateScene()

    self.cmbVars['Color'].setCurrentIndex(self.cmbVars['Color'].count()-1)
    self.updateScene()

    self.resizeEvent(qtg.QResizeEvent(qtc.QSize(1,1),qtc.QSize(100,100)))
    pair = list(self.amsc.GetCurrentLabels())[0]
    self.amsc.SetSelection([pair,pair[0],pair[1]])
    self.updateScene()

    super(ScatterView3D, self).test()
