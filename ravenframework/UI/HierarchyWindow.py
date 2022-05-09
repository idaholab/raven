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
  A UI for visualizing hierarchical objects, specifically the hierarchical
  clustering made available from scipy.
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

from sys import path

from .BaseHierarchicalView import BaseHierarchicalView
from .DendrogramView import DendrogramView
from .ScatterView import ScatterView
from . import colors

import time
import sys
import functools
import re
import random
import numpy as np

class HierarchyWindow(qtw.QMainWindow):
  """
    A UI for visualizing hierarchical objects, specifically the hierarchical
    clustering made available from scipy.
  """
  closed = qtc.Signal(qtc.QObject)
  sigLevelChanged = qtc.Signal(qtc.QObject)
  sigColorChanged = qtc.Signal(qtc.QObject)
  def __init__(self, engine, debug=None, views=None):
    """
      The initialization method for this window.
      @ In, engine, unSupervisedLearning, the object containing the hierarchy
      @ In, debug, boolean, whether we are in debug mode which increases the
        verbosity of this code.
      @ In, views, list(string), list of strings which will be interpretted to
        view class instances based on the class name.
      @ Out, None
    """
    super(HierarchyWindow,self).__init__()
    self.views = []
    self.resize(800,600)
    self.setCentralWidget(None)
    self.setDockOptions(qtw.QMainWindow.AllowNestedDocks)

    self.debug = debug
    self.engine = engine

    self.levels = sorted(set(self.engine.linkage[:,2]))
    self.colorMap = {}

    self.fileMenu = self.menuBar().addMenu('File')
    # self.optionsMenu = self.menuBar().addMenu('Options')
    self.viewMenu = self.menuBar().addMenu('View')
    newMenu = self.viewMenu.addMenu('New...')

    self.setLevel(self.engine.initOptionDict['level'])

    for view in views:
      self.addNewView(view)

    for subclass in BaseHierarchicalView.__subclasses__():
      action = newMenu.addAction(subclass.__name__)
      action.triggered.connect(functools.partial(self.addNewView,action.text()))

  def test(self):
    """
        Method for testing this UI. It will generate one of each of the
        subclass views of the BaseHierarchicalView and call each of the signaled
        events on each view.
        @ In, None
        @ Out, None
    """
    for viewClass in BaseHierarchicalView.__subclasses__():
      self.addNewView(viewClass.__name__)

    labels = self.getLabels()
    self.decreaseLevel()
    self.increaseLevel()
    self.setColor(0,qtg.QColor(255,0,0))

    for view in self.views:
      view.updateScene()
      view.colorChanged()
      view.levelChanged()
      view.selectionChanged()

      view.test()


  def createDockWidget(self,view):
    """
      Method to create a new child dock widget of a specified type.
      @ In, view, an object belonging to a subclass of BaseHierarchicalView
        that will be added to this window.
      @ Out, None
    """
    dockWidget = qtw.QDockWidget()
    dockWidget.setWindowTitle(view.windowTitle())

    if view.scrollable:
      scroller = qtw.QScrollArea()
      scroller.setWidget(view)
      scroller.setWidgetResizable(True)
      dockWidget.setWidget(scroller)
    else:
      dockWidget.setWidget(view)

    self.addDockWidget(qtc.Qt.TopDockWidgetArea,dockWidget)
    self.viewMenu.addAction(dockWidget.toggleViewAction())

  def addNewView(self, viewType):
    """
      Method to create a new child view which will be added as a dock widget
      and thus will call createDockWidget()
      @ In, viewType, a string specifying a subclass of BaseHierarchicalView
        that will be added to this window.
      @ Out, None
    """
    defaultWidgetName = ''
    for subclass in BaseHierarchicalView.__subclasses__():
      if subclass.__name__ == viewType:
        idx = 0
        for view in self.views:
          if isinstance(view,subclass):
            idx += 1

        defaultWidgetName = subclass.__name__.replace('View','')
        if idx > 0:
          defaultWidgetName += ' ' + str(idx)

        self.views.append(subclass(mainWindow=self))
        view = self.views[-1]

        self.createDockWidget(view)

        self.sigLevelChanged.connect(view.levelChanged)
        self.sigColorChanged.connect(view.colorChanged)
        #self.sigSelectionChanged.connect(view.selectionChanged)
        #self.amsc.sigDataChanged.connect(view.dataChanged)

  def closeEvent(self,event):
    """
      Event handler triggered when this window is closed.
      @ In, event, a QCloseEvent specifying the context of this event.
      @ Out, None
    """
    self.closed.emit(self)
    return super(HierarchyWindow,self).closeEvent(event)

  def increaseLevel(self):
    """
      Function to increase the level of the underlying data object.
      @ In, None
      @ Out, None
    """
    level = self.engine.initOptionDict['level']
    for newLevel in self.levels:
      if newLevel > level:
        self.setLevel(newLevel)
        return

  def decreaseLevel(self):
    """
      Function to decrease the level of the underlying data object.
      @ In, None
      @ Out, None
    """
    level = self.engine.initOptionDict['level']
    for newLevel in reversed(self.levels):
      if newLevel < level:
        self.setLevel(newLevel)
        return

  def getLevel(self):
    """
      Function to retrieve the level of the underlying data object.
      @ In, None
      @ Out, level, float, the current level being used by the unsupervised
        engine.
    """
    return self.engine.initOptionDict['level']

  def setLevel(self, newLevel):
    """
      Function to set the level of the underlying data object.
      @ In, level, float, the new level to be used by the unsupervised
        engine.
      @ Out, None
    """
    self.engine.initOptionDict['level'] = newLevel

    linkage = self.engine.linkage

    self.labels = np.zeros(len(self.engine.outputDict['outputs']['labels']))

    heads = {}

    for i in range(len(self.labels)):
      heads[i] = [i]
      self.labels[i] = i

    n = linkage.shape[0]+1
    for i in range(linkage.shape[0]):
      newIdx = n+i
      leftChildIdx,rightChildIdx,level,size = linkage[i,:]
      if level <= newLevel:
        heads[newIdx] = heads.pop(leftChildIdx) + heads.pop(rightChildIdx)
      else:
        break

      for head,children in heads.items():
        for idx in children:
          self.labels[idx] = head


    self.levelChanged()

  def levelChanged(self):
    """
      Function that will propagate the changes to the level to its child views.
      @ In, None
      @ Out, None
    """
    self.sigLevelChanged.emit(self)

  def setColor(self,idx,color):
    """
      Set the color of a specified index in the tree.
      @ In, idx, int, the index to be updated
      @ In, color, QColor, the new color for the index.
      @ Out, None
    """
    self.colorMap[idx] = color
    self.sigColorChanged.emit(self)

  def getColor(self,idx):
    """
      Get the color of a specified index in the tree.
      @ In, idx, int, the index to be updated.
      @ Out, color, QColor, the color for the index.
    """
    if idx not in self.colorMap:
      # self.colorMap[idx] = qtg.QColor(*tuple(255*np.random.rand(3)))
      self.colorMap[idx] = qtg.QColor(next(colors.colorCycle))

    return self.colorMap[idx]

  def getData(self):
    """
      Retrieve all of the data associated to this window.
      @ In, None
      @ Out, data, nparray, the data being used by this window.
    """
    data = np.zeros((len(self.engine.features),
                     len(self.engine.outputDict['inputs'])))
    for col,value in enumerate(self.engine.outputDict['inputs']):
      data[:,col] = value
    return data

  def getDimensions(self):
    """
      Get the dimensionality of the underlying data.
      @ In, None
      @ Out, dimensionality, string list, the dimensions of the data being used.
    """
    return self.engine.features

  def getSelectedIndices(self):
    """
      Get the currently selected indices
      @ In, None
      @ Out, None
    """
    return []

  def getLabels(self):
    """
      Get the labels of this data
      @ In, None
      @ Out, None
    """
    return self.labels
