#!/usr/bin/env python

from PySide import QtCore as qtc
from PySide import QtGui as qtg

from sys import path

from DendrogramView import DendrogramView
from BaseView import BaseView

import time
import sys
import functools
import re
import random
import numpy as np

class HierarchyWindow(qtg.QMainWindow):
  """

  """
  closed = qtc.Signal(qtc.QObject)
  sigLevelChanged = qtc.Signal(qtc.QObject)
  def __init__(self, engine, level=None, debug=None, views=None):
    """

    """
    super(HierarchyWindow,self).__init__()
    self.views = []
    self.resize(800,600)
    self.setCentralWidget(None)
    self.setDockOptions(qtg.QMainWindow.AllowNestedDocks)

    self.debug = debug
    self.engine = engine

    self.levels = sorted(set(self.engine.linkage[:,2]))
    self.level = level or 0

    self.fileMenu = self.menuBar().addMenu('File')
    # self.optionsMenu = self.menuBar().addMenu('Options')
    self.viewMenu = self.menuBar().addMenu('View')
    newMenu = self.viewMenu.addMenu('New...')

    for view in views:
      self.addNewView(view)

    for subclass in BaseView.__subclasses__():
      action = newMenu.addAction(subclass.__name__)
      action.triggered.connect(functools.partial(self.addNewView,action.text()))

  def createDockWidget(self,view):
    """
      Method to create a new child dock widget of a specified type.
      @ In, view, an object belonging to a subclass of BaseView that will
        be added to this window.
    """
    dockWidget = qtg.QDockWidget()
    dockWidget.setWindowTitle(view.windowTitle())

    if view.scrollable:
      scroller = qtg.QScrollArea()
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
      @ In, viewType, a string specifying a subclass of BaseView that will
        be added to this window.
    """
    defaultWidgetName = ''
    for subclass in BaseView.__subclasses__():
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
        #self.sigSelectionChanged.connect(view.selectionChanged)
        #self.amsc.sigDataChanged.connect(view.dataChanged)

  def closeEvent(self,event):
    """
      Event handler triggered when this window is closed.
      @ In, event, a QCloseEvent specifying the context of this event.
    """
    self.closed.emit(self)
    return super(HierarchyWindow,self).closeEvent(event)

  def increaseLevel(self):
    """
    """
    for lvl in self.levels:
      if lvl > self.level:
        self.level = lvl
        self.levelChanged()
        return

  def decreaseLevel(self):
    """
    """
    for lvl in reversed(self.levels):
      if lvl < self.level:
        self.level = lvl
        self.levelChanged()
        return

  def setLevel(self, newLevel):
    """
    """
    self.level = newLevel
    self.levelChanged()

  def levelChanged(self):
    """
    """
    self.sigLevelChanged.emit(self)