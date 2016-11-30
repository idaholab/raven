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
    A Main Window container for holding various subwindows related to the
    visualization and analysis of a dataset according to the approximate
    Morse-Smale complex.
  """
  closed = qtc.Signal(qtc.QObject)
  def __init__(self, **kwargs):
    """
      Initialization method that can optionally specify all of the parameters
      needed for building an underlying AMSC_Object to be used internally by
      this window and its child views.
      @ In, kwargs, a dictionary that will hand the requested data to the
        views.
    """
    super(HierarchyWindow,self).__init__()
    self.views = []
    self.resize(800,600)
    self.setCentralWidget(None)
    self.setDockOptions(qtg.QMainWindow.AllowNestedDocks)
    if 'debug' in kwargs:
      self.debug = kwargs['debug']

    self.fileMenu = self.menuBar().addMenu('File')
    self.optionsMenu = self.menuBar().addMenu('Options')
    self.viewMenu = self.menuBar().addMenu('View')
    newMenu = self.viewMenu.addMenu('New...')
    if 'views' in kwargs:
      views = kwargs.pop('views')
      for view in views:
        self.addNewView(view, kwargs)

    for subclass in BaseView.__subclasses__():
      action = newMenu.addAction(subclass.__name__)
      action.triggered.connect(functools.partial(self.addNewView,action.text(), kwargs))

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

    self.viewMenu.addAction(dockWidget.toggleViewAction())

  def addNewView(self,viewType, **kwargs):
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

        self.views.append(subclass(self,
                                   parent=None,
                                   title=defaultWidgetName,
                                   **kwargs))
        view = self.views[-1]

        self.createDockWidget(view)

        #self.amsc.sigSelectionChanged.connect(view.selectionChanged)
        #self.amsc.sigFilterChanged.connect(view.filterChanged)
        #self.amsc.sigDataChanged.connect(view.dataChanged)

  def closeEvent(self,event):
    """
      Event handler triggered when this window is closed.
      @ In, event, a QCloseEvent specifying the context of this event.
    """
    self.closed.emit(self)
    return super(HierarchyWindow,self).closeEvent(event)
