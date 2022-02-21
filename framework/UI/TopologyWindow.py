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
  A Main Window container for holding various subwindows related to the
  visualization and analysis of a dataset according to the approximate
  Morse-Smale complex.
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3

try:
  from PySide import QtCore as qtc
  from PySide import QtGui as qtw
except ImportError as e:
  from PySide2 import QtCore as qtc
  from PySide2 import QtWidgets as qtw

from sys import path

from AMSC.AMSC_Object import QAMSC_Object
from .BaseTopologicalView import BaseTopologicalView
from .TopologyMapView import TopologyMapView
from .SensitivityView import SensitivityView
from .FitnessView import FitnessView
from .ScatterView2D import ScatterView2D
from .ScatterView3D import ScatterView3D

import time
import sys
import functools
import re
import random
import numpy as np

class TopologyWindow(qtw.QMainWindow):
  """
      A Main Window container for holding various subwindows related to the
      visualization and analysis of a dataset according to the approximate
      Morse-Smale complex.
  """
  views = []
  closed = qtc.Signal(qtc.QObject)
  def __init__(self, X=None, Y=None, w=None, names=None, graph='beta skeleton',
               gradient='steepest', knn=-1, beta=1.0, normalization=None,
               debug=False, views=None):
    """ Initialization method that can optionally specify all of the parameters
        needed for building an underlying AMSC_Object to be used internally by
        this window and its child views.
        @ In, X, an m-by-n array of values specifying m n-dimensional samples
        @ In, Y, a m vector of values specifying the output responses
          corresponding to the m samples specified by X
        @ In, w, an optional m vector of values specifying the weights
          associated to each of the m samples used. Default of None means all
          points will be equally weighted
        @ In, names, an optional list of strings that specify the names to
          associate to the n input dimensions and 1 output dimension. Default of
          None means input variables will be x0,x1...,x(n-1) and the output will
          be y
        @ In, graph, an optional string specifying the type of neighborhood
          graph to use. Default is 'beta skeleton,' but other valid types are:
          'delaunay,' 'relaxed beta skeleton,' 'none', or 'approximate knn'
        @ In, gradient, an optional string specifying the type of gradient
          estimator to use. Currently the only available option is 'steepest'
        @ In, knn, an optional integer value specifying the maximum number of
          k-nearest neighbors used to begin a neighborhood search. In the case
          of graph='[relaxed] beta skeleton', we will begin with the specified
          approximate knn graph and prune edges that do not satisfy the empty
          region criteria.
        @ In, beta, an optional floating point value between 0 and 2. This
          value is only used when graph='[relaxed] beta skeleton' and specifies
          the radius for the empty region graph computation (1=Gabriel graph,
          2=Relative neighbor graph)
        @ In, normalization, an optional string specifying whether the
          inputs/output should be scaled before computing. Currently, two modes
          are supported 'zscore' and 'feature'. 'zscore' will ensure the data
          has a mean of zero and a standard deviation of 1 by subtracting the
          mean and dividing by the variance. 'feature' scales the data into the
          unit hypercube.
        @ In, debug, an optional boolean flag for whether debugging output
          should be enabled.
        @ In, views, list of strings, represents views to be added to this main
          window.  Valid values include a string representation of any of the
          subclasses of the BaseTopologicalView.
    """
    super(TopologyWindow,self).__init__()
    self.resize(800,600)
    self.setCentralWidget(None)
    self.setDockOptions(qtw.QMainWindow.AllowNestedDocks)
    self.debug = debug
    self.amsc = None

    if X is not None:
      self.BuildAMSC(X, Y, w, names, graph, gradient, knn, beta, normalization)
    else:
      self.BuildAMSC(None, None, None, None, None, None, None, None, None)

    self.fileMenu = self.menuBar().addMenu('File')
    self.optionsMenu = self.menuBar().addMenu('Options')
    self.viewMenu = self.menuBar().addMenu('View')
    newMenu = self.viewMenu.addMenu('New...')

    if views is not None:
      for view in views:
        self.addNewView(view)

    for subclass in BaseTopologicalView.__subclasses__():
      action = newMenu.addAction(subclass.__name__)
      action.triggered.connect(functools.partial(self.addNewView,action.text()))

    buildModels = self.optionsMenu.addAction('Build Local Models')
    buildModels.triggered.connect(self.buildModels)

  def test(self):
    """
        Method for testing this UI. It will generate one of each of the
        subclass views of the BaseTopologicalView and call each of the signaled
        events on each view.
        @ In, None
        @ Out, None
    """
    for viewClass in BaseTopologicalView.__subclasses__():
      self.addNewView(viewClass.__name__)

    ## Rebuild the AMSC after all of the views have been added
    self.BuildAMSC(self.amsc.X, self.amsc.Y, self.amsc.w, self.amsc.names,
                   self.amsc.graph, self.amsc.gradient, self.amsc.knn,
                   self.amsc.beta, self.amsc.normalization)

    self.buildModels()

    for view in self.views:
      view.dataChanged()
      view.filterChanged()
      view.selectionChanged()
      view.persistenceChanged()
      view.modelsChanged()
      view.weightsChanged()

      view.test()

    ## Reset the persistence level to the value used by our test case.
    ## Note, if we require multiple test cases, we could remember this state
    ## from before, but as this function should capture all of the functionality
    ## this is a low priority fix and might require persistent storage of this
    ## value on the object.
    self.amsc.Persistence(1.0)

  def BuildAMSC(self, X=None, Y=None, w=None, names=None, graph='beta skeleton',
                gradient='steepest', knn=-1, beta=1.0, normalization=None):
    """ Initialization method that can optionally specify all of the parameters
        needed for building an underlying AMSC_Object to be used internally by
        this window and its child views.
        @ In, X, an m-by-n array of values specifying m n-dimensional samples
        @ In, Y, a m vector of values specifying the output responses
          corresponding to the m samples specified by X
        @ In, w, an optional m vector of values specifying the weights
          associated to each of the m samples used. Default of None means all
          points will be equally weighted
        @ In, names, an optional list of strings that specify the names to
          associate to the n input dimensions and 1 output dimension. Default of
          None means input variables will be x0,x1...,x(n-1) and the output will
          be y
        @ In, graph, an optional string specifying the type of neighborhood
          graph to use. Default is 'beta skeleton,' but other valid types are:
          'delaunay,' 'relaxed beta skeleton,' 'none', or 'approximate knn'
        @ In, gradient, an optional string specifying the type of gradient
          estimator to use. Currently the only available option is 'steepest'
        @ In, knn, an optional integer value specifying the maximum number of
          k-nearest neighbors used to begin a neighborhood search. In the case
          of graph='[relaxed] beta skeleton', we will begin with the specified
          approximate knn graph and prune edges that do not satisfy the empty
          region criteria.
        @ In, beta, an optional floating point value between 0 and 2. This
          value is only used when graph='[relaxed] beta skeleton' and specifies
          the radius for the empty region graph computation (1=Gabriel graph,
          2=Relative neighbor graph)
        @ In, normalization, an optional string specifying whether the
          inputs/output should be scaled before computing. Currently, two modes
          are supported 'zscore' and 'feature'. 'zscore' will ensure the data
          has a mean of zero and a standard deviation of 1 by subtracting the
          mean and dividing by the variance. 'feature' scales the data into the
          unit hypercube.
        @ In, debug, an optional boolean flag for whether debugging output
          should be enabled.
    """
    if self.debug:
      start = time.clock()
      ## Disable non-message handler output for now.
      # sys.stderr.write('Building AMSC...\r')
      # sys.stderr.flush()
    if self.amsc is None:
      self.amsc = QAMSC_Object(X=X, Y=Y, w=w, names=names, graph=graph,
                              gradient=gradient, knn=knn, beta=beta,
                              normalization=normalization, debug=self.debug)
    else:
      self.amsc.Reinitialize(X=X, Y=Y, w=w, names=names, graph=graph,
                             gradient=gradient, knn=knn, beta=beta,
                             normalization=normalization, debug=self.debug)
    if self.debug:
      end = time.clock()
      # sys.stderr.write('Construction Time: %f s\n' % (end-start))

  def buildModels(self):
    """ Method to tell the underlying data object ot construct its local models,
        if they do not yet exist.
    """
    self.amsc.BuildModels()

  def createDockWidget(self,view):
    """ Method to create a new child dock widget of a specified type.
        @ In, view, an object belonging to a subclass of BaseTopologicalView
          that will be added to this window.
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

    #Placement was arbitrarily selected
    self.addDockWidget(qtc.Qt.TopDockWidgetArea,dockWidget)

    self.viewMenu.addAction(dockWidget.toggleViewAction())

  def addNewView(self,viewType):
    """ Method to create a new child view which will be added as a dock widget
        and thus will call createDockWidget()
        @ In, viewType, a string specifying a subclass of BaseTopologicalView
          that will be added to this window.
    """
    defaultWidgetName = ''
    for subclass in BaseTopologicalView.__subclasses__():
      if subclass.__name__ == viewType:
        idx = 0
        for view in self.views:
          if isinstance(view,subclass):
            idx += 1

        defaultWidgetName = subclass.__name__.replace('View','')
        if idx > 0:
          defaultWidgetName += ' ' + str(idx)

        self.views.append(subclass(self,self.amsc,defaultWidgetName))
        view = self.views[-1]

        self.createDockWidget(view)
        self.amsc.sigPersistenceChanged.connect(view.persistenceChanged)
        self.amsc.sigSelectionChanged.connect(view.selectionChanged)
        self.amsc.sigFilterChanged.connect(view.filterChanged)
        self.amsc.sigDataChanged.connect(view.dataChanged)
        self.amsc.sigModelsChanged.connect(view.modelsChanged)
        self.amsc.sigWeightsChanged.connect(view.weightsChanged)

  def closeEvent(self,event):
    """ Event handler triggered when this window is closed.
        @ In, event, a QCloseEvent specifying the context of this event.
    """
    self.closed.emit(self)
    return super(TopologyWindow,self).closeEvent(event)
