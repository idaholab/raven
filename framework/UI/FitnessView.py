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
  A view widget for visualizing the R^2 fitness of the local stepwise
  regression results.
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3

try:
  from PySide import QtCore as qtc
  from PySide import QtGui as qtg
  from PySide import QtGui as qtw
  from PySide import QtSvg as qts
except ImportError as e:
  from PySide2 import QtCore as qtc
  from PySide2 import QtGui as qtg
  from PySide2 import QtWidgets as qtw
  from PySide2 import QtSvg as qts

from .BaseTopologicalView import BaseTopologicalView

import math
import numpy as np

class FitnessView(BaseTopologicalView):
  """
     A view widget for visualizing the R^2 fitness of the local stepwise
     regression results.
  """
  def __init__(self, parent=None, amsc=None, title=None):
    """ Initialization method that can optionally specify the parent widget,
        an AMSC object to reference, and a title for this widget.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, amsc, an optional AMSC_Object specifying the underlying data
          object for this widget to use.
        @ In, title, an optional string specifying the title of this widget.
    """
    super(FitnessView, self).__init__(parent,amsc,title)

  def Reinitialize(self, parent=None, amsc=None, title=None):
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

    self.padding = 2

    ## General Graphics View/Scene setup
    self.scene = qtw.QGraphicsScene()
    self.scene.setSceneRect(0,0,100,100)
    self.gView = qtw.QGraphicsView(self.scene)
    self.gView.setRenderHints(qtg.QPainter.Antialiasing |
                              qtg.QPainter.SmoothPixmapTransform)
    self.gView.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.gView.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.font = qtg.QFont('sans-serif', 12)

    ## Defining the right click menu
    self.rightClickMenu = qtw.QMenu()
    self.fillAction = self.rightClickMenu.addAction('Fill viewport')
    self.fillAction.setCheckable(True)
    self.fillAction.setChecked(True)
    self.fillAction.triggered.connect(self.updateScene)

    self.showNumberAction = self.rightClickMenu.addAction('Show Numeric Values')
    self.showNumberAction.setCheckable(True)
    self.showNumberAction.setChecked(True)
    self.showNumberAction.triggered.connect(self.updateScene)

    captureAction = self.rightClickMenu.addAction('Capture')
    captureAction.triggered.connect(self.saveImage)

    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())

    layout.addWidget(self.gView)
    self.updateScene()

  def saveImage(self, filename=None):
    """
        Saves the current display of this view to a static image by loading a
        file dialog box.
        @ In, filename, string, optional parameter specifying where this image
        will be saved. If None, then a dialog box will prompt the user for a
        name and location.
        @ Out, None
    """
    if filename is None:
      dialog = qtw.QFileDialog(self)
      dialog.setFileMode(qtw.QFileDialog.AnyFile)
      dialog.setAcceptMode(qtw.QFileDialog.AcceptSave)
      dialog.exec_()
      if dialog.result() == qtw.QFileDialog.Accepted:
        filename = dialog.selectedFiles()[0]
      else:
        return

    self.scene.clearSelection()
    self.scene.setSceneRect(self.scene.itemsBoundingRect())
    if filename.endswith('.svg'):
      svgGen = qts.QSvgGenerator()
      svgGen.setFileName(filename)
      svgGen.setSize(self.scene.sceneRect().size().toSize())
      svgGen.setViewBox(self.scene.sceneRect())
      svgGen.setTitle("Screen capture of " + self.__class__.__name__)
      svgGen.setDescription("Generated from RAVEN.")
      painter = qtg.QPainter(svgGen)
    else:
      image = qtg.QImage(self.scene.sceneRect().size().toSize(), qtg.QImage.Format_ARGB32)
      image.fill(qtc.Qt.transparent)
      painter = qtg.QPainter(image)
    self.scene.render(painter)
    if not filename.endswith('.svg'):
      image.save(filename,quality=100)
    del painter

  def contextMenuEvent(self,event):
    """ An event handler triggered when the user right-clicks on this view that
        will force the context menu to appear.
        @ In, event, a QContextMenuEvent specifying the context of this event.
    """
    self.rightClickMenu.popup(event.globalPos())

  def resizeEvent(self,event):
    """ An event handler triggered when the user resizes this view.
        @ In, event, a QResizeEvent specifying the context of this event.
    """
    super(FitnessView, self).resizeEvent(event)
    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())
    self.updateScene()

  def selectionChanged(self):
    """ An event handler triggered when the user changes the selection of the
        data.
    """
    self.updateScene()

  def persistenceChanged(self):
    """ An event handler triggered when the user changes the persistence setting
        of the data.
    """
    self.updateScene()

  def modelsChanged(self):
    """ An event handler triggered when the user requests a new set of local
        models.
    """
    self.updateScene()

  def updateScene(self):
    """ A method for drawing the scene of this view.
    """
    self.scene.clear()

    if self.fillAction.isChecked():
      self.scene.setSceneRect(0,0,100*float(self.gView.width())/float(self.gView.height()),100)
    else:
      self.scene.setSceneRect(0,0,100,100)

    width = self.scene.width()
    height = self.scene.height()
    plotWidth = width - 2*self.padding
    plotHeight = height - 2*self.padding

    axisPen = qtg.QPen(qtc.Qt.black)
    names = self.amsc.GetNames()[:-1]

    if not self.amsc.FitsSynced():
      txtItem = self.scene.addSimpleText('Rebuild Local Models',self.font)
      txtItem.setFlag(qtw.QGraphicsItem.ItemIgnoresTransformations)
      txtItem.setPos(0,0)
      txtItem.setFlag(qtw.QGraphicsItem.ItemIsMovable)
      txtItem.setFlag(qtw.QGraphicsItem.ItemIsSelectable)
      self.scene.changed.connect(self.scene.invalidate)
      self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)
      return

    selection = self.amsc.GetSelectedSegments()
    colorMap = self.amsc.GetColors()

    ## Check if they selected any extrema
    if selection is None or len(selection) == 0:
      selection = []
      selectedExts = self.amsc.GetSelectedExtrema()
      allSegments = self.amsc.GetCurrentLabels()
      for minMaxPair in allSegments:
        for extIdx in selectedExts:
          if extIdx in minMaxPair:
            selection.append(minMaxPair)
      ## Okay, well then we will just plot everything we have for the current
      ## level
      if len(selection) == 0:
        selection = allSegments

    selectionCount = len(selection)

    if selectionCount > 0:
      axisHeight = plotHeight/float(selectionCount)
      axisWidth = plotWidth/float(selectionCount)
    dimCount = len(names)

    fitErrorData = {}

    for j,extPair in enumerate(selection):
      fitErrorData[extPair] = self.amsc.ComputePerDimensionFitErrors(extPair)

    maxValue = 1

    j = 0
    for extPair in selection:
      retValue = fitErrorData[extPair]
      if retValue is not None:
        indexOrder,rSquared,fStatistic = retValue

      myColor = colorMap[extPair]
      myPen = qtg.QPen(qtg.QColor('#000000'))
      brushColor = qtg.QColor(myColor)
      brushColor.setAlpha(127)
      myBrush = qtg.QBrush(brushColor)

      vals = rSquared

      w = axisWidth / dimCount
      self.font.setPointSizeF(np.clip(w-2*self.padding,2,18))
      for i,val in enumerate(vals):
        name = names[indexOrder[i]]
        if val > 0:
          barExtent = (val/maxValue)*plotHeight
        else:
          barExtent = 0
        x = j*axisWidth + i*axisWidth/float(dimCount)+self.padding
        y = height-self.padding
        h = -barExtent
        if self.showNumberAction.isChecked():
          numTxtItem = self.scene.addSimpleText('%.3g' % val, self.font)
          fm = qtg.QFontMetrics(numTxtItem.font())
          fontHeight = fm.height()
          fontWidth = fm.width(numTxtItem.text())

          numTxtItem.setPos(x+(w-fontHeight)/2.,y-plotHeight+fontWidth)
          #numTxtItem.rotate(285) #XXX not in qt5
          numTxtItem.setFlag(qtw.QGraphicsItem.ItemIsMovable)
          numTxtItem.setFlag(qtw.QGraphicsItem.ItemIsSelectable)
          numTxtItem.setZValue(2)

        myRect = self.scene.addRect(x,y,w,h,myPen,myBrush)
        myRect.setToolTip(str(val))
        myRect.setAcceptHoverEvents(True)

        txtItem = self.scene.addSimpleText(' ' + name,self.font)
        fm = qtg.QFontMetrics(txtItem.font())
        fontHeight = fm.height()
        fontWidth = fm.width(name)
        txtItem.setPos(x+(w-fontHeight)/2.,y)
        #txtItem.rotate(270) #XXX not in qt5
        txtItem.setFlag(qtw.QGraphicsItem.ItemIsMovable)
        txtItem.setFlag(qtw.QGraphicsItem.ItemIsSelectable)
        txtItem.setZValue(2)

      x = j*axisWidth+self.padding
      y = height-self.padding
      w = axisWidth
      h = -plotHeight
      self.scene.addRect(x,y,w,h,axisPen)
      j += 1

      self.scene.changed.connect(self.scene.invalidate)
      self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)

  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations. For this class in particular, we will test:
        - Building the models (which allows the actual plot to be displayed)
        - Saving the view buffer in svg and png formats.
        - Triggering the resize event.
        @ In, None
        @ Out, None
    """
    self.amsc.BuildModels()
    self.amsc.ClearSelection()
    self.saveImage(self.windowTitle()+'.svg')
    self.saveImage(self.windowTitle()+'.png')
    self.resizeEvent(qtg.QResizeEvent(qtc.QSize(1,1),qtc.QSize(100,100)))
    super(FitnessView, self).test()
