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
   A view widget for visualizing the sensitivity coefficients of each locally
   constructed model of the data.
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

## TODO: Fix the fonts
## TODO: Make scale make sense
## TODO: Place labels better
class SensitivityView(BaseTopologicalView):
  """
     A view widget for visualizing the sensitivity coefficients of each locally
     constructed model of the data.
  """
  def __init__(self, parent=None, amsc=None, title=None):
    """ Initialization method that can optionally specify the parent widget,
        an AMSC object to reference, and a title for this widget.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, amsc, an optional AMSC_Object specifying the underlying data
          object for this widget to use.
        @ In, title, an optional string specifying the title of this widget.
    """
    super(SensitivityView, self).__init__(parent,amsc,title)

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
    self.shapeMenu = qtw.QMenu('Layout')
    self.shapeGroup = qtw.QActionGroup(self.shapeMenu)
    self.rightClickMenu.addMenu(self.shapeMenu)
    shapeActions = []
    shapeActions.append(self.shapeMenu.addAction('Horizontal Bar'))
    shapeActions.append(self.shapeMenu.addAction('Radial'))
    for act in shapeActions:
      act.setCheckable(True)
      self.shapeGroup.addAction(act)
    shapeActions[0].setChecked(True)
    self.shapeGroup.triggered.connect(self.updateScene)

    ## Ba da ba ba ba I'm lovin' it
    self.valueMenu = qtw.QMenu('Value to Display')
    self.valueGroup = qtw.QActionGroup(self.valueMenu)
    self.rightClickMenu.addMenu(self.valueMenu)
    valueActions = []
    valueActions.append(self.valueMenu.addAction('Linear coefficients'))
    valueActions.append(self.valueMenu.addAction('Pearson correlation'))
    valueActions.append(self.valueMenu.addAction('Spearman rank correlation'))
    for act in valueActions:
      act.setCheckable(True)
      self.valueGroup.addAction(act)
    valueActions[0].setChecked(True)
    self.valueGroup.triggered.connect(self.updateScene)

    self.showLabelsAction = self.rightClickMenu.addAction('Show Labels')
    self.showLabelsAction.setCheckable(True)
    self.showLabelsAction.setChecked(True)
    self.showLabelsAction.triggered.connect(self.updateScene)

    self.showNumberAction = self.rightClickMenu.addAction('Show Numeric Values')
    self.showNumberAction.setCheckable(True)
    self.showNumberAction.setChecked(True)
    self.showNumberAction.triggered.connect(self.updateScene)

    self.bundledAction = self.rightClickMenu.addAction('Bundled on Dimension')
    self.bundledAction.setCheckable(True)
    self.bundledAction.setChecked(False)
    self.bundledAction.triggered.connect(self.updateScene)

    self.signedAction = self.rightClickMenu.addAction('Signed')
    self.signedAction.setCheckable(True)
    self.signedAction.setChecked(True)
    self.signedAction.triggered.connect(self.updateScene)

    self.fillAction = self.rightClickMenu.addAction('Fill viewport')
    self.fillAction.setCheckable(True)
    self.fillAction.setChecked(True)
    self.fillAction.triggered.connect(self.updateScene)

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
      painter = qtg.QPainter (svgGen)
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
    super(SensitivityView, self).resizeEvent(event)
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

  def layoutRadialScene(self):
    """ A convenience method for drawing the sensitivity scene in radial fashion
    """
    self.scene.clear()
    width = self.scene.width()
    height = self.scene.height()
    minDim = min([width,height])-24. # 12 point font * 2 for top and bottom,
                                     ## width is a bit harder...

    centerX = width/2.
    centerY = height/2.
    radius = minDim/2.

    axisPen = qtg.QPen(qtc.Qt.black)

    self.scene.addEllipse(centerX - radius, centerY - radius, minDim, \
                          minDim, axisPen)
    names = self.amsc.GetNames()[:-1]

    for i,name in enumerate(names):
      if len(names) <= 2:
        theta = 3*math.pi*float(i)/2.
      else:
        theta = 2*math.pi*float(i)/float(len(names))
      endX = radius*math.cos(theta)+centerX
      endY = radius*math.sin(theta)+centerY
      self.scene.addLine(centerX,centerY,endX,endY,axisPen)
      if self.showLabelsAction.isChecked():
        txtItem = self.scene.addSimpleText(name,self.font)
        txtItem.setPos(endX,endY)
        txtItem.setFlag(qtw.QGraphicsItem.ItemIsMovable)
        txtItem.setFlag(qtw.QGraphicsItem.ItemIsSelectable)
        txtItem.setFlag(qtw.QGraphicsItem.ItemIgnoresTransformations)

    selection = self.amsc.GetSelectedSegments()
    colorMap = self.amsc.GetColors()
    if self.valueGroup.checkedAction().text() == 'Linear coefficients':
      fits = self.amsc.SegmentFitCoefficients()
    elif self.valueGroup.checkedAction().text() == 'Pearson correlation':
      fits = self.amsc.SegmentPearsonCoefficients()
    elif self.valueGroup.checkedAction().text() == 'Spearman rank correlation':
      fits = self.amsc.SegmentSpearmanCoefficients()

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

    if self.valueGroup.checkedAction().text() == 'Linear coefficients':
      maxValue = 0
      for extPair in selection:
        if maxValue < max(map(abs,fits[extPair])):
          maxValue = max(map(abs,fits[extPair]))
    else:
      maxValue = 1

    for extPair in selection:
      myColor = colorMap[extPair]
      myPen = qtg.QPen(qtg.QColor('#000000'))
      brushColor = qtg.QColor(myColor)
      brushColor.setAlpha(127)
      myBrush = qtg.QBrush(brushColor)
      myPoly = qtg.QPolygonF()
      for i,val in enumerate(map(abs,fits[extPair])):
        if len(names) <= 2:
          theta = 3*math.pi*float(i)/2.
        else:
          theta = 2*math.pi*float(i)/float(len(names))
        dimX = (val/maxValue)*radius*math.cos(theta)+centerX
        dimY = (val/maxValue)*radius*math.sin(theta)+centerY
        myPoly.append(qtc.QPointF(dimX,dimY))
      if len(names) <= 2:
        myPoly.append(qtc.QPointF(centerX,centerY))
      self.scene.addPolygon(myPoly,myPen,myBrush)

  def layoutBarScene(self):
    """ A convenience method for drawing the sensitivity scene in bar fashion.
    """
    self.scene.clear()

    width = self.scene.width()
    height = self.scene.height()

    plotWidth = width - 2*self.padding
    plotHeight = height - 2*self.padding

    maxExtent = plotWidth

    axisPen = qtg.QPen(qtc.Qt.black)
    names = self.amsc.GetNames()[:-1]

    selection = self.amsc.GetSelectedSegments()
    colorMap = self.amsc.GetColors()
    if self.valueGroup.checkedAction().text() == 'Linear coefficients':
      fits = self.amsc.SegmentFitCoefficients()
    elif self.valueGroup.checkedAction().text() == 'Pearson correlation':
      fits = self.amsc.SegmentPearsonCoefficients()
    elif self.valueGroup.checkedAction().text() == 'Spearman rank correlation':
      fits = self.amsc.SegmentSpearmanCoefficients()

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

    if self.valueGroup.checkedAction().text() == 'Linear coefficients':
      maxValue = 0
      for extPair in selection:
        if maxValue < max(map(abs,fits[extPair])):
          maxValue = max(map(abs,fits[extPair]))
    else:
      maxValue = 1

    if self.bundledAction.isChecked():
      axisHeight = plotHeight/float(len(names))
      axisWidth = plotWidth/float(len(names))

      for j,extPair in enumerate(selection):
        myColor = colorMap[extPair]
        myPen = qtg.QPen(qtg.QColor('#000000'))
        brushColor = qtg.QColor(myColor)
        brushColor.setAlpha(127)
        myBrush = qtg.QBrush(brushColor)
        for i,val in enumerate(fits[extPair]):
          absVal = abs(val)
          barExtent = (absVal/maxValue)*maxExtent
          if self.signedAction.isChecked():
            x = self.padding + maxExtent/2.
            if val > 0:
              w = barExtent/2.
            else:
              w = -barExtent/2.
          else:
            x = self.padding
            w = barExtent
          y = (height-self.padding) - i*axisHeight \
              - j*axisHeight/float(len(selection))
          h = -axisHeight / float(len(selection))
          if self.showNumberAction.isChecked():
            numTxtItem = self.scene.addSimpleText('%.3g' % val, self.font)
            numTxtItem.setFlag(qtw.QGraphicsItem.ItemIgnoresTransformations)
            fm = qtg.QFontMetrics(numTxtItem.font())
            fontWidth = fm.width(numTxtItem.text())
            numTxtItem.setPos(self.padding+maxExtent-fontWidth,y+h)
            numTxtItem.setFlag(qtw.QGraphicsItem.ItemIsMovable)
            numTxtItem.setFlag(qtw.QGraphicsItem.ItemIsSelectable)
            numTxtItem.setZValue(2)
          myRect = self.scene.addRect(x,y,w,h,myPen,myBrush)
          myRect.setToolTip(str(val))
          myRect.setAcceptHoverEvents(True)
      for i,name in enumerate(names):
        x = self.padding
        y = height - self.padding - i/float(len(names))*plotHeight
        w = plotWidth
        h = -axisHeight
        if self.showLabelsAction.isChecked():
          txtItem = self.scene.addSimpleText(name,self.font)
          txtItem.setFlag(qtw.QGraphicsItem.ItemIgnoresTransformations)
          fm = qtg.QFontMetrics(txtItem.font())
          fontHeight = fm.height()
          fontWidth = fm.width(txtItem.text())
          txtItem.setPos(self.padding-fontWidth,y+h + (axisHeight-fontHeight)/2.)
          txtItem.setFlag(qtw.QGraphicsItem.ItemIsMovable)
          txtItem.setFlag(qtw.QGraphicsItem.ItemIsSelectable)
          txtItem.setZValue(2)
        myRect = self.scene.addRect(x,y,w,h,axisPen)
        myRect.setZValue(2) # Any value greater than 1 should work to draw on top
    else:
      if len(selection) > 0:
        axisHeight = plotHeight/float(len(selection))
        axisWidth = plotWidth/float(len(selection))
      dimCount = len(names)

      self.font.setPointSizeF(np.clip(axisHeight/float(dimCount)-2*self.padding,2,18))
      for j,extPair in enumerate(selection):
        myColor = colorMap[extPair]
        myPen = qtg.QPen(qtg.QColor('#000000'))
        brushColor = qtg.QColor(myColor)
        brushColor.setAlpha(127)
        myBrush = qtg.QBrush(brushColor)
        for i,val in enumerate(fits[extPair]):
          absVal = abs(val)
          name = names[i]
          barExtent = (absVal/maxValue)*maxExtent
          if self.signedAction.isChecked():
            x = self.padding + maxExtent/2.
            if val > 0:
              w = barExtent/2.
            else:
              w = -barExtent/2.
          else:
            x = self.padding
            w = barExtent
          y = (height-self.padding) - j*axisHeight \
              - i*axisHeight/float(dimCount)
          h = -axisHeight / float(dimCount)

          if self.showLabelsAction.isChecked():
            txtItem = self.scene.addSimpleText(name,self.font)
            ## this line can be useful for text sizing, although we cannot
            ## rotate the text if we ignore the transformations.
            # txtItem.setFlag(qtw.QGraphicsItem.ItemIgnoresTransformations)
            fm = qtg.QFontMetrics(txtItem.font())
            fontHeight = fm.boundingRect(txtItem.text()).height()
            fontWidth = fm.boundingRect(txtItem.text()).width()
            txtItem.setPos(self.padding,y+0.5*(h-fontHeight))
            txtItem.setFlag(qtw.QGraphicsItem.ItemIsMovable)
            txtItem.setFlag(qtw.QGraphicsItem.ItemIsSelectable)
            txtItem.setZValue(2)
          if self.showNumberAction.isChecked():
            numTxtItem = self.scene.addSimpleText('%.3g' % val, self.font)
            ## this line can be useful for text sizing, although we cannot
            ## rotate the text if we ignore the transformations.
            # numTxtItem.setFlag(qtw.QGraphicsItem.ItemIgnoresTransformations)
            fm = qtg.QFontMetrics(numTxtItem.font())
            fontWidth = fm.boundingRect(numTxtItem.text()).width()
            fontHeight = fm.boundingRect(numTxtItem.text()).height()
            numTxtItem.setPos(self.padding+maxExtent-fontWidth,y+0.5*(h-fontHeight))
            numTxtItem.setFlag(qtw.QGraphicsItem.ItemIsMovable)
            numTxtItem.setFlag(qtw.QGraphicsItem.ItemIsSelectable)
            numTxtItem.setZValue(2)
          myRect = self.scene.addRect(x,y,w,h,myPen,myBrush)
          myRect.setToolTip(str(val))
          myRect.setAcceptHoverEvents(True)

        x = self.padding
        y = (height-self.padding) - j*axisHeight
        w = maxExtent
        h = -axisHeight
        myRect = self.scene.addRect(x,y,w,h,axisPen)
        myRect.setZValue(2) # Any value greater than 1 should work to draw on top

    if self.signedAction.isChecked():
      axisPen = qtg.QPen(qtc.Qt.black)
      axisPen.setWidthF(.5)
      x = self.padding + maxExtent/2.
      y = self.padding
      h = plotHeight
      self.scene.addLine(x,y,x,y+h,axisPen)

  def updateScene(self):
    """ A method for drawing the scene of this view.
    """
    if not self.amsc.FitsSynced():
      self.scene.setSceneRect(0,0,self.gView.width(),self.gView.height())
      self.scene.clear()
      txtItem = self.scene.addSimpleText('Rebuild Local Models',self.font)
      txtItem.setPos(self.padding,self.padding)
      txtItem.setFlag(qtw.QGraphicsItem.ItemIgnoresTransformations)
    else:
      if self.fillAction.isChecked():
        self.scene.setSceneRect(0,0,100*float(self.gView.width())/float(self.gView.height()),100)
      else:
        self.scene.setSceneRect(0,0,100,100)

      if self.shapeGroup.checkedAction().text() == 'Radial':
        self.bundledAction.setEnabled(False)
        self.signedAction.setEnabled(False)
        self.showNumberAction.setEnabled(False)
        self.fillAction.setEnabled(False)
        self.layoutRadialScene()
      else:
        self.bundledAction.setEnabled(True)
        self.signedAction.setEnabled(True)
        self.showNumberAction.setEnabled(True)
        self.fillAction.setEnabled(True)
        self.layoutBarScene()
    self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)
    self.scene.changed.connect(self.scene.invalidate)


  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations. For this class in particular, we will test:
        - Building of the models (which allows for the actual display of
          information on this view)
        - Cylcing through all permutations of the display features which
         includes the radial/bar layouts, the bundling of dimensions or segments
         of data, the display of signed or unsigned information, and whether
         the plot fills the viewport or maintains a square aspect ratio.
        - Setting the selection of data and ensuring this view updates.
        - Saving buffer of this view in both svg and png formats.
        - Triggering of the resize event.
        @ In, None
        @ Out, None
    """
    self.amsc.BuildModels()
    for action in self.shapeGroup.actions():
      action.setChecked(True)
      for value in self.valueGroup.actions():
        value.setChecked(True)
        self.amsc.ClearSelection()

        self.signedAction.setChecked(True)
        self.bundledAction.setChecked(True)
        self.fillAction.setChecked(True)
        self.updateScene()
        self.signedAction.setChecked(False)
        self.bundledAction.setChecked(False)
        self.fillAction.setChecked(False)
        self.updateScene()
        pair = list(self.amsc.GetCurrentLabels())[0]
        self.amsc.SetSelection([pair,pair[0],pair[1]])
        self.updateScene()

    self.saveImage(self.windowTitle()+'.svg')
    self.saveImage(self.windowTitle()+'.png')
    self.resizeEvent(qtg.QResizeEvent(qtc.QSize(1,1),qtc.QSize(100,100)))

    super(SensitivityView, self).test()
