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
  A view that shows the topological skeleton, that is, which minima are
  connected to which maxima.  Point locations are determined by persistence
  on the horizontal axis and function value on the y-axis, so that minima
  will tend to occur at the bottom and maxima will occur toward the top. The
  point size of extrema is determined by how many samples flow toward them.
  Color is used to distinguish maxima from minima.

  Lines connecting maxima and minima are given a width based on the number
  of samples occurring in that flow segment.
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

import numpy as np
import math
import time
from . import colors

class CustomGraphicsView(qtw.QGraphicsView):
  """
     A subclass of QGraphicsView where we custom the handling of mouse
     events and drawing of selected items.
  """
  def mousePressEvent(self, event):
    """ An event handler triggered when the user presses a mouse button on this
        view.
        @ In, event, a QMouseEvent specifying the context of this event.
    """
    if event.buttons() != qtc.Qt.MiddleButton:
      super(CustomGraphicsView, self).mousePressEvent(event)
    else:
      self.parent().mousePressEvent(event,True)

  def mouseMoveEvent(self, event):
    """ An event handler triggered when the user moves the mouse on this view.
        @ In, event, a QMouseEvent specifying the context of this event.
    """
    if event.buttons() != qtc.Qt.MiddleButton:
      super(CustomGraphicsView, self).mouseMoveEvent(event)
    else:
      self.parent().mouseMoveEvent(event,True)

  def mouseReleaseEvent(self, event):
    """ An event handler triggered when the user releases a mouse button on this
        view.
        @ In, event, a QMouseEvent specifying the context of this event.
    """
    if event.buttons() != qtc.Qt.MiddleButton:
      super(CustomGraphicsView, self).mouseReleaseEvent(event)
    else:
      self.parent().mouseReleaseEvent(event,True)

class CustomPathItem(qtw.QGraphicsPathItem):
  """
     A subclass of QGraphicsPathItem where we custom the drawing.
  """
  def __init__(self, path, parent=None, scene=None,data=None):
    """ Initialization method specifying the path to store and optionally
        specifies the parent widget, the scene, and some extra data for the
        contextual tooltip window.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, scene, an optional QGraphicsScene specifying the scene to which
          this object will be attached.
        @ In, data, a dictionary of data for this graphical item to display
          in its tooltip.
    """
    #super(CustomPathItem, self).__init__(path,parent,scene)
    super(CustomPathItem, self).__init__(path,parent)
    self.graphics = []
    if scene is not None:
      scene.addItem(self)
    self.tipSize = qtc.QSize(0,0)

    font = qtg.QFont('Courier New',2)
    fm = qtg.QFontMetrics(font)
    fontHeight = fm.height()

    dataNames = []
    if data is not None and 'amsc' in data:
      dataNames = data['amsc'].GetNames()
    partition = data['amsc'].Partitions()[data['key']]

    padding = 1
    self.tipSize = qtc.QSize(30,fontHeight*(len(dataNames)+1)+2*padding)
    xPos = padding
    yPos = padding+2

    cBox = qtg.QPainterPath() # context pop-up box
    cBox.addRoundedRect(0,0,self.tipSize.width(),self.tipSize.height(),1,1)
    gBox = self.scene().addPath(cBox)
    gBox.setVisible(False)
    self.graphics.append((cBox,gBox))

    # We can add more graphics to the tooltip by following this template,
    ##  where you should specify a pen and brush unlike the example above which
    ##  will be modified according to the set pen of the particular object.
    cKey = qtg.QPainterPath() # contextual pop-up text
    if data is not None and 'key' in data:
      text = str(data['key']) + ' Count: ' + str(len(partition))
      fontWidth = fm.width(text)
      ## Center the text on the tooltip
      cKey.addText(qtc.QPointF((self.tipSize.width()-fontWidth)/2.,yPos),font,text)
    gText = self.scene().addPath(cKey)
    gText.setPen(qtg.QPen(qtg.QColor('#333333')))
    gText.setBrush(qtg.QBrush(qtg.QColor('#333333')))
    gText.setVisible(False)
    self.graphics.append((cKey,gText))

    yPos += fontHeight
    ##  End template

    for i,name in enumerate(dataNames):
      if i == len(dataNames)-1:
        vals = data['amsc'].GetY(partition)
        minVal = min(data['amsc'].GetY())
        maxVal = max(data['amsc'].GetY())
      else:
        vals = data['amsc'].GetX(partition,i)
        minVal = min(data['amsc'].GetX(None,i))
        maxVal = max(data['amsc'].GetX(None,i))

      if len(name) > 7:
        label = name[:3] + u"\u2026" + name[-3:]
      else:
        label = name
      ## Ensure everything is 7 characters wide
      while len(label) < 7:
        label = ' ' + label
      label += ' '
      cText = qtg.QPainterPath() # contextual pop-up text
      cText.addText(qtc.QPointF(padding,yPos),font,label)
      gText = self.scene().addPath(cText)
      gText.setPen(qtg.QPen(qtg.QColor('#333333')))
      gText.setBrush(qtg.QBrush(qtg.QColor('#333333')))
      gText.setVisible(False)
      self.graphics.append((cText,gText))

      xPos = padding + fm.width(label)

      boxX = xPos
      boxY = yPos - 2
      boxW = self.tipSize.width()-xPos-padding
      boxH = fontHeight

      cDimBox = qtg.QPainterPath() # contextual pop-up text
      cDimBox.addRect(boxX,boxY,boxW,boxH)
      gDimBox = self.scene().addPath(cDimBox)
      gDimBox.setPen(qtg.QPen(qtg.QColor('#333333')))
      gDimBox.setBrush(qtg.QBrush(qtg.QColor('#FFFFFF')))
      gDimBox.setVisible(False)
      self.graphics.append((cDimBox,gDimBox))

      cDimSpan = qtg.QPainterPath() # contextual pop-up text
      x = (boxW)/(maxVal-minVal)*(min(vals)-minVal) + boxX
      y = boxY
      w = (boxW)/(maxVal-minVal)*(max(vals)-minVal) + boxX - x
      h = boxH
      cDimSpan.addRect(x,y,w,h)
      gDimSpan = self.scene().addPath(cDimSpan)
      gDimSpan.setPen(qtg.QPen(qtg.QColor('#333333')))
      gDimSpan.setBrush(qtg.QBrush(qtg.QColor('#CCCCCC')))
      gDimSpan.setVisible(False)
      self.graphics.append((cDimSpan,gDimSpan))

      ##########################################################################
      ## Histogram Sparkline

      if i == len(dataNames)-1:
        cDimHist = qtg.QPainterPath() # contextual pop-up text
        hist,binEdges = np.histogram(vals)
        maxCount = max(hist)
        for j,bin in enumerate(hist):
          x = (boxW)/(maxVal-minVal)*(binEdges[j]-minVal) + boxX
          w = (boxW)/(maxVal-minVal)*(binEdges[j+1]-minVal) + boxX - x
          h = float(bin)/float(maxCount)*boxH
          y = boxY+boxH - h
          cDimHist.addRect(x,y,w,h)
        gDimHist = self.scene().addPath(cDimHist)
        gDimHist.setPen(qtg.QPen(qtc.Qt.NoPen)) #qtg.QPen(qtg.QColor('#CCCCCC')))
        gDimHist.setBrush(qtg.QBrush(qtg.QColor('#333333')))
        gDimHist.setVisible(False)
        self.graphics.append((cDimHist,gDimHist))

      ##########################################################################
      ## Scatter Sparkline
      else:
        cDimScatter = qtg.QPainterPath() # contextual pop-up text
        Xs = vals
        Ys = data['amsc'].GetY(partition)
        maxY = max(Ys)
        minY = min(Ys)
        for xVal,yVal in zip(Xs,Ys):
          x = (boxW)/(maxVal-minVal)*(xVal-minVal) + boxX
          y = boxY+boxH - boxH/(maxY-minY)*(yVal-minY)
          w = .2
          h = .2
          cDimScatter.addRect(x,y,w,h)
        gDimScatter = self.scene().addPath(cDimScatter)
        gDimScatter.setPen(qtg.QPen(qtc.Qt.NoPen)) #qtg.QPen(qtg.QColor('#CCCCCC')))
        gDimScatter.setBrush(qtg.QBrush(qtg.QColor('#333333')))
        gDimScatter.setVisible(False)
        self.graphics.append((cDimScatter,gDimScatter))

      ##########################################################################

      yPos += fontHeight


    ## This stuff will be performed on all of the tooltip elements
    for path,graphic in self.graphics:
      graphic.setZValue(10)

  def paint(self,painter,option,widget=None):
    """
      A method for painting this item
      @ In, painter, QPainter, the painter responsible for drawing this thing
      @ In, option, QStyleOptionGraphicsItem, the option parameter provides
        style options for the item, such as its state, exposed area and its
        level-of-detail hints (quoted from PySide documentation)
      @ In, widget, QWidget, the widget attached to this.
      @ Out, None
    """
    if self.isSelected():
      selectedOption = qtw.QStyleOptionGraphicsItem(option)
      selectedOption.state &=  (not qtw.QStyle.State_Selected)
      originalPen = self.pen()
      selectedPen = qtg.QPen(originalPen)
      selectedPen.setDashPattern([2,1])
      selectedPen.setWidthF(originalPen.widthF()*1.5)
      self.setPen(selectedPen)
      super(CustomPathItem,self).paint(painter, selectedOption, widget)
      self.setPen(originalPen)
    else:
      super(CustomPathItem,self).paint(painter,option,widget)

  def shape(self):
    """ Returns the shape of this item as a PySide.QtGui.QPainterPath in local
        coordinates. The shape is used for many things, including collision
        detection, hit tests, and for the QGraphicsScene.items() functions.
        This subclass reimplements this function to return a more accurate
        shape than the default bounding box.
        @ In, None
        @ Out, QPainterPath speciyfing the shape of this object in local
          coordinates.
    """
    width = self.pen().widthF()

    downShape = qtg.QPainterPath(self.path())
    upShape = qtg.QPainterPath(self.path())
    downShape.translate(0,-width)
    upShape.translate(0,width)
    downShape.connectPath(upShape.toReversed())

    leftShape = qtg.QPainterPath(self.path())
    rightShape = qtg.QPainterPath(self.path())
    leftShape.translate(-width,0)
    rightShape.translate(width,0)
    leftShape.connectPath(rightShape.toReversed())
    return downShape.united(leftShape)

  def hoverEnterEvent(self,event):
    """ Event handler should enable the contextual tooltip when the mouse is
        on top of this item.
        @ In, event, a QGraphicsSceneHoverEvent specifying the context of this
        event.
        @ Out, None
    """
    scene = self.scene()
    mouse = event.scenePos()
    tipX = np.clip(mouse.x(),0,scene.width() - self.tipSize.width())
    tipY = np.clip(mouse.y(),0,scene.height() - self.tipSize.height())

    for path,graphic in self.graphics:
      graphic.setPath(path.translated(tipX,tipY))
      graphic.setVisible(True)

  def hoverMoveEvent(self,event):
    """ Event handler should update the position of the contextual tooltip when
        the mouse is on top of this item, and disable it otherwise.
        @ In, event, a QGraphicsSceneHoverEvent specifying the context of this
        event.
        @ Out, None
    """
    scene = self.scene()
    mouse = event.scenePos()
    tipX = np.clip(mouse.x(),0,scene.width() - self.tipSize.width())
    tipY = np.clip(mouse.y(),0,scene.height() - self.tipSize.height())

    for path,graphic in self.graphics:
      graphic.setPath(path.translated(tipX,tipY))

  def hoverLeaveEvent(self,event):
    """ Event handler should disable the contextual tooltip when the mouse
        leaves this item.
        @ In, event, a QGraphicsSceneHoverEvent specifying the context of this
        event.
        @ Out, None
    """
    #Hide the popup widget
    if not self.isSelected():
      for path,graphic in self.graphics:
        graphic.setVisible(False)
        graphic.setPath(path)

  def setPen(self,pen):
    """ Sets the pen for this item to pen. The pen is used to draw the item's
        outline.
        @ In, pen, a QPen specifying the drawing characteristics of this path.
        @ Out, None
    """
    super(CustomPathItem,self).setPen(pen)
    bgColor = pen.color()
    bgColor.setAlpha(50)
    for i in range(4,len(self.graphics),4):
      self.graphics[i][1].setBrush(qtg.QBrush(pen.color().lighter()))
      self.graphics[i+1][1].setBrush(qtg.QBrush(pen.color().darker()))

  def itemChange(self, change, value):
    """ This virtual function is called by PySide.QtGui.QGraphicsItem to notify
        custom items that some part of the item's state changes.
        @ In, change, a GraphicsItemChange object speciyifing the parameter of
          the item that is changing.
        @ In, value, object specifying the new value of the parameter where the
          type will depend on change.
        @ Out, object specifying the new state (default is to return value).

    """
    if change == qtw.QGraphicsItem.ItemSceneHasChanged:
      for graphic in self.graphics:
        if graphic not in self.scene().items():
          self.scene().addItem(graphic)
    return super(CustomPathItem,self).itemChange(change,value)

#TODO We can encode glyphs on the extrema to denote spatial location?
#TODO We can encode information on the darkness of the boundary on the points
#     and the lines
class TopologyMapView(BaseTopologicalView):
  """
      A view that shows the topological skeleton, that is, which minima are
      connected to which maxima.  Point locations are determined by persistence
      on the horizontal axis and function value on the y-axis, so that minima
      will tend to occur at the bottom and maxima will occur toward the top. The
      point size of extrema is determined by how many samples flow toward them.
      Color is used to distinguish maxima from minima.

      Lines connecting maxima and minima are given a width based on the number
      of samples occurring in that flow segment.
  """
  polygonMap = {}
  def __init__(self, parent=None, amsc=None, title=None):
    """ Initialization method that can optionally specify the parent widget,
        an AMSC object to reference, and a title for this widget.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, amsc, an optional AMSC_Object specifying the underlying data
          object for this widget to use.
        @ In, title, an optional string specifying the title of this widget.
    """
    super(TopologyMapView, self).__init__(parent,amsc,title)

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
      self.setLayout(qtw.QGridLayout())
    layout = self.layout()
    self.clearLayout(layout)

    self.scene = qtw.QGraphicsScene()
    self.scene.setSceneRect(0,0,100,100)
    self.gView = CustomGraphicsView(self.scene)
    self.gView.setParent(self)
    self.gView.setRenderHints(qtg.QPainter.Antialiasing |
                              qtg.QPainter.SmoothPixmapTransform)
    self.gView.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.gView.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.gView.setDragMode(qtw.QGraphicsView.RubberBandDrag)
    self.scene.selectionChanged.connect(self.select)

    mergeSequence = self.amsc.GetMergeSequence()
    pCount = len(set([p for idx,(parent,p) in mergeSequence.items()]))-1

    self.rightClickMenu = qtw.QMenu()
    persAction = self.rightClickMenu.addAction('Set Persistence Here')
    persAction.triggered.connect(self.setPersistence)
    incAction = self.rightClickMenu.addAction('Increase Persistence')
    incAction.triggered.connect(self.increasePersistence)
    decAction = self.rightClickMenu.addAction('Decrease Persistence')
    decAction.triggered.connect(self.decreasePersistence)

    self.fillAction = self.rightClickMenu.addAction('Fill viewport')
    self.fillAction.setCheckable(True)
    self.fillAction.setChecked(True)
    self.fillAction.triggered.connect(self.updateScene)

    self.glyphMenu = self.rightClickMenu.addMenu('Extremum Glyphs')
    glyphActions = []
    glyphActions.append(self.glyphMenu.addAction('Triangle'))
    glyphActions.append(self.glyphMenu.addAction('Circle'))

    self.glyphGroup = qtw.QActionGroup(self.glyphMenu)
    for act in glyphActions:
      act.setCheckable(True)
      self.glyphGroup.addAction(act)
    glyphActions[0].setChecked(True)
    self.glyphGroup.triggered.connect(self.updateScene)

    self.colorAction = self.rightClickMenu.addAction('Colored Extrema')
    self.colorAction.setCheckable(True)
    self.colorAction.setChecked(False)
    self.colorAction.triggered.connect(self.updateScene)

    captureAction = self.rightClickMenu.addAction('Capture')
    captureAction.triggered.connect(self.saveImage)

    layout.addWidget(self.gView)
    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())

    persistences = [p for idx,(parent,p) in mergeSequence.items()]
    persistences = sorted(set(persistences))
    self.amsc.Persistence(persistences[-1])

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
      image.save(filename, quality=100)

    del painter

  def resizeEvent(self,event):
    """ An event handler triggered when the user resizes this view.
        @ In, event, a QResizeEvent specifying the context of this event.
    """
    super(TopologyMapView,self).resizeEvent(event)
    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())
    self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)
    self.updateScene()

  def setPersistence(self):
    """ A method for setting the persistence according to the mouse's horizontal
        position on the QGraphicsScene.
    """
    mergeSequence = self.amsc.GetMergeSequence()
    position = self.gView.mapFromGlobal(self.rightClickMenu.pos())
    mousePt = self.gView.mapToScene(position.x(),position.y()).x()
    persistences = [p for idx,(parent,p) in mergeSequence.items()]
    persistences = sorted(set(persistences))
    minP = 0
    maxP = max(persistences)

    ## TODO
    ## Inverse of the scaleToScene defined locally in the updateScene method
    ## These two functions should be promoted or I should find a better way to
    ## translate from persistence to screen coordinates
    def scaleFromScene(sx,sy):
      """
      Scales a point from scene coordinates to global (world) coordinates.
      @ In, sx, float, the scene x position
      @ In, sy, float, the scene y position
      @ Out, wx, float, the world x position
      @ Out, wy, float, the world y position
      """
      effectiveWidth = self.scene.width()-2*self.padding
      effectiveHeight = self.scene.height()-2*self.padding

      xBounds = (0,max(self.amsc.Y)-min(self.amsc.Y))
      yBounds = (min(self.amsc.Y),max(self.amsc.Y))

      tx = (sx - self.padding) / effectiveWidth
      ty = (self.padding + effectiveHeight - sy)/effectiveHeight

      wx = tx*float(xBounds[1]-xBounds[0]) + xBounds[0]
      wy = ty*float(yBounds[1]-yBounds[0]) + yBounds[0]

      return (wx,wy)

    persistence = np.clip(scaleFromScene(mousePt,0)[0],minP,maxP)
    self.amsc.Persistence(persistence)

    ################################################
    # width = self.scene.width()
    # height = self.scene.height()
    # minDim = min([width,height])
    # padding = 0.1*minDim/2.
    # effectiveWidth = width-2*padding
    # effectiveHeight = height-2*padding

    # def scaleToScene(wx,wy):
    #   xBounds = (0,max(self.amsc.Y)-min(self.amsc.Y))
    #   yBounds = (min(self.amsc.Y),max(self.amsc.Y))

    #   tx = float(wx-xBounds[0])/float(xBounds[1]-xBounds[0])
    #   ty = float(wy-yBounds[0])/float(yBounds[1]-yBounds[0])

    #   x = tx*effectiveWidth + padding
    #   y = effectiveHeight - ty*effectiveHeight + padding
    #   return (x,y)
    # (px,py) = scaleToScene(persistence,0)

    ################################################

  def increasePersistence(self):
    """ A method for increasing the persistence to the next coarsest level given
        the current setting.
    """
    mergeSequence = self.amsc.GetMergeSequence()
    persistences = [p for idx,(parent,p) in mergeSequence.items()]
    eps = max(persistences)*1e-6
    persistences = sorted(set(persistences))
    persistences.insert(0,0.)
    persistences.pop()

    currentP = self.amsc.Persistence()
    idx = 0
    while persistences[idx]+eps <= currentP and idx < len(persistences)-1:
      idx += 1
    self.amsc.Persistence(persistences[idx]+eps)

  def decreasePersistence(self):
    """ A method for decreasing the persistence to the next finest level given
        the current setting.
    """
    mergeSequence = self.amsc.GetMergeSequence()
    persistences = [p for idx,(parent,p) in mergeSequence.items()]
    eps = max(persistences)*1e-6
    persistences = sorted(set(persistences))
    persistences.insert(0,0.)
    persistences.pop()

    currentP = self.amsc.Persistence()
    idx = len(persistences)-1
    while persistences[idx]+eps >= currentP and idx > 0:
      idx -= 1
    self.amsc.Persistence(persistences[idx]+eps)

  def select(self):
    """ A method for passing this view's selected items to the underlying data
        structure.
    """
    selectedKeys = []
    for key,graphic in self.polygonMap.items():
      if graphic in self.scene.selectedItems():
        selectedKeys.append(key)
    self.amsc.SetSelection(selectedKeys)

  def contextMenuEvent(self,event):
    """ An event handler triggered when the user right-clicks on this view that
        will force the context menu to appear.
        @ In, event, a QContextMenuEvent specifying the context of this event.
    """
    self.rightClickMenu.popup(event.globalPos())

  def mousePressEvent(self, event, fromChild=False):
    """ An event handler triggered when the user presses a mouse button on this
        view. Will keep track of whether this event was initiated by this view's
        child or not.
        @ In, event, a QMouseEvent specifying the context of this event.
        @ In, fromChild, a boolean value specifying whether this request
          originated in a child of this object.
    """
    if fromChild:
      mousePt = self.gView.mapToScene(event.x(),event.y())
    else:
      pass

  def mouseMoveEvent(self, event, fromChild=False):
    """ An event handler triggered when the user moves the mouse on this view.
        Will keep track of whether this event was initiated by this view's
        child or not.
        @ In, event, a QMouseEvent specifying the context of this event.
        @ In, fromChild, a boolean value specifying whether this request
          originated in a child of this object.
    """
    if fromChild:
      if event.buttons() == qtc.Qt.MiddleButton:
        colorMap = self.amsc.GetColors()
        for extPair,graphic in self.polygonMap.items():
          if graphic in self.scene.selectedItems():
            if isinstance(graphic,qtw.QGraphicsPathItem):
              minLabel = extPair[0]
              maxLabel = extPair[1]

              fillColor = qtg.QColor(colorMap[extPair])
              fillColor.setAlpha(200)
              brush = qtg.QBrush(fillColor)

              xMin = self.extLocations[minLabel][0]
              yMin = self.extLocations[minLabel][1]
              xMax = self.extLocations[maxLabel][0]
              yMax = self.extLocations[maxLabel][1]

              mousePt = self.gView.mapToScene(event.x(),event.y())

              path = qtg.QPainterPath()
              path.moveTo(xMin, yMin)
              path.cubicTo(mousePt.x(), yMin, mousePt.x(), yMax,  xMax, yMax)
              graphic.setPath(path)
    else:
      pass

  def mouseReleaseEvent(self, event, fromChild=False):
    """ An event handler triggered when the user releases a mouse button on this
        view. Will keep track of whether this event was initiated by this view's
        child or not.
        @ In, event, a QMouseEvent specifying the context of this event.
        @ In, fromChild, a boolean value specifying whether this request
          originated in a child of this object.
    """
    if fromChild:
      pass
    else:
      pass
      # super(TopologyMapView, self).mouseReleaseEvent(event)

  def persistenceChanged(self):
    """ An event handler triggered when the user changes the persistence setting
        of the data.
    """
    persistence = self.amsc.Persistence()
    mergeSequence = self.amsc.GetMergeSequence()
    persistences = [p for idx,(parent,p) in mergeSequence.items()]
    persistences = sorted(set(persistences))
    persistences.insert(0,0.)
    persistences.pop()
    for i,pers in enumerate(persistences):
      if pers >= persistence:
        idx = i-1
        break
    self.updateScene()

  def updateScene(self):
    """ A method for drawing the scene of this view.
    """
    self.polygonMap = {}
    self.scene.clear()

    if self.fillAction.isChecked():
      self.scene.setSceneRect(0,0,100*float(self.gView.width())/float(self.gView.height()),100)
    else:
      self.scene.setSceneRect(0,0,100,100)

    width = self.scene.width()
    height = self.scene.height()
    self.scene.addRect(0,0,width,height,qtg.QPen(qtc.Qt.black))

    partitions = self.amsc.Partitions()
    mergeSequence = self.amsc.GetMergeSequence()
    colorMap = self.amsc.GetColors()

    #Set the points to have a diameter between 1-10% of the shortest dimension
    minDim = min([width,height])
    self.maxDiameter = 0.1*minDim
    self.minDiameter = 0.01*minDim

    self.padding = self.maxDiameter/2.
    effectiveWidth = width-2*self.padding
    effectiveHeight = height-2*self.padding
    # self.scene.addRect(padding,padding,effectiveWidth,effectiveHeight,qtg.QPen(qtc.Qt.black))
    def scaleDiameter(x):
      """
        Scales the diameter to an appropriate size
        @ In, x, float, unscaled diameter
        @ Out, diameter, float, scaled diameter
      """
      bounds = (0,self.amsc.X.shape[0])
      t = float(x-bounds[0])/float(bounds[1]-bounds[0])
      return (self.maxDiameter-self.minDiameter)*t + self.minDiameter

    ## TODO
    ## This function and its inverse (defined in SetPersistence) should be
    ##  promoted or I should find a better way to translate from persistence to
    ##  screen coordinates. This wouldn't be necessary except for the padding.
    def scaleToScene(wx,wy):
      """
        Function that converts global (world) coordinates into scene
        coordinates.
        @ In, wx, float, world x position
        @ In, wy, float, world y position

         @ Out, sx, float, scene x position
        @ Out, sy, float, scene y position
      """
      xBounds = (0,max(self.amsc.Y)-min(self.amsc.Y))
      yBounds = (min(self.amsc.Y),max(self.amsc.Y))

      tx = float(wx-xBounds[0])/float(xBounds[1]-xBounds[0])
      ty = float(wy-yBounds[0])/float(yBounds[1]-yBounds[0])

      x = tx*effectiveWidth + self.padding
      y = effectiveHeight - ty*effectiveHeight + self.padding
      return (x,y)

    ############################################################################
    ## Draw the background "active" persistence
    gray = qtg.QColor('#999999')
    transparentGray = gray.lighter()
    transparentGray.setAlpha(127)

    currentP = self.amsc.Persistence()
    persPen = qtg.QPen(gray)
    persBrush = qtg.QBrush(transparentGray)
    (px,py) = scaleToScene(currentP,0)
    self.scene.addRect(px,0,width-px,height,persPen,persBrush)

    ############################################################################
    ## First place all of the extrema appropriately
    self.extLocations = {}

    ## The minimum distances we will allow things to overlap
    epsX = effectiveWidth*1e-2
    epsY = effectiveHeight*1e-2
    for extPair,items in partitions.items():
      minLabel = extPair[0]
      maxLabel = extPair[1]

      for extIdx in [minLabel,maxLabel]:
        if extIdx not in self.extLocations:
          xMin = mergeSequence[extIdx][1]
          yMin = self.amsc.Y[extIdx]
          (xMin,yMin) = scaleToScene(xMin,yMin)
          self.extLocations[extIdx] = (xMin,yMin)
    ############################################################################

    for extPair,items in partitions.items():
      minLabel = extPair[0]
      maxLabel = extPair[1]
      ys = self.amsc.Y[np.array(items)]
      lineWidth = (self.maxDiameter+self.minDiameter)/2.

      pen = qtg.QPen(qtc.Qt.NoPen) #qtg.QPen(qtc.Qt.black)

      fillColor = qtg.QColor(colorMap[extPair])
      fillColor.setAlpha(200)
      brush = qtg.QBrush(fillColor)

      xMin = self.extLocations[minLabel][0]
      yMin = self.extLocations[minLabel][1]
      xMax = self.extLocations[maxLabel][0]
      yMax = self.extLocations[maxLabel][1]

      path = qtg.QPainterPath()
      path.moveTo(xMin, yMin)
      path.cubicTo((xMax+xMin)/2., yMin, (xMax+xMin)/2., yMax,  xMax, yMax)

      partitionData = {'key': (minLabel,maxLabel), 'amsc': self.amsc}

      self.polygonMap[(minLabel,maxLabel)] = CustomPathItem(path,None,self.scene,partitionData)
      pen = qtg.QPen(brush,self.minDiameter)
      self.polygonMap[(minLabel,maxLabel)].setPen(pen)
      self.polygonMap[(minLabel,maxLabel)].setFlag(qtw.QGraphicsItem.ItemIsSelectable)
      self.polygonMap[(minLabel,maxLabel)].setAcceptHoverEvents(True)
      self.polygonMap[(minLabel,maxLabel)].setFlag(qtw.QGraphicsItem.ItemClipsToShape)

    for key,(parentIdx,persistence) in mergeSequence.items():
      if key in self.extLocations:
        x = self.extLocations[key][0]
        y = self.extLocations[key][1]
      else:
        x = persistence
        y = self.amsc.Y[key]
        (x,y) = scaleToScene(x,y)

      if self.amsc.GetClassification(key) == 'minimum':
        if persistence >= currentP:
          pen = qtg.QPen(colors.minPenColor)
          brush = qtg.QBrush(colors.minBrushColor)
        else:
          pen = qtg.QPen(colors.inactiveMinPenColor)
          brush = qtg.QBrush(colors.inactiveMinBrushColor)
        labelIndex = 0
      elif self.amsc.GetClassification(key) == 'maximum':
        if persistence >= currentP:
          pen = qtg.QPen(colors.maxPenColor)
          brush = qtg.QBrush(colors.maxBrushColor)
        else:
          pen = qtg.QPen(colors.inactiveMaxPenColor)
          brush = qtg.QBrush(colors.inactiveMaxBrushColor)
        labelIndex = 1
      else:
        # Debug for errors, this should not be seen
        col = qtg.QColor('#cccccc')
        pen = qtg.QPen(col.darker())
        brush = qtg.QBrush(col)
        labelIndex = 2

      if self.colorAction.isChecked():
        if key in colorMap:
          extremaColor = qtg.QColor(colorMap[key])
        else:
          extremaColor = qtg.QColor('#cccccc')
        pen = qtg.QPen(extremaColor.darker())
        brush = qtg.QBrush(extremaColor)

      count = 0
      startT = time.clock()
      for extPair,items in partitions.items():
        if key in extPair:
          count += len(items)
      # print('counting finished: (%f s)' % (time.clock()-startT))

      diameter = scaleDiameter(count)

      if self.glyphGroup.checkedAction().text() == 'Triangle':
        triangle = qtg.QPolygonF()
        if labelIndex == 0:
          startTheta = 30
        elif labelIndex == 1:
          startTheta = 90
        else:
          startTheta = 0

        radius = diameter/2.
        for i in range(3):
          theta = startTheta+i*120
          theta = theta*math.pi/180.
          triangle.append(qtc.QPointF(x+radius*math.cos(theta),y-radius*math.sin(theta)))

        self.polygonMap[key] = self.scene.addPolygon(triangle,pen,brush)
      elif self.glyphGroup.checkedAction().text() == 'Circle':
        self.polygonMap[key] = self.scene.addEllipse(x-diameter/2., \
                                                     y-diameter/2., diameter, \
                                                     diameter, pen, brush)
      if currentP <= persistence:
        self.polygonMap[key].setFlag(qtw.QGraphicsItem.ItemIsSelectable)
        self.polygonMap[key].setZValue(1/diameter)
      self.polygonMap[key].setToolTip(str(key))
      self.polygonMap[key].setAcceptHoverEvents(True)
    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())
    self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)

  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations. For this class in particular, we will test:
        - Saving the view buffer in both svg and png formats.
        - Triggering of the resize event.
        - Subselecting data and updating this view to reflect those changes.
        - Toggling the color of the extrema.
        - Toggling the fill viewport action.
        - Increasing, decreasing, and explicitly setting the persistence level.
        - Changing the shape of the extremum glyphs
        - Triggering the mouse move, press, and release events.
        - Triggering the right-click context menu
        @ In, None
        @ Out, None
    """
    self.amsc.ClearSelection()

    self.saveImage(self.windowTitle()+'.svg')
    self.saveImage(self.windowTitle()+'.png')

    self.resizeEvent(qtg.QResizeEvent(qtc.QSize(1,1),qtc.QSize(100,100)))
    pair = list(self.amsc.GetCurrentLabels())[0]
    self.amsc.SetSelection([pair,pair[0],pair[1]])
    self.colorAction.setChecked(True)
    self.fillAction.setChecked(True)
    self.updateScene()

    self.increasePersistence()
    self.decreasePersistence()
    self.colorAction.setChecked(False)
    self.fillAction.setChecked(False)
    for action in self.glyphGroup.actions():
      action.setChecked(True)
      self.updateScene()

    pair = list(self.amsc.GetCurrentLabels())[0]
    self.amsc.SetSelection([pair,pair[0],pair[1]])

    genericMouseEvent = qtg.QMouseEvent(qtc.QEvent.MouseMove, qtc.QPoint(0,0), qtc.Qt.MiddleButton, qtc.Qt.MiddleButton, qtc.Qt.NoModifier)
    self.setPersistence()
    self.mouseReleaseEvent(genericMouseEvent, True)
    self.mouseReleaseEvent(genericMouseEvent, False)
    self.mousePressEvent(genericMouseEvent, True)
    self.mousePressEvent(genericMouseEvent, False)
    self.mouseMoveEvent(genericMouseEvent, True)
    self.mouseMoveEvent(genericMouseEvent, False)
    self.contextMenuEvent(genericMouseEvent)

    # self.mouseMoveEvent(qtg.QMouseEvent(qtc.Qt.MouseButton), qtc.QPoint(0,0), None, qtc.QEvent.Type, qtc.QPoint(0,0), qtc.Qt.)
    self.updateScene()

    super(TopologyMapView, self).test()
