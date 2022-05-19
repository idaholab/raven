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
  This file contains the Zoomable graphics view which can be inherited by
  others. This generic class provides some common UI elements that would
  otherwise need to be replicated, such as zooming, panning, saving the UI scene
  to an image, and selecting things on the scene.
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

import os

################################################################################
## This adds another library, but makes pretty buttons, we can probably
## accomplish the same thing with small pngs that are stored in a resources
## directory
# import os
# os.environ['QT_API'] = 'pyside'
# import qtawesome as qta
## These icons can be generated from here: http://fa2png.io/
## I choose to do 32 pixels as it is the smallest power of 2 that provides a
## large enough button to select with relative ease.
################################################################################

class OverlayButton(qtw.QPushButton):
  """
    A UI element that can be overlayed on top of other UI elements and will
    render transparently until a user's mouse hovers over it.
  """
  def __init__(self, *args, **kwargs):
    """
      Initializer for the Overlay Button that will apply a default style,
      otherwise this is the same as a QPushButton.
    """
    super(OverlayButton, self).__init__(*args, **kwargs)
    self.setDefaults()

  def setDefaults(self):
    """
      Applies a look and feel to the button that includes color and transparency
      @ In, None
      @ Out, None
    """
    self.setStyleSheet("background-color: rgba( 51, 73, 96, 25%); color: rgba(236,240,241, 75%);  opacity: 0.25;")

  def enterEvent(self, event):
    """
      When the user hovers over the button this callback will remove the
      transparency from the button
      @ In, event, PySide.QtCore.QEvent, the event that triggers this callback,
        namely a mouse over event.
      @ Out, None
    """
    super(OverlayButton, self).enterEvent(event)
    self.setStyleSheet("background-color: rgba( 51, 73, 96, 75%); color: rgba(236,240,241, 100%);  opacity: 0.75;")

  def leaveEvent(self, event):
    """
      When the user's mouse leave the button this callback will remove the
      transparency from the button
      @ In, event, PySide.QtCore.QEvent, the event that triggers this callback,
        namely a mouse over event.
      @ Out, None
    """
    super(OverlayButton, self).leaveEvent(event)
    self.setDefaults()

################################################################################
## Defining the qtawesome icons is no simpler than loading the images from files
# defaultIconColor = 'white'
# resetIcon = qta.icon('fa.rotate-left', color=defaultIconColor)
# handIcon = qta.icon('fa.hand-paper-o', color=defaultIconColor)
# mouseIcon = qta.icon('fa.mouse-pointer', color=defaultIconColor)
# screenshotIcon = qta.icon('fa.camera', color=defaultIconColor)

resourceLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)),'resources')
resetIcon      = qtg.QIcon(os.path.join(resourceLocation,'fa-rotate-left_32.png'))
handIcon       = qtg.QIcon(os.path.join(resourceLocation,'fa-hand-paper-o_32.png'))
mouseIcon      = qtg.QIcon(os.path.join(resourceLocation,'fa-mouse-pointer_32.png'))
screenshotIcon = qtg.QIcon(os.path.join(resourceLocation,'fa-camera_32.png'))
################################################################################

class ZoomableGraphicsView(qtw.QGraphicsView):
  """
    This is a generic class for providing some basic UI functionality that can
    be inherited by other classes. It provides functionality such as
    zooming/panning, selecting, and taking screen captures of a QGraphicsScene.
  """
  defaultSceneDimension = 1000

  def __init__(self, parent):
    """
      Initializer that will set L&F defaults and initialize the UI elements
      @ In, parent, PySide.QtGui.QWidget, the parent widget of this widget.
    """
    #super(ZoomableGraphicsView, self).__init__(parent)
    qtw.QGraphicsView.__init__(self, parent)
    self._zoom = 0
    self.padding = 10
    self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
    self.setResizeAnchor(qtw.QGraphicsView.AnchorUnderMouse)
    self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.setFrameShape(qtw.QFrame.NoFrame)
    self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)

    self.setRenderHints(qtg.QPainter.Antialiasing |
                        qtg.QPainter.SmoothPixmapTransform)

    scene = qtw.QGraphicsScene(self)
    scene.setSceneRect(0,0,ZoomableGraphicsView.defaultSceneDimension,ZoomableGraphicsView.defaultSceneDimension)
    self.setScene(scene)

    self.rightClickMenu = qtw.QMenu()
    self.fillAction = self.rightClickMenu.addAction('Fill viewport')
    self.fillAction.setVisible(False)
    self.fillAction.setCheckable(True)
    self.fillAction.setChecked(False)
    # self.fillAction.triggered.connect(self.createScene)

    self.resetButton = OverlayButton(self)
    self.resetButton.setToolTip('Reset View')
    self.resetButton.setIcon(resetIcon)
    self.resetButton.clicked.connect(self.resetView)

    resetAction = self.rightClickMenu.addAction('Reset View')
    resetAction.triggered.connect(self.resetView)

    self.modeButton = OverlayButton(parent=self)
    self.modeButton.setToolTip('Switch to Selection Mode')
    self.modeButton.setIcon(handIcon)
    self.modeButton.clicked.connect(self.toggleMouseMode)
    self.modeAction = self.rightClickMenu.addAction('Toggle Mouse Mode')
    self.modeAction.triggered.connect(self.toggleMouseMode)

    self.cameraButton = OverlayButton(parent=self)
    self.cameraButton.setToolTip('Capture Image')
    self.cameraButton.setIcon(screenshotIcon)
    self.cameraButton.clicked.connect(self.saveImage)
    cameraAction = self.rightClickMenu.addAction('Capture Image')
    cameraAction.triggered.connect(self.saveImage)

    x = self.width()
    self.placeButtons()

  # def fitInView(self,rect):
  #   if not rect.isNull():
  #     unity = self.transform().mapRect(qtc.QRectF(0, 0, 1, 1))
  #     self.scale(1 / unity.width(), 1 / unity.height())
  #     viewrect = self.viewport().rect()
  #     scenerect = self.transform().mapRect(rect)
  #     factor = min(viewrect.width() / scenerect.width(),
  #                  viewrect.height() / scenerect.height())
  #     self.scale(factor, factor)
  #     self.centerOn(rect.center())
  #     self._zoom = 0

  def saveImage(self, filename=None):
    """
      Method for saving the contents of this view to an image file.
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
        myFile = dialog.selectedFiles()[0]
      else:
        return

    self.scene().clearSelection()
    if filename.endswith('.svg'):
      svgGen = qts.QSvgGenerator()
      svgGen.setFileName(filename)
      svgGen.setSize(self.sceneRect().size().toSize())
      svgGen.setViewBox(self.sceneRect())
      svgGen.setTitle("Screen capture of " + self.__class__.__name__)
      svgGen.setDescription("Generated from RAVEN.")
      painter = qtg.QPainter(svgGen)
    else:
      image = qtg.QImage(self.sceneRect().size().toSize(), qtg.QImage.Format_ARGB32)
      image.fill(qtc.Qt.transparent)
      painter = qtg.QPainter(image)

    self.scene().render(painter)
    if not filename.endswith('.svg'):
      image.save(filename, quality=100)

    del painter

  def placeButtons(self):
    """
      This function will position the overlay buttons in the correct place on
      the view which should be the upper left corner.
      @ In, None
      @ Out, None
    """
    xPos = 0
    self.resetButton.move(0, 0)
    xPos += self.resetButton.width()
    self.modeButton.move(xPos, 0)
    xPos += self.modeButton.width()
    self.cameraButton.move(xPos, 0)

  def resizeEvent(self, event):
    """
      A callback function that will be triggered when this view is resized.
      @ In, event, PySide.QtGui.QResizeEvent, the resize event that triggered
        this callback.
      @ Out, None
    """
    super(ZoomableGraphicsView, self).resizeEvent(event)
    self._zoom = 0
    self.fitInView(self.sceneRect(),qtc.Qt.KeepAspectRatio)
    # if self._zoom == 0:
    #   self.fitInView(self.sceneRect(),qtc.Qt.KeepAspectRatio)
    self.placeButtons()

  def zoomFactor(self):
    """
      Returns the zoom level of this view.
      @ In, None
      @ Out, self._zoom, int, the level of zoom currently being used by this
        view.
    """
    return self._zoom

  def resetView(self):
    """
      Resets this view to its default viewing settings. This will typically
      fit everything on the scene onscreen.
      @ In, None
      @ Out, None
    """
    self._zoom = 0
    self.fitInView(self.sceneRect(),qtc.Qt.KeepAspectRatio)
    self.placeButtons()

  def toggleMouseMode(self):
    """
      Function that will switch the mouse mode from pan/zoom to selection mode
      or vice versa
      @ In, None
      @ Out, None
    """
    if self.dragMode() == qtw.QGraphicsView.ScrollHandDrag:
      self.setDragMode(qtw.QGraphicsView.RubberBandDrag)
      toolTipText = 'Switch to Pan and Zoom'
      icon = mouseIcon
    else:
      self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
      toolTipText = 'Switch to Selection Mode'
      icon = handIcon

    self.modeButton.setToolTip(toolTipText)
    self.modeButton.setIcon(icon)

  def wheelEvent(self, event):
    """
      Callback for handling a mouse wheel event
      @ In, event, PySide.QtGui.QWheelEvent, event that triggered this callback
      @ Out, None
    """
    if self.dragMode() != qtw.QGraphicsView.ScrollHandDrag:
      return ## Ignore if we are not in pan and zoom mode
    if event.delta() > 0:
      factor = 1.1
      self._zoom += 1
    else:
      factor = 0.9
      self._zoom -= 1

    if self._zoom > 0:
      self.scale(factor, factor)
    elif self._zoom == 0:
      self.fitInView(self.sceneRect(),qtc.Qt.KeepAspectRatio)
    else:
      self._zoom = 0
      self.fitInView(self.sceneRect(),qtc.Qt.KeepAspectRatio)

  def contextMenuEvent(self,event):
    """
      Callback for initiating the context menu traditionally this is through a
      right mouse button click.
      @ In, event, PySide.QtGui.QContextMenuEvent, the triggering event
    """
    self.rightClickMenu.popup(event.globalPos())


  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations. For this class in particular, we will test:
         - The mouse wheel events to ensure that zooming is appropriately
           handled and ignored in the case where the user is in pan mode.
         - Toggling the mouse mode between zooming and panning.
         - Saving the view buffer to both an svg and png format.
        @ In, None
        @ Out, None
    """
    zoom = ZoomableGraphicsView.zoomFactor(self)
    ZoomableGraphicsView.resetView(self)
    ZoomableGraphicsView.toggleMouseMode(self)
    ZoomableGraphicsView.toggleMouseMode(self)
    ZoomableGraphicsView.saveImage(self, self.windowTitle()+'.svg')
    ZoomableGraphicsView.saveImage(self, self.windowTitle()+'.png')

    genericMouseEvent = qtg.QMouseEvent(qtc.QEvent.MouseMove, qtc.QPoint(0,0), qtc.Qt.MiddleButton, qtc.Qt.MiddleButton, qtc.Qt.NoModifier)
    ZoomableGraphicsView.contextMenuEvent(self, genericMouseEvent)
    genericMouseEvent = qtg.QWheelEvent(qtc.QPoint(0,0), 1, qtc.Qt.MiddleButton, qtc.Qt.NoModifier)
    ZoomableGraphicsView.wheelEvent(self, genericMouseEvent)
    genericMouseEvent = qtg.QWheelEvent(qtc.QPoint(0,0), -1, qtc.Qt.MiddleButton, qtc.Qt.NoModifier)
    ZoomableGraphicsView.wheelEvent(self, genericMouseEvent)
    genericMouseEvent = qtg.QWheelEvent(qtc.QPoint(0,0), -1, qtc.Qt.MiddleButton, qtc.Qt.NoModifier)
    ZoomableGraphicsView.wheelEvent(self, genericMouseEvent)
    ZoomableGraphicsView.toggleMouseMode(self)
    ZoomableGraphicsView.wheelEvent(self, genericMouseEvent)
