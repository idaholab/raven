from PySide import QtCore as qtc
from PySide import QtGui as qtg
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

from PySide import QtSvg as qts

class OverlayButton(qtg.QPushButton):
  def __init__(self, *args, **kwargs):
    super(OverlayButton, self).__init__(*args, **kwargs)
    self.setDefaults()

  def setDefaults(self):
    self.setStyleSheet("background-color: rgba( 51, 73, 96, 25%); color: rgba(236,240,241, 75%);  opacity: 0.25;")

  def enterEvent(self, event):
    super(OverlayButton, self).enterEvent(event)
    self.setStyleSheet("background-color: rgba( 51, 73, 96, 75%); color: rgba(236,240,241, 100%);  opacity: 0.75;")

  def leaveEvent(self, event):
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

class ZoomableGraphicsView(qtg.QGraphicsView):
  defaultSceneDimension = 1000

  def __init__(self, parent):
    super(ZoomableGraphicsView, self).__init__(parent)
    self._zoom = 0
    self.padding = 10
    self.setTransformationAnchor(qtg.QGraphicsView.AnchorUnderMouse)
    self.setResizeAnchor(qtg.QGraphicsView.AnchorUnderMouse)
    self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.setFrameShape(qtg.QFrame.NoFrame)
    self.setDragMode(qtg.QGraphicsView.ScrollHandDrag)

    self.setRenderHints(qtg.QPainter.Antialiasing |
                        qtg.QPainter.SmoothPixmapTransform)

    scene = qtg.QGraphicsScene(self)
    scene.setSceneRect(0,0,ZoomableGraphicsView.defaultSceneDimension,ZoomableGraphicsView.defaultSceneDimension)
    self.setScene(scene)

    self.rightClickMenu = qtg.QMenu()
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

  def saveImage(self):
    dialog = qtg.QFileDialog(self)
    dialog.setFileMode(qtg.QFileDialog.AnyFile)
    dialog.setAcceptMode(qtg.QFileDialog.AcceptSave)
    dialog.exec_()
    if dialog.result() == qtg.QFileDialog.Accepted:
      myFile = dialog.selectedFiles()[0]
      self.scene().clearSelection()
      if myFile.endswith('.svg'):
        svgGen = qts.QSvgGenerator()
        svgGen.setFileName(myFile)
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
      if not myFile.endswith('.svg'):
        image.save(myFile,quality=100)
      del painter

  def placeButtons(self):
    xPos = 0
    self.resetButton.move(0, 0)
    xPos += self.resetButton.width()
    self.modeButton.move(xPos, 0)
    xPos += self.modeButton.width()
    self.cameraButton.move(xPos, 0)

  def resizeEvent(self, event):
    super(ZoomableGraphicsView, self).resizeEvent(event)
    self._zoom = 0
    self.fitInView(self.sceneRect(),qtc.Qt.KeepAspectRatio)
    # if self._zoom == 0:
    #   self.fitInView(self.sceneRect(),qtc.Qt.KeepAspectRatio)
    self.placeButtons()

  def zoomFactor(self):
    return self._zoom

  def resetView(self):
    self._zoom = 0
    self.fitInView(self.sceneRect(),qtc.Qt.KeepAspectRatio)
    self.placeButtons()

  def toggleMouseMode(self):
    if self.dragMode() == qtg.QGraphicsView.ScrollHandDrag:
      self.setDragMode(qtg.QGraphicsView.RubberBandDrag)
      toolTipText = 'Switch to Pan and Zoom'
      icon = mouseIcon
    else:
      self.setDragMode(qtg.QGraphicsView.ScrollHandDrag)
      toolTipText = 'Switch to Selection Mode'
      icon = handIcon

    self.modeButton.setToolTip(toolTipText)
    self.modeButton.setIcon(icon)

  def wheelEvent(self, event):
    if self.dragMode() != qtg.QGraphicsView.ScrollHandDrag:
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
    # if self.dragMode() == qtg.QGraphicsView.ScrollHandDrag:
    #   ## Do something else with the right clicks
    #   pass
    # else:
    self.rightClickMenu.popup(event.globalPos())
