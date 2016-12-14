#!/usr/bin/env python

import numpy as np
import sys

from PySide import QtCore as qtc
from PySide import QtGui as qtg
from PySide import QtSvg as qts

from BaseView import BaseView
from ZoomableGraphicsView import ZoomableGraphicsView
from Tree import Node
import colors

gray = qtg.QColor('#999999')
black = qtg.QColor('#000000')
transparentGray = gray.lighter()
transparentGray.setAlpha(127)

def linakgeToTree(*linkages):
  """
    Convert a linkage matrix into a tree that knows how to perform its own
    layout.
    In @ linkages, np.array(s), one or more linkage matrices that will each be
    made into a single tree with a null node used to connect them allowing
    them to be tied together into a single data structure.
  """

  maxLevel = 0

  for linkage in linkages:
    maxLevel = max(maxLevel, linkage[-1,2])

  ## This empty root allows us to create a fake node that connects the forest
  root = Node('None', None, maxLevel,0)

  for linkage in linkages:
    ## Iterate through the linkage matrix in reverse order since the last merge
    ## will be the root node of this tree
    n = linkage.shape[0]+1

    for i in range(linkage.shape[0]-1,-1,-1):
      newIdx = n+i
      leftChildIdx,rightChildIdx,level,size = linkage[i,:]
      leftChildIdx = int(leftChildIdx)
      rightChildIdx = int(rightChildIdx)
      size = int(size)

      node = root.getNode(newIdx)

      if node is None:
        ## If the node does not exist yet, then we are constructing a new
        ## subtree
        node = root.addChild(newIdx, level, size)
        ## This is a new subtree, so in order to get the correct size at the
        ## root, we need to add the total size of this subtree.
        root.size += size
      else:
        ## Otherwise, we need to update this node's level and size with the
        ## information from this merge of the linkage matrix
        node.level = level
        node.size = size

      ## We will update these children when we get to their merges, if they are
      ## leaves, then they are at level zero and have only one item, themselves.
      node.addChild(leftChildIdx, 0, 1)
      node.addChild(rightChildIdx, 0, 1)

  return root

class DendrogramView(ZoomableGraphicsView,BaseView):
  """
    A view that shows a hierarchical data object.
  """
  # minDiameterMultiplier = 0.001
  def __init__(self, mainWindow=None):
    """
    """
    ZoomableGraphicsView.__init__(self, mainWindow)
    BaseView.__init__(self, mainWindow)

    self.mainWindow = mainWindow
    self.tree = linakgeToTree(self.mainWindow.engine.linkage)

    ## User-editable parameters
    self.truncationSize = 1
    self.truncationLevel = 0
    self.maxDiameterMultiplier = 0.05

    ## Setup right click menu for user customization
    self.rightClickMenu.addSeparator()

    levelAction = self.rightClickMenu.addAction('Set Level')
    levelAction.triggered.connect(self.setLevel)
    incAction = self.rightClickMenu.addAction('Raise Threshold Level')
    incAction.triggered.connect(self.increaseLevel)
    decAction = self.rightClickMenu.addAction('Lower Threshold Level')
    decAction.triggered.connect(self.decreaseLevel)

    self.rightClickMenu.addSeparator()

    setTruncationAction = self.rightClickMenu.addAction('Truncate...')
    setTruncationAction.triggered.connect(self.setTruncation)
    resizePointsAction = self.rightClickMenu.addAction('Resize Points')
    resizePointsAction.triggered.connect(self.setDiameterMultiplier)

    self.edgeAction = self.rightClickMenu.addAction('Smooth Edges')
    self.edgeAction.setCheckable(True)
    self.edgeAction.setChecked(True)
    self.edgeAction.triggered.connect(self.updateScene)

    self.scene().selectionChanged.connect(self.select)

    ## Okay, let's draw the thing
    self.createScene()

  def setTruncation(self):
    """
      Opens a dialog box that allows the user to set the truncation of the
      dendrogram by setting both the minimum node size (count), and the minimum
      level used.
      @In, None
      @Out, None
    """
    dialog = qtg.QDialog(self)
    layout = qtg.QVBoxLayout()
    dialog.setLayout(layout)

    ## Put the label and its associated value label in one row using a sublayout
    sublayout = qtg.QHBoxLayout()
    layout.addLayout(sublayout)

    sublayout.addWidget(qtg.QLabel('Minimum Node Size:'))
    nodeSize = qtg.QLabel('%d' % self.truncationSize)
    sublayout.addWidget(nodeSize)

    ## Next place the associated slider underneath that
    minNodeSizeSlider = qtg.QSlider(qtc.Qt.Horizontal)
    minNodeSizeSlider.setMinimum(1)
    minNodeSizeSlider.setMaximum(self.tree.size)
    ## Use a lambda function to keep the label in sync with the slider
    updateNodeLabel = lambda: nodeSize.setText('%d' % minNodeSizeSlider.value())
    minNodeSizeSlider.valueChanged.connect(updateNodeLabel)
    layout.addWidget(minNodeSizeSlider)

    ## Next item, do the same thing, first put the two labels on one row.
    sublayout = qtg.QHBoxLayout()
    layout.addLayout(sublayout)

    sublayout.addWidget(qtg.QLabel('Minimum Level:'))
    levels = self.getLevels()
    for i,lvl in enumerate(reversed(levels)):
      if lvl < self.truncationLevel:
        break
    idx = len(levels)-1-i
    minLevel = qtg.QLabel('%f' % lvl)
    sublayout.addWidget(minLevel)

    ## Next, create the slider to go underneath them
    minLevelSlider = qtg.QSlider(qtc.Qt.Horizontal)
    minLevelSlider.setMinimum(0)
    minLevelSlider.setMaximum(len(levels)-1)
    minLevelSlider.setSliderPosition(idx)
    ## Use a lambda function to keep the label in sync with the slider
    updateLevelSlider = lambda: minLevel.setText('%f' % levels[minLevelSlider.value()])
    minLevelSlider.valueChanged.connect(updateLevelSlider)
    layout.addWidget(minLevelSlider)

    ## Add the buttons for accepting/rejecting the proposed values
    buttons = qtg.QDialogButtonBox(qtg.QDialogButtonBox.Ok |
                                   qtg.QDialogButtonBox.Cancel,
                                   qtc.Qt.Horizontal, dialog)

    def localAccept():
      """
        This callback will be registered to when the user selects "OK" and will
        update the truncation values. Otherwise, this will never get called and
        the results of the dialog created here will be destroyed.
        @In, None
        @Out, None
      """
      self.truncationSize = int(nodeSize.text())
      self.truncationLevel = float(minLevel.text())
      self.createScene()

    ## Register the buttons within the dialog itself, and our own method for
    ## updating the dendrogram.
    buttons.accepted.connect(localAccept)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    ## Using .open() creates a modal window, but does not block, so thus why
    ## we have registered the callback above, otherwise we could call exec_(),
    ## and then have something like: if dialog.result() == qtg.QDialog.Accepted:
    ## that has the same code as localAccept.
    dialog.open()

  def setDiameterMultiplier(self):
    """
      Opens a dialog box that allows the user to set the maximum point size of
      the dendrogram.
      @In, None
      @Out, None
    """

    ## The code here is going to be very similar to that used in to create the
    ## dialog in setTruncation, so use that function as a model, and I will
    ## forego the verbosity here.

    dialog = qtg.QDialog(self)
    layout = qtg.QVBoxLayout()

    sublayout = qtg.QHBoxLayout()
    staticLabel = qtg.QLabel('Maximum Point Diameter (%% of Window Width):')
    sublayout.addWidget(staticLabel)

    pointSizeSpinner = qtg.QDoubleSpinBox()
    pointSizeSpinner.setMinimum(0.01)
    pointSizeSpinner.setMaximum(0.25)
    pointSizeSpinner.setSingleStep(0.01)
    pointSizeSpinner.setValue(self.maxDiameterMultiplier)
    sublayout.addWidget(pointSizeSpinner)

    layout.addLayout(sublayout)

    buttons = qtg.QDialogButtonBox(qtg.QDialogButtonBox.Ok |
                                   qtg.QDialogButtonBox.Cancel,
                                   qtc.Qt.Horizontal, dialog)

    def localAccept():
      """
        This callback will be registered to when the user selects "OK" and will
        update the maximum node size. Otherwise, this will never get called and
        the results of the dialog created here will be destroyed.
        @In, None
        @Out, None
      """
      self.maxDiameterMultiplier = pointSizeSpinner.value()
      self.updateScene()

    buttons.accepted.connect(localAccept)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    dialog.setLayout(layout)
    dialog.open()

  def increaseLevel(self):
    """
    """
    self.mainWindow.increaseLevel()

  def decreaseLevel(self):
    """
    """
    self.mainWindow.decreaseLevel()

  def contextMenuEvent(self,event):
    """
      Overload the contextMenuEvent to register the y location of the click, in
      case the user selects to set the level. We will need it then.
      @ In, event, QContextMenuEvent, the parameters of the event triggering
      this callback
      @ Out, None
    """
    self.currentY = event.y()
    super(DendrogramView,self).contextMenuEvent(event)

  def setLevel(self):
    """
    """
    # position = self.mapFromGlobal(self.rightClickMenu.pos())
    # mousePt = self.mapToScene(position.x(),position.y()).y()
    mousePt = self.mapToScene(0,self.currentY).y()

    minLevel = 0
    maxLevel = self.maxLevel()
    effectiveHeight = self.scene().height()-2*self.padding
    ty = (self.padding + effectiveHeight - mousePt)/effectiveHeight
    wy = ty*float(maxLevel)

    self.mainWindow.setLevel(np.clip(wy,minLevel,maxLevel))
    self.levelChanged()

  def select(self):
    """
    """
    selectedKeys = []
    for key,graphic in self.nodes.iteritems():
      if graphic in self.scene().selectedItems():
        selectedKeys.append(key)
    # self.tree.SetSelection(selectedKeys)

  def getLevels(self):
    """
      Convenience function for extracting the level data common to the parent
      window.
      @ In, None
      @ Out, level, list, an ordered list of the levels available in increasing
        value
    """
    return self.mainWindow.levels

  def getCurrentLevel(self):
    """
      Convenience function for extracting the level data common to the parent
      window.
      @ In, None
      @ Out, level, list, an ordered list of the levels available in increasing
        value
    """
    return self.mainWindow.level

  def maxLevel(self):
    """
      Convenience function for extracting the largest level common to the parent
      window.
      @ In, None
      @ Out, float, the largest available level in the hierarchy.
    """
    return self.mainWindow.levels[-1]

  def selectionChanged(self):
    ## Disable the communication so we don't end up in infinite callbacks
    self.scene().selectionChanged.disconnect(self.select)

    selectedKeys = self.segmentation.selectedSegments
    for key,graphic in self.nodes.iteritems():
      if key in selectedKeys:
        graphic.setSelected(True)
      else:
        graphic.setSelected(False)

    ## Re-enable the communication
    self.scene().selectionChanged.connect(self.select)

  def levelChanged(self):
    """
    """
    self.updateScene()

  def updateScene(self):
    """
    """
    self.updateScreenParameters()
    self.updateNodes()
    self.updateArcs()
    self.updateActiveBox()

  def updateScreenParameters(self):
    """
    """
    scene = self.scene()
    width = scene.width()
    height = scene.height()
    minDim = min([width,height])

    totalCount = self.tree.getLeafCount()

    ## At most, the minimum diameter will be 1% of the smallest scene dimension,
    ## otherwise, it will be small enough to fit the entire count of ellipses
    ## end to end. This worst case scenario will be witnessed in an untruncated
    ## hierarchical clustering where at the bottom level each data point is a
    ## singleton cluster.
    self.minDiameter = min(minDim / float(totalCount),0.01 * minDim)
    self.maxDiameter = self.maxDiameterMultiplier*minDim

    self.padding = self.maxDiameter / 2.

    self.usableWidth = width - 2 * self.padding
    self.usableHeight = height - 2 * self.padding

  def updateNodes(self, newNodes=None):
    """
    """
    maxLevel = self.maxLevel()

    if newNodes is not None:
      for idx,(x,y) in newNodes:
        if idx not in self.colorMap:
          # self.colorMap[idx] = qtg.QColor(*tuple(255*np.random.rand(3)))
          self.colorMap[idx] = qtg.QColor(colors.colorCycle.next())

        brush = qtg.QBrush(self.colorMap[idx])
        pen = qtc.Qt.NoPen

        ## Don't worry about placing or sizing the ellipse, we will do that
        ## below according to current settings of the view, right now we want
        ## to establish the non-transient properties.
        self.nodes[idx] = self.scene().addEllipse(0,0,0,0,pen,brush)
        self.nodes[idx].setFlag(qtg.QGraphicsItem.ItemIsSelectable)
        ## Dynamically adding some members to the class to make this easier to
        ## recompute in case the usable screen area changes as in the case when
        ## the user adjusts the size of the ellipses.
        self.nodes[idx].rawX = x
        self.nodes[idx].rawY = y / maxLevel


    totalCount = self.tree.getLeafCount()
    ## Needed in order to invert the y-axis
    height = self.scene().height()

    for idx,glyph in self.nodes.items():
      node = self.tree.getNode(idx)
      count = node.size

      diameter = float(count)/totalCount*(self.maxDiameter-self.minDiameter)+self.minDiameter

      oldRect = glyph.rect()

      color = glyph.brush().color()
      if node.level < self.getCurrentLevel():
        color.setAlpha(64)
        diameter = self.minDiameter
        glyph.setFlag(qtg.QGraphicsItem.ItemIsSelectable)
        glyph.setToolTip('   id: %d\nlevel: %f\nInactive' %(idx, node.level))
      else:
        color.setAlpha(255)
        glyph.setToolTip('   id: %d\nlevel: %f\ncount: %d' %(idx, node.level, int(count)))
      brush = qtg.QBrush(color)

      if self.nodes[idx].isSelected():
        pen = qtg.QPen(black)
      else:
        pen = qtc.Qt.NoPen

      newX = glyph.rawX*self.usableWidth + self.padding - diameter/2.
      newY = height - self.usableHeight * glyph.rawY - self.padding - diameter/2.

      glyph.setRect(newX, newY, diameter, diameter)
      glyph.setPen(pen)
      glyph.setBrush(brush)


      ## Ensure smaller nodes are drawn on top
      if diameter == 0:
        glyph.setZValue(sys.float_info.max)
      else:
        glyph.setZValue(1./diameter)

  def updateArcs(self, newEdges = None):
    """
    """

    if newEdges is not None:
      for edge in newEdges:
        path = qtg.QPainterPath()
        self.arcs[edge] = self.scene().addPath(path,qtg.QPen(gray))
        self.arcs[edge].setZValue(0)

    maxLevel = self.maxLevel()

    ## Needed in order to invert the y-axis
    height = self.scene().height()

    for (idx1,idx2),pathGraphic in self.arcs.items():
      rect1 = self.nodes[idx1].rect()
      rect2 = self.nodes[idx2].rect()

      x1 = rect1.x()+rect1.width()/2.
      y1 = rect1.y()+rect1.height()/2.

      x2 = rect2.x()+rect2.width()/2.
      y2 = rect2.y()+rect2.height()/2.

      path = qtg.QPainterPath()
      path.moveTo(x1,y1)
      if self.edgeAction.isChecked():
        ## idx1 should always be the parent
        parent = self.tree.getNode(idx1)
        maxChild = parent.maximumChild(self.truncationSize, self.truncationLevel)
        if maxChild is None:
          maxChildLevel = y2
          intermediateY = height - self.usableHeight*maxChildLevel/maxLevel-self.padding
        else:
          maxChildRect = self.nodes[maxChild.id].rect()

          # maxChildLevel = maxChild.level
          ## or
          intermediateY = maxChildRect.y()+maxChildRect.height()/2.

        path.cubicTo(x1,y1+(intermediateY-y1)*0.33, x2,intermediateY-(intermediateY-y1)*0.33,x2,intermediateY)
        path.lineTo(x2,y2)
      else:
        path.lineTo(x2,y1)
        path.lineTo(x2,y2)

      pathGraphic.setPath(path)

  def updateActiveBox(self):
    """
    """

    ## If the active box does not exist yet, then create it, don't worry about
    ## the height, we will calculate that in the remainder of this function
    if self.activeBox is None:
      pen = qtg.QPen(gray)
      brush = qtg.QBrush(transparentGray)
      width = self.scene().width()
      self.activeBox = self.scene().addRect(0, 0, width, 0, pen, brush)

    height = self.scene().height()
    minDim = min([self.scene().width(),self.scene().height()])

    maxDiameter = self.maxDiameterMultiplier*minDim
    levelRatio = self.getCurrentLevel()/self.maxLevel()

    level = height - self.usableHeight*levelRatio-self.padding

    rect = self.activeBox.rect()
    rect.setHeight(level)
    self.activeBox.setRect(rect)

  def createScene(self):
    """
    """
    ## Clear data for saving state
    self.colorMap = {}
    self.nodes = {}
    self.arcs = {}
    self.activeBox = None

    scene = self.scene()
    scene.clear()
    sz = ZoomableGraphicsView.defaultSceneDimension
    scene.setSceneRect(0, 0, sz, sz)

    # if self.fillAction.isChecked():
    #   aspectRatio = float(self.width())/float(self.height())
    #   scene.setSceneRect(0, 0, sz*aspectRatio, sz)
    # else:
    #   scene.setSceneRect(0, 0, sz, sz)

    self.updateScreenParameters()

    pen = qtg.QPen(qtc.Qt.black)
    self.boundingBox = scene.addRect(0, 0, scene.width(), scene.height(), pen)

    xOffset = 0
    ids = []
    points = []
    edges = []

    ## Not necessary, just shorthanding the variable names
    tSize = self.truncationSize
    tLevel = self.truncationLevel
    totalCount = self.tree.getLeafCount()

    for node in self.tree.children:
      count = node.getLeafCount()
      rootWidth = float(count)/totalCount

      (myIds,myPoints,myEdges) = node.Layout(xOffset, rootWidth, tSize, tLevel)

      ids.extend(myIds)
      points.extend(myPoints)
      edges.extend(myEdges)
      xOffset += rootWidth

    self.updateNodes(zip(ids,points))
    self.updateArcs(edges)
    self.updateActiveBox()

    self.fitInView(scene.sceneRect(),qtc.Qt.KeepAspectRatio)