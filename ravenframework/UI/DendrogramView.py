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
    This file contains the Dendrogram view that visualizes trees and hierarchies
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3

import numpy as np
import sys
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

from .BaseHierarchicalView import BaseHierarchicalView
from .ZoomableGraphicsView import ZoomableGraphicsView
from .Tree import Node

# import timeit

gray = qtg.QColor('#999999')
black = qtg.QColor('#000000')
transparentGray = gray.lighter()
transparentGray.setAlpha(127)

def linakgeToTree(*linkages):
  """
    Convert a linkage matrix into a tree that knows how to perform its own
    layout.
    @ In, linkages, np.array(s), one or more linkage matrices that will each be
    made into a single tree with a null node used to connect them allowing
    them to be tied together into a single data structure. The columns of this
    linkage matrix are described here:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    The first two columns represent the clusters being merged.
    The third column represents the level at where the first two columns merge.
    The fourth column is the size of the newly formed cluster.
    The implicit fifth number is the row which when combined with the original
    size of the data yields the the new cluster's identifier.
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

class DendrogramView(ZoomableGraphicsView,BaseHierarchicalView):
  """
    A view that shows a hierarchical data object.
  """
  # minDiameterMultiplier = 0.001
  def __init__(self, mainWindow=None):
    """
      The initialization method for the DendrogramView that sets some default
      parameters, initializes the UI, and constructs the scene
      @In, mainWindow, HierarchicalWindow, the parent window of this view
      @Out, None
    """
    # BaseHierarchicalView.__init__(self, mainWindow)
    ZoomableGraphicsView.__init__(self, mainWindow)

    self.setWindowTitle(self.__class__.__name__)
    self.scrollable = False
    self.mainWindow = mainWindow


    self.tree = linakgeToTree(self.mainWindow.engine.linkage)

    ## User-editable parameters
    self.truncationSize = 1
    self.truncationLevel = 0
    self.maxDiameterMultiplier = 0.05
    self.totalCount = self.tree.getLeafCount()

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
    dialog = qtw.QDialog(self)
    layout = qtw.QVBoxLayout()
    dialog.setLayout(layout)

    ## Put the label and its associated value label in one row using a sublayout
    sublayout = qtw.QHBoxLayout()
    layout.addLayout(sublayout)

    sublayout.addWidget(qtw.QLabel('Minimum Node Size:'))
    nodeSize = qtw.QLabel('%d' % self.truncationSize)
    sublayout.addWidget(nodeSize)

    ## Next place the associated slider underneath that
    minNodeSizeSlider = qtw.QSlider(qtc.Qt.Horizontal)
    minNodeSizeSlider.setMinimum(1)
    minNodeSizeSlider.setMaximum(self.tree.size)
    ## Use a lambda function to keep the label in sync with the slider
    updateNodeLabel = lambda: nodeSize.setText('%d' % minNodeSizeSlider.value())
    minNodeSizeSlider.valueChanged.connect(updateNodeLabel)
    layout.addWidget(minNodeSizeSlider)

    ## Next item, do the same thing, first put the two labels on one row.
    sublayout = qtw.QHBoxLayout()
    layout.addLayout(sublayout)

    sublayout.addWidget(qtw.QLabel('Minimum Level:'))
    levels = self.getLevels()
    for i,lvl in enumerate(reversed(levels)):
      if lvl < self.truncationLevel:
        break
    idx = len(levels)-1-i
    minLevel = qtw.QLabel('%f' % lvl)
    sublayout.addWidget(minLevel)

    ## Next, create the slider to go underneath them
    minLevelSlider = qtw.QSlider(qtc.Qt.Horizontal)
    minLevelSlider.setMinimum(0)
    minLevelSlider.setMaximum(len(levels)-1)
    minLevelSlider.setSliderPosition(idx)
    ## Use a lambda function to keep the label in sync with the slider
    updateLevelSlider = lambda: minLevel.setText('%f' % levels[minLevelSlider.value()])
    minLevelSlider.valueChanged.connect(updateLevelSlider)
    layout.addWidget(minLevelSlider)

    ## Add the buttons for accepting/rejecting the proposed values
    buttons = qtw.QDialogButtonBox(qtw.QDialogButtonBox.Ok |
                                   qtw.QDialogButtonBox.Cancel,
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
    ## and then have something like: if dialog.result() == qtw.QDialog.Accepted:
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

    dialog = qtw.QDialog(self)
    layout = qtw.QVBoxLayout()

    sublayout = qtw.QHBoxLayout()
    staticLabel = qtw.QLabel('Maximum Point Diameter (%% of Window Width):')
    sublayout.addWidget(staticLabel)

    pointSizeSpinner = qtw.QDoubleSpinBox()
    pointSizeSpinner.setMinimum(0.01)
    pointSizeSpinner.setMaximum(0.25)
    pointSizeSpinner.setSingleStep(0.01)
    pointSizeSpinner.setValue(self.maxDiameterMultiplier)
    sublayout.addWidget(pointSizeSpinner)

    layout.addLayout(sublayout)

    buttons = qtw.QDialogButtonBox(qtw.QDialogButtonBox.Ok |
                                   qtw.QDialogButtonBox.Cancel,
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
      Callback function that will trigger when the data oject's current level is
      increased. This function will propagate this information to the
      mainWindow.
      @ In, None
      @ Out, None
    """
    self.mainWindow.increaseLevel()

  def decreaseLevel(self):
    """
      Callback function that will trigger when the data oject's current level is
      decreased. This function will propagate this information to the
      mainWindow.
      @ In, None
      @ Out, None
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

    mousePt = self.mapToScene(event.pos())

    onSomething = False
    for idx,graphic in self.nodes.items():
      if graphic.contains(mousePt):
        menu = qtw.QMenu()
        colorAction = menu.addAction('Change Color')

        def pickColor():
          """
            A function that will execute a dialog and if accepted will update
            the color of the currently selected item.
            @ In, None
            @ Out, None
          """
          dialog = qtw.QColorDialog()
          dialog.setCurrentColor(self.getColor(idx))
          dialog.exec_()
          if dialog.result() == qtw.QDialog.Accepted:
            self.setColor(idx, dialog.currentColor())
            self.updateScene()

        colorAction.triggered.connect(pickColor)
        menu.exec_(event.globalPos())
        return

    if not onSomething:
      super(DendrogramView,self).contextMenuEvent(event)

  def setLevel(self):
    """
      Callback function that will trigger when the data oject's current level is
      set to a specified level. This function will propagate this information
      to the mainWindow.
      @ In, None
      @ Out, None
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
      Function for registering a user's selection.
      @ In, None
      @ Out, None
    """
    selectedKeys = []
    for key,graphic in self.nodes.items():
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
    return self.mainWindow.getLevel()

  def maxLevel(self):
    """
      Convenience function for extracting the largest level common to the parent
      window.
      @ In, None
      @ Out, float, the largest available level in the hierarchy.
    """
    return self.mainWindow.levels[-1]

  def selectionChanged(self):
    """
      A callback function that will trigger when the user changes the onscreen
      selection of nodes.
      @ In, None
      @ Out, None
    """
    # ## Disable the communication so we don't end up in infinite callbacks
    # self.scene().selectionChanged.disconnect(self.select)

    # selectedKeys = self.segmentation.selectedSegments
    # for key,graphic in self.nodes.items():
    #   if key in selectedKeys:
    #     graphic.setSelected(True)
    #   else:
    #     graphic.setSelected(False)

    # ## Re-enable the communication
    # self.scene().selectionChanged.connect(self.select)
    pass

  def updateScene(self):
    """
      This will update the scene given changes to the underlying data object.
      @ In, None
      @ Out, None
    """
    # x = timeit.timeit(self.updateScreenParameters, number=5)
    # print('updateScreenParameters: %f' % x)

    # x = timeit.timeit(self.updateNodes, number=5)
    # print('updateNodes: %f' % x)

    # x = timeit.timeit(self.updateArcs, number=5)
    # print('updateArcs: %f' % x)

    # x = timeit.timeit(self.updateActiveLine, number=5)
    # print('updateActiveLine: %f' % x)

    self.updateScreenParameters()
    self.updateNodes()
    self.updateArcs()
    self.updateActiveLine()

  def updateScreenParameters(self):
    """
      This will update the scene parameters such as width and height.
      @ In, None
      @ Out, None
    """
    scene = self.scene()
    width = scene.width()
    height = scene.height()
    minDim = min([width,height])

    ## At most, the minimum diameter will be 1% of the smallest scene dimension,
    ## otherwise, it will be small enough to fit the entire count of ellipses
    ## end to end. This worst case scenario will be witnessed in an untruncated
    ## hierarchical clustering where at the bottom level each data point is a
    ## singleton cluster.
    self.minDiameter = min(minDim / float(self.totalCount),0.01 * minDim)
    self.maxDiameter = self.maxDiameterMultiplier*minDim

    self.padding = self.maxDiameter / 2.

    self.usableWidth = width - 2 * self.padding
    self.usableHeight = height - 2 * self.padding

  def setColor(self,idx, color):
    """
      This will set the color of a given index.
      @ In, idx, int, the unique id to update
      @ In, color, QColor, the color you are setting for the associated id.
      @ Out, None
    """
    self.mainWindow.setColor(idx, color)

  def getColor(self,idx):
    """
      Returns the color of the specified index.
      @ In, idx, int, the unique id of this color.
      @ Out, None
    """
    return self.mainWindow.getColor(idx)

  def updateNodes(self, newNodes=None):
    """
      Update the drawing of the nodes.
      @ In, newNodes, list(Node), a list of potentially new ndoes that will be
        added to the drawing.
      @ Out, None
    """
    maxLevel = self.maxLevel()
    currentLevel = self.getCurrentLevel()

    if newNodes is not None:
      for idx,(x,y) in newNodes:
        color = self.getColor(idx)
        brush = qtg.QBrush(color)
        pen = qtg.QPen(qtc.Qt.NoPen)

        ## Don't worry about placing or sizing the ellipse, we will do that
        ## below according to current settings of the view, right now we want
        ## to establish the non-transient properties.
        self.nodes[idx] = self.scene().addEllipse(0,0,0,0,pen,brush)
        self.nodes[idx].setFlag(qtw.QGraphicsItem.ItemIsSelectable)
        ## Dynamically adding some members to the class to make this easier to
        ## recompute in case the usable screen area changes as in the case when
        ## the user adjusts the size of the ellipses.
        self.nodes[idx].rawX = x
        self.nodes[idx].rawY = y / maxLevel
        self.nodes[idx].treeNode = self.tree.getNode(idx)

    ## Needed in order to invert the y-axis
    height = self.scene().height()

    for idx,glyph in self.nodes.items():
      node = glyph.treeNode
      count = node.size

      diameter = float(count)/self.totalCount*(self.maxDiameter-self.minDiameter)+self.minDiameter

      oldRect = glyph.rect()

      color = self.getColor(idx)
      ## If the node's parent is less than the current level, or if this node
      ## is above the current level
      if node.parent != self.tree and node.parent.level <= currentLevel or (node.level > currentLevel and node.getLeafCount(self.truncationSize, self.truncationLevel) > 1):
        color.setAlpha(64)
        diameter = self.minDiameter
        glyph.setFlag(qtw.QGraphicsItem.ItemIsSelectable, False)
        glyph.setToolTip('   id: %d\nlevel: %f\nInactive' %(idx, node.level))
      else:
        color.setAlpha(255)
        glyph.setFlag(qtw.QGraphicsItem.ItemIsSelectable, True)
        glyph.setToolTip('   id: %d\nlevel: %f\ncount: %d' %(idx, node.level, int(count)))
      brush = qtg.QBrush(color)

      if self.nodes[idx].isSelected():
        pen = qtg.QPen(black)
      else:
        pen = qtg.QPen(qtc.Qt.NoPen)

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
      Update the drawing of the arcs.
      @ In, newEdges, list((int,int))), a list of potentially new edges that
        will be added to the drawing.
      @ Out, None
    """

    if newEdges is not None:
      for edge in newEdges:
        path = qtg.QPainterPath()
        self.arcs[edge] = self.scene().addPath(path,qtg.QPen(gray))
        self.arcs[edge].setZValue(0)
        ## The recursive tree traversal can be cumbersome, so we will attach
        ## pointers to the correct nodes to each path graphic.
        self.arcs[edge].source = self.tree.getNode(edge[0])
        self.arcs[edge].sink = self.tree.getNode(edge[1])

    maxLevel = self.maxLevel()
    currentLevel = self.getCurrentLevel()

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
        parent = pathGraphic.source
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

      parent = pathGraphic.source
      child = pathGraphic.sink
      if parent.level > currentLevel and child.level <= currentLevel:
        pathGraphic.setPen(qtg.QPen(self.getColor(idx2),min(2,rect2.width()),cap=qtc.Qt.RoundCap))
      else:
        pathGraphic.setPen(qtg.QPen(gray,0))
      pathGraphic.setPath(path)

  def updateActiveLine(self):
    """
      Update the drawing of the actively set level line, everything below this
      will be considered inactive.
      @ In, None
      @ Out, None
    """
    width = self.scene().width()
    ## If the active box does not exist yet, then create it, don't worry about
    ## the height, we will calculate that in the remainder of this function
    if self.activeLine is None:
      pen = qtg.QPen(gray)
      self.activeLine = self.scene().addLine(0, 0, width, 0, pen)

    height = self.scene().height()
    minDim = min(width,height)

    maxDiameter = self.maxDiameterMultiplier*minDim
    levelRatio = self.getCurrentLevel()/self.maxLevel()

    level = height - self.usableHeight*levelRatio-self.padding

    self.activeLine.setLine(0,level,width,level)

  def createScene(self):
    """
      Function to create the scene including initialization of ui element data
      structures.
      @ In, None
      @ Out, None
    """
    ## Clear data for saving state
    self.nodes = {}
    self.arcs = {}
    self.activeLine = None

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

    for node in self.tree.children:
      count = node.getLeafCount()
      rootWidth = float(count)/self.totalCount

      (myIds,myPoints,myEdges) = node.Layout(xOffset, rootWidth, tSize, tLevel)

      ids.extend(myIds)
      points.extend(myPoints)
      edges.extend(myEdges)
      xOffset += rootWidth

    self.updateNodes(zip(ids,points))
    self.updateArcs(edges)
    self.updateActiveLine()

    self.fitInView(scene.sceneRect(),qtc.Qt.KeepAspectRatio)


  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations. For this class in particular, we will test:
        - Setting the color of one of the nodes by its key value
        - Increasing and decreasing the assigned level of the hierarchy
        - Toggling the edge display and updating the scene after each
        - Triggering the dialog box that allows setting of the truncation size
        - Triggering the dialog box that adjusts the scaling the glyphs
        - Triggering the right-click context menu
        - Triggering the setting of a new level
        - Triggering the selecting of a subset of data
        @ In, None
        @ Out, None
    """
    self.setColor(0,qtg.QColor(255,0,0))
    self.increaseLevel()
    self.decreaseLevel()

    self.edgeAction.setChecked(True)
    self.updateScene()

    self.edgeAction.setChecked(False)
    self.updateScene()

    levels = self.getLevels()
    self.setTruncation()
    self.setDiameterMultiplier()

    genericMouseEvent = qtg.QMouseEvent(qtc.QEvent.MouseMove, qtc.QPoint(0,0), qtc.Qt.MiddleButton, qtc.Qt.MiddleButton, qtc.Qt.NoModifier)
    self.contextMenuEvent(genericMouseEvent)
    self.setLevel()
    self.select()

    super(DendrogramView, self).test()
    BaseHierarchicalView.test(self)
