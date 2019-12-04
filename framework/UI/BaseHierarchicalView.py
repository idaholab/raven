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
  This BaseHierarchicalView is used by the HierarchicalWindow
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3
try:
  from PySide.QtCore import QSize
  from PySide.QtGui import QWidget
except ImportError as e:
  from PySide2.QtCore import QSize
  from PySide2.QtWidgets import QWidget


from .ZoomableGraphicsView import ZoomableGraphicsView

class BaseHierarchicalView(QWidget):
  """ A base class for all widgets in this package.
  """
  def __init__(self, mainWindow=None, title=None):
    """
      Initialization method that can optionally specify the parent widget,
        a title for this widget, and specific data used by this view.
        @ In, mainWindow, an optional QWidget that will be the parent of this widget
        @ In, title, an optional string specifying the title of this widget.
    """
    ## This is a stupid hack around the problem of multiple inheritance, maybe
    ## I should rethink the class hierarchy here?
    #if not isinstance(self, ZoomableGraphicsView):
    #  super(BaseHierarchicalView, self).__init__(mainWindow)
    QWidget.__init__(self, mainWindow)

    if title is None:
      self.setWindowTitle(self.__class__.__name__)
    else:
      self.setWindowTitle(title)
    self.scrollable = False
    self.mainWindow = mainWindow

  def sizeHint(self):
    """
      Specifies the default size hint for this widget
      @ In, None
      @ Out, size, QSize, suggested size
    """
    return QSize(200,200)

  def clearLayout(self, layout):
    """
      Clears the layout and marks each child widget for deletion.
      @ In, layout, QLayout, the layout to clear
      @ Out, None
    """
    if layout is not None:
      while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
          widget.deleteLater()
        else:
          self.clearLayout(item.layout())


  def updateScene(self):
    """
      This method will perform necessary operations in order to update the
      scene drawn on this view's canvas
      @ In, None
      @ Out, None
    """
    pass

  def colorChanged(self):
    """
      This callback will ensure the UI is appropriately updated when a user
      triggers a color change to one of the partitions
      @ In, None
      @ Out, None
    """
    self.updateScene()

  def levelChanged(self):
    """
      This callback will ensure the UI is appropriately updated when a user
      triggers a level change to the hierarchy
      @ In, None
      @ Out, None
    """
    self.updateScene()

  def selectionChanged(self):
    """
      This callback will ensure the UI is appropriately updated when a user
      triggers a change to the selected data of the hierarchy
      @ In, None
      @ Out, None
    """
    self.updateScene()

  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations.  For this class in particular, we will test:
        - Retrieving the size hint of this view.
        - A generic update
        - Clearing the layout of this view.
        @ In, None
        @ Out, None
    """
    sizeHint = BaseHierarchicalView.sizeHint(self)
    BaseHierarchicalView.updateScene(self)
    BaseHierarchicalView.clearLayout(self, self.layout())
