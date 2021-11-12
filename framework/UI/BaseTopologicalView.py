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
  A base class for all widgets associated to the TopologicalWindow.
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

class BaseTopologicalView(QWidget):
  """
    A base class for all widgets associated to the TopologicalWindow.
  """
  def __init__(self,parent=None,amsc=None,title=None):
    """ Initialization method that can optionally specify the parent widget,
        an AMSC object to reference, and a title for this widget.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, amsc, an optional AMSC_Object specifying the underlying data
          object for this widget to use.
        @ In, title, an optional string specifying the title of this widget.
    """
    super(BaseTopologicalView, self).__init__(parent)
    self.amsc = amsc
    if title is None:
      self.setWindowTitle(self.__class__.__name__)
    else:
      self.setWindowTitle(title)
    self.scrollable = False
    self.Reinitialize(parent,amsc,title)

  def Reinitialize(self):
    """ Will restore defaults and clear internal data structures on this widget.
    """
    pass

  def clearLayout(self, layout):
    """ Clears the layout and marks each child widget for deletion.
    """
    if layout is not None:
      while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
          widget.deleteLater()
        else:
          self.clearLayout(item.layout())

  def sizeHint(self):
    """ Specifies the default size hint for this widget
    """
    return QSize(200,200)

  def dataChanged(self):
    """ Fired when the AMSC loads a new dataset
    """
    self.Reinitialize()

  def filterChanged(self):
    """ Fired when the user filters data on a different view
        (the AMSC must store this somehow?)
    """
    pass

  def selectionChanged(self):
    """ Fired when the user selects a piece of data on a different view
        (the AMSC must store this somehow?)
    """
    pass

  def persistenceChanged(self):
    """ Fired when the AMSC changes its current persistence value
        (This could be the same as changing the 'filter' of the data?)
    """
    pass

  def modelsChanged(self):
    """ Fired when the AMSC rebuilds its local models
    """
    pass

  def weightsChanged(self):
    """ Fired when the data weights are changed
    """
    pass

  def test(self):
    """
        A test function for performing operations on this class that need to be
        automatically tested such as simulating mouse and keyboard events, and
        other internal operations. For this class in particular, we will test:
        - Reinitialization of this view
        - The generic slots of this method that correspond to actions on the
          underlying data such as changing the data iteself, filtering the data,
          selecting the data, changing the persistence, updating the models, and
          adjusting the weights of the data.
        @ In, None
        @ Out, None
    """
    BaseTopologicalView.Reinitialize(self)
    BaseTopologicalView.dataChanged(self)
    BaseTopologicalView.filterChanged(self)
    BaseTopologicalView.selectionChanged(self)
    BaseTopologicalView.persistenceChanged(self)
    BaseTopologicalView.modelsChanged(self)
    BaseTopologicalView.weightsChanged(self)
