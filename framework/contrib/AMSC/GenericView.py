#!/usr/bin/env python

from PySide.QtCore import QSize
from PySide.QtGui import QWidget

class GenericView(QWidget):
  """ A base class for all widgets in this package.
  """
  def __init__(self,parent=None,amsc=None,title=None):
    """ Initialization method that can optionally specify the parent widget,
        an AMSC object to reference, and a title for this widget.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, amsc, an optional AMSC_Object specifying the underlying data
          object for this widget to use.
        @ In, title, an optional string specifying the title of this widget.
    """
    super(GenericView, self).__init__(parent)
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
