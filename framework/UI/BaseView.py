#!/usr/bin/env python

from PySide.QtCore import QSize
from PySide.QtGui import QWidget

class BaseView(QWidget):
  """ A base class for all widgets in this package.
  """
  def __init__(self,parent=None, title=None, **kwargs):
    """ Initialization method that can optionally specify the parent widget,
        a title for this widget, and specific data used by this view.
        @ In, parent, an optional QWidget that will be the parent of this widget
        @ In, title, an optional string specifying the title of this widget.
        @ In, kwargs, a dictionary holding any specific data needed by this
          view
    """
    super(BaseView, self).__init__(parent)
    if title is None:
      self.setWindowTitle(self.__class__.__name__)
    else:
      self.setWindowTitle(title)
    self.scrollable = False
    self.Reinitialize(parent,title)

  def Reinitialize(self, obj):
    """
      Will restore defaults and clear internal data structures on this widget.
      @ In, obj, an optional object that holds the state of the data being
        manipulated.
    """
    pass

  def clearLayout(self, layout):
    """
      Clears the layout and marks each child widget for deletion.
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
    """
      Specifies the default size hint for this widget
    """
    return QSize(200,200)

  def dataChanged(self):
    """
      Fired when the data being visualized changes
    """
    self.Reinitialize()

  def selectionChanged(self):
    """
      Fired when the user changes the selected data
    """
    pass
