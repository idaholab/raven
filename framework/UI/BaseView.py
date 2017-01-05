#!/usr/bin/env python

from PySide.QtCore import QSize
from PySide.QtGui import QWidget

from ZoomableGraphicsView import ZoomableGraphicsView

class BaseView(QWidget):
  """ A base class for all widgets in this package.
  """
  def __init__(self, mainWindow=None, title=None):
    """ Initialization method that can optionally specify the parent widget,
        a title for this widget, and specific data used by this view.
        @ In, mainWindow, an optional QWidget that will be the parent of this widget
        @ In, title, an optional string specifying the title of this widget.
    """

    ## This is a stupid hack around the problem of multiple inheritance, maybe
    ## I should rethink the class hierarchy here?
    if not isinstance(self, ZoomableGraphicsView):
      super(BaseView, self).__init__(mainWindow)

    if title is None:
      self.setWindowTitle(self.__class__.__name__)
    else:
      self.setWindowTitle(title)
    self.scrollable = False
    self.mainWindow = mainWindow

  def sizeHint(self):
    """
      Specifies the default size hint for this widget
    """
    return QSize(200,200)

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


  def updateScene(self):
    """
    """
    pass

  def colorChanged(self):
    """
    """
    self.updateScene()

  def levelChanged(self):
    """
    """
    self.updateScene()
