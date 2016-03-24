"""
Created on January 12, 2016

@author: maljdp
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3-------------------------------------------

#External Modules---------------------------------------------------------------
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
import MessageHandler
#Internal Modules End-----------------------------------------------------------

try:
  import PySide.QtGui as qtg
  import PySide.QtCore as qtc

  class InteractiveApplication(qtg.QApplication):
    """
      Application - A subclass of the base QApplication where we can instantiate
      our own signals and slots, create UI elements, and manage inter-thread
      communication
    """
    windowClosed = qtc.Signal(str)
    def __init__(self,arg__1,messageHandler):
      """
        A default constructor which will initialize an empty dictionary of user
        interfaces that will be managed by this instance.
      """
      self.UIs = {}
      qtg.QApplication.__init__(self,arg__1)
      self.messageHandler = messageHandler
      self.setQuitOnLastWindowClosed(False)

    def createUI(self, uiType, uiID, params):
      """
        Generates a new user interface element.
          @ In, uiType, names a module/class of user interface type recognized
          by RAVEN, currently the only option is 'MainWindow'
          @ In, uiID, a unique identifier for the UI being generated which will
          be its key value in the internally stored UIs dictionary.
          @ In, params, the params being passed into the user interface
          element's constructor.
          @ Out, None
      """
      ## Note: This assumes that the class is contained in a Module of the same
      ## name. This should be the standard interface for UIs in RAVEN, otherwise
      ## this line of code should change to accommodate whatever standard is used.
      self.UIs[uiID] = getattr(__import__(uiType),uiType)(**params)
      self.UIs[uiID].show()
      self.UIs[uiID].closed.connect(self.closeEvent)

    def closeEvent(self, window):
      """
        Slot for accepting close events of UI windows.
          @ In, window, the window emitting the closed signal.
          @ Out, None
      """
      for key,value in self.UIs.items():
        if value == window:
          self.windowClosed.emit(key)

  __PySideAvailable = True
except ImportError as e:
  __PySideAvailable = False
  pass
