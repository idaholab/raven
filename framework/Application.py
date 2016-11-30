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

  class InteractiveApplication(qtg.QApplication,MessageHandler.MessageUser):
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
      self.printTag = 'RAVEN Application'
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
      # self.UIs[uiID] = getattr(__import__(uiType),uiType)(**params)
      try:
        self.UIs[uiID] = getattr(__import__('UI.'+uiType),uiType)(**params)
        self.UIs[uiID].show()
        self.UIs[uiID].closed.connect(self.closeEvent)
      except ImportError as e:

        message = 'The requested interactive UI is unavailable. '
        message += 'RAVEN will continue in non-interactive mode for this step. '
        message += 'Please send the following debug information to the '
        message += 'developer list:\n\t' + str(e) + '\n'

        ## This will ensure that the waiting threads are released.
        self.windowClosed.emit(uiID)

        ## We will execute a warning since the system can recover and proceed as
        ## if in a non-interactive mode for this step, and potentially recover
        ## and run more UIs in a later step. This is a failure in some sense, so
        ## I am elevating the verbosity to be silent for this warning.
        self.raiseAWarning(message,verbosity='silent')

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
