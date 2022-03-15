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
Created on January 12, 2016

@author: maljdp
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3-------------------------------------------

#External Modules---------------------------------------------------------------
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .BaseClasses import MessageUser
from .Interaction import Interaction
#Internal Modules End-----------------------------------------------------------

try:
  import PySide.QtGui as qtw
  import PySide.QtCore as qtc
  __QtAvailable = True
except ImportError as e:
  try:
    import PySide2.QtWidgets as qtw
    import PySide2.QtCore as qtc
    __QtAvailable = True
  except ImportError as e:
    __QtAvailable = False

if __QtAvailable:
  class InteractiveApplication(qtw.QApplication, MessageUser):
    """
      Application - A subclass of the base QApplication where we can instantiate
      our own signals and slots, create UI elements, and manage inter-thread
      communication
    """
    windowClosed = qtc.Signal(str)

    def __init__(self, arg1, interactionType=Interaction.Yes):
      """
        A default constructor which will initialize an empty dictionary of user
        interfaces that will be managed by this instance.
        @ In, arg1, list, unknown
        @ In, interactionType, Interaction, boolean-like
        @ Out, None
      """
      super().__init__()
      self.printTag = 'RAVEN Application'
      self.UIs = {}
      self.interactionType = interactionType
      qtw.QApplication.__init__(self, arg1)
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
        ## We are going to add the debug parameter based on what the user
        ## requested from the RAVEN command line.
        params['debug'] = (self.interactionType in [Interaction.Debug, Interaction.Test])

        self.UIs[uiID] = getattr(__import__('UI.'+uiType),uiType)(**params)
        self.UIs[uiID].closed.connect(self.closeEvent)
        self.UIs[uiID].show()

        if self.interactionType == Interaction.Test:
          message = 'Test mode: the UI will be closed auotmatically.'
          self.raiseAWarning(message, verbosity='silent')
          self.UIs[uiID].test()
          ## We may want to come up with a way of ensuring that each signal has
          ## been fully processed by the UI, maybe it handles it internally or
          ## maybe we query it here before calling close() -- DPM 5/9/2017
          # time.sleep(10)
          self.UIs[uiID].close()

      except ImportError as e:

        message = 'The requested interactive UI is unavailable. RAVEN will continue in non-interactive mode for this step. Please file an issue on gitlab with the following debug information:\n\t' + str(e) + '\n'

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
