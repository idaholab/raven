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
Created on March 22, 2021

Moved from MessageHandler

@author: talbpaul
"""

import inspect
from pprint import pp

from .. import MessageHandler

class MessageUser(object):
  """
    Inheriting from this class grants access to methods used by the MessageHandler.
  """
  def __init__(self):
    """
      Construct.
      @ In, messageHandler, MessageHandler.MessageHandler, optional, message handler
      @ In, verbosity, MessageHandler.MessageHandler, optional, message handler
      @ Out, None
    """
    # NOTE getMessageHandler is inserted into the builtins by the MessageHandler module
    self.messageHandler = getMessageHandler() # instance responsible for handling messages
    self.verbosity = None                     # message verbosity for this instance

  def setMessageHandler(self, mh):
    """
      Set/replace the message handler.
      @ In, mh, MessageHandler.MessageHandler, message handler
      @ Out, None
    """
    assert isinstance(mh, MessageHandler.MessageHandler)
    self.messageHandler = mh

  def setVerbosity(self, verbosity):
    """
      Set/replace verbosity.
      @ In, verbosity, string, requested verbosity level
      @ Out, None
    """
    self.verbosity = verbosity

  def getVerbosity(self):
    """
      Attempts to learn the local verbosity level of itself
      @ In, None
      @ Out, verbosity, string, verbosity type (e.g. 'all')
    """
    return self.verbosity

  def raiseAnError(self, etype, *args, **kwargs):
    """
      Raises an error. By default shows in all verbosity levels.
      @ In, etype, Exception, Exception class to raise (e.g. IOError)
      @ In, *args, dict, comma-separated list of things to put in message (as print() function)
      @ In, **kwargs, dict, optional arguments, which can be:
                            verbosity, the priority of the message (default 'silent')
                            tag, the message label (default 'ERROR')
      @ Out, None
    """
    verbosity = kwargs.get('verbosity', 'silent')
    tag       = kwargs.get('tag'      , 'ERROR' )
    color     = kwargs.get('color'    , None     )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.error(self, etype, msg, str(tag), verbosity, color)

  def raiseAWarning(self, *args, **kwargs):
    """
      Prints a warning. By default shows in 'quiet', 'all', and 'debug'
      @ In, *args, dict, comma-separated list of things to put in message (as print() function)
      @ In, **kwargs, dict, optional arguments, which can be:
                            verbosity, the priority of the message (default 'quiet')
                            tag, the message label (default 'Warning')
      @ Out, None
    """
    verbosity = kwargs.get('verbosity', 'quiet'  )
    tag       = kwargs.get('tag'      , 'Warning')
    color     = kwargs.get('color'    , None     )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self, msg, str(tag), verbosity, color)

  def raiseAMessage(self, *args, **kwargs):
    """
      Prints a message. By default shows in 'all' and 'debug'
      @ In, *args, dict, comma-separated list of things to put in message (as print() function)
      @ In, **kwargs, dict, optional arguments, which can be:
                            verbosity, the priority of the message (default 'all')
                            tag, the message label (default 'Message')
      @ Out, None
    """
    verbosity  = kwargs.get('verbosity' , 'all'    )
    tag        = kwargs.get('tag'       , 'Message')
    color      = kwargs.get('color'     , None     )
    forcePrint = kwargs.get('forcePrint', False     )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self, msg, str(tag), verbosity, color, forcePrint=forcePrint)

  def raiseADebug(self, *args, **kwargs):
    """
      Prints a debug message. By default shows only in 'debug'
      @ In, *args, dict, comma-separated list of things to put in message (as print() function)
      @ In, **kwargs, dict, optional arguments, which can be:
                            verbosity, the priority of the message (default 'debug')
                            tag, the message label (default 'DEBUG')
      @ Out, None
    """
    verbosity = kwargs.get('verbosity', 'debug')
    tag = kwargs.get('tag', 'DEBUG')
    color = kwargs.get('color', None)
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self, msg, str(tag), verbosity, color)

  def raiseWhatsThis(self, label, obj, tag="DEBUGG"):
    """
      Shortcut for a commonly-used print command for developers.
      @ In, label, str, string description of object
      @ In, obj, object, this is the thing to print
      @ In, tag, str, optional, identifying prefix to include with print
    """
    stack = inspect.stack()
    print('/'+'*'*80)
    print(f'{tag} WHAT THIS IS:')
    for f, frame in enumerate(stack[:0:-1]):
      if f == 0:
        starter = 'started by'
      else:
        starter = 'called'
      print(f'{tag} -> {starter} "{frame[3]}" line {frame[2]} in file "{frame[1]}"')
    print(f'{tag} {label}: type "{type(obj)}"')
    print(pp(obj))
    print('\\'+'*'*80)

  def raiseMarker(self, tag=""):
    """
      Shortcut for a commonly-used print command for developers.
      @ In, label, str, string description of object
      @ In, obj, object, this is the thing to print
      @ In, tag, str, optional, identifying prefix to include with print
    """
    stack = inspect.stack()
    print(f'*** MARKER {tag} method "{stack[1][3]}" line {stack[1][2]} file "{stack[1][1]}"')
