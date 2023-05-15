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
Created on Apr 20, 2015

@author: talbpaul
"""
import sys
import time
import bisect
import builtins

from .utils import utils

_starttime = time.time()

"""
HOW THIS MODULE WORKS

The intention is for a single instance of the MessageHandler class to exist in any simulation.
Currently, that instance is created in the Simulation initialization and propogated through
all the RAVEN objects.  This usually happens by passing it to BaseClass.readXML, but for
objects that don't inherit from BaseClass, the messageHandler instance should be passed
and set via instantiation or initialization.  The appropriate class member to point at the
messageHandler instance reference is "self.messageHandler," for reasons that will be made clear
with the BaseClasses.MessageUser superclass.

While an object can access the messageHandler to raise messages and errors, for convienience
RAVEN provides the MessageUser superclass, which BaseType and (almost?) all other Raven objects
inherit from.  This provides simplistic hooks for a developer to raise an error or message
with the standard message priorities, as

self.raiseAnError(IOError, 'Input value is invalid:', value)

There are currently 4 verbosity levels/message priorities.  They are:
 - silent: only errors are displayed
 - quiet : errors and warnings are displayed
 - all   : (default) errors, warnings, and messages are displayed
 - debug : errors, warnings, messages, and debug messages are displayed

The developer can change the priority level of their raised messages through the 'verbosity'
keyword.  For example,

self.raiseAMessage('Hello, World', verbosity='silent')

will be printed along with errors if the simulation verbosity is set to 'silent', as well as
all other levels.

TL;DR: BaseClasses/MessageUser is a superclass that gives access to hooks to the simulation's MessageHandler
instance, while the MessageHandler is an output stream control tool.

In an effort to make the MH more flexible, we insert getMessageHandler into the python "builtins" module.
This means that any time after this module (MessageHandler) is imported, you can use
"getMessageHandler(name='default')" to retrieve a particular message handler as identified by "name".
"""

class MessageHandler(object):
  """
    Class for handling messages, warnings, and errors in RAVEN.  One instance of this
    class should be created at the start of the Simulation and propagated through
    the readMoreXML function of the BaseClass, and initialization of other classes.
  """
  def __init__(self):
    """
      Class constructor
      @ In, None
      @ Out, None
    """
    self.starttime    = _starttime
    self.printTag     = 'MESSAGE HANDLER'
    self.verbosity    = 'all'
    self.callerLength = 25
    self.tagLength    = 15
    self.suppressErrs = False
    self.printTime    = True
    self.inColor      = False
    self.verbCode     = {'silent' : 0,
                         'quiet'  : 1,
                         'all'    : 2,
                         'debug'  : 3}
    self.colorDict    = {'debug'  : 'yellow',
                         'message': 'neutral',
                         'warning': 'magenta',
                         'error'  : 'red'}
    self.colors       = {'neutral': '\033[0m',
                         'red'    : '\033[31m',
                         'green'  : '\033[32m',
                         'yellow' : '\033[33m',
                         'blue'   : '\033[34m',
                         'magenta': '\033[35m',
                         'cyan'   : '\033[36m'}
    self.warnings     = [] # collection of warnings that were raised during this run
    self.warningCount = [] # count of the collections of warning above

  def initialize(self, initDict):
    """
      Initializes basic instance attributes
      @ In, initDict, dict, dictionary of global options
      @ Out, None
    """
    self.verbosity = initDict.get('verbosity', 'all').lower()
    self.callerLength = initDict.get('callerLength', 25)
    self.tagLength = initDict.get('tagLength', 15)
    self.suppressErrs = utils.stringIsTrue(initDict.get('suppressErrs', 'False'))

  def printWarnings(self):
    """
      Prints a summary of warnings collected during the run.
      @ In, None
      @ Out, None
    """
    if len(self.warnings)>0:
      if self.verbCode[self.verbosity] > 0:
        print('-'*50)
        print(f'There were {sum(self.warningCount)} warnings during the simulation run:')
        for w, warning in enumerate(self.warnings):
          count = self.warningCount[w]
          if count > 1:
            print(f'({self.warningCount[w]} times) {warning}')
          else:
            print(f'({self.warningCount[w]} time) {warning}')
        print('-'*50)
      else:
        print(f'There were {sum(self.warningCount)} warnings during the simulation run.')

  def paint(self, string, color):
    """
      Formats string with color
      @ In, string, string, string
      @ In, color, string, color name
      @ Out, paint, string, formatted string
    """
    if color.lower() not in self.colors:
      self.message(self, f'Requested color {color} not recognized!  Skipping...', 'Warning', 'quiet')
      return string
    return self.colors[color.lower()] + string + self.colors['neutral']

  def setTimePrint(self, msg):
    """
      Allows the code to toggle timestamp printing.
      @ In, msg, string, the string that means true or false
      @ Out, None
    """
    if utils.stringIsTrue(msg):
      self.callerLength = 40
      self.tagLength = 30
      self.printTime = True
    elif utils.stringIsFalse(msg):
      self.callerLength = 25
      self.tagLength = 15
      self.printTime = False

  def setColor(self, inColor):
    """
      Allows output to screen to be colorized.
      @ In, inColor, string, boolean value
      @ Out, None
    """
    if utils.stringIsTrue(inColor):
      self.inColor = True

  def getStringFromCaller(self, obj):
    """
      Determines the appropriate print string from an object
      @ In, obj, instance, preferably an object with a printTag method; otherwise, a string or an object
      @ Out, tag, string, string to print
    """
    if type(obj).__name__ in ['str', 'unicode']: # ?when is this ever not true?
      return obj
    if hasattr(obj,'printTag'):
      tag = str(obj.printTag)
    else:
      tag = str(obj)

    return tag

  def getDesiredVerbosity(self, caller):
    """
      Tries to use local verbosity; otherwise uses global
      @ In, caller, instance, the object desiring to print
      @ Out, desVerbosity, int, integer equivalent to verbosity level
    """
    if hasattr(caller, 'getVerbosity'):
      localVerb = caller.getVerbosity()
    else:
      localVerb = None
    if localVerb is None:
      localVerb = self.verbosity
    desVerbosity = self.checkVerbosity(localVerb)

    return desVerbosity

  def checkVerbosity(self, verb):
    """
      Converts English-readable verbosity to computer-legible integer
      @ In, verb, string, the string verbosity equivalent
      @ Out, currentVerb, int, integer equivalent to verbosity level
    """
    if str(verb).strip().lower() not in self.verbCode:
      raise IOError(f'Verbosity key {verb} not recognized!  Options are {list(self.verbCode.keys())}')
    currentVerb = self.verbCode[str(verb).strip().lower()]

    return currentVerb

  def error(self, caller, etype, message, tag='ERROR', verbosity='silent', color=None):
    """
      Raise an error message, unless errors are suppressed.
      @ In, caller, object, the entity desiring to print a message
      @ In, etype, Error, the type of error to throw
      @ In, message, string, the message to print
      @ In, tag, string, optional, the printed message type (usually Message, Debug, or Warning, and sometimes FIXME)
      @ In, verbosity, string, optional, the print priority of the message
      @ In, color, string, optional, color to apply to message
      @ Out, None
    """
    verbval = max(self.getDesiredVerbosity(caller), self.checkVerbosity(self.verbosity))
    self.message(caller, message, tag, verbosity, color=color)
    if not self.suppressErrs:
      self.printWarnings()
      # debug mode gets full traceback, others quieted
      if verbval < 3:
        #all, quiet, silent
        sys.tracebacklimit = 0
      raise etype(message)

  def message(self, caller, message, tag, verbosity, color=None, writeTo=sys.stdout, forcePrint=False):
    """
      Print a message
      @ In, caller, object, the entity desiring to print a message
      @ In, message, string, the message to print
      @ In, tag, string, the printed message type (usually Message, Debug, or Warning, and sometimes FIXME)
      @ In, verbosity, string, the print priority of the message
      @ In, color, string, optional, color to apply to message
      @ In, forcePrint, bool, optional, force the print independetly on the verbosity level? Defaul False
      @ Out, None
    """
    verbval = self.checkVerbosity(verbosity)
    okay, msg = self._printMessage(caller, message, tag, verbval, color, forcePrint)
    if tag.lower().strip() == 'warning':
      self.addWarning(message)
    if okay:
      print(msg, file=writeTo)
    sys.stdout.flush()

  def addWarning(self, msg):
    """
      Stores warnings so that they can be reported in summary later.
      @ In, msg, string, only the main part of the message, used to determine uniqueness
      @ Out, None
    """
    index = bisect.bisect_left(self.warnings, msg)
    if len(self.warnings) == 0 or index == len(self.warnings) or self.warnings[index] != msg:
      self.warnings.insert(index,msg)
      self.warningCount.insert(index,1)
    else:
      self.warningCount[index] += 1

  def _printMessage(self, caller, message, tag, verbval, color=None, forcePrint=False):
    """
      Checks verbosity to determine whether something should be printed, and formats message
      @ In, caller , object, the entity desiring to print a message
      @ In, message, string, the message to print
      @ In, tag    , string, the printed message type (usually Message, Debug, or Warning, and sometimes FIXME)
      @ In, verbval, int   , the print priority of the message
      @ In, color, string, optional, color to apply to message
      @ In, forcePrint, bool, optional, force the print independetly on the verbosity level? Defaul False
      @ Out, (shouldIPrint,msg), tuple, shouldIPrint -> bool, indication if the print should be allowed
                                        msg          -> string, the formatted message
    """
    # allows raising standardized messages
    shouldIPrint = False
    desired = self.getDesiredVerbosity(caller)
    if verbval <= desired or forcePrint:
      shouldIPrint = True
    if not shouldIPrint:
      return False, ''
    ctag = self.getStringFromCaller(caller)
    msg=self.stdMessage(ctag,tag,message,color)

    return shouldIPrint, msg

  def stdMessage(self, pre, tag, post, color=None):
    """
      Formats string for pretty printing
      @ In, pre  , string, who is printing the message
      @ In, tag  , string, the type of message being printed (Error, Warning, Message, Debug, FIXME, etc)
      @ In, post , string, the actual message body
      @ In, color, string, optional, color to apply to message
      @ Out, msg, string, formatted message
    """
    msg = ''
    if self.printTime:
      curtime = time.time() - self.starttime
      msg += f'({curtime:8.2f} sec) '
      if self.inColor:
        msg = self.paint(msg, 'cyan')
    msgend = pre.ljust(self.callerLength)[0:self.callerLength] + ': ' + tag.ljust(self.tagLength)[0:self.tagLength] + ' -> ' + post
    if self.inColor:
      if color is not None:
        #overrides other options
        msgend = self.paint(msgend, color)
      elif tag.lower() in self.colorDict:
        msgend = self.paint(msgend,self.colorDict[tag.lower()])
    msg += msgend

    return msg

def timePrint(message):
  """
    Prints the time since start then the message
    @ In, message, string
    @ Out, None
  """
  curtime = time.time() - _starttime
  msg = f'({curtime:8.2f} sec) '
  print(msg + message)

_handlers = {}

def makeHandler(name):
  """
    Instantiate and register new instance of message handler
    @ In, name, str, identifying name for new handler
    @ Out, makeHandler, MessageHandler, instance
  """
  handler = MessageHandler()
  _handlers[name] = handler

  return handler

# default handler
makeHandler('default')

def getHandler(name='default'):
  """
    Retrieve a message handling instance.
    Styled after the Python logging module, maybe we should be switching to that.
    @ In, name, str, optional, identifying name of handler to return
    @ Out, getHandler, MessageHandler, instance (created if not existing)
  """
  h = _handlers.get(name, None)
  if h is None:
    h = makeHandler(name)
  # NOTE: idk why, but h = _handlers.get(name, makeHandler(name)) does not work.
  # I think it's because it executes makeHandler(name) regardless of if name is present or not.
  return h

builtins.getMessageHandler = getHandler
