'''
Created on Apr 20, 2015

@author: talbpaul
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import platform
import os
import time
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
#Internal Modules End--------------------------------------------------------------------------------

# set a global variable for backend default setting
if platform.system() == 'Windows': disAvail = True
else:
  if os.getenv('DISPLAY'): disAvail = True
  else:                    disAvail = False

#custom exceptions
class NoMoreSamplesNeeded(GeneratorExit): pass

'''
HOW THIS MODULE WORKS

The intention is for a single instance of the MessageHandler class to exist in any simulation.
Currently, that instance is created in the Simulation initialization and propogated through
all the RAVEN objects.  This usually happens by passing it to BaseClass.readXML, but for
objects that don't inherit from BaseClass, the messageHandler instance should be passed
and set via instantiation or initialization.  The appropriate class member to point at the
messageHandler instance reference is "self.messageHandler," for reasons that will be made clear
with the  MessageUser superclass.

While an object can access the messageHandler to raise messages and errors, for convienience
we provide the MessageUser superclass, which BaseType and (almost?) all other Raven objects
inherit from.  This provides simplistic hooks for a developer to raise an error or message
with the standard message priorities, as

self.raiseAnError(IOError,'Input value is invalid:',value)

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

TL;DR: MessageUser is a superclass that gives access to hooks to the simulation's MessageHandler
instance, while the MessageHandler is an output stream control tool.
'''

class MessageUser(object):
  """
    Inheriting from this class grants access to methods used by the MessageHandler.
    In order to work properly, a subclass of this superclass should have a member
    'self.messageHandler' that references a MessageHandler instance.
  """
  def raiseAnError(self,etype,*args,**kwargs):
    """
      Raises an error. By default shows in all verbosity levels.
      @ In, etype, Exception class to raise (e.g. IOError)
      @ In, *args, comma-seperated list of things to put in message (as print() function)
      @ In, **kwargs, optional arguments, which can be:
                      verbosity, the priority of the message (default 'silent')
                      tag, the message label (default 'ERROR')
      @ Out, None
    """
    verbosity = kwargs.get('verbosity','silent')
    tag       = kwargs.get('tag'      ,'ERROR' )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.error(self,etype,msg,str(tag),verbosity)

  def raiseAWarning(self,*args,**kwargs):
    """
      Prints a warning. By default shows in 'quiet', 'all', and 'debug'
      @ In, *args, comma-seperated list of things to put in message (as print() function)
      @ In, **kwargs, optional arguments, which can be:
                      verbosity, the priority of the message (default 'quiet')
                      tag, the message label (default 'Warning')
      @ Out, None
    """
    verbosity = kwargs.get('verbosity','quiet'  )
    tag       = kwargs.get('tag'      ,'Warning')
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self,msg,str(tag),verbosity)

  def raiseAMessage(self,*args,**kwargs):
    """
      Prints a message. By default shows in 'all' and 'debug'
      @ In, *args, comma-seperated list of things to put in message (as print() function)
      @ In, **kwargs, optional arguments, which can be:
                      verbosity, the priority of the message (default 'all')
                      tag, the message label (default 'Message')
      @ Out, None
    """
    verbosity = kwargs.get('verbosity','all'    )
    tag       = kwargs.get('tag'      ,'Message')
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self,msg,str(tag),verbosity)

  def raiseADebug(self,*args,**kwargs):
    """
      Prints a debug message. By default shows only in 'debug'
      @ In, *args, comma-seperated list of things to put in message (as print() function)
      @ In, **kwargs, optional arguments, which can be:
                      verbosity, the priority of the message (default 'debug')
                      tag, the message label (default 'DEBUG')
      @ Out, None
    """
    verbosity = kwargs.get('verbosity','debug')
    tag       = kwargs.get('tag'      ,'DEBUG')
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self,msg,str(tag),verbosity)

  def getLocalVerbosity(self,default=None):
    """
      Attempts to learn the local verbosity level of itself
      @ OPTIONAL In, default, the verbosity level to return if not found
      @ Out, string, verbosity type (e.g. 'all')
    """
    try: return self.verbosity
    except AttributeError: return default


class MessageHandler(object):
  """
  Class for handling messages, warnings, and errors in RAVEN.  One instance of this
  class should be created at the start of the Simulation and propagated through
  the readMoreXML function of the BaseClass, and initialization of other classes.
  """
  def __init__(self):
    """
      Init of class
      @In, None
      @Out, None
    """
    self.starttime    = time.time()
    self.printTag     = 'MESSAGE HANDLER'
    self.verbosity    = None
    self.suppressErrs = False
    self.printTime    = True
    self.verbCode     = {'silent':0, 'quiet':1, 'all':2, 'debug':3}

  def initialize(self,initDict):
    """
      Initializes basic instance attributes
      @ In, initDict, dictionary of global options
      @ Out, None
    """
    self.verbosity     = initDict.get('verbosity','all')
    self.callerLength  = initDict.get('callerLength',40)
    self.tagLength     = initDict.get('tagLength',30)
    self.suppressErrs  = initDict['suppressErrs'] in utils.stringsThatMeanTrue() if 'suppressErrs' in initDict.keys() else False

  def setTimePrint(self,msg):
      '''
        Allows the code to toggle timestamp printing.
        @ In, msg, the string that means true or false
        @ Out, None
      '''
      if msg in utils.stringsThatMeanTrue():
          self.callerLength = 40
          self.tagLength = 30
          self.printTime = True
      elif msg in utils.stringsThatMeanFalse():
          self.callerLength = 25
          self.tagLength = 15
          self.printTime = False

  def getStringFromCaller(self,obj):
    """
      Determines the appropriate print string from an object
      @ In, obj, preferably an object with a printTag method; otherwise, a string or an object
      @ Out, tag, string to print
    """
    if type(obj) in [str,unicode]: return obj
    try: obj.printTag
    except AttributeError: tag = str(obj)
    else: tag = str(obj.printTag)
    return tag

  def getDesiredVerbosity(self,caller):
    """
      Tries to use local verbosity; otherwise uses global
      @ In, caller, the object desiring to print
      @ Out, integer, integer equivalent to verbosity level
    """
    localVerb = caller.getLocalVerbosity(default=self.verbosity)
    if localVerb == None: localVerb = self.verbosity
    return self.checkVerbosity(localVerb) #self.verbCode[str(localVerb).strip().lower()]

  def checkVerbosity(self,verb):
    """
      Converts English-readable verbosity to computer-legible integer
      @ In, verb, the string verbosity equivalent
      @ Out, integer, integer equivalent to verbosity level
    """
    if str(verb).strip().lower() not in self.verbCode.keys():
      raise IOError('Verbosity key '+str(verb)+' not recognized!  Options are '+str(self.verbCode.keys()+[None]),'ERROR','silent')
    return self.verbCode[str(verb).strip().lower()]

  def error(self,caller,etype,message,tag='ERROR',verbosity='silent'):
    """
      Raise an error message, unless errors are suppressed.
      @ In, caller, the entity desiring to raise an error
      @ In, etype, the type of error to throw
      @ In, message, the accompanying message for the error
      @ OPTIONAL In, tag, the printed error type (default ERROR)
      @ OPTIONAL In, verbosity, the print priority of the message (default 'silent', highest priority)
      @ Out, None
    """
    verbval = self.checkVerbosity(verbosity)
    okay,msg = self._printMessage(caller,message,tag,verbval)
    if okay:
      if not self.suppressErrs: raise etype(msg)
      else: print(msg)

  def message(self,caller,message,tag,verbosity):
    """
      Print a message
      @ In, caller, the entity desiring to print a message
      @ In, message, the message to print
      @ In, tag, the printed message type (usually Message, Debug, or Warning, and sometimes FIXME)
      @ In, verbosity, the print priority of the message
      @ Out, None
    """
    verbval = self.checkVerbosity(verbosity)
    okay,msg = self._printMessage(caller,message,tag,verbval)
    if okay: print(msg)

  def _printMessage(self,caller,message,tag,verbval):
    """
      Checks verbosity to determine whether something should be printed, and formats message
      @ In, caller, the entity desiring to print a message
      @ In, message, the message to print
      @ In, tag, the printed message type (usually Message, Debug, or Warning, and sometimes FIXME)
      @ In, verbval, the print priority of the message
      @ Out, bool, indication if the print should be allowed
      @ Out, msg, the formatted message
    """
    #allows raising standardized messages
    shouldIPrint = False
    desired = self.getDesiredVerbosity(caller)
    if verbval <= desired: shouldIPrint=True
    if not shouldIPrint: return False,''
    ctag = self.getStringFromCaller(caller)
    msg=self.stdMessage(ctag,tag,message)
    return shouldIPrint,msg

  def stdMessage(self,pre,tag,post):
    """
      Formats string for pretty printing
      @ In, pre, string of who is printing the message
      @ In, tag, the type of message being printed (Error, Warning, Message, Debug, FIXME, etc)
      @ In, post, the actual message body
      @ Out, string, formatted message
    """
    msg = ''
    if self.printTime:
      curtime = time.time()-self.starttime
      msg+='('+'{:8.2f}'.format(curtime)+' sec) '
    msg+=pre.ljust(self.callerLength)[0:self.callerLength] + ': '
    msg+=tag.ljust(self.tagLength)[0:self.tagLength]+' -> '
    msg+=post
    return msg
