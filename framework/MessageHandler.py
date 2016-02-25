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
import sys
import os
import time
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
#Internal Modules End--------------------------------------------------------------------------------


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
    color     = kwargs.get('color'    ,None     )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.error(self,etype,msg,str(tag),verbosity,color)

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
    color     = kwargs.get('color'    ,None     )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self,msg,str(tag),verbosity,color)

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
    color     = kwargs.get('color'    ,None     )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self,msg,str(tag),verbosity,color)

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
    color     = kwargs.get('color'    ,None   )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self,msg,str(tag),verbosity,color)

  def getLocalVerbosity(self,default=None):
    """
      Attempts to learn the local verbosity level of itself
      @ OPTIONAL In, default, the verbosity level to return if not found
      @ Out, string, verbosity type (e.g. 'all')
    """
    if hasattr(self,'verbosity'): return self.verbosity
    else: return default


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
      @ Out, None
    """
    self.starttime    = time.time()
    self.printTag     = 'MESSAGE HANDLER'
    self.verbosity    = None
    self.suppressErrs = False
    self.printTime    = True
    self.inColor      = False
    self.verbCode     = {'silent':0, 'quiet':1, 'all':2, 'debug':3}
    self.colorDict    = {'debug':'yellow', 'message':'neutral', 'warning':'magenta', 'error':'red'}
    self.colors={
      'neutral' : '\033[0m',
      'red'     : '\033[31m',
      'green'   : '\033[32m',
      'yellow'  : '\033[33m',
      'blue'    : '\033[34m',
      'magenta' : '\033[35m',
      'cyan'    : '\033[36m'}
    self.warnings     = [] #collection of warnings that were raised during this run


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

  def printWarnings(self):
    """
      Destructor.
      @ In, None
      @ Out, None
    """
    if len(self.warnings)>0:
      print('-'*50)
      print('There were warnings during the simulation run:')
      for w in self.warnings:
        print(w)
      print('-'*50)

  def paint(self,str,color):
    """
      Formats string with color
      @ In, str, string, string
      @ In, color, string, color name
      @ Out, string, formatted string
    """
    if color.lower() not in self.colors.keys():
      self.messaage(self,'Requested color %s not recognized!  Skipping...' %color,'Warning','quiet')
      return str
    return self.colors[color.lower()]+str+self.colors['neutral']

  def setTimePrint(self,msg):
      '''
        Allows the code to toggle timestamp printing.
        @ In, msg, the string that means true or false
        @ Out, None
      '''
      if msg.lower() in utils.stringsThatMeanTrue():
          self.callerLength = 40
          self.tagLength = 30
          self.printTime = True
      elif msg.lower() in utils.stringsThatMeanFalse():
          self.callerLength = 25
          self.tagLength = 15
          self.printTime = False

  def setColor(self,inColor):
    """
      Allows output to screen to be colorized.
      @ In, inColor, string, boolean value
      @ Out, None
    """
    if inColor.lower() in utils.stringsThatMeanTrue():
      self.inColor = True

  def getStringFromCaller(self,obj):
    """
      Determines the appropriate print string from an object
      @ In, obj, preferably an object with a printTag method; otherwise, a string or an object
      @ Out, tag, string to print
    """
    if type(obj).__name__ in ['str','unicode']: return obj
    if hasattr(obj,'printTag'):
      tag = str(obj.printTag)
    else:
      tag = str(obj)
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

  def error(self,caller,etype,message,tag='ERROR',verbosity='silent',color=None):
    """
      Raise an error message, unless errors are suppressed.
      @ In, caller, object, the entity desiring to print a message
      @ In, etype, Error, the type of error to throw
      @ In, message, string, the message to print
      @ In, tag, string, the printed message type (usually Message, Debug, or Warning, and sometimes FIXME)
      @ In, verbosity, string, the print priority of the message
      @ In, color, optional string, color to apply to message
      @ Out, None
    """
    #okay,msg = self._printMessage(caller,message,tag,self.checkVerbosity(verbosity))
    verbval = max(self.getDesiredVerbosity(caller),self.checkVerbosity(self.verbosity))
    #if okay:
    self.message(caller,message,tag,verbosity,color)
    # if in debug mode, raise error so user gets trace
    if not self.suppressErrs and verbval==3:
      self.__del__()
      raise etype(message) #DEBUG mode without suppression
    #otherwise, just exit
    if not self.suppressErrs: #exit after print
      self.__del__()
      sys.exit(1)

  def message(self,caller,message,tag,verbosity,color=None):
    """
      Print a message
      @ In, caller, object, the entity desiring to print a message
      @ In, message, string, the message to print
      @ In, tag, string, the printed message type (usually Message, Debug, or Warning, and sometimes FIXME)
      @ In, verbosity, string, the print priority of the message
      @ In, color, optional string, color to apply to message
      @ Out, None
    """
    verbval = self.checkVerbosity(verbosity)
    okay,msg = self._printMessage(caller,message,tag,verbval,color)
    if tag.lower().strip() == 'warning': self.warnings.append(msg)
    if okay:
      print(msg)
    sys.stdout.flush()

  def _printMessage(self,caller,message,tag,verbval,color=None):
    """
      Checks verbosity to determine whether something should be printed, and formats message
      @ In, caller , object, the entity desiring to print a message
      @ In, message, string, the message to print
      @ In, tag    , string, the printed message type (usually Message, Debug, or Warning, and sometimes FIXME)
      @ In, verbval, int   , the print priority of the message
      @ In, color  , optional string, color to apply to message
      @ Out, bool  , indication if the print should be allowed
      @ Out, msg   , the formatted message
    """
    #allows raising standardized messages
    shouldIPrint = False
    desired = self.getDesiredVerbosity(caller)
    if verbval <= desired: shouldIPrint=True
    if not shouldIPrint: return False,''
    ctag = self.getStringFromCaller(caller)
    msg=self.stdMessage(ctag,tag,message,color)
    return shouldIPrint,msg

  def stdMessage(self,pre,tag,post,color=None):
    """
      Formats string for pretty printing
      @ In, pre  , string, who is printing the message
      @ In, tag  , string, the type of message being printed (Error, Warning, Message, Debug, FIXME, etc)
      @ In, post , string, the actual message body
      @ In, color, optional string, the color to apply to the message
      @ Out, string, formatted message
    """
    msg = ''
    if self.printTime:
      curtime = time.time()-self.starttime
      msg+='('+'{:8.2f}'.format(curtime)+' sec) '
      if self.inColor: msg = self.paint(msg,'cyan')
    msgend = pre.ljust(self.callerLength)[0:self.callerLength] + ': '+tag.ljust(self.tagLength)[0:self.tagLength]+' -> ' + post
    if self.inColor:
      if color is not None: #overrides other options
        msgend = self.paint(msgend,color)
      elif tag.lower() in self.colorDict.keys():
        msgend = self.paint(msgend,self.colorDict[tag.lower()])
    msg+=msgend
    return msg
