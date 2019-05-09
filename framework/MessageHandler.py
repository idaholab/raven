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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import time
import bisect
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
#Internal Modules End--------------------------------------------------------------------------------


#custom exceptions
class NoMoreSamplesNeeded(GeneratorExit):
  """
    Custom exception class for no more samples
  """
  pass

"""
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
"""

class MessageUser(object):
  """
    Inheriting from this class grants access to methods used by the MessageHandler.
    In order to work properly, a subclass of this superclass should have a member
    'self.messageHandler' that references a MessageHandler instance.
  """
  def raiseAnError(self,etype,*args,**kwargs):
    """
      Raises an error. By default shows in all verbosity levels.
      @ In, etype, Exception, Exception class to raise (e.g. IOError)
      @ In, *args, dict, comma-seperated list of things to put in message (as print() function)
      @ In, **kwargs, dict, optional arguments, which can be:
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
      @ In, *args, dict, comma-seperated list of things to put in message (as print() function)
      @ In, **kwargs, dict, optional arguments, which can be:
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
      @ In, *args, dict, comma-seperated list of things to put in message (as print() function)
      @ In, **kwargs, dict, optional arguments, which can be:
                            verbosity, the priority of the message (default 'all')
                            tag, the message label (default 'Message')
      @ Out, None
    """
    verbosity  = kwargs.get('verbosity' ,'all'    )
    tag        = kwargs.get('tag'       ,'Message')
    color      = kwargs.get('color'     ,None     )
    forcePrint = kwargs.get('forcePrint',False     )
    msg = ' '.join(str(a) for a in args)
    self.messageHandler.message(self,msg,str(tag),verbosity,color,forcePrint=forcePrint)

  def raiseADebug(self,*args,**kwargs):
    """
      Prints a debug message. By default shows only in 'debug'
      @ In, *args, dict, comma-seperated list of things to put in message (as print() function)
      @ In, **kwargs, dict, optional arguments, which can be:
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
      @ In, default, string, optional, the verbosity level to return if not found
      @ Out, verbosity, string, verbosity type (e.g. 'all')
    """
    if hasattr(self,'verbosity'):
      return self.verbosity
    else:
      return default


class MessageHandler(object):
  """
    Class for handling messages, warnings, and errors in RAVEN.  One instance of this
    class should be created at the start of the Simulation and propagated through
    the readMoreXML function of the BaseClass, and initialization of other classes.
  """
  def __init__(self):
    """
      Init of class
      @ In, None
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
    self.warningCount = [] #count of the collections of warning above

  def initialize(self,initDict):
    """
      Initializes basic instance attributes
      @ In, initDict, dict, dictionary of global options
      @ Out, None
    """
    self.verbosity     = initDict.get('verbosity','all').lower()
    self.callerLength  = initDict.get('callerLength',40)
    self.tagLength     = initDict.get('tagLength',30)
    self.suppressErrs  = initDict['suppressErrs'] in utils.stringsThatMeanTrue() if 'suppressErrs' in initDict.keys() else False

  def printWarnings(self):
    """
      Prints a summary of warnings collected during the run.
      @ In, None
      @ Out, None
    """
    if len(self.warnings)>0:
      if self.verbCode[self.verbosity]>0:
        print('-'*50)
        print('There were %i warnings during the simulation run:' %sum(self.warningCount))
        for w,warning in enumerate(self.warnings):
          count = self.warningCount[w]
          time = 'time'
          if count > 1:
            time += 's'
          print('(%i %s) %s' %(self.warningCount[w],time,warning))
        print('-'*50)
      else:
        print('There were %i warnings during the simulation run.' %sum(self.warningCount))

  def paint(self,str,color):
    """
      Formats string with color
      @ In, str, string, string
      @ In, color, string, color name
      @ Out, paint, string, formatted string
    """
    if color.lower() not in self.colors.keys():
      self.message(self,'Requested color %s not recognized!  Skipping...' %color,'Warning','quiet')
      return str
    return self.colors[color.lower()]+str+self.colors['neutral']

  def setTimePrint(self,msg):
    """
      Allows the code to toggle timestamp printing.
      @ In, msg, string, the string that means true or false
      @ Out, None
    """
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
      @ In, obj, instance, preferably an object with a printTag method; otherwise, a string or an object
      @ Out, tag, string, string to print
    """
    if type(obj).__name__ in ['str','unicode']:
      return obj
    if hasattr(obj,'printTag'):
      tag = str(obj.printTag)
    else:
      tag = str(obj)
    return tag

  def getDesiredVerbosity(self,caller):
    """
      Tries to use local verbosity; otherwise uses global
      @ In, caller, instance, the object desiring to print
      @ Out, desVerbosity, int, integer equivalent to verbosity level
    """
    localVerb = caller.getLocalVerbosity(default=self.verbosity)
    if localVerb == None:
      localVerb = self.verbosity
    desVerbosity = self.checkVerbosity(localVerb)
    return desVerbosity

  def checkVerbosity(self,verb):
    """
      Converts English-readable verbosity to computer-legible integer
      @ In, verb, string, the string verbosity equivalent
      @ Out, currentVerb, int, integer equivalent to verbosity level
    """
    if str(verb).strip().lower() not in self.verbCode.keys():
      raise IOError('Verbosity key '+str(verb)+' not recognized!  Options are '+str(list(self.verbCode.keys())+[None]))
    currentVerb = self.verbCode[str(verb).strip().lower()]
    return currentVerb

  def error(self,caller,etype,message,tag='ERROR',verbosity='silent',color=None):
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
    verbval = max(self.getDesiredVerbosity(caller),self.checkVerbosity(self.verbosity))
    self.message(caller,message,tag,verbosity,color=color)
    if not self.suppressErrs:
      self.printWarnings()
      # debug mode gets full traceback, others quieted
      if verbval<3:
        #all, quiet, silent
        sys.tracebacklimit=0
      raise etype(message)

  def message(self,caller,message,tag,verbosity,color=None,writeTo=sys.stdout, forcePrint=False):
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
    okay,msg = self._printMessage(caller,message,tag,verbval,color,forcePrint)
    if tag.lower().strip() == 'warning':
      self.addWarning(message)
    if okay:
      print(msg,file=writeTo)
    sys.stdout.flush()

  def addWarning(self,msg):
    """
      Stores warnings so that they can be reported in summary later.
      @ In, msg, string, only the main part of the message, used to determine uniqueness
      @ Out, None
    """
    index = bisect.bisect_left(self.warnings,msg)
    if len(self.warnings) == 0 or index == len(self.warnings) or self.warnings[index] != msg:
      self.warnings.insert(index,msg)
      self.warningCount.insert(index,1)
    else:
      self.warningCount[index] += 1

  def _printMessage(self,caller,message,tag,verbval,color=None,forcePrint=False):
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
    #allows raising standardized messages
    shouldIPrint = False
    desired = self.getDesiredVerbosity(caller)
    if verbval <= desired or forcePrint:
      shouldIPrint=True
    if not shouldIPrint:
      return False,''
    ctag = self.getStringFromCaller(caller)
    msg=self.stdMessage(ctag,tag,message,color)
    return shouldIPrint,msg

  def stdMessage(self,pre,tag,post,color=None):
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
      curtime = time.time()-self.starttime
      msg+='('+'{:8.2f}'.format(curtime)+' sec) '
      if self.inColor:
        msg = self.paint(msg,'cyan')
    msgend = pre.ljust(self.callerLength)[0:self.callerLength] + ': '+tag.ljust(self.tagLength)[0:self.tagLength]+' -> ' + post
    if self.inColor:
      if color is not None:
        #overrides other options
        msgend = self.paint(msgend,color)
      elif tag.lower() in self.colorDict.keys():
        msgend = self.paint(msgend,self.colorDict[tag.lower()])
    msg+=msgend
    return msg
