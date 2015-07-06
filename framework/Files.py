'''
Created on Apr 30, 2015

@author: alfoa
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
#Internal Modules End--------------------------------------------------------------------------------

class File(BaseType):
  """
  This class is the base implementation of the file object entity in RAVEN.
  This is needed in order to standardize the object manipulation in the RAVEN code
  """
  def __init__(self):
    """
    Constructor
    @ In,  None
    @ Out, None
    """
    BaseType.__init__(self)
    self.__isOpen = False
    self.__file   = None  #when open, refers to open file, else None
    self.path=''
    self.base=''
    self.ext=''
    self.filename=''
    # TODO HOWTO self.workingDir = runInfoDict['WorkingDir']
    #the source of the initialization input determines the class you want
    #  if read from XML, you want the UserGenerated class, initialized by _readMoreXML(XMLNode)
    #  if created internally by RAVEN, you want the RAVENGenerated class, initialized by initialize(filename)

  def __del__(self):
    """
    Destructor.  Ensures file is closed before exit.
    @ In,  None
    @ Out, None
    """
    try:
      if self.isOpen(): self.__file.close()
    except AttributeError as e:
      print('Had a problem with closing file',self.filename,'|',e)

  def __getstate__(self):
    """Pickle dump method hook.
    @ In, None
    @ Out, dict, dict of objets needed to restore instance
    """
    statedict={'path':self.path,
               'base':self.base,
               'ext' :self.ext}
    return statedict

  def __setstate__(self,statedict):
    """Pickle load method hook.
    @ In, statedict, dict of objets needed to restore instance
    @ Out, None
    """
    self.path = statedict['path']
    self.base = statedict['base']
    self.ext  = statedict['ext' ]
    self.updateFilename()

  ### HELPER FUNCTIONS ###
  def updateFilename(self):
    """
    Based on changes, recreates filename from self.base and self.ext
    @ In, None
    @Out, None
    """
    self.filename = '.'.join([self.base,self.ext])

  def setFilename(self,filename):
    """
    Sets name, extension from filename = 'name.ext'
    @ In, filename, string, full filename
    @ Out, None
    """
    self.raiseADebug('CHANGING MY FILENAME! ...')
    self.raiseADebug('...from',self.getAbsFile())
    if self.__isOpen: self.raiseAnError('Tried to change the name of an open file: %s! Close it first.' %self.getAbsFile())
    self.filename = filename.strip()
    if self.filename != '.': self.base = os.path.basename(self.filename).split()[0].split('.')[0]
    else                   : self.base = self.filename
    if len(filename.split(".")) > 1: self.ext = filename.split(".")[1].lower()
    else                           : self.ext = 'unknown'
    self.raiseADebug('... to ',self.getAbsFile())

  def setExtension(self,ext):
    """Sets the extension of the file.
    @ In, ext, string, extension to change file to
    @Out, None
    """
    if self.__isOpen: self.raiseAnError('Tried to change the name of an open file: %s! Close it first.' %self.getAbsFile())
    self.ext = ext
    self.updateFilename()

  def setPath(self,path=None):
    """Sets the path to the file object.
    @ In, path, string, optional, path to set
    @Out, None
    """
    if self.__isOpen: self.raiseAnError('Tried to change the path of an open file: %s! Close it first.' %self.getAbsFile())
    if path==None: path=self.path
    if '~' in path:
      path = os.path.expanduser(path)
    self.path = path

  def setAbsFile(self,pathandfile):
    """Sets the path AND the filename.
    @ In, pathandfile, string, path to file and the filename itself in a single string
    @Out, None
    """
    self.raiseADebug('SETTING ABSOLUTE FILE!')
    if self.__isOpen: self.raiseAnError('Tried to change the path/name of an open file: %s! Close it first.' %self.getAbsFile())
    self.path,filename = os.path.split(pathandfile)
    self.setFilename(filename)

  def getPath(self):
    return self.path

  def getAbsFile(self):
    self.updateFilename()
    return os.path.normpath(os.path.join(self.path,self.filename))

  ### ACCESS FUNCTIONS ###
  def isOpen(self):
    """
    Checks the open status of the internal file
    @ In,  None
    @ Out, bool, True if file is open
    """
    return self.__isOpen

  def checkExists(self):
    """
    Checks path for existence of the file, errors if not found.
    @ In,  None
    @ Out, None
    """
    path = os.path.normpath(os.path.join(self.path,self.filename))
    if not os.path.exists(path): self.raiseAnError(IOError,'File not found:',path)

  ### FILE-LIKE FUNCTIONS ###
  def __iter__(self):
    """Acts like iterating over file
    @ In, None
    @ Out, iterator
    """
    if not self.isOpen(): self.open('r')
    self.__file.seek(0)
    return (l for l in self.__file)

  def open(self,mode='rw'):
    """
    Opens the file if not open, else throws a warning
    @ In,  mode, string (optional) the read-write mode according to python "file" method ('r','a','w','rw',etc) (default 'rw')
    @ Out, None
    """
    #TODO check if file exists
    if not self.__isOpen:
      self.__file = file(self.getAbsFile(),mode)
      self.__isOpen = True
    else: self.raiseAWarning('Tried to open',self.filename,'but file already open!')

  def close(self):
    """
    Closes the file if open, else throws a warning.
    @ In,  None
    @ Out, None
    """
    if self.__isOpen:
      self.__file.close()
      self.__file = None
      self.__isOpen = False
    else: self.raiseAWarning('Tried to close',self.filename,'but file not open!')

  def write(self,string,overwrite=False):
    """
      Mimics the "write" function of a python file object.
      @ In, string, the string to write to file
      @ In, overwrite, bool (optional), whether to overwrite the existing file if not open
      @Out, None
    """
    if not self.isOpen(): self.open('a' if not overwrite else 'w')
    self.__file.write(string)

  def writelines(self,string,overwrite=False):
    """
    Writes to the file whose name is being stored
    @ In, string or list of string, the string to write to the file
    @ In, overwrite, bool (optional), if true will open file in write mode instead of append
    @ Out, None
    """
    if not self.isOpen(): self.open('a' if not overwrite else 'w')
    self.__file.writelines(string)

  def read(self,size=None):
    """
      Mimics the "read" function of a python file object.
      @ In, size, integer (optional), number of bytes to read
      @Out, string, bytes read from file
    """
    if size is None:
      return self.__file.read()
    else:
      return self.__file.read(size)

  def readline(self):
    """
      Mimics the "readline" function of a python file object.
      @ In, None
      @Out, string, next line from file
    """
    return self.__file.readline()
    self.type = 'internal'
    self.printTag = 'Internal File'

#
#
#
#
class RAVENGenerated(File):
  """
  This class is for file objects that are created and used internally by RAVEN.
  Initialization is through calling self.initialize
  """
  def initialize(self,filename,messageHandler,path='.',subtype=None):
    """Since this is internally generated, set up all the basic information.
    @ In, filename, string, name of the file
    @ In, messageHandler, MessageHandler object, message handler
    @ In, path, string (optional), path to file object
    @ In, subtype, string (optional), subtype for labeling
    @Out, None
    """
    self.messageHandler = messageHandler
    self.type = 'internal'
    self.printTag = 'Internal File'
    self.path=path
    self.raiseADebug('FIRST FILENAME SET!')
    self.setFilename(filename)
    self.perturbed = False
    self.subtype   = subtype
    self.name      = filename
    #self.checkExists()
    #TODO we need a way to distinguish if it should already exist or not

class CSV(RAVENGenerated):
  """Specialized class specific to CSVs.  Was useful, may not be now."""
  def initialize(self,filename,messageHandler,path='.',subtype=None):
    """Since this is internally generated, set up all the basic information.
    @ In, filename, string, name of the file
    @ In, messageHandler, MessageHandler object, message handler
    @ In, path, string (optional), path to file object
    @ In, subtype, string (optional), subtype for labeling
    @Out, None
    """
    RAVENGenerated.initialize(self,filename,messageHandler,path,subtype)
    self.type='csv'
    self.printTag = 'Internal CSV'
#
#
#
#
class UserGenerated(File):
  """
  This class is for file objects that are created and used internally by RAVEN.
  Initialization is through self._readMoreXML
  """
  def _readMoreXML(self,node):
    """
      reads the xmlNode and sets parameters
      @ In,  xmlNode, XML node
      @ In,  msgHandler, MessageHandler object
      @ Out, None
    """
    self.raiseADebug('READING UserGen xml!!!!!!!!')
    #for node in xmlNode:
    self.raiseADebug('NODE:',node.tag)
    self.type = node.tag #XSD should confirm types as Input only valid type so far
    self.printTag = self.type+' File'
    self.setFilename(node.text.strip())
    self.perturbed = node.attrib.get('perturbable',True)
    self.subtype   = node.attrib.get('type'       ,None)
    self.alias     = node.attrib.get('name'       ,self.filename)
#
#
#
#
  ### STRING-LIKE FUNCTIONS ###
#  def __add__(self, other) :
#    """
#    Overload add "+"
#    """
#    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
#    return Files(self.filename + other)
#  def __radd__(self, other):
#    """
#    Overload radd "+"
#    """
#    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
#    return Files(other + self.filename)
#  def __lt__(self, other)  :
#    """
#    Overload lt "<"
#    """
#    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
#    return len(self.filename) < len(str(other))
#  def __le__(self, other) :
#    """
#    Overload le "<="
#    """
#    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
#    return len(self.filename) <= len(str(other))
#  def __eq__(self, other)  :
#    """
#    Overload eq "=="
#    """
#    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes','NoneType']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
#    return self.filename == other
#  def __ne__(self, other)  :
#    """
#    Overload ne "!="
#    """
#    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes','NoneType']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
#    return self.filename != other
#  def __gt__(self, other)  :
#    """
#    Overload gt ">"
#    """
#    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
#    return len(self.filename) > len(str(other))
#  def __ge__(self, other)  :
#    """
#    Overload ge ">"
#    """
#    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
#    return len(self.filename) >= len(str(other))
#

"""
  Interface Dictionary (factory)(private)
"""
__base                        = 'Data'
__interFaceDict               = {}
__interFaceDict['RAVEN']      = RAVENGenerated
__interFaceDict['CSV']        = CSV
__interFaceDict['Input']      = UserGenerated
__knownTypes                  = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  try: return __interFaceDict[Type]()
  except KeyError: caller.raiseAnError(NameError,'Files module does not recognize '+__base+' type '+Type)
