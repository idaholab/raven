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
from copy import copy,deepcopy
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
    self.__file   = None  #when open, refers to open file, else None
    self.__path=''
    self.__base=''
    self.__ext=None
    self.subtype=None
    self.perturbable=False

  def __del__(self):
    """
    Destructor.  Ensures file is closed before exit.
    @ In,  None
    @ Out, None
    """
    try:
      if self.isOpen(): self.close()
    except AttributeError as e:
      print('Had a problem with closing file',self.getFilename(),'|',e)

  def __copy__(self):
    """
    Overwite of shallow copy method, to ensure less pass-by-reference.
    @ In, None
    @ Out, new File instance
    """
    cls = self.__class__
    new = cls.__new__(cls)
    new.__dict__.update(self.__dict__)
    return new

  def __deepcopy__(self,memo):
    """
    Overwite of deep copy method, to ensure no pass-by-reference.
    @ In, memo, dictionary to fill (see copy module documentation)
    @ Out, new File instance
    """
    cls = self.__class__
    new = cls.__new__(cls)
    memo[id(self)] = new
    for k,v in self.__dict__.items():
      setattr(new,k,deepcopy(v,memo))
    return new

  def __getstate__(self):
    """Pickle dump method hook.
    @ In, None
    @ Out, dict, dict of objets needed to restore instance
    """
    statedict={'path':self.__path,
               'base':self.__base,
               'ext' :self.__ext,
               'subtype':self.subtype}
    return statedict

  def __setstate__(self,statedict):
    """Pickle load method hook.
    @ In, statedict, dict of objets needed to restore instance
    @ Out, None
    """
    self.__path  = statedict['path']
    self.__base  = statedict['base']
    self.__ext   = statedict['ext' ]
    self.subtype = statedict['subtype' ]
    self.updateFilename()

  def __repr__(self):
    """Overwrite of string representation.
    @ In, None
    @ Out, string, full file path and name in string
    """
    return "(FILE) "+self.getAbsFile()+" (END FILE)"

  def __enter__(self):
    self.__file.open(self.getAbsFile())
    return self.__file

  def __exit__(self,*args):
    self.__file.close()

  ### HELPER FUNCTIONS ###
  ## the base elements for the file are path, base, and extension ##
  # retrieval tools #
  def getPath(self):
    """Retriever for path.
    @ In, None
    @ Out, string, path
    """
    return self.__path

  def getBase(self):
    """Retriever for file base.
    @ In, None
    @ Out, string path
    """
    return self.__base

  def getExt(self):
    """Retriever for file base.
    @ In, None
    @ Out, string path
    """
    return self.__ext

  # setting tools #
  def setPath(self,path):
    """Sets the path to the file object.
    @ In, path, string, optional, path to set
    @ Out, None
    """
    if self.isOpen(): self.raiseAnError('Tried to change the path of an open file: %s! Close it first.' %self.getAbsFile())
    if '~' in path: path = os.path.expanduser(path)
    self.__path = path

  def prependPath(self,addpath):
    """Prepends path to existing path.
    @ In, addpath, string, new path to prepend
    @ Out, None
    """
    if self.isOpen(): self.raiseAnError('Tried to change the path of an open file: %s! Close it first.' %self.getAbsFile())
    if '~' in addpath: addpath = os.path.expanduser(addpath)
    self.__path = os.path.join(addpath,self.getPath())

  def setBase(self,base):
    """Sets the base name of the file.
    @ In, base, string, base to change file to
    @ Out, None
    """
    if self.isOpen(): self.raiseAnError('Tried to change the base name of an open file: %s! Close it first.' %self.getAbsFile())
    self.__base = base

  def setExt(self,ext):
    """Sets the extension of the file.
    @ In, ext, string, extension to change file to
    @ Out, None
    """
    if self.isOpen(): self.raiseAnError('Tried to change the extension of an open file: %s! Close it first.' %self.getAbsFile())
    self.__ext = ext

  ## the base elements for the file are path, base, and extension ##
  # getters #
  def getFilename(self):
    """Retriever for full filename.
    @ In, None
    @ Out, string, filename
    """
    if self.__ext is not None: return '.'.join([self.__base,self.__ext])
    else: return self.__base

  def getAbsFile(self):
    """Retriever for path/file.
    @ In, None
    @ Out, string, path/file
    """
    return os.path.normpath(os.path.join(self.getPath(),self.getFilename()))

  def getType(self):
    """Retrieves the subtype set in the XML (UserGenerated) or by the developer.
       Note that this gives the subtype, since type is reserved for internal RAVEN use.
       @ In, None
       @ Out, string, subtype if not None, else ''
    """
    if self.subtype is None: return ''
    else: return self.subtype

  def getPerturbable(self):
    """Retrieves the "perturbable" boolean attribute.  Defaults to True for UserGenerated, False for others.
       @ In, None
       @ Out, boolean, perturbable
    """
    return self.perturbable

  # setters #
  def setFilename(self,filename):
    """
    Sets base, extension from filename = 'name.ext'
    @ In, filename, string, full filename (without path)
    @ Out, None
    """
    if self.isOpen(): self.raiseAnError('Tried to change the name of an open file: %s! Close it first.' %self.getAbsFile())
    filename = filename.strip()
    if filename != '.': self.setBase(os.path.basename(filename).split()[0].split('.')[0])
    else: self.setBase(filename)
    if len(filename.split(".")) > 1: self.setExt(filename.split(".")[-1].lower())
    else: self.setExt(None)

  def setAbsFile(self,pathandfile):
    """Sets the path AND the filename.
    @ In, pathandfile, string, path to file and the filename itself in a single string
    @ Out, None
    """
    if self.isOpen(): self.raiseAnError('Tried to change the path/name of an open file: %s! Close it first.' %self.getAbsFile())
    path,filename = os.path.split(pathandfile)
    self.setFilename(filename)
    self.setPath(path)

  ### ACCESS FUNCTIONS ###
  def isOpen(self):
    """
    Checks the open status of the internal file
    @ In,  None
    @ Out, bool, True if file is open
    """
    return self.__file is not None

  def checkExists(self):
    """
    Checks path for existence of the file, errors if not found.
    @ In,  None
    @ Out, None
    """
    path = os.path.normpath(os.path.join(self.path,self.getFilename()))
    if not os.path.exists(path): self.raiseAnError(IOError,'File not found:',path)

  ### FILE-LIKE FUNCTIONS ###
  def close(self):
    """
    Closes the file if open, else throws a warning.
    @ In,  None
    @ Out, None
    """
    if self.isOpen():
      self.__file.close()
      del self.__file
      self.__file = None
    else: self.raiseAWarning('Tried to close',self.getFilename(),'but file not open!')

  def flush(self):
    """Provides access to the python file method of the same name.
      @  In, None
      @ Out, integer file descriptor
    """
    return self.__file.flush()

  def fileno(self):
    """Provides access to the python file method of the same name.
      @  In, None
      @ Out, integer, file descriptor
    """
    return self.__file.fileno()

  def isatty(self):
    """Provides access to the python file method of the same name.
      @  In, None
      @ Out, Boolean, true of file connected to tty-like device
    """
    return self.__file.isatty()

  def next(self):
    """Provides access to the python file method of the same name.
      @  In, None
      @ Out, next line in iteration (or StopIteration if EOF)
    """
    return self.__file.next()

  def read(self,mode='r',size=None):
    """
      Mimics the "read" function of a python file object.
      @ In, size, integer (optional), number of bytes to read
      @ Out, string, bytes read from file
    """
    if not self.isOpen(): self.open(mode)
    if size is None: return self.__file.read()
    else: return self.__file.read(size)

  def readline(self,mode='r',size=None):
    """
      Mimics the "readline" function of a python file object.
      @ In, mode, string, the mode (r,a,w) with which to interact with the file
      @ In, size, int, the number of bytes to read in, as per the Python file object
      @ Out, string, next line from file
    """
    if not self.isOpen(): self.open(mode)
    if size is None: return self.__file.readline()
    else: return self.__file.readline(size)

  def readlines(self,sizehint=None,mode='r'):
    """Provides access to the python file method of the same name.
      @  In, sizehint, bytes to read up to
      @ Out, list, lines read
    """
    if not self.isOpen(): self.open(mode)
    if sizehint is None: return self.__file.readlines()
    else: return self.__file.readlines(sizehint)

  def seek(self,offset,whence=None):
    """Provides access to the python file method of the same name.
      @  In, offset, location in file to seek
      @  In, whence, integer indicator (see python file documentation)
      @ Out, None
    """
    if whence is None: return self.__file.seek(offset)
    else: return self.__file.seek(offset,whence)

  def tell(self):
    """Provides access to the python file method of the same name.
      @  In, None
      @ Out, int, file's current position
    """
    return self.__file.tell()

  def truncate(self,size=None):
    """Provides access to the python file method of the same name.
      @  In, size, maximum file size after truncation
      @ Out, None
    """
    if size is None: return self.__file.truncate()
    else: return self.__file.truncate(size)

  def write(self,string,overwrite=False):
    """
      Mimics the "write" function of a python file object.
      @ In, string, the string to write to file
      @ In, overwrite, bool (optional), whether to overwrite the existing file if not open
      @ Out, None
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

  ### FILE-EXPECTED FUNCTIONS ###
  # N.B. these don't show up in the python file docs, but are needed to act like files
  def open(self,mode='rw'):
    """
    Opens the file if not open, else throws a warning
    @ In,  mode, string (optional) the read-write mode according to python "file" method ('r','a','w','rw',etc) (default 'rw')
    @ Out, None
    """
    if not self.isOpen(): self.__file = open(self.getAbsFile(),mode)
    else: self.raiseAWarning('Tried to open',self.getFilename(),'but file already open!')

  def __iter__(self): #MIGHT NEED TO REMOVE
    """Acts like iterating over file
    @ In, None
    @ Out, iterator
    """
    if not self.isOpen(): self.open('r')
    self.__file.seek(0)
    return self.__file.__iter__()
    #return (l for l in self.__file)
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
    @ Out, None
    """
    self.messageHandler = messageHandler
    self.type = 'internal'
    self.printTag = 'Internal File'
    self.setPath(path)
    self.setFilename(filename)
    self.perturbed = False
    self.subtype   = subtype
    self.name      = filename

#
#
#
#
class CSV(RAVENGenerated):
  """Specialized class specific to CSVs.  Was useful, may not be now, might be again."""
  def initialize(self,filename,messageHandler,path='.',subtype=None):
    """Since this is internally generated, set up all the basic information.
    @ In, filename, string, name of the file
    @ In, messageHandler, MessageHandler object, message handler
    @ In, path, string (optional), path to file object
    @ In, subtype, string (optional), subtype for labeling
    @ Out, None
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
    self.type = node.tag #XSD should confirm valid types
    self.printTag = self.type+' File'
    self.setAbsFile(node.text.strip())
    self.perturbed = node.attrib.get('perturbable',True)
    self.subtype   = node.attrib.get('type'       ,None)
    self.alias     = node.attrib.get('name'       ,self.getFilename())

#
#
#
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
