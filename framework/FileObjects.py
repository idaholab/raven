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

class FileObject(BaseType,str):
  """
  This class is the implementation of the file object entity in RAVEN.
  This is needed in order to standardize the object manipulation in the RAVEN code
  """
  def __init__(self,filename=None):
    """
    Constructor
    """
    BaseType.__init__(self)
    if filename is not None: #constructor for no-xml case; else use _readMoreXML
      self._setFilename(filename)

  def __iter__(self):
    """Acts like iterating over file
    @ In, None
    @ Out, iterator
    """
    return (l for l in file(self.filename,'r'))

  def __add__(self, other) :
    """
    Overload add "+"
    """
    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
    return FileObject(self.filename + other)
  def __radd__(self, other):
    """
    Overload radd "+"
    """
    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
    return FileObject(other + self.filename)
  def __lt__(self, other)  :
    """
    Overload lt "<"
    """
    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
    return len(self.filename) < len(str(other))
  def ___le__(self, other) :
    """
    Overload le "<="
    """
    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
    return len(self.filename) <= len(str(other))
  def __eq__(self, other)  :
    """
    Overload eq "=="
    """
    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes','NoneType']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
    return self.filename == other
  def __ne__(self, other)  :
    """
    Overload ne "!="
    """
    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes','NoneType']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
    return self.filename != other
  def __gt__(self, other)  :
    """
    Overload gt ">"
    """
    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
    return len(self.filename) > len(str(other))
  def __ge__(self, other)  :
    """
    Overload ge ">"
    """
    if type(other).__name__ not in [type(self).__name__,'str','unicode','bytes']: self.raiseAnError(ValueError,"other is not a string like type! Got "+ type(other).__name__)
    return len(self.filename) >= len(str(other))

  def _readMoreXML(self,xmlNode,msgHandler):
    """
      reads the xmlNode and sets parameters
      @ In, xmlNode, XML node
      @ In, msgHandler, MessageHandler object
    """
    self.messageHandler = msgHandler
    for node in xmlNode:
      if node.tag=='Input':
        self.type = 'Input'
        self._setFilename(node.attrib.get('name')) #TODO errorchecking or XSD?
      elif node.tag=='Auxiliary':pass
      if self.name is None: self.raiseAnError('Missing "name" attribute!')

  def _setFilename(self,filename):
    """
    Sets name, extension from filename = 'name.ext'
    @ In, filename, string, full filename
    @ Out, None
    """
    str.__init__(filename.strip())
    self.filename = filename.strip()
    if self.filename != '.': self.name = os.path.basename(self.filename).split()[0]
    else                   : self.name = self.filename
    if len(filename.split(".")) > 1: self.ext = filename.split(".")[1].lower()
    else                           : self.ext = 'unknown'

  def writelines(self,string,overwrite=False):
    """
    Writes to the file whose name is being stored
    @ In, string or list of string, the string to write to the file
    @ In, overwrite, bool (optional), if true will open file in write mode instead of append
    @ Out, None
    """
    mode = 'a' if not overwrite else 'w'
    writeto = file(self.filename,mode)
    writeto.writelines(string)
    writeto.close()

def Input(FileObjects):
  pass

def Output(FileObjects):
  pass

"""
  Interface Dictionary (factory)(private)
"""
__base                        = 'Data'
__interFaceDict               = {}
__interFaceDict['Input']      = Input
__interFaceDict['Output']     = Output
__knownTypes                  = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  try: return __interFaceDict[Type]()
  except KeyError: caller.raiseAnError(NameError,'FileObjects: not known '+__base+' type '+Type)
