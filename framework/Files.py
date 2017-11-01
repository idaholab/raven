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
from copy import deepcopy
from utils import xmlUtils
import bisect
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from utils import utils, mathUtils
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
    self.__file             = None  # when open, refers to open file, else None
    self.__path             = ''    # file path
    self.__base             = ''    # file base
    self.__ext              = None  # file extension
    self.__linkedModel      = None  # hard link to a certain Code subtype (e.g. RELAP-7, MooseBasedApp, etc,)
    self.type               = None  # type ("type" in the input) to label a file to any particular subcode in the code interface
    self.perturbable        = False # is this file perturbable by a sampling strategy?

  def __del__(self):
    """
      Destructor.  Ensures file is closed before exit.
      @ In,  None
      @ Out, None
    """
    try:
      if self.isOpen():
        self.close()
    except AttributeError as e:
      print('Had a problem with closing file',self.getFilename(),'|',e)

  def __copy__(self):
    """
      Overwite of shallow copy method, to ensure less pass-by-reference.
      @ In, None
      @ Out, new, File, new File instance
    """
    cls = self.__class__
    new = cls.__new__(cls)
    new.__dict__.update(self.__dict__)
    return new

  def __deepcopy__(self,memo):
    """
      Overwite of deep copy method, to ensure no pass-by-reference.
      @ In, memo, dict, dictionary to fill (see copy module documentation)
      @ Out, new, File, new File instance
    """
    cls = self.__class__
    new = cls.__new__(cls)
    memo[id(self)] = new
    for k,v in self.__dict__.items():
      setattr(new,k,deepcopy(v,memo))
    return new

  def __getstate__(self):
    """
      Pickle dump method hook.
      @ In, None
      @ Out, statedict, dict, dict of objets needed to restore instance
    """
    statedict={'path':self.__path,
               'base':self.__base,
               'ext' :self.__ext,
               'type':self.type,
               'linkedModel':self.__linkedModel}
    return statedict

  def __setstate__(self,statedict):
    """
      Pickle load method hook.
      @ In, statedict, dict, of objets needed to restore instance
      @ Out, None
    """
    self.__file  = None
    self.__path  = statedict['path']
    self.__base  = statedict['base']
    self.__ext   = statedict['ext' ]
    self.type    = statedict['type' ]
    self.__linkedModel = statedict['linkedModel' ]

  def __repr__(self):
    """
      Overwrite of string representation.
      @ In, None
      @ Out, newRepr, string, full file path and name in string
    """
    newRepr = "(FILE) "+self.getAbsFile()+" (END FILE)"
    return newRepr

  def __enter__(self):
    """
      Needed to simulate Python file object.
      @ In, None
      @ Out, __file, object, file object
    """
    if not self.isOpen():
      self.open()
    return self.__file

  def __exit__(self,*args):
    """
      Needed to simulate Python file object.
      @ In, args, dict, for future usage
      @ Out, None
    """
    self.__file.close()

  ### HELPER FUNCTIONS ###
  ## the base elements for the file are path, base, and extension ##
  # retrieval tools #
  def getPath(self):
    """
      Retriever for path.
      @ In, None
      @ Out, __path, string, path
    """
    return self.__path

  def getBase(self):
    """
      Retriever for file base.
      @ In, None
      @ Out, __base, string path
    """
    return self.__base

  def getExt(self):
    """
      Retriever for file extension.
      @ In, None
      @ Out, __ext, string, extension of the file name (e.g. txt, csv)
    """
    return self.__ext

  def getLinkedCode(self):
    """
      Retriever for code name associated with this file.
      @ In, None
      @ Out, getLinkedCode, string, string path
    """
    return self.__linkedModel

  # setting tools #
  def setPath(self,path):
    """
      Sets the path to the file object.
      @ In, path, string (optional), path to set
      @ Out, None
    """
    if self.isOpen():
      self.raiseAnError('Tried to change the path of an open file: %s! Close it first.' %self.getAbsFile())
    if '~' in path:
      path = os.path.expanduser(path)
    self.__path = path

  def prependPath(self,addpath):
    """
      Prepends path to existing path.
      @ In, addpath, string, new path to prepend
      @ Out, None
    """
    if self.isOpen():
      self.raiseAnError('Tried to change the path of an open file: %s! Close it first.' %self.getAbsFile())
    if '~' in addpath:
      addpath = os.path.expanduser(addpath)
    self.__path = os.path.join(addpath,self.getPath())

  def setBase(self,base):
    """
      Sets the base name of the file.
      @ In, base, string, base to change file to
      @ Out, None
    """
    if self.isOpen():
      self.raiseAnError('Tried to change the base name of an open file: %s! Close it first.' %self.getAbsFile())
    self.__base = base

  def setExt(self,ext):
    """
      Sets the extension of the file.
      @ In, ext, string, extension to change file to
      @ Out, None
    """
    if self.isOpen():
      self.raiseAnError('Tried to change the extension of an open file: %s! Close it first.' %self.getAbsFile())
    self.__ext = ext

  ## the base elements for the file are path, base, and extension ##
  # getters #
  def getFilename(self):
    """
      Retriever for full filename.
      @ In, None
      @ Out, __base, string, filename
    """
    if self.__ext is not None:
      return '.'.join([self.__base,self.__ext])
    else:
      return self.__base

  def getAbsFile(self):
    """
      Retriever for path/file.
      @ In, None
      @ Out, absPathFile, string, path/file
    """
    absPathFile = os.path.normpath(os.path.join(self.getPath(),self.getFilename()))
    return absPathFile

  def getType(self):
    """
      Retrieves the type set in the XML (UserGenerated) or by the developer.
      Note that this gives the type, since type is reserved for internal RAVEN use.
      @ In, None
      @ Out, type, string, type if not None, else ''
    """
    type = '' if self.type is None else self.type
    return type

  def getPerturbable(self):
    """
      Retrieves the "perturbable" boolean attribute.  Defaults to True for UserGenerated, False for others.
      @ In, None
      @ Out, perturbable, bool, perturbable
    """
    return self.perturbable

  # setters #
  def setFilename(self,filename):
    """
      Sets base, extension from filename = 'name.ext'
      @ In, filename, string, full filename (without path)
      @ Out, None
    """
    if self.isOpen():
      self.raiseAnError('Tried to change the name of an open file: %s! Close it first.' %self.getAbsFile())
    filename = filename.strip()

    # This will split the file name at the rightmost '.'
    base, ext = os.path.splitext(filename)

    # The rightmost '.' will be the first character of ext
    # (unless there is no '.' in the file name, in which case ext is '')
    self.setBase(base)

    if (ext == ''):
      self.setExt(None)
    else:
      self.setExt(ext.lstrip('.'))

  def setAbsFile(self,pathandfile):
    """
      Sets the path AND the filename.
      @ In, pathandfile, string, path to file and the filename itself in a single string
      @ Out, None
    """
    if self.isOpen():
      self.raiseAnError('Tried to change the path/name of an open file: %s! Close it first.' %self.getAbsFile())
    path,filename = os.path.split(pathandfile)
    self.setFilename(filename)
    self.setPath(path)

  ### ACCESS FUNCTIONS ###
  def isOpen(self):
    """
      Checks the open status of the internal file
      @ In,  None
      @ Out, response, bool, True if file is open
    """
    response = self.__file is not None
    return response

  def checkExists(self):
    """
      Checks path for existence of the file, errors if not found.
      @ In,  None
      @ Out, None
    """
    path = os.path.normpath(os.path.join(self.path,self.getFilename()))
    if not os.path.exists(path):
      self.raiseAnError(IOError,'File not found:',path)

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
    else:
      self.raiseAWarning('Tried to close',self.getFilename(),'but file not open!')

  def flush(self):
    """
      Provides access to the python file method of the same name.
      @ In, None
      @ Out, flush, int, integer file descriptor
    """
    return self.__file.flush()

  def fileno(self):
    """
      Provides access to the python file method of the same name.
      @  In, None
      @ Out, fileno, int, integer file descriptor
    """
    return self.__file.fileno()

  def isatty(self):
    """
      Provides access to the python file method of the same name.
      @  In, None
      @ Out, isatty, bool, true if file connected to tty-like device
    """
    return self.__file.isatty()

  def next(self):
    """
      Provides access to the python file method of the same name.
      @  In, None
      @ Out, next, string, next line in iteration (or StopIteration if EOF)
    """
    return self.__file.next()

  def read(self,mode='r',size=None):
    """
      Mimics the "read" function of a python file object.
      @ In, mode, string, the mode (r,a,w) with which to interact with the file
      @ In, size, int, optional, number of bytes to read
      @ Out, bytesRead, string, bytes read from file
    """
    if not self.isOpen():
      self.open(mode)
    bytesRead = self.__file.read() if size is None else self.__file.read(size)
    return bytesRead

  def readline(self,mode='r',size=None):
    """
      Mimics the "readline" function of a python file object.
      @ In, mode, string, the mode (r,a,w) with which to interact with the file
      @ In, size, int, the number of bytes to read in, as per the Python file object
      @ Out, lineRead, string, next line from file
    """
    if not self.isOpen():
      self.open(mode)
    lineRead = self.__file.readline() if size is None else self.__file.readline(size)
    return lineRead

  def readlines(self,sizehint=None,mode='r'):
    """
      Provides access to the python file method of the same name.
      @ In, sizehint, int, bytes to read up to
      @ In, mode, string, the mode (r,a,w) with which to interact with the file
      @ Out, lines, list, lines read
    """
    if not self.isOpen():
      self.open(mode)
    lines = self.__file.readlines() if sizehint is None else self.__file.readlines(sizehint)
    return lines

  def seek(self,offset,whence=None):
    """
      Provides access to the python file method of the same name.
      @ In, offset, int, location in file to seek
      @ In, whence, int, optional, integer indicator (see python file documentation)
      @ Out, None
    """
    if whence is None:
      return self.__file.seek(offset)
    else:
      return self.__file.seek(offset,whence)

  def tell(self):
    """
      Provides access to the python file method of the same name.
      @ In, None
      @ Out, posit, int, file's current position
    """
    posit = self.__file.tell()
    return posit

  def truncate(self,size=None):
    """
      Provides access to the python file method of the same name.
      @ In, size, int, optional, maximum file size after truncation
      @ Out, None
    """
    if size is None:
      return self.__file.truncate()
    else:
      return self.__file.truncate(size)

  def write(self,string,overwrite=False):
    """
      Mimics the "write" function of a python file object.
      @ In, string, string, the string to write to file
      @ In, overwrite, bool, optional, whether to overwrite the existing file if not open
      @ Out, None
    """
    if not self.isOpen():
      self.open('a' if not overwrite else 'w')
    self.__file.write(string)

  def writelines(self,string,overwrite=False):
    """
      Writes to the file whose name is being stored
      @ In, string or list of string, the string to write to the file
      @ In, overwrite, bool, optional, if true will open file in write mode instead of append
      @ Out, None
    """
    if not self.isOpen():
      self.open('a' if not overwrite else 'w')
    self.__file.writelines(string)

  ### FILE-EXPECTED FUNCTIONS ###
  # N.B. these don't show up in the python file docs, but are needed to act like files
  def open(self,mode='rw'):
    """
      Opens the file if not open, else throws a warning
      @ In,  mode, string, optional, the read-write mode according to python "file" method ('r','a','w','rw',etc) (default 'rw')
      @ Out, None
    """
    if not self.isOpen():
      self.__file = open(self.getAbsFile(),mode)
    else:
      self.raiseAWarning('Tried to open',self.getFilename(),'but file already open!')

  def __iter__(self): #MIGHT NEED TO REMOVE
    """
      Acts like iterating over file
      @ In, None
      @ Out, __iter__, iterator, file iterator
    """
    if not self.isOpen():
      self.open('r')
    self.__file.seek(0)
    return self.__file.__iter__()
#
#
#
#
class RAVENGenerated(File):
  """
    This class is for file objects that are created and used internally by RAVEN.
    Initialization is through calling self.initialize
  """
  def initialize(self,filename,messageHandler,path='.',type='internal'):
    """
      Since this is internally generated, set up all the basic information.
      @ In, filename, string, name of the file
      @ In, messageHandler, MessageHandler object, message handler
      @ In, path, string, optional, path to file object
      @ In, type, string, optional, type for labeling
      @ Out, None
    """
    self.messageHandler = messageHandler
    self.type = type
    self.printTag = 'Internal File'
    self.setPath(path)
    self.setFilename(filename)
    self.perturbed = False
    self.name      = filename

#
#
#
#
#   Form for StaticXMLOutput
#    Static form:
#    <root type='Static'>
#      <parameter>
#        <single-value properties>value</single-value properties>
#        <multi-value properties>
#          <w.r.t. parameter2>value</w.r.t. paramter2>
#          <w.r.t. parameter2>value</w.r.t. paramter2>
#        </multi-value properties>
#      <parameter>
#    </root>
#
class StaticXMLOutput(RAVENGenerated):
  """
    Specialized class for consistent RAVEN XML outputs.  See forms in comments above.
  """

  def __getstate__(self):
    """
      Pickle dump method hook.
      @ In, None
      @ Out, statedict, dict, dict of objects needed to restore instance
    """
    statedict = RAVENGenerated.__getstate__(self)
    if hasattr(self,'tree'):
      statedict['tree'] = self.tree
    return statedict

  def __setstate__(self,statedict):
    """
      Pickle load method hook.
      @ In, statedict, dict, of objects needed to restore instance
      @ Out, None
    """
    if 'tree' in statedict.keys():
      self.tree = statedict.pop('tree')
    RAVENGenerated.__setstate__(self,statedict)

  def newTree(self,root,pivotParam=None):
    """
      Sets up a new internal tree.
      @ In, root, string, name of root node
      @ In, pivotParam, string, unused (for consistency with dynamic method)
      @ Out, None
    """
    self.tree = xmlUtils.newTree(root,attrib={'type':'Static'})

  def addScalar(self,target,name,value,root=None,pivotVal=None):
    """
      Adds a node entry named "name" with value "value" to "target" node, such as
      <root>
        <target>
          <name>value<name>
      @ In, target, string, target parameter to add node value to
      @ In, name, string, name of characteristic of target to add
      @ In, value, string/float/etc, value of characteristic
      @ In, root, xml.etree.ElementTree.Element, optional, root to add to
      @ In, pivotVal, float, optional, unused (for consistency with dynamic mode)
      @ Out, None
    """
    if root is None:
      root = self.tree.getroot()
    #note: illegal unicode characters are checked for in xmlUtils methods
    targ = self._findTarg(root,target)
    targ.append(xmlUtils.newNode(name,text=value))

  def addVector(self,target,name,valueDict,root=None,pivotVal=None):
    """
      Adds a node entry named "name" with value "value" to "target" node, such as
      <root>
        <target>
          <name>
            <with_respect_to_name1>value1
            <with_respect_to_name2>value2
            <with_respect_to_name3>value3
      The valueDict should be as {with_respect_to_name1:value1, with_respect_to_name2:value2}
      For example, if the "name" is sensitivity_coefficients, each entry would be the
          sensitivity of "target" to "with_respect_to" parameters.
      @ In, target, string, target parameter to add node value to
      @ In, name, string, name of characteristic of target to add
      @ In, valueDict, dict, name:value
      @ In, root, xml.etree.ElementTree.Element, optional, root to add to
      @ In, pivotVal, float, optional, unused (for consistency with dynamic mode)
      @ Out, None
    """
    if root is None:
      root = self.tree.getroot()
    targ = self._findTarg(root,target)
    nameNode = xmlUtils.newNode(name)
    for key,value in valueDict.items():
      nameNode.append(xmlUtils.newNode(key,text=value))
    targ.append(nameNode)

  def _findTarg(self,root,target):
    """
      Searches "root" for "target" node and makes it if not found
      @ In, root, xml.etree.ElementTree.Element, node to search under
      @ In, target, string, name of target to find
      @ Out, targ, xml.etree.ElementTree.Element, desired target node
    """
    #find target node
    targ = xmlUtils.findPath(root,target)
    #if it doesn't exist, make it and add it
    if targ is None:
      targ = xmlUtils.newNode(target)
      root.append(targ)
    return targ

  def writeFile(self):
    """
      Writes the input file to disk.
      @ In, None
      @ Out, None
    """
    #prettify tree
    pretty = xmlUtils.prettify(self.tree)
    #make sure file is written cleanly and anew
    if self.isOpen():
      self.close()
    self.writelines(pretty,overwrite=True)
    self.close()

#    Form for DynamicXMLOutput
#    <root type='Static'>
#      <pivot value="value">
#        <parameter>
#          <single-value properties>value</single-value properties>
#          <multi-value properties>
#            <w.r.t. parameter2>value</w.r.t. paramter2>
#            <w.r.t. parameter2>value</w.r.t. paramter2>
#          </multi-value properties>
#        <parameter>
#    </root>
class DynamicXMLOutput(StaticXMLOutput):
  """
    Specialized class for consistent RAVEN XML outputs.  See forms in comments above.
  """
  def __init__(self):
    StaticXMLOutput.__init__(self)
    # while maintaining two parallel lists is not very pythonic, it's much faster for searching for large lists.
    self.pivotNodes = [] #list of pivot nodes, maintained in the order of pivotVals below
    self.pivotVals = [] #RBTree (self maintained) of unique pivot values
    #we don't save the root name, so make a variable for saving it
    self.rootName = 'DynamicTree'

  def newTree(self,root,pivotParam=None):
    """
      Sets up a new internal tree.
      @ In, root, string, name of root node
      @ In, pivotParam, string, name of time-like parameter
      @ Out, None
    """
    if pivotParam is None:
      self.raiseAnError(RuntimeError,'newTree argument pivotParam is None but output is in dynamic mode!')
    self.pivotParam = pivotParam
    self.pivotNodes = []
    self.rootName = root

  def addScalar(self,target,name,value,pivotVal=None):
    """
      Adds a node entry named "name" with value "value" to "target" node, such as
      <root>
        <pivotParam value=pivotVal>
          <target>
            <name>value<name>
      @ In, pivotVal, float, value of the pivot parameter
      @ In, target, string, target parameter to add node value to
      @ In, name, string, name of characteristic of target to add
      @ In, value, string/float/etc, value of characteristic
      @ Out, None
    """
    if pivotVal is None:
      self.raiseAnError(RuntimeError,'In addScalar no pivotVal specificied, but in dynamic mode!')
    pivotNode = self.findPivotNode(pivotVal)
    #use addScalar methods to add parameters
    StaticXMLOutput.addScalar(self,target,name,value,root=pivotNode)

  def addVector(self,target,name,valueDict,pivotVal=None):
    """
      Adds a node entry named "name" with value "value" to "target" node, such as
      <root>
        <target>
          <name>
            <with_respect_to_name1>value1
            <with_respect_to_name2>value2
            <with_respect_to_name3>value3
      The valueDict should be as {with_respect_to_name1:value1, with_respect_to_name2:value2}
      For example, if the "name" is sensitivity_coefficients, each entry would be the
          sensitivity of "target" to "with_respect_to" parameters.
      @ In, target, string, target parameter to add node value to
      @ In, name, string, name of characteristic of target to add
      @ In, valueDict, dict, name:value
      @ Out, None
    """
    if pivotVal is None:
      self.raiseAnError(RuntimeError,'In addScalar no pivotVal specificied, but in dynamic mode!')

    pivotNode = self.findPivotNode(pivotVal)
    StaticXMLOutput.addVector(self,target,name,valueDict,root=pivotNode)

  def findPivotNode(self,pivotVal):
    """
      Searches pivot nodes for node with value pivotVal, or adds it if it doesn't exist
      @ In, pivotVal, float, value of pivot to find
      @ Out, pivotNode, xml.etree.ElementTree.Element, node desired
    """
    #find the right dynamic node
    self.pivotVals,pivotIndex,pivotVal = utils.getRelativeSortedListEntry(self.pivotVals,pivotVal,tol=1e-10)
    #check if an insertion was performed
    if len(self.pivotVals) > len(self.pivotNodes):
      #create new node
      pivotNode = xmlUtils.newNode(self.pivotParam, attrib={'value':pivotVal})
      self.pivotNodes.insert(pivotIndex,pivotNode)
    else:
      pivotNode = self.pivotNodes[pivotIndex]
    return pivotNode

  def writeFile(self):
    """
      Writes the input file to disk.  Because we store a list of dynamic nodes
      instead of a full tree, requires some special treatment
      @ In, None
      @ Out, None
    """
    if self.isOpen():
      self.close()
    #write the root node manually
    self.writelines('<'+self.rootName+' type="Dynamic">\n',overwrite=True)
    #write out each time step node
    for node in self.pivotNodes:
      pretty = xmlUtils.prettify(node,startingTabs=1,addRavenNewlines=False)
      self.writelines(pretty)
    self.writelines('</'+self.rootName+'>\n')
    self.close()
#
#
#
#
class CSV(RAVENGenerated):
  """
    Specialized class specific to CSVs.  Was useful, may not be now, might be again.
  """
  def initialize(self,filename,messageHandler,path='.',type='csv'):
    """
      Since this is internally generated, set up all the basic information.
      @ In, filename, string, name of the file
      @ In, messageHandler, MessageHandler object, message handler
      @ In, path, string, optional, path to file object
      @ In, type, string, optional, type for labeling
      @ Out, None
    """
    RAVENGenerated.initialize(self,filename,messageHandler,path,type)
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
      @ In, xmlNode, XML node
      @ Out, None
    """
    self.type = node.attrib.get('type','UserGenerated') #XSD should confirm valid types
    #used to be node.tag, but this caused issues, since many things in raven
    #access "type" directly instead of through an accessor like getType
    self.printTag = self.type+' File'
    self.__linkedModel = node.attrib.get('linkedCode' ,None)
    self.perturbed     = node.attrib.get('perturbable',True)
    self.subDirectory  = node.attrib.get('subDirectory',"")
    self.setAbsFile(os.path.join(self.subDirectory,node.text.strip()))
    self.alias         = node.attrib.get('name'       ,self.getFilename())

  def __getstate__(self):
    """
      Pickle dump method hook.
      @ In, None
      @ Out, statedict, dict, dict of objets needed to restore instance
    """
    statedict = File.__getstate__(self)
    statedict['perturbed'   ] = self.perturbed
    statedict['subDirectory'] = self.subDirectory
    statedict['alias'       ] = self.alias
    return statedict

  def __setstate__(self,statedict):
    """
      Pickle load method hook.
      @ In, statedict, dict, of objets needed to restore instance
      @ Out, None
    """
    File.__setstate__(self,statedict)
    self.perturbed     = statedict['perturbed'   ]
    self.subDirectory  = statedict['subDirectory']
    self.alias         = statedict['alias'       ]


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
__interFaceDict['StaticXMLOutput']  = StaticXMLOutput
__interFaceDict['DynamicXMLOutput'] = DynamicXMLOutput
__knownTypes                  = __interFaceDict.keys()

def knownTypes():
  """
    Returns known types.
    @ In, None
    @ Out, __knownTypes, list, list of known types
  """
  return __knownTypes

def returnInstance(Type,caller):
  """
    Returns an object construction pointer from this module.
    @ In, Type, string, requested object
    @ In, caller, object, requesting object
    @ Out, __interFaceDict, instance, instance of the object
  """
  try:
    return __interFaceDict[Type]()
  except KeyError:
    caller.raiseAnError(NameError,'Files module does not recognize '+__base+' type '+Type)
