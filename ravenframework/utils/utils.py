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
  Utility module containing methods commonly used throughout the Python framework.
"""
# NOTE we still import these from __future__ here because many machines still running
# python 2.X need to use this file (for example the plugin installer)
from __future__ import division, print_function, absolute_import

# *************************** NOTE FOR DEVELOPERS ***************************
# Do not import numpy or scipy or other libraries that are not              *
# built into python.  Otherwise the import can fail, and since utils        *
# are used by --library-report, this can cause diagnostic messages to fail. *
# ***************************************************************************
import bisect
import sys
import os
import glob
import errno
import shutil
import inspect
import subprocess
import platform
from importlib import import_module
# import numpy # DO NOT import! See note above.
# import six   # DO NOT import! see note above.
from difflib import SequenceMatcher

class Object(object):
  """
    Simple custom inheritance object.
  """
  pass

#custom errors
class NoMoreSamplesNeeded(GeneratorExit):
  """
    Custom RAVEN error available for use in the framework.
  """
  pass

class byPass(object):
  """
    This is dummy class that is needed to emulate the "dataObject" resetData method
  """
  def __init__(self):
    self.name = ""

  def resetData(self):
    """
      This is dummy method that is needed to emulate the "dataObject" resetData method
      @ In, None
      @ Out, None
    """
    pass

class StringPartialFormatDict(dict):
  """
    Allows partially formatting a template string.
    See https://stackoverflow.com/questions/17215400/python-format-string-unused-named-arguments
    Use as '{a} {b} {a}'.format_map(StringPartialFormatDict(a='one')) -> 'one {b} one'
  """
  def __missing__(self, key):
    """
      Replaces missing keys with formatting entries. May not work for any formats like {b:1.3e}.
      @ In, key, str, formatting string key (the friend between the braces)
      @ Out, key, str, re-formatted string
    """
    return '{' + key + '}'

def partialFormat(msg, info):
  """
    Automates the partial formatting of a string (msg) with a format dictionary (info).
    Example: '{a} {b} {c}'.partialFormat({b:'two'}) -> '{a} two {c}'
    Note formatting is lost or may cause errors; that is,
    Example: '{a:3s} {b:2d} {c:3s}'.partialFormat({b=2}) -> '{a}  2 {c}'
    @ In, msg, string, string to partially format
    @ In, info, dict, keywords to apply
  """
  return msg.format_map(StringPartialFormatDict(**info))

# ID separator that should be used cross the code when combined ids need to be assembled.
# For example, when the "EnsembleModel" creates new  ``prefix`` ids for sub-models
__idSeparator = "++"

def identifyIfExternalModelExists(caller, moduleIn, workingDir):
  """
    Method to check if a external module exists and in case return the module that needs to be loaded with
    the correct path
    @ In, caller,object, the RAVEN caller (i.e. self)
    @ In, moduleIn, string, module read from the XML file
    @ In, workingDir, string, the path of the working directory
    @ Out, (moduleToLoad, fileName), tuple, a tuple containing the module to load (that should be used in method importFromPath) and the filename (no path)
  """
  if moduleIn.endswith('.py'):
    moduleToLoadString = moduleIn[:-3]
  else:
    moduleToLoadString = moduleIn
  workingDirModule = os.path.abspath(os.path.join(workingDir,moduleToLoadString))
  if os.path.exists(workingDirModule+".py"):
    moduleToLoadString = workingDirModule
    path, filename = os.path.split(workingDirModule)
    os.sys.path.append(os.path.abspath(path))
  else:
    path, filename = os.path.split(moduleToLoadString)
    if (path != ''):
      abspath = os.path.abspath(path)
      if '~' in abspath:
        abspath = os.path.expanduser(abspath)
      if os.path.exists(abspath):
        caller.raiseAWarning('file '+moduleToLoadString+' should be relative to working directory. Working directory: '+workingDir+' Module expected at '+abspath)
        os.sys.path.append(abspath)
      else:
        caller.raiseAnError(IOError,'The path provided for the' + caller.type + ' named '+ caller.name +' does not exist!!! Got: ' + abspath + ' and ' + workingDirModule)
  return moduleToLoadString, filename

def checkIfUnknowElementsinList(referenceList,listToTest):
  """
    Method to check if a list contains elements not contained in another
    @ In, referenceList, list, reference list
    @ In, listToTest, list, list to test
    @ Out, unknownElements, list, list of elements of 'listToTest' not contained in 'referenceList'
  """
  unknownElements = []
  for elem in listToTest:
    if elem not in referenceList:
      unknownElements.append(elem)
  return unknownElements

def checkIfPathAreAccessedByAnotherProgram(pathname, timelapse = 10.0):
  """
    Method to check if a path (file or directory) is currently
    used by another program. It is based on accessing time...
    Probably there is a better way.
    @ In, pathname, string containing the all path
    @ In, timelapse, float, tollerance on time modification
    @ Out, boolReturn, bool, True if it is used by another program, False otherwise
  """
  import stat
  import time
  mode = os.stat(pathname).st_mode
  if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
    raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '->  path '+pathname+ ' is neither a file nor a dir!')
  boolReturn = abs(os.stat(pathname).st_mtime - time.time()) < timelapse
  return boolReturn

def checkIfLockedRavenFileIsPresent(pathName,fileName="ravenLockedKey.raven"):
  """
    Method to check if a path (directory) contains an hidden raven file
    @ In, pathName, string, string containing the path
    @ In, fileName, string, optional, string containing the file name
    @ Out, filePresent, bool, True if it is present, False otherwise
  """
  filePresent = os.path.isfile(os.path.join(pathName,fileName))
  if not filePresent:
    open(os.path.join(pathName,fileName), 'w')
  return filePresent

def removeFile(pathAndFileName):
  """
    Method to remove a file
    @ In, pathAndFileName, string, string containing the path and filename
    @ Out, None
  """
  if os.path.isfile(pathAndFileName):
    os.remove(pathAndFileName)

def removeDir(strPath):
  """
    Method to remove a directory.
    @ In, strPath, string, path to directory to remove
    @ Out, None
  """
  path = os.path.abspath(os.path.expanduser(strPath))
  shutil.rmtree(path, onerror=_removeDirErrorHandler)

def _removeDirErrorHandler(func, path, excinfo):
  """
    Handles errors arising from using shutil.rmtree
    Argument descriptions from shutil documentation
    @ In, func, is the function which raised the exception; it depends on the platform and
                implementation
    @ In, path, will be the path name passed to function
    @ In, excinfo, will be the exception information returned by sys.exc_info()
    @ Out, None
  """
  print('utils.removeDir WARNING: unable to remove {path} using {func}, ' +
        'raising the following exception: {excinfo}. Continuing ...'
        .format(path=path, func=func, excinfo=excinfo))

def returnImportModuleString(obj,moduleOnly=False):
  """
    Method to return a list of strings that represent the
    modules on which the 'obj' depends on. It already implements
    the 'import' statement or the 'from x import y'
    @ In, obj, instance, the object that needs to be inquired
    @ In, moduleOnly, bool, optional, get the modules only (True) or also the function dependencies(False)
    @ Out, mods, list, list of string containing the modules
  """
  mods = []
  for key, value in dict(inspect.getmembers(obj)).items():
    if moduleOnly:
      if not inspect.ismodule(value):
        continue
    else:
      if not (inspect.ismodule(value) or inspect.ismethod(value)):
        continue
    if key != value.__name__:
      if value.__name__.split(".")[-1] != key:
        mods.append(str('import ' + value.__name__ + ' as '+ key))
      else:
        mods.append(str('from ' + '.'.join(value.__name__.split(".")[:-1]) + ' import '+ key))
    else:
      mods.append(str(key))
  return mods

def getPrintTagLenght():
  """
    Method to return the length of the strings used for Screen output
    @ In, None,
    @ Out, tagLenght, int, the default tag length
  """
  tagLenght = 25
  return tagLenght

def UreturnPrintTag(intag):
  """
    Method to return the a string formatted with respect to the length
    obtained by the method getPrintTagLenght() (generally used for pre tag)
    @ In, intag, string, string that needs to be formatted
    @ Out, returnString, string, the formatted string
  """
  returnString = intag.ljust(getPrintTagLenght())[0:getPrintTagLenght()]
  return returnString

def UreturnPrintPostTag(intag):
  """
    Method to return the a string formatted with respect to the length
    obtained by the method getPrintTagLenght() - 15 (generally used for post tag)
    @ In, intag, string, string that needs to be formatted
    @ Out, returnString, string, the formatted string
  """
  returnString = intag.ljust(getPrintTagLenght()-15)[0:(getPrintTagLenght()-15)]
  return returnString

def convertMultipleToBytes(sizeString):
  """
    Convert multiple (e.g. Mbytes, Gbytes,Kbytes) in bytes
    International system type (e.g., 1 Mb = 10^6)
    @ In, sizeString, string, string that needs to be converted in bytes
    @ Out, convertMultipleToBytes, integer, the number of bytes
  """
  if   'mb' in sizeString:
    return int(sizeString.replace("mb",""))*10**6
  elif 'kb' in sizeString:
    return int(sizeString.replace("kb",""))*10**3
  elif 'gb' in sizeString:
    return int(sizeString.replace("gb",""))*10**9
  else:
    try:
      return int(sizeString)
    except:
      raise IOError(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '->  can not understand how to convert expression '+str(sizeString)+' to number of bytes. Accepted Mb,Gb,Kb (no case sentive)!')

# I don't think there's a reason to make this an enum, but it could be done.
trueThingsFull = ('True', 'Yes', '1')
trueThings = tuple(x[0].lower() for x in trueThingsFull)

def stringIsTrue(s):
  """
    Determines if provided entity corresponds to a truth statement
    @ In, s, string or castable, entity to check
    @ Out, stringIsTrue, bool, True if string is recognized by RAVEN as evaluating to True
  """
  # as far as I know, nothing in python cannot be cast as a string.
  s = str(s).strip()
  return s.lower().startswith(trueThings)

# I don't think there's a reason to make this an enum, but it could be done.
falseThingsFull = ('False', 'No', '0')
falseThings = tuple(x[0].lower() for x in falseThingsFull)
def stringIsFalse(s):
  """
    Determines if provided entity corresponds to a falsehood statement
    @ In, s, string or castable, entity to check
    @ Out, stringIsFalse, bool, False if string is recognized by RAVEN as evaluating to False
  """
  # as far as I know, nothing in python cannot be cast as a string.
  s = str(s).strip()
  return s.lower().startswith(falseThings)

boolThingsFull = tuple(list(trueThingsFull)+list(falseThingsFull))

def stringsThatMeanSilent():
  """
    Return list of strings that indicate a verbosity of the lowest level (just errors). You linguists add what you wish
    @ In, None
    @ Out, listOfStrings, list, list of strings that mean Silent in RAVEN
  """
  listOfStrings = list(['0','silent','false','f','n','no','none'])
  return listOfStrings

def stringsThatMeanPartiallyVerbose():
  """
    Return list of strings that indicate a verbosity of the medium level (errors and warnings). You linguists add what you wish.
    @ In, None
    @ Out, listOfStrings, list, list of strings that mean Quiet in RAVEN
  """
  listOfStrings = list(['1','quiet','some'])
  return listOfStrings

def stringsThatMeanVerbose():
  """
    Return list of strings that indicate full verbosity (errors warnings, messages). You linguists add what you wish.
    @ In, None
    @ Out, listOfStrings, list, list of strings that mean Full Verbosity in RAVEN
  """
  listOfStrings = list(['2','loud','true','t','y','yes','all'])
  return listOfStrings

def interpretBoolean(inArg):
  """
    Utility method to convert an inArg into a boolean.
    The inArg can be either a string or integer
    @ In, inArg, object, object to convert
    @ Out, interpretedObject, bool, the interpreted boolean
  """
  if type(inArg).__name__ == "bool":
    return inArg
  elif type(inArg).__name__ == "integer":
    if inArg == 0:
      return False
    else:
      return True
  elif type(inArg).__name__ in ['str','bytes','unicode']:
    if stringIsTrue(inArg):
      return True
    elif stringIsFalse(inArg):
      return False
    else:
      raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag("ERROR") + '-> can not convert string to boolean in method interpretBoolean!!!!')
  else:
    raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag("ERROR") + '-> type unknown in method interpretBoolean. Got' + type(inArg).__name__)

def isClose(f1, f2, relTolerance=1e-14, absTolerance=0.0):
  """
    Method to compare two floats
    @ In, f1, float, first float
    @ In, f2, float, first float
    @ In, relTolerance, float, optional, relative tolerance
    @ In, absTolerance, float, optional, absolute tolerance
    @ Out, isClose, bool, is it close enough?
  """
  return abs(f1-f2) <= max(relTolerance * max(abs(f1), abs(f2)), absTolerance)

def compare(s1,s2,relTolerance = 1e-14):
  """
    Method aimed to compare two strings. This method tries to convert the 2
    strings in float and uses an integer representation to compare them.
    In case the conversion is not possible (string or only one of the strings is
    convertable), the method compares strings as they are.
    @ In, s1, string, first string to be compared
    @ In, s2, string, second string to be compared
    @ In, relTolerance, float, relative tolerance
    @ Out, response, bool, the boolean response (True if s1==s2, False otherwise)
  """
  w1, w2 = floatConversion(s1), floatConversion(s2)
  if   type(w1) == type(w2) and type(w1) != float:
    return s1 == s2
  elif type(w1) == type(w2) and type(w1) == float:
    from . import mathUtils
    return mathUtils.compareFloats(w1,w2,relTolerance)
  elif type(w1) != type(w2) and type(w1) in [float,int] and type(w2) in [float,int]:
    w1, w2 = float(w1), float(w2)
    return compare(w1,w2)
  else:
    return (w1 == w2)

def intConversion (s):
  """
    Method aimed to cast a string as integer. If the conversion is not possible,
    it returns None
    @ In, s, string,  string to be converted
    @ Out, response, int or None, the casted value
  """
  try:
    return int(s)
  except (ValueError,TypeError) as e:
    return None

def floatConversion (s):
  """
    Method aimed to cast a string as float. If the conversion is not possible,
    it returns None
    @ In, s, string,  string to be converted
    @ Out, response, float or None, the casted value
  """
  try:
    return float(s)
  except (ValueError,TypeError) as e:
    return None

def partialEval(s):
  """
    Method aimed to evaluate a string as float or integer.
    If neither a float nor an integer can be casted, return
    the un-casted string
    @ In, s, string,  string to be converted
    @ Out, response, float or int or string, the casted value
  """
  evalS = intConversion(s)
  if evalS is None:
    evalS = floatConversion(s)
  if evalS is None:
    return s
  else:
    return evalS

def toString(s):
  """
    Method aimed to convert a string in type str
    @ In, s, string,  string to be converted
    @ Out, response, string, the casted value
  """
  if type(s) == type(""):
    return s
  else:
    return s.decode()

def toBytes(s):
  """
    Method aimed to convert a string in type bytes
    @ In, s, string,  string to be converted
    @ Out, response, bytes, the casted value
  """
  if type(s) == type(""):
    return s.encode()
  elif type(s).__name__ in ['unicode','str','bytes']:
    return bytes(s)
  else:
    return s

def toBytesIterative(s):
  """
    Method aimed to convert all the string-compatible content of
    an object (dict, list, or string) in type bytes (recursively call toBytes(s))
    @ In, s, object,  object whose content needs to be converted
    @ Out, response, object, a copy of the object in which the string-compatible has been converted
  """
  if type(s) == list:
    return [toBytes(x) for x in s]
  elif type(s) == dict:
    if len(s) == 0:
      return None
    tempdict = {}
    for key,value in s.items():
      tempdict[toBytes(key)] = toBytesIterative(value)
    return tempdict
  else:
    return toBytes(s)

def toStrish(s):
  """
    Method aimed to convert a string in str type
    @ In, s, string,  string to be converted
    @ Out, response, str, the casted value
  """
  if type(s) == type(""):
    return s
  elif type(s) == type(b""):
    return s
  else:
    return str(s)

def keyIn(dictionary,key):
  """
    Method that return the key or toBytes key if in, else returns None.
    Use like
    inKey = keyIn(adict,key)
    if inKey is not None:
      foo = adict[inKey]
    else:
      pass #not found
    @ In, dictionary, dict, the dictionary whose key needs to be returned
    @ Out, response, str or bytes, the key (converted in bytes if needed)
  """
  if key in dictionary:
    return key
  else:
    bin_key = toBytes(key)
    if bin_key in dictionary:
      return bin_key
    else:
      return None

def first(c):
  """
    Method to return the first element of collections,
    for a list this is equivalent to c[0], but this also
    work for things that are views
    @ In, c, collection, the collection
    @ Out, response, item, the next item in the collection
  """
  return next(iter(c))

def importFromPath(filename, printImporting = True):
  """
    Method to import a module from a given path
    @ In, filename, str, the full path of the module to import
    @ In, printImporting, bool, True if information about the importing needs to be printed out
    @ Out, importedModule, module, the imported module
  """
  if printImporting:
    print('(            ) '+UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('Message')+ '      -> importing module '+ filename)
  import imp, os.path
  try:
    (path, name) = os.path.split(filename)
    (name, ext) = os.path.splitext(name)
    (file, filename, data) = imp.find_module(name, [path])
    importedModule = imp.load_module(name, file, filename, data)
    pythonPath = os.environ.get("PYTHONPATH","")
    absPath = os.path.abspath(path)
    if absPath not in pythonPath:
      os.environ['PYTHONPATH'] = pythonPath+ os.pathsep + absPath
  except Exception as ae:
    raise Exception('(            ) '+ UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ '-> importing module '+ filename + ' at '+path+os.sep+name+' failed with error '+str(ae))
  return importedModule

def getRelativeSortedListEntry(sortedList,value,tol=1e-15):
  """
    !!WARNING!! This method expects "sortedList" to already be a sorted list of float values!
    There are faster methods if they are not floats, and this will NOT work at all on unsorted lists.
    - Looks for a (close enough) match to "value" in "sortedList" using binomial search.  If found,
    returns the index and value of the matching entry.  If not found, adds a new entry to the sortedList
    and returns the new index with the original value.
    It is recommended that this method be used to add ALL entries into the sorted list to keep it sorted.
    @ In, sortedList, list, list of __sorted__ float values
    @ In, value, float, value to search for match
    @ Out, sortedList, list, possibly modified by still ordered list of floats
    @ Out, match_index, int, index of match in sortedList
    @ Out, match, float, matching float
  """
  from .mathUtils import compareFloats #necessary to prevent errors at module load
  index = bisect.bisect_left(sortedList,value)
  match_index = None
  match = None
  #if "value" is smallest value in list...
  if index == 0:
    if len(sortedList)>0:
    #check if current first matches
      if compareFloats(sortedList[0], value, tol=tol):
        match = sortedList[0]
        match_index = index
  #if "value" is largest value in list...
  elif index > len(sortedList)-1:
    #check if current last matches
    if compareFloats(sortedList[-1], value, tol=tol):
      match = sortedList[-1]
      match_index = len(sortedList)-1
  #if "value" is in the middle...
  else:
    #check both neighbors (left and right) for a match
    for idx in [index-1, index]:
      if compareFloats(sortedList[idx], value, tol=tol):
        match = sortedList[idx]
        match_index = idx
  #if no match found, add it
  if match is None:
    sortedList.insert(index,value)
    match = value
    match_index = index
  return sortedList,match_index,match

# def metaclass_insert__getstate__(self):
#   """
#   Overwrite state (for pickle-ing)
#   we do not pickle the HDF5 (C++) instance
#   but only the info to re-load it
#   """
#   # capture what is normally pickled
#   state = self.__dict__.copy()
#   # we pop the database instance and close it
#   state.pop("database")
#   self.database.closeDatabaseW()
#   # what we return here will be stored in the pickle
#   return state
#
# def metaclass_insert__setstate__(self, newstate):
#   self.__dict__.update(newstate)
#   self.exist    = True

def metaclass_insert(metaclass,*baseClasses):
  """
    This allows a metaclass to be inserted as a base class.
    Metaclasses substitute in as a type(name,bases,namespace) function,
    and can be anywhere in the hierarchy.  This instantiates the
    metaclass so it can be used as a base class.
    Example use:
    class Foo(metaclass_insert(Metaclass)):
    This function is based on the method used in Benjamin Peterson's six.py
    @ In, metaclass, abc, the metaclass
    @ In, baseClasses, args*, base classes
    @ Out, metaclass, class, the new metaclass
  """
  namespace={}
  return metaclass("NewMiddleClass",baseClasses,namespace)

class abstractstatic(staticmethod):
  """
    This can be make an abstract static method
    import abc
    class A(metaclass_insert(abc.ABCMeta)):
      @abstractstatic
      def test():
        pass
    class B(A):
      @staticmethod
      def test():
        return 5
  """
  def __init__(self, function):
    """
      Constructor
      @ In, function, pointer, the function to 'abstract'
      @ Out, None
    """
    super(abstractstatic, self).__init__(function)
    function.__isabstractmethod__ = True
  __isabstractmethod__ = True

def find_crow(framework_dir):
  """
    Make sure that the crow path is in the python path. If not, add the path.
    @ In, framework_dir, string, the absolute path of the framework
    @ Out, None
  """
  try:
    import crow_modules.distribution1D
    return
  except:
    ravenDir = os.path.dirname(framework_dir)
    #Add the module directory to the search path.
    crowDirs = [os.path.join(ravenDir,"crow"),
                os.path.join(os.path.dirname(ravenDir),"crow"),
                ravenDir]
    if "CROW_DIR" in os.environ:
      crowDirs.insert(0,os.path.join(os.environ["CROW_DIR"]))
    #Check for editable install
    if len(glob.glob(os.path.join(ravenDir, "src", "crow_modules", "_randomENG*"))) > 0:
      sys.path.append(os.path.join(ravenDir, "src"))
      return
    for crowDir in crowDirs:
      pmoduleDir = os.path.join(crowDir,"install")
      if os.path.exists(pmoduleDir):
        sys.path.append(pmoduleDir)
        # we add it in pythonpath too
        os.environ['PYTHONPATH'] = os.environ.get("PYTHONPATH","") + os.pathsep + pmoduleDir
        return
    for crowDir in crowDirs:
      if os.path.exists(os.path.join(crowDir,"tests")):
        raise IOError(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ ' -> Crow was found in '+crowDir+' but does not seem to be compiled')
    raise IOError(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ ' -> Crow has not been found. It location is supposed to be one of '+str(crowDirs)+'. Has RAVEN been built?')

def add_path(absolutepath):
  """
    Method to add a path in the PYTHON PATH
    @ In, absolutepath, string, the absolute path to be added
    @ Out, None
  """
  if not os.path.exists(absolutepath):
    raise IOError(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ ' -> "'+absolutepath+ '" directory has not been found!')
  sys.path.append(absolutepath)
  # we add it in pythonpath too
  newPath = os.environ.get("PYTHONPATH","") + os.pathsep + absolutepath
  if len(newPath) >= 32000: #Some OS's have a limit of 2**15 for environ
    print("WARNING: excessive length PYTHONPATH:'"+str(newPath)+"'")
  os.environ['PYTHONPATH'] = newPath

def add_path_recursively(absoluteInitialPath):
  """
    Method to recursively add all the path and subpaths contained in absoluteInitialPath in the pythonpath
    @ In, absoluteInitialPath, string, the absolute path to add
    @ Out, None
  """
  for dirr,_,_ in os.walk(absoluteInitialPath):
    add_path(dirr)

def findCrowModule(name):
  """
    Method to find one of the crow module (e.g. distribution1D, interpolationNDpy, randomENG, etc.) and return it.
    @ In, name, str, the name of the module
    @ Out, module, instance, the instance of module of "name"
  """
  availableCrowModules = ['distribution1D','interpolationND','randomENG']
  # assert
  assert(name in availableCrowModules)
  # find the module
  try:
    module = import_module("crow_modules.{}".format(name))
  except (ImportError, ModuleNotFoundError) as ie:
    if not str(ie).startswith("No module named"):
      print('sys.path:', sys.path)
      raise ie
    module = import_module("{}".format(name))
  return module

def getPythonCommand():
  """
    Method to get the prefered python command.
    @ In, None
    @ Out, pythonCommand, str, the name of the command to use.
  """
  if os.name == "nt":
    pythonCommand = "python"
  else:
    pythonCommand = sys.executable
  ## Alternative method.  However, if called by run_tests or raven_framework
  ## sys.executable is already taken into account PYTHON_COMMAND and this
  ## logic
  #if sys.version_info.major > 2:
  #  if os.name == "nt":
  #    #Command is python on windows in conda and Python.org install
  #    pythonCommand = "python"
  #  else:
  #    pythonCommand = "python3"
  #else:
  #  pythonCommand = "python"
  #pythonCommand = os.environ.get("PYTHON_COMMAND", pythonCommand)
  return pythonCommand

def printCsv(csv,*args):
  """
    Writes the values contained in args to a csv file specified by csv
    @ In, csv, File instance, an open file object to which we will be writing
    @ In, args, dict, an arbitrary collection of values to write to the file
    @ Out, None
  """
  print(*args,file=csv,sep=',')

def printCsvPart(csv,*args):
  """
    Writes the values contained in args to a csv file specified by csv appending a comma
    to the end to allow more data to be written to the line.
    @ In, csv, File instance, an open file object to which we will be writing
    @ In, args, dict, an arbitrary collection of values to write to the file
    @ Out, None
  """
  print(*args,file=csv,sep=',',end=',')

def tryParse(text):
  """
    A convenience function for attempting to parse a string as a number (first,
    attempts to create an integer, and falls back to a float if the value has
    a decimal, and finally resorting to just returning the string in the case
    where the data cannot be converted).
    @ In, text, string we are trying to parse
    @ Out, value, int/float/string, the possibly converted type
  """

  ## FIXME is there anything that is a float that will raise an
  ## exception for int?

  ## Yes, inf and nan do not convert well to int, but would you
  ## ever have these in an input file? - dpm 6/8/16
  try:
    value = int(text)
  except ValueError:
    try:
      value = float(text)
    except ValueError:
      value = text
  ## If this tag exists, but has no internal text, then it is most likely
  ## a boolean value
  except TypeError:
    return True
  return value

def makeDir(dirName):
  """
    Function that will attempt to create a directory. If the directory already
    exists, this function will return silently with no error, however if it
    fails to create the directory for any other reason, then an error is
    raised.
    @ In, dirName, string, specifying the new directory to be created
    @ Out, None
  """
  try:
    os.makedirs(dirName)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(dirName):
      ## The path already exists so we can safely ignore this exception
      pass
    else:
      ## If it failed for some other reason, we want to see what the
      ## error is still
      raise

class pickleSafeSubprocessPopen(subprocess.Popen):
  """
    Subclass of subprocess.Popen used internally to prevent _handle member from being pickled.  On
    Windows, _handle contains an operating system reference that throws an exception when deep copied.
  """
  # Only define these methods on Windows to override deep copy/pickle (member may not exist on other
  #   platforms.
  if platform.system() == 'Windows':
    def __getstate__(self):
      """
        Returns a dictionary of the object state for pickling/deep copying.  Omits member '_handle',
        which cannot be deep copied when non-None.
        @ In, None
        @ Out, result, dict, the get state dict
      """
      result = self.__dict__.copy()
      del result['_handle']
      return result

    def __setstate__(self, d):
      """
        Used to load an object dictionary when unpickling.  Since member '_handle' could not be
        deep copied, load it back as value None.
        @ In, d, dict, previously stored namespace to restore
        @ Out, None
      """
      self.__dict__ = d
      self._handle = None

def removeDuplicates(objectList):
  """
    Method to efficiently remove duplicates from a list and maintain their
    order based on first appearance. See the url below for a description of why
    this is optimal:
    http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
    @ In, objectList, list, list from which to remove duplicates
    @ Out, uniqueObjectList, list, list with unique values ordered by their
      first appearance in objectList
  """
  seen = set()
  ## Store this locally so it doesn't have to be re-evaluated at each iteration
  ## below
  seen_add = seen.add
  ## Iterate through the list and only take x if it has not been seen.
  ## The 'or' here acts as a short circuit if the first condition is True, then
  ## the second will not be executed, otherwise x will be added to seen and
  ## since adding to a set is not a conditional operation, it will always return
  ## False, so in conjunction with the 'not' this will ensure that the first
  ## occurrence of x is added to seen and uniqueObjectList. Long explanation,
  ## but efficient computation.
  uniqueObjectList = [x for x in objectList if not (x in seen or seen_add(x))]
  return uniqueObjectList

def typeMatch(var,varTypeStr):
  """
    This method is aimed to check if a variable changed datatype
    @ In, var, python datatype, the first variable to compare
    @ In, varTypeStr, string, the type that this variable should have
    @ Out, match, bool, is the datatype changed?
  """
  typeVar = type(var)
  match = typeVar.__name__ == varTypeStr or typeVar.__module__+"."+typeVar.__name__ == varTypeStr
  if not match:
    # check if the types start with the same root
    if len(typeVar.__name__) <= len(varTypeStr):
      if varTypeStr.startswith(typeVar.__name__):
        match = True
    else:
      if typeVar.__name__.startswith(varTypeStr):
        match = True
  return match

def isASubset(setToTest,pileList):
  """
    Check if setToTest is ordered subset of pileList in O(n)
    @ In, setToTest, list, set that needs to be tested
    @ In, pileList, list, pile of sets
    @ Out, isASubset, bool, True if setToTest is a subset
  """

  if len(pileList) < len(setToTest):
    return False

  index = 0
  for element in setToTest:
    try:
      index = pileList.index(element, index) + 1
    except ValueError:
      return False
  else:
    return True

def filterAllSubSets(listOfLists):
  """
    Given list of listOfLists, return new list of listOfLists without subsets
    @ In, listOfLists, list of lists, all lists to check
    @ Out, setToTest, iterator, iterator over the list without subsets
  """
  for setToTest in listOfLists:
    if not any(isASubset(setToTest, pileList) for pileList in listOfLists
      if setToTest is not pileList):
      yield setToTest

def mergeDictionaries(*dictArgs):
  """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    Adapted from: http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
    @ In, dictArgs, dict, a list of dictionaries to merge
    @ Out, mergedDict, dict, merged dictionary including keys from everything in dictArgs.
  """
  mergedDict = {}
  for dictionary in dictArgs:
    overlap = set(dictionary.keys()).intersection(mergedDict.keys())
    if len(overlap):
      raise IOError(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ ' -> mergeDictionaries: the dictionaries being merged have the following overlapping keys: ' + ', '.join(overlap))
    mergedDict.update(dictionary)
  return mergedDict

def mergeSequences(seq1,seq2):
  """
    This method has been taken from "http://stackoverflow.com"
    It is aimed to merge two sequences (lists) into one preserving the order in the two lists
    e.g. ['A', 'B', 'C', 'D', 'E',           'H', 'I']
         ['A', 'B',           'E', 'F', 'G', 'H',      'J', 'K']
    will become
         ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    @ In, seq1, list, the first sequence to be merged
    @ In, seq2, list, the second sequence to be merged
    @ Out, merged, list, the merged list of elements
  """
  sm=SequenceMatcher(a=seq1,b=seq2)
  merged = []
  for (op, start1, end1, start2, end2) in sm.get_opcodes():
    if op == 'equal' or op=='delete':
      #This range appears in both sequences, or only in the first one.
      merged += seq1[start1:end1]
    elif op == 'insert':
      #This range appears in only the second sequence.
      merged += seq2[start2:end2]
    elif op == 'replace':
      #There are different ranges in each sequence - add both.
      merged += seq1[start1:end1]
      merged += seq2[start2:end2]
  return merged

def checkTypeRecursively(inObject):
  """
    This method check the type of the inner object in the inObject.
    If inObject is an interable, this method returns the type of the first element
    @ In, inObject, object, a pyhon object
    @ Out, returnType, str, the type of the inner object

  """
  returnType = type(inObject).__name__
  try:
    for val in inObject:
      returnType = checkTypeRecursively(val)
      break
  except:
    pass
  return returnType

def returnIdSeparator():
  """
    This method is aimed to return the ID separator that should be used cross the code when
    combined ids need to be assembled. For example, when the "EnsembleModel" creates new
    ``prefix`` ids for sub-models
    @ In, None
    @ Out, __idSeparator, string, the id separator
  """
  return __idSeparator

def getAllSubclasses(cls):
  """
    Recursively collect all of the classes that are a subclass of cls
    @ In, cls, the class to retrieve sub-classes.
    @ Out, getAllSubclasses, list of class objects for each subclass of cls.
  """
  return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in getAllSubclasses(s)]

def displayAvailable():
  """
    The return variable for backend default setting of whether a display is
    available or not. For instance, if we are running on the HPC without an X11
    instance, then we don't have the ability to display the plot, only to save it
    to a file
    @ In, None
    @ Out, dispaly, bool, return True if platform is Windows or environment varialbe
      'DISPLAY' is available, otherwise return False
  """
  display = False
  if platform.system() == 'Windows':
    display = True
  else:
    if os.getenv('DISPLAY'):
      display = True
    else:
      display = False
  return display

def which(cmd):
  """
    Emulate the which method in shutil.
    Return the path to an executable which would be run if the given cmd was called.
    If no cmd would be called, return None.
    @ In, cmd, str, the exe to check
    @ Out, which, str, the full path or None if not found
  """
  def _access_check(fn):
    """
      Just check if the path is executable
      @ In, fn, string, the file to check
      @ Out, _access_check, bool, if accessable or not?
    """
    return (os.path.exists(fn) and os.access(fn, os.X_OK) and not os.path.isdir(fn))
  if os.path.dirname(cmd):
    if _access_check(cmd):
      return cmd
    return None
  path = os.environ.get("PATH", os.defpath)
  if not path:
    return None
  path = path.split(os.pathsep)
  if sys.platform == "win32":
    if not os.curdir in path:
      path.insert(0, os.curdir)
    pathext = os.environ.get("PATHEXT", "").split(os.pathsep)
    if any(cmd.lower().endswith(ext.lower()) for ext in pathext):
      files = [cmd]
    else:
      files = [cmd + ext for ext in pathext]
  else:
    files = [cmd]
  seen = set()
  for dir in path:
    normdir = os.path.normcase(dir)
    if not normdir in seen:
      seen.add(normdir)
      for thefile in files:
        name = os.path.join(dir, thefile)
        if _access_check(name):
          return name
  return None

