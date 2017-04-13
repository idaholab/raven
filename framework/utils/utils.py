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

from __future__ import division, print_function, absolute_import
# WARNING if you import unicode_literals here, we fail tests (e.g. framework.testFactorials).  This may be a future-proofing problem. 2015-04.
import warnings
warnings.simplefilter('default',DeprecationWarning)


#Do not import numpy or scipy or other libraries that are not
# built into python.  Otherwise the import can fail, and since utils
# are used by --library-report, this can cause diagnostic messages to fail.
import bisect
import sys, os, errno
import inspect
import subprocess
import platform
import copy
import numpy
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
  if moduleIn.endswith('.py') : moduleToLoadString = moduleIn[:-3]
  else                        : moduleToLoadString = moduleIn
  workingDirModule = os.path.abspath(os.path.join(workingDir,moduleToLoadString))
  if os.path.exists(workingDirModule+".py"):
    moduleToLoadString = workingDirModule
    path, filename = os.path.split(workingDirModule)
    os.sys.path.append(os.path.abspath(path))
  else:
    path, filename = os.path.split(moduleToLoadString)
    if (path != ''):
      abspath = os.path.abspath(path)
      if '~' in abspath:abspath = os.path.expanduser(abspath)
      if os.path.exists(abspath):
        caller.raiseAWarning('file '+moduleToLoadString+' should be relative to working directory. Working directory: '+workingDir+' Module expected at '+abspath)
        os.sys.path.append(abspath)
      else: caller.raiseAnError(IOError,'The path provided for the' + caller.type + ' named '+ caller.name +' does not exist!!! Got: ' + abspath + ' and ' + workingDirModule)
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
    if elem not in referenceList: unknownElements.append(elem)
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
  if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)): raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '->  path '+pathname+ ' is neither a file nor a dir!')
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
  if os.path.isfile(pathAndFileName): os.remove(pathAndFileName)

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
      if not inspect.ismodule(value): continue
    else:
      if not (inspect.ismodule(value) or inspect.ismethod(value)): continue
    if key != value.__name__:
      if value.__name__.split(".")[-1] != key: mods.append(str('import ' + value.__name__ + ' as '+ key))
      else                                   : mods.append(str('from ' + '.'.join(value.__name__.split(".")[:-1]) + ' import '+ key))
    else: mods.append(str(key))
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
  if   'mb' in sizeString: return int(sizeString.replace("mb",""))*10**6
  elif 'kb' in sizeString: return int(sizeString.replace("kb",""))*10**3
  elif 'gb' in sizeString: return int(sizeString.replace("gb",""))*10**9
  else:
    try   : return int(sizeString)
    except: raise IOError(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '->  can not understand how to convert expression '+str(sizeString)+' to number of bytes. Accepted Mb,Gb,Kb (no case sentive)!')

def stringsThatMeanTrue():
  """
    Return list of strings with the meaning of true in RAVEN (eng,ita,roman,french,german,chinese,latin, turkish, bool)
    @ In, None
    @ Out, listOfStrings, list, list of strings that mean True in RAVEN
  """
  listOfStrings = list(['yes','y','true','t','si','vero','dajie','oui','ja','yao','verum', 'evet', 'dogru', '1', 'on'])
  return listOfStrings

def stringsThatMeanFalse():
  """
    Return list of strings with the meaning of true in RAVEN (eng,ita,roman,french,german,chinese,latin, turkish, bool)
    @ In, None
    @ Out, listOfStrings, list, list of strings that mean False in RAVEN
  """
  listOfStrings = list(['no','n','false','f','nono','falso','nahh','non','nicht','bu','falsus', 'hayir', 'yanlis', '0', 'off'])
  return listOfStrings

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
  if type(inArg).__name__ == "bool": return inArg
  elif type(inArg).__name__ == "integer":
    if inArg == 0: return False
    else         : return True
  elif type(inArg).__name__ in ['str','bytes','unicode']:
    if inArg.lower().strip() in stringsThatMeanTrue()   : return True
    elif inArg.lower().strip() in stringsThatMeanFalse(): return False
    else                                                : raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag("ERROR") + '-> can not convert string to boolean in method interpretBoolean!!!!')
  else: raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag("ERROR") + '-> type unknown in method interpretBoolean. Got' + type(inArg).__name__)

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
  if   type(w1) == type(w2) and type(w1) != float: return s1 == s2
  elif type(w1) == type(w2) and type(w1) == float:
    from utils import mathUtils
    return mathUtils.compareFloats(w1,w2,relTolerance)
  elif type(w1) != type(w2) and type(w1) in [float,int] and type(w2) in [float,int]:
    w1, w2 = float(w1), float(w2)
    return compare(w1,w2)
  else: return (w1 == w2)

def intConversion (s):
  """
    Method aimed to cast a string as integer. If the conversion is not possible,
    it returns None
    @ In, s, string,  string to be converted
    @ Out, response, int or None, the casted value
  """
  try              : return int(s)
  except ValueError: return None

def floatConversion (s):
  """
    Method aimed to cast a string as float. If the conversion is not possible,
    it returns None
    @ In, s, string,  string to be converted
    @ Out, response, float or None, the casted value
  """
  try              : return float(s)
  except ValueError: return None

def partialEval(s):
  """
    Method aimed to evaluate a string as float or integer.
    If neither a float nor an integer can be casted, return
    the un-casted string
    @ In, s, string,  string to be converted
    @ Out, response, float or int or string, the casted value
  """
  evalS = intConversion(s)
  if evalS is None: evalS = floatConversion(s)
  if evalS is None: return s
  else            : return evalS

def toString(s):
  """
    Method aimed to convert a string in type str
    @ In, s, string,  string to be converted
    @ Out, response, string, the casted value
  """
  if type(s) == type(""): return s
  else                  : return s.decode()

def toBytes(s):
  """
    Method aimed to convert a string in type bytes
    @ In, s, string,  string to be converted
    @ Out, response, bytes, the casted value
  """
  if type(s) == type("")                            : return s.encode()
  elif type(s).__name__ in ['unicode','str','bytes']: return bytes(s)
  else                                              : return s

def toBytesIterative(s):
  """
    Method aimed to convert all the string-compatible content of
    an object (dict, list, or string) in type bytes (recursively call toBytes(s))
    @ In, s, object,  object whose content needs to be converted
    @ Out, response, object, a copy of the object in which the string-compatible has been converted
  """
  if type(s) == list: return [toBytes(x) for x in s]
  elif type(s) == dict:
    if len(s.keys()) == 0: return None
    tempdict = {}
    for key,value in s.items(): tempdict[toBytes(key)] = toBytesIterative(value)
    return tempdict
  else: return toBytes(s)

def toListFromNumpyOrC1array(array):
  """
    This method converts a numpy or c1darray into list
    @ In, array, numpy or c1array,  array to be converted
    @ Out, response, list, the casted value
  """
  response = array
  if type(array).__name__ == 'ndarray':
    response = array.tolist()
  elif type(array).__name__.split(".")[0] == 'c1darray':
    response = numpy.asarray(array).tolist()
  return response

def toListFromNumpyOrC1arrayIterative(array):
  """
    Method aimed to convert all the string-compatible content of
    an object (dict, list, or string) in type list from numpy and c1darray types (recursively call toBytes(s))
    @ In, array, object,  object whose content needs to be converted
    @ Out, response, object, a copy of the object in which the string-compatible has been converted
  """
  if type(array) == list: return [toListFromNumpyOrC1array(x) for x in array]
  elif type(array) == dict:
    if len(array.keys()) == 0: return None
    tempdict = {}
    for key,value in array.items(): tempdict[toBytes(key)] = toListFromNumpyOrC1arrayIterative(value)
    return tempdict
  else: return toBytes(array)

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
    if inKey is not None: foo = adict[inKey]
    else                : pass #not found
    @ In, dictionary, dict, the dictionary whose key needs to be returned
    @ Out, response, str or bytes, the key (converted in bytes if needed)
  """
  if key in dictionary:
    return key
  else:
    bin_key = toBytes(key)
    if bin_key in dictionary: return bin_key
    else                    : return None

def first(c):
  """
    Method to return the first element of collections,
    for a list this is equivalent to c[0], but this also
    work for things that are views
    @ In, c, collection, the collection
    @ Out, response, item, the next item in the collection
  """
  return next(iter(c))

def iter_len(c):
  """
    Method to count the number of elements in an iterable.
    @ In, c, the iterable
    @ Out, the number of items in the first level of the iterable
  """
  return sum(1 for _ in c)

def importFromPath(filename, printImporting = True):
  """
    Method to import a module from a given path
    @ In, filename, str, the full path of the module to import
    @ In, printImporting, bool, True if information about the importing needs to be printed out
    @ Out, importedModule, module, the imported module
  """
  if printImporting: print('(            ) '+UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('Message')+ '      -> importing module '+ filename)
  import imp, os.path
  try:
    (path, name) = os.path.split(filename)
    (name, ext) = os.path.splitext(name)
    (file, filename, data) = imp.find_module(name, [path])
    importedModule = imp.load_module(name, file, filename, data)
  except Exception as ae:
    raise Exception('(            ) '+ UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ '-> importing module '+ filename + ' at '+path+os.sep+name+' failed with error '+str(ae))
  return importedModule

def index(a, x):
  """
    Method to locate the leftmost value exactly equal to x in the list a (assumed to be sorted)
    @ In, a, list, the list that needs to be inquired
    @ In, x, float, the inquiring value
    @ Out, i, int, the index of the leftmost value exactly equal to x
  """
  i = bisect.bisect_left(a, x)
  if i != len(a) and a[i] == x: return i
  return None

def find_lt(a, x):
  """
    Method to Find rightmost value less than x in the list a (assumed to be sorted)
    @ In, a, list, the list that needs to be inquired
    @ In, x, float, the inquiring value
    @ Out, i, int, the index of the Find rightmost value less than x
  """
  i = bisect.bisect_left(a, x)
  if i: return a[i-1],i-1
  return None,None

def find_le_index(a,x):
  """
    Method to Find the index of the rightmost value less than or equal to x in the list a (assumed to be sorted)
    @ In, a, list, the list that needs to be inquired
    @ In, x, float, the inquiring value
    @ Out, i, int, the index of the rightmost value less than or equal to x
  """
  i = bisect.bisect_right(a, x)
  if i: return i
  return None

def find_le(a, x):
  """
    Method to Find the rightmost value less than or equal to x in the list a (assumed to be sorted)
    @ In, a, list, the list that needs to be inquired
    @ In, x, float, the inquiring value
    @ Out, i, tuple, tuple[0] -> the rightmost value less than or equal to x, tuple[1] -> index
  """
  i = bisect.bisect_right(a, x)
  if i: return a[i-1],i-1
  return None,None

def find_gt(a, x):
  """
    Method to Find the leftmost value greater than x in the list a (assumed to be sorted)
    @ In, a, list, the list that needs to be inquired
    @ In, x, float, the inquiring value
    @ Out, i, tuple, tuple[0] -> the leftmost value greater than x, tuple[1] -> index
  """
  i = bisect.bisect_right(a, x)
  if i != len(a): return a[i],i
  return None,None

def find_ge(a, x):
  """
    Method to Find the leftmost item greater than or equal to x in the list a (assumed to be sorted)
    @ In, a, list, the list that needs to be inquired
    @ In, x, float, the inquiring value
    @ Out, i, tuple, tuple[0] ->leftmost item greater than or equal to x, tuple[1] -> index
  """
  i = bisect.bisect_left(a, x)
  if i != len(a): return a[i],i
  return None,None

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
  from utils.mathUtils import compareFloats #necessary to prevent errors at module load
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

def interpolateFunction(x,y,option,z=None,returnCoordinate=False):
  """
    Method to interpolate 2D/3D points
    @ In, x, ndarray or cached_ndarray, the array of x coordinates
    @ In, y, ndarray or cached_ndarray, the array of y coordinates
    @ In, z, ndarray or cached_ndarray, optional, the array of z coordinates
    @ In, returnCoordinate, boolean, optional, true if the new coordinates need to be returned
    @ Out, i, ndarray or cached_ndarray or tuple, the interpolated values
  """
  options = copy.copy(option)
  if x.size <= 2:
    xi = x
  else:
    xi = np.linspace(x.min(),x.max(),int(options['interpPointsX']))
  if z != None:
    if y.size <= 2:
      yi = y
    else:
      yi = np.linspace(y.min(),y.max(),int(options['interpPointsY']))
    xig, yig = np.meshgrid(xi, yi)
    try:
      if ['nearest','linear','cubic'].count(options['interpolationType']) > 0 or z.size <= 3:
        if options['interpolationType'] != 'nearest' and z.size > 3:
          zi = griddata((x,y), z, (xi[None,:], yi[:,None]), method=options['interpolationType'])
        else:
          zi = griddata((x,y), z, (xi[None,:], yi[:,None]), method='nearest')
      else:
        rbf = Rbf(x,y,z,function=str(str(options['interpolationType']).replace('Rbf', '')), epsilon=int(options.pop('epsilon',2)), smooth=float(options.pop('smooth',0.0)))
        zi  = rbf(xig, yig)
    except Exception as ae:
      if 'interpolationTypeBackUp' in options.keys():
        print(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        zi = interpolateFunction(x,y,z,options)
      else:
        raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
    if returnCoordinate: return xig,yig,zi
    else               : return zi
  else:
    try:
      if ['nearest','linear','cubic'].count(options['interpolationType']) > 0 or y.size <= 3:
        if options['interpolationType'] != 'nearest' and y.size > 3: yi = griddata((x), y, (xi[:]), method=options['interpolationType'])
        else: yi = griddata((x), y, (xi[:]), method='nearest')
      else:
        xig, yig = np.meshgrid(xi, yi)
        rbf = Rbf(x, y,function=str(str(options['interpolationType']).replace('Rbf', '')),epsilon=int(options.pop('epsilon',2)), smooth=float(options.pop('smooth',0.0)))
        yi  = rbf(xi)
    except Exception as ae:
      if 'interpolationTypeBackUp' in options.keys():
        print(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        yi = interpolateFunction(x,y,options)
      else: raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
    if returnCoordinate:
      return xi,yi
    else:
      return yi

def line3DInterpolation(x,y,z,nPoints):
  """
    Method to interpolate 3D points on a line
    @ In, x, ndarray or cached_ndarray, the array of x coordinates
    @ In, y, ndarray or cached_ndarray, the array of y coordinates
    @ In, z, ndarray or cached_ndarray, the array of z coordinates
    @ In, nPoints, int, number of desired inteporlation points
    @ Out, i, ndarray or cached_ndarray or tuple, the interpolated values
  """
  options = copy.copy(option)
  data = np.vstack((x,y,z))
  tck , u= interpolate.splprep(data, s=1e-6, k=3)
  new = interpolate.splev(np.linspace(0,1,nPoints), tck)
  return new[0], new[1], new[2]

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
    import crow_modules.distribution1Dpy2
    return
  except:
    ravenDir = os.path.dirname(framework_dir)
    #Add the module directory to the search path.
    crowDirs = [os.path.join(ravenDir,"crow"),
                os.path.join(os.path.dirname(ravenDir),"crow")]
    if "CROW_DIR" in os.environ:
      crowDirs.insert(0,os.path.join(os.environ["CROW_DIR"]))
    for crowDir in crowDirs:
      pmoduleDir = os.path.join(crowDir,"install")
      if os.path.exists(pmoduleDir):
        sys.path.append(pmoduleDir)
        return
    for crowDir in crowDirs:
      if os.path.exists(os.path.join(crowDir,"tests")):
        raise IOError(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ ' -> Crow was found in '+crowDir+' but does not seem to be compiled')
    raise IOError(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ ' -> Crow has not been found. It location is supposed to be one of '+str(crowDirs))

def add_path(absolutepath):
  """
    Method to add a path in the PYTHON PATH
    @ In, absolutepath, string, the absolute path to be added
    @ Out, None
  """
  if not os.path.exists(absolutepath):
    raise IOError(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ ' -> "'+absolutepath+ '" directory has not been found!')
  sys.path.append(absolutepath)

def add_path_recursively(absoluteInitialPath):
  """
    Method to recursively add all the path and subpaths contained in absoluteInitialPath in the pythonpath
    @ In, absoluteInitialPath, string, the absolute path to add
    @ Out, None
  """
  for dirr,_,_ in os.walk(absoluteInitialPath): add_path(dirr)

def find_distribution1D():
  """
    Method to find the crow distribution1D module and return it.
    @ In, None
    @ Out, module, instance, the module of distribution1D
  """
  if sys.version_info.major > 2:
    try:
      import crow_modules.distribution1Dpy3
      return crow_modules.distribution1Dpy3
    except ImportError as ie:
      if not str(ie).startswith("No module named"):
        raise ie
      import distribution1Dpy3
      return distribution1Dpy3
  else:
    try:
      import crow_modules.distribution1Dpy2
      return crow_modules.distribution1Dpy2
    except ImportError as ie:
      if not str(ie).startswith("No module named"):
        raise ie
      import distribution1Dpy2
      return distribution1Dpy2

def find_interpolationND():
  """
    Method to find the crow interpolationND module and return it.
    @ In, None
    @ Out, module, instance, the module of interpolationND
  """
  if sys.version_info.major > 2:
    try:
      import crow_modules.interpolationNDpy3
      return crow_modules.interpolationNDpy3
    except ImportError as ie:
      if not str(ie).startswith("No module named"):
        raise ie
      import interpolationNDpy3
      return interpolationNDpy3
  else:
    try:
      import crow_modules.interpolationNDpy2
      return crow_modules.interpolationNDpy2
    except ImportError as ie:
      if not str(ie).startswith("No module named"):
        raise ie
      import interpolationNDpy2
      return interpolationNDpy2

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
      if varTypeStr.startswith(typeVar.__name__): match = True
    else:
      if typeVar.__name__.startswith(varTypeStr): match = True
  return match

def sizeMatch(var,sizeToCheck):
  """
    This method is aimed to check if a variable has an expected size
    @ In, var, python datatype, the first variable to compare
    @ In, sizeToCheck, int, the size this variable should have
    @ Out, sizeMatched, bool, is the size ok?
  """
  sizeMatched = True
  if len(numpy.atleast_1d(var)) != sizeToCheck: sizeMatched = False
  return sizeMatched


def isASubset(setToTest,pileList):
  """
    Check if setToTest is ordered subset of pileList in O(n)
    @ In, setToTest, list, set that needs to be tested
    @ In, pileList, list, pile of sets
    @ Out, isASubset, bool, True if setToTest is a subset
  """

  if len(pileList) < len(setToTest): return False

  index = 0
  for element in setToTest:
    try              : index = pileList.index(element, index) + 1
    except ValueError: return False
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
  except: pass
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
