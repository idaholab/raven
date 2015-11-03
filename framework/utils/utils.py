from __future__ import division, print_function, absolute_import
# WARNING if you import unicode_literals here, we fail tests (e.g. framework.testFactorials).  This may be a future-proofing problem. 2015-04.
import warnings
warnings.simplefilter('default',DeprecationWarning)


import numpy as np
import bisect
import sys, os
from scipy.interpolate import Rbf, griddata
import copy
import inspect

class Object(object):pass

#custom errors
class NoMoreSamplesNeeded(GeneratorExit): pass


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


def checkIfPathAreAccessedByAnotherProgram(pathname, timelapse = 10.0):
  """
    Method to check if a path (file or directory) is currently
    used by another program. It is based on accessing time...
    Probably there is a better way.
    @ In, pathname, string containing the all path
    @ In, timelapse, float, tollerance on time modification
    @ Out, boolean, True if it is used by another program, False otherwise
  """
  import stat
  import time
  mode = os.stat(pathname).st_mode
  if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)): raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '->  path '+pathname+ ' is neither a file nor a dir!')
  return abs(os.stat(pathname).st_mtime - time.time()) < timelapse

def checkIfLockedRavenFileIsPresent(pathname,filename="ravenLockedKey.raven"):
  """
    Method to check if a path (directory) contains an hidden raven file
    @ In, pathname, string containing the path
    @ In, filename, string containing the file name
    @ Out, boolean, True if it is present, False otherwise
  """
  filePresent = os.path.isfile(os.path.join(pathname,filename))
  open(os.path.join(pathname,filename), 'w')
  return filePresent

def returnImportModuleString(obj,moduleOnly=False):
  """
    Method to return a list of strings that represent the
    modules on which the 'obj' depends on. It already implements
    the 'import' statement or the 'from x import y'
    @ In, obj, instance, the object that needs to be inquired
    @ In, moduleOnly, bool, optional, get the modules only (True) or also the function dependencies(False)
    @ Out, list, list of string containing the modules
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
    @ Out, int, the default tag length
  """
  return 25

def UreturnPrintTag(intag):
  """
    Method to return the a string formatted with respect to the length
    obtained by the method getPrintTagLenght() (generally used for pre tag)
    @ In, intag, string, string that needs to be formatted
    @ Out, returnString, string, the formatted string
  """
  return intag.ljust(getPrintTagLenght())[0:getPrintTagLenght()]

def UreturnPrintPostTag(intag):
  """
    Method to return the a string formatted with respect to the length
    obtained by the method getPrintTagLenght() - 15 (generally used for post tag)
    @ In, intag, string, string that needs to be formatted
    @ Out, returnString, string, the formatted string
  """
  return intag.ljust(getPrintTagLenght()-15)[0:(getPrintTagLenght()-15)]

def convertMultipleToBytes(sizeString):
  """
    Convert multiple (e.g. Mbytes, Gbytes,Kbytes) in bytes
    International system type (e.g., 1 Mb = 10^6)
    @ In, sizeString, string, string that needs to be converted in bytes
    @ Out, bytes, integer, the number of bytes
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
    @ Out, listofstrings, list of strings that mean True in RAVEN
  """
  return list(['yes','y','true','t','si','vero','dajie','oui','ja','yao','verum', 'evet', 'dogru', '1', 'on'])

def stringsThatMeanFalse():
  """
    Return list of strings with the meaning of true in RAVEN (eng,ita,roman,french,german,chinese,latin, turkish, bool)
    @ In, None
    @ Out, listofstrings, list of strings that mean False in RAVEN
  """
  return list(['no','n','false','f','nono','falso','nahh','non','nicht','bu','falsus', 'hayir', 'yanlis', '0', 'off'])

def stringsThatMeanSilent():
  """
    Return list of strings that indicate a verbosity of the lowest level (just errors). You linguists add what you wish
    @ In, None
    @ Out, listofstrings, list of strings that mean Silent in RAVEN
  """
  return list(['0','silent','false','f','n','no','none'])

def stringsThatMeanPartiallyVerbose():
  """
    Return list of strings that indicate a verbosity of the medium level (errors and warnings). You linguists add what you wish.
    @ In, None
    @ Out, listofstrings, list of strings that mean Quiet in RAVEN
  """
  return list(['1','quiet','some'])

def stringsThatMeanVerbose():
  """
    Return list of strings that indicate full verbosity (errors warnings, messages). You linguists add what you wish.
    @ In, None
    @ Out, listofstrings, list of strings that mean Full Verbosity in RAVEN
  """
  return list(['2','loud','true','t','y','yes','all'])

def interpretBoolean(inarg):
  """
    Utility method to convert an inarg into a boolean.
    The inarg can be either a string or integer
    @ In, object, object to convert
    @ Out, interpretedObject, bool, the interpreted boolean
  """
  if type(inarg).__name__ == "bool": return inarg
  elif type(inarg).__name__ == "integer":
    if inarg == 0: return False
    else         : return True
  elif type(inarg).__name__ in ['str','bytes','unicode']:
      if inarg.lower().strip() in stringsThatMeanTrue()   : return True
      elif inarg.lower().strip() in stringsThatMeanFalse(): return False
      else                                                : raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag("ERROR") + '-> can not convert string to boolean in method interpretBoolean!!!!')
  else: raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag("ERROR") + '-> type unknown in method interpretBoolean. Got' + type(inarg).__name__)

def compare(s1,s2,sig_fig = 6):
  """
    Method aimed to compare two strings. This method tries to convert the 2
    strings in float and uses an integer representation to compare them.
    In case the conversion is not possible (string or only one of the strings is
    convertable), the method compares strings as they are.
    @ In, s1, string, first string to be compared
    @ In, s2, string, second string to be compared
    @ In, sig_fig, int, minimum number of digits that need to match
    @ Out, response, bool, the boolean response (True if s1==s2, False otherwise)
  """
  w1, w2 = floatConversion(s1), floatConversion(s2)
  if   type(w1) == type(w2) and type(w1) != float: return s1 == s2
  elif type(w1) == type(w2) and type(w1) == float: return int(w1*10**sig_fig) == int(w2*10**sig_fig)
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
    @ Out, response, str, the casted value
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

def convertNumpyToLists(inputDict):
  """
    Method aimed to convert a dictionary containing numpy
    arrays or a single numpy array in list
    @ In, inputDict, dict or numpy array,  object whose content needs to be converted
    @ Out, response, dict or list, same object with its content converted
  """
  returnDict = inputDict
  if type(inputDict) == dict:
    for key, value in inputDict.items():
      if   type(value) == np.ndarray: returnDict[key] = value.tolist()
      elif type(value) == dict      : returnDict[key] = (convertNumpyToLists(value))
      else                          : returnDict[key] = value
  elif type(inputDict) == np.ndarray: returnDict = inputDict.tolist()
  return returnDict

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

def importFromPath(filename, printImporting = True):
  """
    Method to import a module from a given path
    @ In, filename, str, the full path of the module to import
    @ In, printImporting, bool, True if information about the importing needs to be printed out
    @ Out, importedModule, module, the imported module
  """
  if printImporting: print(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('Message')+ '-> importing module '+ filename)
  import imp, os.path
  try:
    (path, name) = os.path.split(filename)
    (name, ext) = os.path.splitext(name)
    (file, filename, data) = imp.find_module(name, [path])
    importedModule = imp.load_module(name, file, filename, data)
  except Exception as ae:
    raise Exception(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ '-> importing module '+ filename + ' at '+path+os.sep+name+' failed with error '+str(ae))
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

def metaclass_insert(metaclass,*base_classes):
  """
    This allows a metaclass to be inserted as a base class.
    Metaclasses substitute in as a type(name,bases,namespace) function,
    and can be anywhere in the hierarchy.  This instantiates the
    metaclass so it can be used as a base class.
    Example use:
    class Foo(metaclass_insert(Metaclass)):
    This function is based on the method used in Benjamin Peterson's six.py
  """
  namespace={}
  return metaclass("NewMiddleClass",base_classes,namespace)

def interpolateFunction(x,y,option,z = None,returnCoordinate=False):
  """
    Method to interpolate 2D/3D points
    @ In, x, ndarray or cached_ndarray, the array of x coordinates
    @ In, y, ndarray or cached_ndarray, the array of y coordinates
    @ In, z, ndarray or cached_ndarray, optional, the array of z coordinates
    @ In, returnCoordinate, boolean, optional, true if the new coordinates need to be returned
    @ Out, i, ndarray or cached_ndarray or tuple, the interpolated values
  """
  options = copy.copy(option)
  if x.size <= 2: xi = x
  else          : xi = np.linspace(x.min(),x.max(),int(options['interpPointsX']))
  if z != None:
    if y.size <= 2: yi = y
    else          : yi = np.linspace(y.min(),y.max(),int(options['interpPointsY']))
    xig, yig = np.meshgrid(xi, yi)
    try:
      if ['nearest','linear','cubic'].count(options['interpolationType']) > 0 or z.size <= 3:
        if options['interpolationType'] != 'nearest' and z.size > 3: zi = griddata((x,y), z, (xi[None,:], yi[:,None]), method=options['interpolationType'])
        else: zi = griddata((x,y), z, (xi[None,:], yi[:,None]), method='nearest')
      else:
        rbf = Rbf(x,y,z,function=str(str(options['interpolationType']).replace('Rbf', '')), epsilon=int(options.pop('epsilon',2)), smooth=float(options.pop('smooth',0.0)))
        zi  = rbf(xig, yig)
    except Exception as ae:
      if 'interpolationTypeBackUp' in options.keys():
        print(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        zi = interpolateFunction(x,y,z,options)
      else: raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
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
    if returnCoordinate: return xi,yi
    else               : return yi

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
    pmoduleDirs = [os.path.join(os.path.dirname(ravenDir),"crow","install"),
                   os.path.join(ravenDir,"crow","install"),
                   os.path.join(os.path.dirname(ravenDir),"crow","crow_modules"),
                   os.path.join(ravenDir,"crow","crow_modules")]
    for pmoduleDir in pmoduleDirs:
      if os.path.exists(pmoduleDir):
        sys.path.append(pmoduleDir)
        return
    raise IOError(UreturnPrintTag('UTILS') + ': '+UreturnPrintPostTag('ERROR')+ ' -> The directory "crow_modules" has not been found. It location is supposed to be one of '+str(pmoduleDirs))

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
    @ Out, module, the module of distribution1D
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
    @ Out, module, the module of interpolationND
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
      @ In, csv, an open file object to which we will be writing
      @ In, args, an arbitrary collection of values to write to the file
    """
    print(*args,file=csv,sep=',')

def printCsvPart(csv,*args):
    """
      Writes the values contained in args to a csv file specified by csv appending a comma
      to the end to allow more data to be written to the line.
      @ In, csv, an open file object to which we will be writing
      @ In, args, an arbitrary collection of values to write to the file
    """
    print(*args,file=csv,sep=',',end=',')

def numpyNearestMatch(findIn,val):
  """
    Given an array, find the entry that most nearly matches the given value.
    @ In, findIn, the array to look in
    @ In, val, the value for which to find a match
    @ Out, tuple, index where match is and the match itself
  """
  idx = (np.abs(findIn-val)).argmin()
  return idx,findIn[idx]

def NDInArray(findIn,val,tol=1e-12):
  """
    checks a numpy array of numpy arrays for a near match, then returns info.
    @ In, findIn, numpy array of numpy arrays (both arrays can be any length)
    @ In, val, tuple/list/numpy array, entry to look for in findIn
    @ In, tol, float, tolerance to check match within
    @ Out, (bool,idx,val) -> (found/not found, index where found or None, findIn entry or None)
  """
  loc = np.where(np.all(np.abs(findIn-val)<tol,axis=1)==1)
  if len(loc[0])>0:
    found = True
    idx = loc[0][0]
    val = findIn[idx]
  else:
    found = False
    idx = val = None
  return found,idx,val

