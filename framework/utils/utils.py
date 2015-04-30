from __future__ import print_function
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
  if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)): raise Exception(returnPrintTag('UTILITIES')+': ' +returnPrintPostTag('ERROR') + '->  path '+pathname+ ' is neither a file nor a dir!')
  return abs(os.stat(pathname).st_mtime - time.time()) < timelapse

def checkIfLockedRavenFileIsPresent(pathname,filename="ravenLockedKey.raven"):
  """
  Method to check if a path (directory) contains an hidden raven file
  @ In, pathname, string containing the path
  @ In, filename, string containing the file name
  @ Out, boolean, True if it is present, False otherwise
  """
  import fcntl
  finm = os.path.join(pathname,filename)
  fp = open(finm, 'w')
  try           : fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except IOError: return True
  return False

def returnImportModuleString(obj,moduleOnly=False):
  mods = []
  globs = dict(inspect.getmembers(obj))
  for key, value in globs.items():
    if moduleOnly:
      if not inspect.ismodule(value): continue
    else:
      if not (inspect.ismodule(value) or inspect.ismethod(value)): continue
    if key != value.__name__:
      if value.__name__.split(".")[-1] != key: mods.append(str('import ' + value.__name__ + ' as '+ key))
      else                                   : mods.append(str('from ' + '.'.join(value.__name__.split(".")[:-1]) + ' import '+ key))
    else: mods.append(str(key))
  return mods

def getPrintTagLenght(): return 25

def returnPrintTag(intag): return intag.ljust(getPrintTagLenght())[0:getPrintTagLenght()]

def returnPrintPostTag(intag): return intag.ljust(getPrintTagLenght()-15)[0:(getPrintTagLenght()-15)]

def raiseAnError(etype,obj,msg):
  '''
    Standardized error raising. Currently halts code.
    @ In, etype, the error type to raise
    @ In, obj, either a string or a class instance to determine the label for the error
    @ In, msg, the error message to display
    @ Out, None
  '''
  if type(obj) in [str,unicode]:
    tag = obj
  else:
    try: obj.printTag
    except AttributeError: tag = str(obj)
    else: tag = str(obj.printTag)
  raise etype(returnPrintTag(tag)+': '+returnPrintPostTag('ERROR')+' -> '+str(msg))

def raiseAWarning(obj,msg,wtag='WARNING'):
  '''
    Standardized warning printing.
    @ In, obj, either a string or a class instance to determine the label for the warning
    @ In, msg, the warning message to display
    @ In, wtag, optional, the type of warning to display (default "WARNING")
    @ Out, None
  '''
  if type(obj) in [str,unicode]:
    tag = obj
  else:
    try: obj.printTag
    except AttributeError: tag = str(obj)
    else: tag = str(obj.printTag)
  print(returnPrintTag(tag)+': '+returnPrintPostTag(str(wtag))+' -> '+str(msg))

def raiseAMessage(obj,msg,wtag='Message'):
  '''
    Standardized message printing.
    @ In, obj, either a string or a class instance to determine the label for the message
    @ In, msg, the message to display
    @ In, wtag, optional, the type of warning to display (default "Message")
    @ Out, None
  '''
  raiseAWarning(obj,msg,wtag)

def convertMultipleToBytes(sizeString):
  '''
  Convert multiple (e.g. Mbytes, Gbytes,Kbytes) in bytes
  International system type (e.g., 1 Mb = 10^6)
  '''
  if   'mb' in sizeString: return int(sizeString.replace("mb",""))*10**6
  elif 'kb' in sizeString: return int(sizeString.replace("kb",""))*10**3
  elif 'gb' in sizeString: return int(sizeString.replace("gb",""))*10**9
  else:
    try   : return int(sizeString)
    except: raise IOError(returnPrintTag('UTILITIES')+': ' +returnPrintPostTag('ERROR') + '->  can not understand how to convert expression '+str(sizeString)+' to number of bytes. Accepted Mb,Gb,Kb (no case sentive)!')

def stringsThatMeanTrue():
  '''return list of strings with the meaning of true in RAVEN (eng,ita,roman,french,german,chinese,latin, turkish)'''
  return list(['yes','y','true','t','si','vero','dajie','oui','ja','yao','etiam', 'evet', 'dogru'])

def stringsThatMeanFalse():
  '''return list of strings with the meaning of true in RAVEN (eng,ita,roman,french,german,chinese,latin, turkish)'''
  return list(['no','n','false','f','nono','falso','nahh','non','nicht','bu','falsus', 'hayir', 'yanlis'])

def interpretBoolean(inarg):
  """
   Utility method to convert an inarg into a boolean.
   The inarg can be either a string or integer
   @ In, object, object to convert (
  """
  if type(inarg).__name__ == "bool": return inarg
  elif type(inarg).__name__ == "integer":
    if inarg == 0: return False
    else         : return True
  elif type(inarg).__name__ in ['str','bytes','unicode']:
      if inarg.lower().strip() in stringsThatMeanTrue()   : return True
      elif inarg.lower().strip() in stringsThatMeanFalse(): return False
      else                                                : raise Exception(returnPrintTag('UTILITIES')+': ' +returnPrintPostTag("ERROR") + '-> can not convert string to boolean in method interpretBoolean!!!!')
  else: raise Exception(returnPrintTag('UTILITIES')+': ' +returnPrintPostTag("ERROR") + '-> type unknown in method interpretBoolean. Got' + type(inarg).__name__)


def compare(s1,s2):
  sig_fig=6
  w1 = partialEval(s1)
  w2 = partialEval(s2)
  if type(w1) == type(w2) and type(w1) != float: return w1 == w2
  elif type(w1) == type(w2) and type(w1) == float: return int(w1*10**sig_fig) == int(w2*10**sig_fig)
  elif type(w1) != type(w2) and type(w1) in [float,int] and type(w2) in [float,int]:
    w1 = float(w1)
    w2 = float(w2)
    return compare(w1,w2)
  else: return (w1 == w2)

def partialEval(s):
  try:
    r = int(s)
    return r
  except ValueError:
    pass
  try:
    r = float(s)
    return r
  except ValueError:
    pass
  return s

def toString(s):
  if type(s) == type(""):
    return s
  else:
    return s.decode()

def toBytes(s):
  if type(s) == type(""):
    return s.encode()
  elif type(s).__name__ in ['unicode','str','bytes']: return bytes(s)
  else:
    return s

def toBytesIterative(s):
  if type(s) == list: return [toBytes(x) for x in s]
  elif type(s) == dict:
    if len(s.keys()) == 0: return None
    tempdict = {}
    for key,value in s.items(): tempdict[toBytes(key)] = toBytesIterative(value)
    return tempdict
  else: return toBytes(s)

def toStrish(s):
  if type(s) == type(""):
    return s
  elif type(s) == type(b""):
    return s
  else:
    return str(s)

def convertDictToListOfLists(inputDict):
  if type(inputDict) == dict:
    returnList = [[],[]]
    for key, value in inputDict.items():
      returnList[0].append(key)
      if type(value) == dict: returnList[1].append(convertDictToListOfLists(value))
      else: returnList[1].append(value)
  else:
    print(returnPrintTag('UTILS') + ': '+returnPrintPostTag('WARNING')+ ' -> in method "convertDictToListOfLists", inputDict is not a dictionary!')
    returnList = None
  return returnList

def convertNumpyToLists(inputDict):
  returnDict = inputDict
  if type(inputDict) == dict:
    for key, value in inputDict.items():
      if   type(value) == np.ndarray: returnDict[key] = value.tolist()
      elif type(value) == dict      : returnDict[key] = (convertNumpyToLists(value))
      else                          : returnDict[key] = value
  elif type(inputDict) == np.ndarray:
    returnDict = inputDict.tolist()
  return returnDict

def keyIn(dictionary,key):
  """Returns the key or toBytes key if in,
  else returns none.  Use like
  inKey = keyIn(adict,key)
  if inKey is not None:
     foo = adict[inKey]
  else:
     pass #not found"""
  if key in dictionary:
    return key
  else:
    bin_key = toBytes(key)
    if bin_key in dictionary:
      return bin_key
    else:
      return None

def first(c):
  """Returns the first element of collections,
  for a list this is equivalent to c[0], but this also
  work for things that are views"""
  return next(iter(c))

def importFromPath(filename, printImporting = True):
    if printImporting: print(returnPrintTag('UTILS') + ': '+returnPrintPostTag('Message')+ '-> importing module '+ filename)
    import imp, os.path
    try:
      (path, name) = os.path.split(filename)
      (name, ext) = os.path.splitext(name)
      (file, filename, data) = imp.find_module(name, [path])
      importedModule = imp.load_module(name, file, filename, data)
    except Exception as ae:
      raise Exception(returnPrintTag('UTILS') + ': '+returnPrintPostTag('ERROR')+ '-> importing module '+ filename + 'failed with error '+str(ae))
    return importedModule

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x: return i
    return None

def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i: return a[i-1],i-1
    return None,None

def find_le_index(a,x):
    'Find the index of the rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i: return i
    return None

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i: return a[i-1],i-1
    return None,None

def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a): return a[i],i
    return None,None

def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
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
  """This allows a metaclass to be inserted as a base class.
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
   Function to interpolate 2D/3D points
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
        print(returnPrintTag('UTILITIES')+': ' +returnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        zi = interpolateFunction(x,y,z,options)
      else: raise Exception(returnPrintTag('UTILITIES')+': ' +returnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
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
        print(returnPrintTag('UTILITIES')+': ' +returnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        yi = interpolateFunction(x,y,options)
      else: raise Exception(returnPrintTag('UTILITIES')+': ' +returnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
    if returnCoordinate: return xi,yi
    else               : return yi

class abstractstatic(staticmethod):
  """This can be make an abstract static method
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
  """ Make sure that the crow path is in the python path. """
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
    raise IOError(returnPrintTag('UTILS') + ': '+returnPrintPostTag('ERROR')+ ' -> The directory "crow_modules" has not been found. It location is supposed to be one of '+pmoduleDirs)

def add_path(absolutepath):
  """ Add absolutepath path is in the python path. """
  if not os.path.exists(absolutepath):
    raise IOError(returnPrintTag('UTILS') + ': '+returnPrintPostTag('ERROR')+ ' -> "'+absolutepath+ '" directory has not been found!')
  sys.path.append(absolutepath)

def add_path_recursively(absoluteInitialPath):
  for dirr,_,_ in os.walk(absoluteInitialPath): add_path(dirr)

def find_distribution1D():
  """ find the crow distribution1D module and return it. """
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
  """ find the crow interpolationND module and return it. """
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
    '''
      Writes the values contained in args to a csv file specified by csv
      @ In, csv, an open file object to which we will be writing
      @ In, args, an arbitrary collection of values to write to the file
    '''
    print(*args,file=csv,sep=',')

def printCsvPart(csv,*args):
    '''
      Writes the values contained in args to a csv file specified by csv appending a comma
      to the end to allow more data to be written to the line.
      @ In, csv, an open file object to which we will be writing
      @ In, args, an arbitrary collection of values to write to the file
    '''
    print(*args,file=csv,sep=',',end=',')

def numpyNearestMatch(findIn,val):
  '''
    Given an array, find the entry that most nearly matches the given value.
    @ In, findIn, the array to look in
    @ In, val, the value for which to find a match
    @ Out, tuple, index where match is and the match itself
  '''
  idx = (np.abs(findIn-val)).argmin()
  return idx,findIn[idx]

def NDInArray(findIn,val,tol=1e-12):
  '''
    checks a numpy array of numpy arrays for a near match, then returns info.
    @ In, findIn, numpy array of numpy arrays (both arrays can be any length)
    @ In, val, tuple/list/numpy array, entry to look for in findIn
    @ In, tol, float, tolerance to check match within
    @ Out, (bool,idx,val) -> (found/not found, index where found or None, findIn entry or None)
  '''
  loc = np.where(np.all(np.abs(findIn-val)<tol,axis=1)==1)
  if len(loc[0])>0:
    found = True
    idx = loc[0][0]
    val = findIn[idx]
  else:
    found = False
    idx = val = None
  return found,idx,val

