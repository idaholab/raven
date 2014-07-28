import numpy as np

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
    print('UTILS         : WARNING -> in method "convertDictToListOfLists", inputDict is not a dictionary!')
    returnList = None
  return returnList


def convertNumpyToLists(inputDict):
  returnDict = inputDict
  if type(inputDict) == dict:
    for key, value in inputDict.items():
      if   type(value) == np.ndarray: returnDict[key] = value.tolist() 
      elif type(value) == dict      : returnDict[key] = (convertNumpyToLists(value))
      else                          : returnDict[key] = value 
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
    if printImporting: print('importing module '+ filename)
    import imp, os.path
    try:
      (path, name) = os.path.split(filename)
      (name, ext) = os.path.splitext(name)
      (file, filename, data) = imp.find_module(name, [path])
      importedModule = imp.load_module(name, file, filename, data)
    except: importedModule = None   
    return importedModule
 

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
  return metaclass("NewMiddleMeta",base_classes,namespace)

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
