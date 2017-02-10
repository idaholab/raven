import re,sys

#A float consists of possibly a + or -, followed possibly by some digits
# followed by one of ( digit. | .digit | or digit) possibly followed by some
# more digits possibly followed by an exponent
floatRe = re.compile("([-+]?\d*(?:(?:\d[.])|(?:[.]\d)|(?:\d))\d*(?:[eE][+-]\d+)?)")

def splitIntoParts(s):
  """Splits the string into floating parts and not float parts
  s: the string
  returns a list where the even indexs are string and the odd
  indexs are floating point number strings.
  """
  return floatRe.split(s)

def shortText(a,b):
  """Returns a short portion of the text that shows the first difference
  a: the first text element
  b: the second text element
  """
  a = repr(a)
  b = repr(b)
  displayLen = 20
  halfDisplay = displayLen//2
  if len(a)+len(b) < displayLen:
    return a+" "+b
  firstDiff = -1
  i = 0
  while i < len(a) and i < len(b):
    if a[i] == b[i]:
      i += 1
    else:
      firstDiff = i
      break
  if firstDiff >= 0:
    #diff in content
    start = max(0,firstDiff - halfDisplay)
  else:
    #diff in length
    firstDiff = min(len(a),len(b))
    start = max(0,firstDiff - halfDisplay)
  if start > 0:
    prefix = "..."
  else:
    prefix = ""
  return prefix+a[start:firstDiff+halfDisplay]+" "+prefix+b[start:firstDiff+halfDisplay]

def setDefaultOptions(options):
  """ sets all the options to defaults
  options: dictionary to add default options to
  """
  options["rel_err"] = float(options.get("rel_err",1.e-10))
  options["zero_threshold"] = float(options.get("zero_threshold",sys.float_info.min*4.0))
  options["remove_whitespace"] = options.get("remove_whitespace",False)
  options["remove_unicode_identifier"] = options.get("remove_unicode_identifier",False)

def removeWhitespaceChars(s):
  """ Removes whitespace characters
  s: string to remove characters from
  """
  s = s.replace(" ","")
  s = s.replace("\t","")
  s = s.replace("\n","")
  #if this were python3 this would work:
  #removeWhitespaceTrans = "".maketrans("",""," \t\n")
  #s = s.translate(removeWhitespaceTrans)
  return s

def removeUnicodeIdentifiers(s):
  """ Removes the u infrount of a unicode string: u'string' -> 'string'
  Note that this also removes a u at the end of string 'stru' -> 'str'
  which is not intended.
  s: string to remove characters from
  """
  s = s.replace("u'","'")
  return s

def compareStringsWithFloats(a,b,numTol = 1e-10, zeroThreshold = sys.float_info.min*4.0, removeWhitespace = False, removeUnicodeIdentifier = False):
  """ Compares two strings that have floats inside them.  This searches for
  floating point numbers, and compares them with a numeric tolerance.
  a: first string to use
  b: second string to use
  numTol: the numerical tolerance.
  zeroThershold: it represents the value below which a float is considered zero (XML comparison only). For example, if zeroThershold = 0.1, a float = 0.01 will be considered as it was 0.0
  removeWhitespace: if True, remove all whitespace before comparing.
  Return (succeeded, note) where succeeded is a boolean that is true if the
  strings match, and note is a comment on the comparison.
  """
  if a == b:
    return (True,"Strings Match")
  if a is None or b is None: return (False,"One of the strings contain a None")
  if removeWhitespace:
    a = removeWhitespaceChars(a)
    b = removeWhitespaceChars(b)
  if removeUnicodeIdentifier:
    a = removeUnicodeIdentifiers(a)
    b = removeUnicodeIdentifiers(b)
  aList = splitIntoParts(a)
  bList = splitIntoParts(b)
  if len(aList) != len(bList):
    return (False,"Different numbers of float point numbers")
  for i in range(len(aList)):
    aPart = aList[i].strip()
    bPart = bList[i].strip()
    if i % 2 == 0:
      #In string
      if aPart != bPart:
        return (False,"Mismatch of "+shortText(aPart,bPart))
    else:
      #In number
      aFloat = float(aPart)
      bFloat = float(bPart)
      aFloat = aFloat if abs(aFloat) > zeroThreshold else 0.0
      bFloat = bFloat if abs(bFloat) > zeroThreshold else 0.0
      if abs(aFloat - bFloat) > numTol:
        return (False,"Numeric Mismatch of '"+aPart+"' and '"+bPart+"'")

  return (True, "Strings Match Floatwise")

def isANumber(x):
  '''Checks if x can be converted to a float.
  @ In, x, a variable or value
  @ Out, bool, True if x can be converted to a float.
  '''
  try:
    float(x)
    return True
  except ValueError:
    return False
