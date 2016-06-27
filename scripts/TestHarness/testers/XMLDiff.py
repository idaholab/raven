from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os,re
import xml.etree.ElementTree as ET

numTol = 1e-10 #effectively zero for our purposes

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

def compareStringsWithFloats(a,b,numTol = 1e-10, zeroThreshold = sys.float_info.min*4.0, removeWhitespace = False):
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

def findBranches(node,path,finished):
  """
    Iterative process to convert XML tree into list of entries
    node: ET.Element whose children need sorting
    path: list(ET.Element) leading to node
    finished: list(list(ET.Element)) full entries
    returns list(list(ET.Element)) of full entries
  """
  for child in node:
    npath = path[:]+[child]
    if len(child)==0:
      finished.append(npath)
    else:
      finished = findBranches(child,npath,finished)
  return finished

def treeToList(node):
  """
    Converts XML tree to list of entries.  Useful to start recursive search.
    node: the xml tree root node to convert
    returns list(list(ET.Element)) of full paths to entries in xml tree
  """
  flattened = findBranches(node,[node],[])
  return list(tuple(f) for f in flattened)

def compareListEntry(aList,bList):
  """
    Comparse flattened XML entries for equality
    aList: list(ET.Element)
    bList: list(ET.Element)
    returns (bool,val)
    bool is True if all tag, text, and attributes match, False otherwise
    qual is percent of matching terms
  """
  numMatch = 0       #number of matching points between entries
  totalMatchable = 0 #total tag, text, and attributes available to match
  match = True        #True if entries match
  diff = []           #tuple of (element, diff code, correct (a) value, test (b) value)
  for i in range(len(aList)):
    if i > len(bList) - 1:
      match = False
      diff.append((bList[-1],XMLDiff.missingChildNode,aList[i].tag,None))
      #could have matched the tag and attributes
      totalMatchable += 1 + len(aList[i].attrib.keys())
      #if text isn't empty, could have matched text, too
      if aList[i].text is not None and len(aList[i].text.strip())>0: totalMatchable+=1
      continue
    a = aList[i]
    b = bList[i]
    #match tag
    same,note = compareStringsWithFloats(a.tag,b.tag)
    totalMatchable += 1
    if not same:
      match = False
      diff.append((b,XMLDiff.notMatchTag,a.tag,b.tag))
    else:
      numMatch += 1
    #match text
    #if (a.text is None or len(a.text)>0) and (b.text is None or len(b.text)>0):
    same,note = compareStringsWithFloats(a.text,b.text)
    if not same:
      match = False
      diff.append((b,XMLDiff.notMatchText,str(a.text),str(b.text)))
      totalMatchable += 1
    else:
      if not(a.text is None or a.text.strip()!=''):
        numMatch += 1
        totalMatchable += 1
    #match attributes
    for attrib in a.attrib.keys():
      totalMatchable += 1
      if attrib not in b.attrib.keys():
        match = False
        diff.append((b,XMLDiff.missingAttribute,attrib,None))
        continue
      same,note = compareStringsWithFloats(a.attrib[attrib],b.attrib[attrib])
      if not same:
        match = False
        diff.append((b,XMLDiff.notMatchAttribute,(a,attrib),(b,attrib)))
      else:
        numMatch += 1
    #note attributes in b not in a
    for attrib in b.attrib.keys():
      if attrib not in a.attrib.keys():
        match = False
        diff.append((b,XMLDiff.extraAttribute,attrib,None))
        totalMatchable += 1
  # note elements in b not in a
  if len(bList) > len(aList):
    match = False
    for j in range(i,len(bList)):
      diff.append((aList[-1],XMLDiff.extraChildNode,bList[j].tag,None))
      #count tag and attributes
      totalMatchable += 1 + len(bList[j].attrib.keys())
      #if text isn't empty, count text, too
      if bList[i].text is not None and len(bList[i].text.strip())>0: totalMatchable+=1
  return (match,float(numMatch)/float(totalMatchable),diff)

def compareUnorderedElement(a,b,*args,**kwargs):
  """
    Compares two element trees and returns (same,message)
    where same is true if they are the same,
    and message is a list of the differences.
    Uses list of tree entries to find best match, instead of climbing the tree
    a: the first element tree
    b: the second element tree
  """
  same = True
  message = []
  matchvals = {}
  diffs = {}
  def failMessage(*args):
    """ adds the fail message to the list
    args: The arguments to the fail message (will be converted with str())
    """
    printArgs = [path]
    printArgs.extend(args)
    argsExpanded = " ".join([str(x) for x in printArgs])
    message.append(argsExpanded)
  aList = treeToList(a)
  bList = treeToList(b)
  #search a for matches in b
  for aEntry in aList:
    matchvals[aEntry] = {}
    diffs[aEntry] = {}
    for bEntry in bList:
      same,matchval,diff = compareListEntry(aEntry,bEntry)
      if same:
        bList.remove(bEntry)
        del matchvals[aEntry]
        del diffs[aEntry]
        #since we found the match, remove from other near matches
        for closeKey in diffs.keys():
          if bEntry in diffs[closeKey].keys():
            del diffs[closeKey][bEntry]
            del matchvals[closeKey][bEntry]
        break
      else:
        matchvals[aEntry][bEntry] = matchval
        diffs[aEntry][bEntry] = diff
  if len(matchvals)==0: #all matches found
    return (True,'')
  else:
    note = ''
    for unmatched,close in matchvals.items():
      #print the path without a match
      note+='No match for '+'/'.join(list(m.tag for m in unmatched))+'\n'
      #print the tree of the nearest match
      note+='  Nearest unused match: '
      close = sorted(list(close.items()),key=lambda x:x[1],reverse=True)
      note+='    '+'/'.join(list(c.tag for c in close[0][0])) +'\n'#+', %2.1f %% match' %(100*close[0][1])+'\n'
      #print what was different between them
      diff =  diffs[unmatched][close[0][0]]
      for b,code,right,miss in diff:
        if b is None:
          b = str(b)
        if code is None:
          code = str(code)
        if right is None:
          right = str(right)
        if miss is None:
          miss = str(miss)
        if code == XMLDiff.missingChildNode:
          note+='    <'+b.tag+'> is missing child node: <'+str(right)+'> vs <'+miss+'>\n'
        elif code == XMLDiff.missingAttribute:
          note+='    <'+b.tag+'> is missing attribute: "'+right+'"\n'
        elif code == XMLDiff.extraChildNode:
          note+='    <'+b.tag+'> has extra child node: <'+right+'>\n'
        elif code == XMLDiff.extraAttribute:
          note+='    <'+b.tag+'> has extra attribute: "'+right+'" = "'+b.attrib[right]+'"\n'
        elif code == XMLDiff.notMatchTag:
          note+='    <'+b.tag+'> tag does not match: <'+right+'> vs <'+miss+'>\n'
        elif code == XMLDiff.notMatchAttribute:
          note+='    <'+b.tag+'> attribute does not match: "'+right[1]+'" = "'+right[0].attrib[right[1]]+'" vs "'+miss[0].attrib[miss[1]]+'"\n'
        elif code == XMLDiff.notMatchText:
          note+='    <'+b.tag+'> text does not match: "'+right+'" vs "'+miss+'"\n'
        else:
          note+='     UNRECOGNIZED OPTION: "'+b.tag+'" "'+str(code)+'": "'+str(right)+'" vs "'+str(miss)+'"\n'

    return (False,[note])

def compareOrderedElement(a,b,*args,**kwargs):
  """ Compares two element trees and returns (same,message)
  where same is true if they are the same,
  and message is a list of the differences
  a: the first element tree
  b: the second element tree
  accepted args:
    - none -
  accepted kwargs:
    path: a string to describe where the element trees are located (mainly
          used recursively)
  """
  same = True
  message = []
  options = kwargs
  path = kwargs.get('path','')
  counter = kwargs.get('counter',0)

  def failMessage(*args):
    """ adds the fail message to the list
    args: The arguments to the fail message (will be converted with str())
    """
    printArgs = [path]
    printArgs.extend(args)
    argsExpanded = " ".join([str(x) for x in printArgs])
    message.append(argsExpanded)

  if a.tag != b.tag:
    same = False
    failMessage("mismatch tags ",a.tag,b.tag)
  else:
    path += a.tag + "/"
  if a.text != b.text:
    succeeded, note = compareStringsWithFloats(a.text, b.text, float(options.get("rel_err",1.e-10)), float(options.get("zero_threshold",sys.float_info.min*4.0)),options.get("remove_whitespace",False))
    if not succeeded:
      same = False
      failMessage(note)
      return (same, message)
  differentKeys = set(a.keys()).symmetric_difference(set(b.keys()))
  sameKeys = set(a.keys()).intersection(set(b.keys()))
  if len(differentKeys) != 0:
    same = False
    failMessage("mismatch attribute keys ",differentKeys)
  for key in sameKeys:
    if a.attrib[key] != b.attrib[key]:
      same = False
      failMessage("mismatch attribute ",key,a.attrib[key],b.attrib[key])
  if len(a) != len(b):
    same = False
    failMessage("mismatch number of children ",len(a),len(b))
  else:
    if a.tag == b.tag:
      #find all matching XML paths
      #WARNING: this will mangle the XML, so other testing should happen above this!
      found=[]
      for i in range(len(a)):
        subOptions = dict(options)
        subOptions["path"] = path
        (sameChild,messageChild) = compareOrderedElement(a[i],b[i],*args,**subOptions)
        if sameChild: found.append((a[i],b[i]))
        same = same and sameChild
      #prune matches from trees
      for children in found:
        a.remove(children[0])
        b.remove(children[1])
      #once all pruning done, error on any remaining structure
      if counter==0: #on head now, recursion is finished
        if len(a)>0:
          aString = ET.tostring(a)
          if len(aString) > 80:
            message.append('Branches in gold not matching test...\n'+path)
          else:
            message.append('Branches in gold not matching test...\n'+path+
                           " "+aString)
        if len(b)>0:
          bString = ET.tostring(b)
          if len(bString) > 80:
            message.append('Branches in test not matching gold...\n'+path)
          else:
            message.append('Branches in test not matching gold...\n'+path+
                           " "+bString)
  return (same,message)

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

class XMLDiff:
  """ XMLDiff is used for comparing a bunch of xml files.
  """
  #codes for differences
  missingChildNode  = 0
  missingAttribute   = 1
  extraChildNode    = 2
  extraAttribute     = 3
  notMatchTag       = 4
  notMatchAttribute = 5
  notMatchText      = 6

  def __init__(self, testDir, outFile,**kwargs):
    """ Create an XMLDiff class
    testDir: the directory where the test takes place
    outFile: the files to be compared.  They will be in testDir + outFile
               and testDir + gold + outFile
    args: other arguments that may be included:
          - 'unordered': indicates unordered sorting
    """
    self.__outFile = outFile
    self.__messages = ""
    self.__same = True
    self.__testDir = testDir
    self.__options = kwargs

  def diff(self):
    """ Run the comparison.
    returns (same,messages) where same is true if all the
    xml files are the same, and messages is a string with all the
    differences.
    """
    # read in files
    for outfile in self.__outFile:
      testFilename = os.path.join(self.__testDir,outfile)
      goldFilename = os.path.join(self.__testDir, 'gold', outfile)
      if not os.path.exists(testFilename):
        self.__same = False
        self.__messages += 'Test file does not exist: '+testFilename
      elif not os.path.exists(goldFilename):
        self.__same = False
        self.__messages += 'Gold file does not exist: '+goldFilename
      else:
        filesRead = True
        try:
          testRoot = ET.parse( testFilename ).getroot()
        except Exception as e:
          filesRead = False
          self.__messages += 'Exception reading file '+testFilename+': '+str(e.args)
        try:
          goldRoot = ET.parse( goldFilename ).getroot()
        except Exception as e:
          filesRead = False
          self.__messages += 'Exception reading file '+goldFilename+': '+str(e.args)
        if filesRead:
          if 'unordered' in self.__options.keys() and self.__options['unordered']:
            same,messages = compareUnorderedElement(goldRoot,testRoot,**self.__options)
          else:
            same,messages = compareOrderedElement(testRoot, goldRoot,**self.__options)
          if not same:
            self.__same = False
            separator = "\n"+" "*4
            self.__messages += "Mismatch between "+testFilename+" and "+goldFilename+separator
            self.__messages += separator.join(messages) + "\n"
        else:
          self.__same = False
    if '[' in self.__messages or ']' in self.__messages:
      self.__messages = self.__messages.replace('[','(')
      self.__messages = self.__messages.replace(']',')')
    return (self.__same,self.__messages)
