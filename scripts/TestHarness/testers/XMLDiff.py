from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os,re
import xml.etree.ElementTree as ET

num_tol = 1e-10 #effectively zero for our purposes

#A float consists of possibly a + or -, followed possibly by some digits
# followed by one of ( digit. | .digit | or digit) possibly followed by some
# more digits possibly followed by an exponent
float_re = re.compile("([-+]?\d*(?:(?:\d[.])|(?:[.]\d)|(?:\d))\d*(?:[eE][+-]\d+)?)")

def splitIntoParts(s):
  """Splits the string into floating parts and not float parts
  s: the string
  returns a list where the even indexs are string and the odd
  indexs are floating point number strings.
  """
  return float_re.split(s)

def short_text(a,b):
  """Returns a short portion of the text that shows the first difference
  a: the first text element
  b: the second text element
  """
  a = repr(a)
  b = repr(b)
  display_len = 20
  half_display = display_len//2
  if len(a)+len(b) < display_len:
    return a+" "+b
  first_diff = -1
  i = 0
  while i < len(a) and i < len(b):
    if a[i] == b[i]:
      i += 1
    else:
      first_diff = i
      break
  if first_diff >= 0:
    #diff in content
    start = max(0,first_diff - half_display)
  else:
    #diff in length
    first_diff = min(len(a),len(b))
    start = max(0,first_diff - half_display_len)
  if start > 0:
    prefix = "..."
  else:
    prefix = ""
  return prefix+a[start:first_diff+half_display]+" "+prefix+b[start:first_diff+half_display]


def compareStringsWithFloats(a,b,num_tol = 1e-10, zero_threshold = sys.float_info.min*4.0):
  """ Compares two strings that have floats inside them.  This searches for
  floating point numbers, and compares them with a numeric tolerance.
  a: first string to use
  b: second string to use
  num_tol: the numerical tolerance.
  zero_thershold: it represents the value below which a float is considered zero (XML comparison only). For example, if zero_thershold = 0.1, a float = 0.01 will be considered as it was 0.0
  Return (succeeded, note) where succeeded is a boolean that is true if the
  strings match, and note is a comment on the comparison.
  """
  if a == b:
    return (True,"Strings Match")
  if a is None or b is None: return (False,"One of the strings contain a None")
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
        return (False,"Mismatch of "+short_text(aPart,bPart))
    else:
      #In number
      aFloat = float(aPart)
      bFloat = float(bPart)
      aFloat = aFloat if abs(aFloat) > zero_threshold else 0.0
      bFloat = bFloat if abs(bFloat) > zero_threshold else 0.0
      if abs(aFloat - bFloat) > num_tol:
        return (False,"Numeric Mismatch of '"+aPart+"' and '"+bPart+"'")

  return (True, "Strings Match Floatwise")

def find_branches(node,path,finished):
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
      finished = find_branches(child,npath,finished)
  return finished

def tree_to_list(node):
  """
    Converts XML tree to list of entries.  Useful to start recursive search.
    node: the xml tree root node to convert
    returns list(list(ET.Element)) of full paths to entries in xml tree
  """
  flattened = find_branches(node,[node],[])
  return list(tuple(f) for f in flattened)

def compare_list_entry(a_list,b_list):
  """
    Comparse flattened XML entries for equality
    a_list: list(ET.Element)
    b_list: list(ET.Element)
    returns (bool,val)
    bool is True if all tag, text, and attributes match, False otherwise
    qual is percent of matching terms
  """
  num_match = 0       #number of matching points between entries
  total_matchable = 0 #total tag, text, and attributes available to match
  match = True        #True if entries match
  diff = []           #tuple of (element, diff code, correct (a) value, test (b) value)
  for i in range(len(a_list)):
    if i > len(b_list) - 1:
      match = False
      diff.append((b_list[-1],XMLDiff.missing_child_node,a_list[i].tag,None))
      #could have matched the tag and attributes
      total_matchable += 1 + len(a_list[i].attrib.keys())
      #if text isn't empty, could have matched text, too
      if a_list[i].text is not None and len(a_list[i].text.strip())>0: total_matchable+=1
      continue
    a = a_list[i]
    b = b_list[i]
    #match tag
    same,note = compareStringsWithFloats(a.tag,b.tag)
    total_matchable += 1
    if not same:
      match = False
      diff.append((b,XMLDiff.not_match_tag,a.tag,b.tag))
    else:
      num_match += 1
    #match text
    #if (a.text is None or len(a.text)>0) and (b.text is None or len(b.text)>0):
    same,note = compareStringsWithFloats(a.text,b.text)
    if not same:
      match = False
      diff.append((b,XMLDiff.not_match_text,str(a.text),str(b.text)))
      total_matchable += 1
    else:
      if not(a.text is None or a.text.strip()!=''):
        num_match += 1
        total_matchable += 1
    #match attributes
    for attrib in a.attrib.keys():
      total_matchable += 1
      if attrib not in b.attrib.keys():
        match = False
        diff.append((b,XMLDiff.missing_attribute,attrib,None))
        continue
      same,note = compareStringsWithFloats(a.attrib[attrib],b.attrib[attrib])
      if not same:
        match = False
        diff.append((b,XMLDiff.not_match_attribute,(a,attrib),(b,attrib)))
      else:
        num_match += 1
    #note attributes in b not in a
    for attrib in b.attrib.keys():
      if attrib not in a.attrib.keys():
        match = False
        diff.append((b,XMLDiff.extra_attribute,attrib,None))
        total_matchable += 1
  # note elements in b not in a
  if len(b_list) > len(a_list):
    match = False
    for j in range(i,len(b_list)):
      diff.append((a_list[-1],XMLDiff.extra_child_node,b_list[j].tag,None))
      #count tag and attributes
      total_matchable += 1 + len(b_list[j].attrib.keys())
      #if text isn't empty, count text, too
      if b_list[i].text is not None and len(b_list[i].text.strip())>0: total_matchable+=1
  return (match,float(num_match)/float(total_matchable),diff)

def compare_unordered_element(a,b,*args,**kwargs):
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
  def fail_message(*args):
    """ adds the fail message to the list
    args: The arguments to the fail message (will be converted with str())
    """
    print_args = [path]
    print_args.extend(args)
    args_expanded = " ".join([str(x) for x in print_args])
    message.append(args_expanded)
  a_list = tree_to_list(a)
  b_list = tree_to_list(b)
  #search a for matches in b
  for a_entry in a_list:
    matchvals[a_entry] = {}
    diffs[a_entry] = {}
    for b_entry in b_list:
      same,matchval,diff = compare_list_entry(a_entry,b_entry)
      if same:
        b_list.remove(b_entry)
        del matchvals[a_entry]
        del diffs[a_entry]
        #since we found the match, remove from other near matches
        for close_key in diffs.keys():
          if b_entry in diffs[close_key].keys():
            del diffs[close_key][b_entry]
            del matchvals[close_key][b_entry]
        break
      else:
        matchvals[a_entry][b_entry] = matchval
        diffs[a_entry][b_entry] = diff
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
        if code == XMLDiff.missing_child_node:
          note+='    <'+b.tag+'> is missing child node: <'+right+'> vs <'+miss+'\n'
        elif code == XMLDiff.missing_attribute:
          note+='    <'+b.tag+'> is missing attribute: "'+right+'"\n'
        elif code == XMLDiff.extra_child_node:
          note+='    <'+b.tag+'> has extra child node: <'+right+'>\n'
        elif code == XMLDiff.extra_attribute:
          note+='    <'+b.tag+'> has extra attribute: "'+right+'" = "'+b.attrib[right]+'"\n'
        elif code == XMLDiff.not_match_tag:
          note+='    <'+b.tag+'> tag does not match: <'+right+'> vs <'+miss+'>\n'
        elif code == XMLDiff.not_match_attribute:
          note+='    <'+b.tag+'> attribute does not match: "'+right[1]+'" = "'+right[0].attrib[right[1]]+'" vs "'+miss[0].attrib[miss[1]]+'"\n'
        elif code == XMLDiff.not_match_text:
          note+='    <'+b.tag+'> text does not match: "'+right+'" vs "'+miss+'"\n'
        else:
          note+='     UNRECOGNIZED OPTION: "'+b.tag+'" "'+str(code)+'": "'+str(right)+'" vs "'+str(miss)+'"\n'

    return (False,[note])

def compare_ordered_element(a,b,*args,**kwargs):
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

  def fail_message(*args):
    """ adds the fail message to the list
    args: The arguments to the fail message (will be converted with str())
    """
    print_args = [path]
    print_args.extend(args)
    args_expanded = " ".join([str(x) for x in print_args])
    message.append(args_expanded)

  if a.tag != b.tag:
    same = False
    fail_message("mismatch tags ",a.tag,b.tag)
  else:
    path += a.tag + "/"
  if a.text != b.text:
    succeeded, note = compareStringsWithFloats(a.text, b.text, float(options.get("rel_err",1.e-10)), float(options.get("zero_threshold",sys.float_info.min*4.0)))
    if not succeeded:
      same = False
      fail_message(note)
      return (same, message)
  different_keys = set(a.keys()).symmetric_difference(set(b.keys()))
  same_keys = set(a.keys()).intersection(set(b.keys()))
  if len(different_keys) != 0:
    same = False
    fail_message("mismatch attribute keys ",different_keys)
  for key in same_keys:
    if a.attrib[key] != b.attrib[key]:
      same = False
      fail_message("mismatch attribute ",key,a.attrib[key],b.attrib[key])
  if len(a) != len(b):
    same = False
    fail_message("mismatch number of children ",len(a),len(b))
  else:
    if a.tag == b.tag:
      #find all matching XML paths
      #WARNING: this will mangle the XML, so other testing should happen above this!
      found=[]
      for i in range(len(a)):
        (same_child,message_child) = compare_ordered_element(a[i],b[i],*options,path=path)
        if same_child: found.append((a[i],b[i]))
        same = same and same_child
      #prune matches from trees
      for children in found:
        a.remove(children[0])
        b.remove(children[1])
      #once all pruning done, error on any remaining structure
      if counter==0: #on head now, recursion is finished
        if len(a)>0:
          a_string = ET.tostring(a)
          if len(a_string) > 80:
            message.append('Branches in gold not matching test...\n'+path)
          else:
            message.append('Branches in gold not matching test...\n'+path+
                           " "+a_string)
        if len(b)>0:
          b_string = ET.tostring(b)
          if len(b_string) > 80:
            message.append('Branches in test not matching gold...\n'+path)
          else:
            message.append('Branches in test not matching gold...\n'+path+
                           " "+b_string)
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
  missing_child_node  = 0
  missing_attribute   = 1
  extra_child_node    = 2
  extra_attribute     = 3
  not_match_tag       = 4
  not_match_attribute = 5
  not_match_text      = 6

  def __init__(self, test_dir, out_files,**kwargs):
    """ Create an XMLDiff class
    test_dir: the directory where the test takes place
    out_files: the files to be compared.  They will be in test_dir + out_files
               and test_dir + gold + out_files
    args: other arguments that may be included:
          - 'unordered': indicates unordered sorting
    """
    self.__out_files = out_files
    self.__messages = ""
    self.__same = True
    self.__test_dir = test_dir
    self.__options = kwargs

  def diff(self):
    """ Run the comparison.
    returns (same,messages) where same is true if all the
    xml files are the same, and messages is a string with all the
    differences.
    """
    # read in files
    for out_file in self.__out_files:
      test_filename = os.path.join(self.__test_dir,out_file)
      gold_filename = os.path.join(self.__test_dir, 'gold', out_file)
      if not os.path.exists(test_filename):
        self.__same = False
        self.__messages += 'Test file does not exist: '+test_filename
      elif not os.path.exists(gold_filename):
        self.__same = False
        self.__messages += 'Gold file does not exist: '+gold_filename
      else:
        files_read = True
        try:
          test_root = ET.parse( test_filename ).getroot()
        except Exception as e:
          files_read = False
          self.__messages += 'Exception reading file '+test_filename+': '+str(e.args)
        try:
          gold_root = ET.parse( gold_filename ).getroot()
        except Exception as e:
          files_read = False
          self.__messages += 'Exception reading file '+gold_filename+': '+str(e.args)
        if files_read:
          if 'unordered' in self.__options.keys() and self.__options['unordered']:
            same,messages = compare_unordered_element(gold_root,test_root,**self.__options)
          else:
            same,messages = compare_ordered_element(test_root, gold_root,**self.__options)
          if not same:
            self.__same = False
            separator = "\n"+" "*4
            self.__messages += "Mismatch between "+test_filename+" and "+gold_filename+separator
            self.__messages += separator.join(messages) + "\n"
        else:
          self.__same = False
    if '[' in self.__messages or ']' in self.__messages:
      self.__messages = self.__messages.replace('[','(')
      self.__messages = self.__messages.replace(']',')')
    return (self.__same,self.__messages)
