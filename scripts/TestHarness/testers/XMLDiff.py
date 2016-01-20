from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os,re
import xml.etree.ElementTree as ET

num_tol = 1e-10 #effectively zero for our purposes

float_re = re.compile("([-+]?(?:\d*[.])?\d+(?:[eE][+-]\d+)?)")

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


def compareStringsWithFloats(a,b,num_tol = 1e-10):
  """ Compares two strings that have floats inside them.  This searches for
  floating point numbers, and compares them with a numeric tolerance.
  a: first string to use
  b: second string to use
  num_tol: the numerical tolerance.
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
    aPart = aList[i]
    bPart = bList[i]
    if i % 2 == 0:
      #In string
      if aPart != bPart:
        return (False,"Mismatch of "+short_text(aPart,bPart))
    else:
      #In number
      aFloat = float(aPart)
      bFloat = float(bPart)
      if abs(aFloat - bFloat) > num_tol:
        return (False,"Numeric Mismatch of '"+aPart+"' and '"+bPart+"'")
  return (True, "Strings Match Floatwise")


def compare_element(a,b,*args,**kwargs):
  """ Compares two element trees and returns (same,message)
  where same is true if they are the same,
  and message is a list of the differences
  differ: XMLdiff object
  a: the first element tree
  b: the second element tree
  accepted args:
    'unordered': indicate test does not require ordered CSV
  accepted kwargs:
    path: a string to describe where the element trees are located (mainly
          used recursively)
  """
  same = True
  message = []
  options = args
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
    succeeded, note = compareStringsWithFloats(a.text, b.text)
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
        if 'unordered' in options:
          for j in range(len(b)):
            (same_child,message_child) = compare_element(a[i],b[j],*options,counter=counter+1,path=path)
            if same_child:
              found.append((a[i],b[i]))
              break
          if not same_child:
            same = False
        else:
          (same_child,message_child) = compare_element(a[i],b[i],*options,path=path)
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
  def __init__(self, test_dir, out_files,*args):
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
    self.__options = args

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
          same,messages = compare_element(test_root, gold_root,*self.__options)
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

