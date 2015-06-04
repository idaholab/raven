from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os
import xml.etree.ElementTree as ET

def compare_element(a,b,path=""):
  """ Compares two element trees and returns (same,message)
  where same is true if they are the same,
  and message is a list of the differences
  a: the first element tree
  b: the second element tree
  path: a string to describe where the element trees are located (mainly
  used recursively)
  """
  same = True
  message = []
  def fail_message(*args):
    """ adds the fail message to the list
    args: The arguments to the fail message (will be converted with str())
    """
    print_args = [path]
    print_args.extend(args)
    args_expanded = " ".join([str(x) for x in print_args])
    #print(*print_args)
    message.append(args_expanded)
  #print("processing ",path,a.tag,b.tag,len(a),len(b))
  if a.tag != b.tag:
    same = False
    fail_message("mismatch tags ",a.tag,b.tag)
  else:
    path += a.tag + "/"
  if a.text != b.text:
    same = False
    fail_message("mismatch text ",repr(a.text),repr(b.text))
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
      for i in range(len(a)):
        (same_child,message_child) = compare_element(a[i],b[i],path)
        same = same and same_child
        message.extend(message_child)
  return (same,message)


class XMLDiff:
  """ XMLDiff is used for comparing a bunch of xml files.
  """
  def __init__(self, test_dir, out_files):
    """ Create an XMLDiff class
    test_dir: the directory where the test takes place
    out_files: the files to be compared.  They will be in test_dir + out_files
    and test_dir + gold + out_files
    """
    self.__out_files = out_files
    self.__messages = ""
    self.__same = True
    self.__test_dir = test_dir

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
        self.addError(test_filename, 'File does not exist!')
      elif not os.path.exists(gold_filename):
        self.addError(gold_filename, 'Gold file does not exist!')
      else:
        try:
          test_root = ET.parse( test_filename ).getroot()
          gold_root = ET.parse( gold_filename ).getroot()
          same,messages = compare_element(test_root, gold_root)
          if not same:
            self.__same = False
            separator = "\n"+" "*4
            self.__messages += "Mismatch between "+test_filename+" and "+gold_filename+separator
            self.__messages += separator.join(messages) + "\n"
        except Exception as e:
          self.addError(out_file, 'Exception reading files '+str(e.args))
    return (self.__same,self.__messages)

