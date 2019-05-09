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
Utilities for comparing strings.
"""
import re
import sys

#A float consists of possibly a + or -, followed possibly by some digits
# followed by one of ( digit. | .digit | or digit) possibly followed by some
# more digits possibly followed by an exponent
floatRe = re.compile("([-+]?\\d*(?:(?:\\d[.])|(?:[.]\\d)|(?:\\d))\\d*(?:[eE][+-]\\d+)?)")

def split_into_parts(s_var):
  """
    Splits the string into floating parts and not float parts
    @ In, s_var, string, the string to split
    @ Out, split_into_parts, string, a list where the even indexs are
       string and the odd indexs are floating point number strings.
  """
  return floatRe.split(s_var)

def short_text(a_str, b_str):
  """
    Returns a short portion of the text that shows the first difference
    @ In, a_str, string, the first text element
    @ In, b_str, string, the second text element
    @ Out, short_text, string, resulting shortened diff
  """
  a_str = repr(a_str)
  b_str = repr(b_str)
  display_len = 20
  half_display = display_len//2
  if len(a_str)+len(b_str) < display_len:
    return a_str+" "+b_str
  first_diff = -1
  i = 0
  while i < len(a_str) and i < len(b_str):
    if a_str[i] == b_str[i]:
      i += 1
    else:
      first_diff = i
      break
  if first_diff >= 0:
    #diff in content
    start = max(0, first_diff - half_display)
  else:
    #diff in length
    first_diff = min(len(a_str), len(b_str))
    start = max(0, first_diff - half_display)
  if start > 0:
    prefix = "..."
  else:
    prefix = ""
  return prefix+a_str[start:first_diff+half_display]+" "+prefix+\
    b_str[start:first_diff+half_display]

def set_default_options(options):
  """
    sets all the options to defaults
    @ In, options, dict, dictionary to add default options to
    @ Out, None
  """
  options["rel_err"] = float(options.get("rel_err", 1.e-10))
  options["zero_threshold"] = float(options.get("zero_threshold", sys.float_info.min*4.0))
  options["remove_whitespace"] = options.get("remove_whitespace", False)
  options["remove_unicode_identifier"] = options.get("remove_unicode_identifier", False)

def remove_whitespace_chars(s_var):
  """
    Removes whitespace characters
    @ In, s, string, to remove characters from
    @ Out, s, string, removed whitespace string
  """
  s_var = s_var.replace(" ", "")
  s_var = s_var.replace("\t", "")
  s_var = s_var.replace("\n", "")
  #if this were python3 this would work:
  #remove_whitespaceTrans = "".maketrans("",""," \t\n")
  #s = s.translate(remove_whitespaceTrans)
  return s_var

def remove_unicode_identifiers(s_var):
  """
    Removes the u infrount of a unicode string: u'string' -> 'string'
    Note that this also removes a u at the end of string 'stru' -> 'str'
      which is not intended.
    @ In, s_var, string, string to remove characters from
    @ Out, s_var, string, cleaned string
  """
  s_var = s_var.replace("u'", "'")
  return s_var

def compare_strings_with_floats(a_str, b_str, num_tol=1e-10,
                                zero_threshold=sys.float_info.min*4.0,
                                remove_whitespace=False,
                                remove_unicode_identifier=False):
  """
    Compares two strings that have floats inside them.
    This searches for floating point numbers, and compares them with a
    numeric tolerance.
    @ In, a_str, string, first string to use
    @ In, b_str, string, second string to use
    @ In, num_tol, float, the numerical tolerance.
    @ In, zeroThershold, float, it represents the value below which a float
       is considered zero (XML comparison only). For example, if
       zeroThershold = 0.1, a float = 0.01 will be considered as it was 0.0
    @ In, remove_whitespace, bool, if True, remove all whitespace before comparing.
    @ Out, compareStringWithFloats, (bool,string), (succeeded, note) where
      succeeded is a boolean that is true if the strings match, and note is
      a comment on the comparison.
  """
  if a_str == b_str:
    return (True, "Strings Match")
  if a_str is None or b_str is None:
    return (False, "One of the strings contain a None")
  if remove_whitespace:
    a_str = remove_whitespace_chars(a_str)
    b_str = remove_whitespace_chars(b_str)
  if remove_unicode_identifier:
    a_str = remove_unicode_identifiers(a_str)
    b_str = remove_unicode_identifiers(b_str)
  a_list = split_into_parts(a_str)
  b_list = split_into_parts(b_str)
  if len(a_list) != len(b_list):
    return (False, "Different numbers of float point numbers")
  for i in range(len(a_list)):
    a_part = a_list[i].strip()
    b_part = b_list[i].strip()
    if i % 2 == 0:
      #In string
      if a_part != b_part:
        return (False, "Mismatch of "+short_text(a_part, b_part))
    else:
      #In number
      a_float = float(a_part)
      b_float = float(b_part)
      a_float = a_float if abs(a_float) > zero_threshold else 0.0
      b_float = b_float if abs(b_float) > zero_threshold else 0.0
      if abs(a_float - b_float) > num_tol:
        return (False, "Numeric Mismatch of '"+a_part+"' and '"+b_part+"'")

  return (True, "Strings Match Floatwise")

def is_a_number(x_var):
  """
    Checks if x_var can be converted to a float.
    @ In, x_var, object, a variable or value
    @ Out, is_a_number, bool, True if x can be converted to a float.
  """
  try:
    float(x_var)
    return True
  except ValueError:
    return False
