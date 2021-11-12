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
from __future__ import print_function, unicode_literals
import xml.dom.minidom as pxml
import xml.etree.ElementTree as ET
import os
import sys
import shutil

def createBackup(filename):
  """
    Creates a backup file based on the file at filename.  If it exists, prints an error message and returns.
    @ In, filename, string (to be appended with '.bak')
    @Out, bool, False on success or True on fail
  """
  bakname = filename+'.bak'
  if not os.path.isfile(bakname):
    shutil.copyfile(filename, bakname)
    return False
  else:
    print('ERROR! Backup file already exists:', bakname)
    print('    If you wish to continue, remove the backup and rerun the script.')
    return True


def prettify(tree):
  """
    Script for turning XML tree into something mostly RAVEN-preferred.  Does not align attributes as some devs like (yet).
    The output can be written directly to a file, as open('whatever.who','w').writelines(prettify(mytree))
    @ In, tree, xml.etree.ElementTree object, the tree form of an input file
    @Out, towrite, string, the entire contents of the desired file to write, including newlines
  """
  #make the first pass at pretty.  This will insert way too many newlines, because of how we maintain XML format.
  pretty = pxml.parseString(ET.tostring(tree.getroot())).toprettyxml(indent='  ')
  #loop over each "line" and toss empty ones, but for ending main nodes, insert a newline after.
  towrite=''
  for line in pretty.split('\n'):
    if line.strip()=='':continue
    towrite += line.rstrip()+'\n'
    if line.startswith('  </'): towrite+='\n'
  return towrite


def standardMain(argv,convert):
  """
    For most scripts, this should be the desired behavior of the 'if __name__=="__main__"' block of a converstion script.
    It covers getting the filename list, checking files exist, creating backups, converting, prettifying, and writing.
    @ In, argv, the system arguments from sys.argv
    @ In, convert, the convert method to be applied
    @Out, integer, 0 on success or (number of failures) on failure

    The possible arguments to pass through argv and their effects are:
    --remove-comments , does not preserve comments in the file
  """
  #require a list of files to act on
  if len(argv)<2:
    raise IOError('No filenames listed to modify!\n    Usage: python path/to/script.py infile1.xml infile2.xml infile3.xml\n'+
                                                  '       or: python path/to/script.py --tests')
  #keep comments?  True by default, turn off with argv '--remove-comments'
  if '--remove-comments' in argv:
    keep_comments=False
    argv.remove('--remove-comments')
  else: keep_comments = True
  always_rewrite = True
  if '--no-rewrite' in argv:
    always_rewrite = False
    argv.remove('--no-rewrite')
  #offer option to apply to all framework tests
  if '--tests' in argv:
    #get list of all 'tests' files
    pathToGCT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),'developer_tools')
    sys.path.append(pathToGCT)
    import get_coverage_tests as gct
    fileDict = gct.getRegressionTests()
    filelist = []
    for dir,files in fileDict.items():
      for f in files:
        filelist.append(os.path.join(dir,f))
  else: #explicilty list files to run
    #remove the script name itself from the list
    filelist = argv[1:]

  #track the failed attempts
  failures = 0
  maxname = max(len(fname) for fname in filelist)
  #iterate over files
  not_converted_files = 0
  for fname in filelist:
    if not os.path.isfile(fname):
      #file doesn't exist, but do continue on to others
      print('ERROR!  File not found:', fname)
      failures+=1
      continue
    if createBackup(fname)==False: #sucessful operation
      #change comments to comment nodes
      strfile = ''.join(line for line in open(fname,'r'))
      if keep_comments: strfile = convertToRavenComment(strfile)

      tree = ET.ElementTree(ET.fromstring(strfile))
      if not always_rewrite:
        tree_copy = prettify(tree)
      convert(tree,fileName=fname)
      if not always_rewrite:
        if prettify(tree) == tree_copy:
          print('File '+fname+ ' not converted since no syntax modifications have been detected')
          not_converted_files+=1
          continue
      convMessage = ('Converting '+fname+'...').ljust(14+maxname,'.')
      towrite = prettify(tree)
      if keep_comments: towrite = convertFromRavenComment(towrite)
      open(fname,'w').writelines(towrite)
      convMessage += 'converted.'
      print(convMessage)
    else:
      #backup was not successfully created
      failures+=1
  if failures>0:
    print('\n%i files converted, but there were %i failures.  See messages above.' %(len(filelist)-not_converted_files-failures,failures))
  else:
    print('\nConversion script completed successfully.  %i files converted.' %(len(filelist)-not_converted_files))
  return failures

def convertFromRavenComment(msg):
  """
    Converts fake comment nodes back into real comments
    @ In, msg, converted file contents as a string (with line seperators)
    @ Out, string, string contents of a file
  """
  msg=msg.replace('<ravenTEMPcomment>','<!--')
  msg=msg.replace('</ravenTEMPcomment>','-->')
  return msg

def convertToRavenComment(msg):
  """
    Converts existing comments temporarily into nodes.
    @ In, msg, string contents of a file
    @ Out, string, converted file contents as a string (with line seperators)
  """
  msg=msg.replace('<!--','<ravenTEMPcomment>')
  msg=msg.replace('-->' ,'</ravenTEMPcomment>')
  return msg


def oldconvertToRavenComment(msg):
  """
    Old way that Converts existing comments temporarily into nodes.  Keeping code just in case parts of it are useful later.
    @ In, msg, string contents of a file
    @ Out, string, converted file contents as a string (with line seperators)
  """
  depth = 0                                                #for nested comments
  numreplaced=0                                            #tracks number of edits made, for tracking reading index position
  #find first comment-related event
  next_open = msg.find('<!--',0)
  next_close = msg.find('-->',0)
  if next_close == next_open == -1: next_idx=-1            #no more comment events
  elif next_open<0: next_idx = next_close                  #no more opens, but close still exist
  elif next_close<0: raise IOError('Mismatched comments!') #more opens but no close events!
  else: next_idx = min(next_open,next_close)               #both opens and closes still exist, so just take nearest
  while next_idx > 0:
    if next_idx == next_open:                              #open event is next
      if depth==0: start_place = next_idx                  #new comment patch, so track start place
      depth+=1
    else:                                                  #next event is a closure
      depth -= 1
      if depth==0:                                         #close off comment, make replacement
        end_place = next_idx+len('-->')
        msg=msg[:start_place]+'<ravenTEMPcomment>'+msg[start_place+len('<!--'):end_place-len('-->')]+'</ravenTEMPcomment>'+msg[end_place:]
        next_idx += 2*len('<ravenTEMPcomment>')+1-len('<!--')-len('<--')
    #find next event, as above, reading from new index place
    startreadplace = next_idx + 2
    next_open = msg.find('<!--',startreadplace)
    next_close= msg.find('-->' ,startreadplace)
    if next_close==next_open and next_close==-1: next_idx=-1
    elif next_open<0: next_idx = next_close
    elif next_close<0: raise IOError('Mismatched comments!')
    else: next_idx = min(next_open,next_close)
  return msg
