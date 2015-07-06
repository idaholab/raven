import xml.dom.minidom as pxml
import xml.etree.ElementTree as ET
import os
import sys


def createBackup(filename):
  """
    Creates a backup file based on the file at filename.  If it exists, prints an error message and returns.
    @ In, filename, string (to be appended with '.bak')
    @Out, bool int, 0 on success or 1 on fail
  """
  bakname = filename+'.bak'
  if not os.path.isfile(bakname):
    bak = file(bakname,'w')
    for line in file(filename,'r'):
      bak.writelines(line)
    bak.close()
    return 0
  else:
    print 'ERROR! Backup file already exists:',bakname
    print '    If you wish to continue, remove the backup and rerun the script.'
    return 1


def prettify(tree):
  """
    Script for turning XML tree into something mostly RAVEN-preferred.  Does not align attributes as some devs like (yet).
    The output can be written directly to a file, as file('whatever.who','w').writelines(prettify(mytree))
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
  """
  #require a list of files to act on
  if len(argv)==0:
    raise IOError('No filenames listed to modify! Usage: python path/to/script.py infile1.xml infile2.xml infile3.xml')
  #track the failed attempts
  failures = 0
  #remove the script name itself from the list
  filelist = argv[1:]
  maxname = max(len(fname) for fname in filelist)
  #iterate over files
  for fname in filelist:
    if not os.path.isfile(fname):
      #file doesn't exist, but do continue on to others
      print 'ERROR!  File not found:',fname
      failures+=1
      continue
    if createBackup(fname)==0:
      print ('Converting '+fname+'...').ljust(14+maxname,'.'),
      tree = convert(ET.parse(fname))
      file(fname,'w').writelines(prettify(tree))
      print 'converted.'
    else:
      #backup was not successfully created
      failures+=1
  if failures>0: print '\n%i files converted, but there were %i failures.  See messages above.' %(len(filelist)-failures,failures)
  else: print '\nConversion script completed successfully.  %i files converted.' %len(filelist)
  return failures

