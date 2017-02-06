import xml.etree.ElementTree as ET
#load XML navigation tools
import sys, os, time
sys.path.append(os.path.join(os.getcwd(),'..','..','framework','utils'))
import xmlUtils

def getNode(fname,nodepath):
  #TODO add option to include parent nodes with ellipses
  root = ET.parse(fname).getroot()
  #format nodepath
  nodepath = nodepath.replace('.','|')
  #check if root is desired node
  if root.tag == nodepath:
    print '\nfound in root\n'
    node = root
    docLevel = 0
  else:
    docLevel = len(nodepath.split('|'))
    print '\nfound in nodes\n'
    node = xmlUtils.findPathEllipsesParents(root,nodepath,docLevel)
    if node is None:
      raise IOError('Unable to find '+nodepath+' in '+fname)
  return xmlUtils.prettify(node,doc=True,docLevel=docLevel)


def chooseSize(string):
  longest = 0
  for line in string.split('\n'):
    longest = max(longest,len(line.rstrip()))
  if longest < 64:
    return '\\normalsize'
  elif longest < 70:
    return '\\small'
  elif longest < 77:
    return '\\footnotesize'
  elif longest < 96:
    return '\\scriptsize'
  else:
    return '\\tiny'


if __name__=='__main__':
  #if len(sys.argv)<5 or '-f' not in sys.argv or '-n' not in sys.argv:
  try:
    fname = sys.argv[sys.argv.index('-f')+1]
    nname = sys.argv[sys.argv.index('-n')+1]
    if '-h' in sys.argv:
      highlight = sys.argv[sys.argv.index('-h')+1]
    else:
      highlight = 'asdfghhjkl'
  except IndexError:
    print 'Error calling writeTex.py!\n  Syntax is: python writeTex.py -f /path/to/file.xml -n name_of_node -h additional,keywords'
    raise IOError()
  #re-parse filename and path to be system dependent
  fname = os.path.join(*fname.split('/'))
  strNode = getNode(fname,nname)
  outFile = file('raven_temp_tex_xml.tex','w')
  printName = os.path.join('raven',fname.replace('../','')).replace('_','\_')
  toWrite = '\\FloatBarrier\n'+\
            chooseSize(printName)+'\n\\texttt{'+printName+'}\n'+\
            chooseSize(strNode)+'\n'+\
            '\\begin{lstlisting}[style=XML,morekeywords={'+highlight+'}]\n'+\
            strNode+'\n'+\
            '\\end{lstlisting}\n'+\
            '\\normalsize'
  outFile.writelines(toWrite)
  outFile.close()

