import xml.etree.ElementTree as ET

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

def newNode(tag,text='',attrib={}):
  """
    Creates a new node with the text and attributes provided.
    @ In, tag, string, tag of node
    @ In, text, optional string, text of node
    @ In, attrib, optional dict, attributes for node
    @ Out, ET.Element object
  """
  el = ET.Element(tag,attrib=attrib)
  el.text = text
  return el

def findPathTags(root,path):
  """
    Finds the Element object along a given path within a tree, or returns None.
    @ In, root, the tree root node to search from
    @ In, path, string, |-seperated list of tags to search down
    @ Out, ET.Element or None
  """
  path = path.split('|')
  if len(path)>1:
    oneup = findPath(root,'|'.join(path[:-1]))
    return oneup.find(path[-1])
  else:
    return root.find(path[-1])

def findPathAttrib(root,path):
  """
    Finds the Element object matching the path by tag and attributes
    @ In, root, the tree root node to search from
    @ In, path, array(tuple(string,dict)), list of (tag, attribs) to search
    @ Out, ET.Element or None
  """
  findTag,findAttrib = path[0]
  possibles = root.findall(findTag)
  if possibles is None: return None
  for p in possibles:
    found = True
    for attr,val in findAttrib.items():
      if p.get(attr) != val:
        found = False
        break
    if found:
      if len(path)>1:
        keepLooking = findPathAttrib(p,path[1:])
        if keepLooking is not None:
          return keepLooking
          break
        else: return None
      else: return p

def findPathText(root,path):
  """
    Finds the Element object matching the path by tag and text
    @ In, root, the tree root node to search from
    @ In, path, array(tuple(string,string)), list of (tag, text) to search
    @ Out, ET.Element or None
  """
  findTag,findText = path[0]
  possibles = root.findall(findTag)
  if possibles is None: return None
  for p in possibles:
    found = True
    if p.text.strip() != findText.strip():
      found = False
    if found:
      if len(path)>1:
        keepLooking = findPathText(p,path[1:])
        if keepLooking is not None:
          return keepLooking
          break
        else: return None
      else: return p

def findPathMixed(root,path):
  """
    Finds the Element object matching the path by tag and either attrib, text, or both
    @ In, root, the tree root node to search from
    @ In, path, array(tuple(string,string,tuple(string,string|dict))), list of (tag, 'tag'|'attrib'|'text', None|attrib dict|text)
    @ Out, ET.Element or None
  """
  findTag,findType,findObj = path[0]
  possibles = root.findall(findTag)
  if possibles is None: return None
  for p in possibles:
    found = True
    if findType == 'text':
      if p.text.strip() != findObj.strip():
        found = False
    elif findType == 'attrib':
      for attr,val in findObj.items():
        if p.get(attr) != val:
          found = False
          break
    elif findType == 'tag':
      pass
    else: raise(IOerror,'findType can only be "text" or "attrib"!  Got "'+str(findType)+'" instead...')
    if found:
      if len(path)>1:
        keepLooking = findPathMixed(p,path[1:])
        if keepLooking is not None:
          return keepLooking
          break
        else: return None
      else: return p

if __name__=='__main__':
  src = ET.parse(file('test.xml','r'))
  root = src.getroot()
  #el = findPathAttrib(root,[('second',{'a1':'three','a2':'four'}),('third',{'b1':'beta'})])
  #print el.tag,el.text,el.attrib
  #el = findPathText(root,[('second','s2'),('third','t2')])
  #print el.tag,el.text,el.attrib
  el = findPathMixed(root,[('right','tag',None),('second','text','s2'),('third','attrib',{'b1':'epsilon'})])
  print el.tag,el.text,el.attrib
