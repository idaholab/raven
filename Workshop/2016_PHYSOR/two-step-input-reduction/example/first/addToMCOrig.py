import xml.etree.ElementTree as ET

tree = ET.parse(file('samplerMonteCarlo_latent.xml','r'))
root = tree.getroot()
for child in root[:]:
  print 'tag:',child.tag
  if child.tag in ['variable','variablesTransformation']:
    root.remove(child)
for v in range(308):
  varnode = ET.Element('variable')
  varnode.attrib['name'] = 'x_'+str(v)
  varnode.text='\n    '
  disnode = ET.Element('distribution')
  disnode.attrib['dim'] = str(v+1)
  disnode.text='MultivariateNormalReduction'
  varnode.append(disnode)
  root.append(varnode)

s = ET.tostring(root)
s = s.replace('</variable>','</variable>\n  ')
s = s.replace('</distribution>','</distribution>\n  ')
file('samplerMonteCarlo_latent_orig.xml','w').writelines(s)

