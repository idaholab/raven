import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

#first order Sobol using first-order polys
tree = ET.parse(file('sobol_dump.xml','r'))
root = tree.getroot()
indchild = root.find('response').find('indices')

contribs = []
n=0
for child in indchild:
  if child.tag != 'variables': continue
  n+=1
  dim = int(child.text.strip().split('_')[1])
  sob = float(child.find('Sobol_index').text)
  contribs.append( [n,dim,sob] )

#second order Sobol using second-order polys
tree = ET.parse(file('sobol_dump2.xml','r'))
root = tree.getroot()
indchild = root.find('response').find('indices')

n=0
for child in indchild:
  if child.tag != 'variables': continue
  if ',' in child.text: continue
  n+=1
  dim = int(child.text.strip().split('_')[1])
  sob = float(child.find('Sobol_index').text)
  contribs[n-1].append(dim)
  contribs[n-1].append(sob)

outFile = file('sobol_rank.csv','w')
outFile.writelines('n, dim, sobol, total\n')
tot = [0,0]
print 'n, dim1, sobol1, total1'
for n,dim1,sob1,dim2,sob2 in contribs:
  tot[0] += sob1
  tot[1] += sob2
  print n,dim1,sob1,tot[0],'|',dim2,sob2,tot[1]
  outFile.writelines('%2i,%2i,%0.8f,%0.8f | %2i,%0.8f,%0.8f\n' %(n,dim1,sob1,tot[0],dim2,sob2,tot[1]))

outFile.close()

