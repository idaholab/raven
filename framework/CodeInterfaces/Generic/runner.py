import numpy as np
import sys

def run(inputFileName,outputFileName=None):
  infile = file(inputFileName,'rb')
  for line in infile:
    if line.startswith('xval'): x = float(line.split('=')[1].strip())
    elif line.startswith('yval'): y = float(line.split('=')[1].strip())
    elif line.startswith('zval'): z = float(line.split('=')[1].strip())
  infile.close()
  if outputFileName==None: outputFileName = 'out'
  outfile = file(outputFileName+'.csv','w')
  outfile.writelines('t,x,y,z,dist\n')
  msg=''
  for ti in range(10):
    t = ti*0.01
    x+= 0.01
    d = np.sqrt(x**x + y**y + z**z)
    for n in [t,x,y,z,d]:
      msg+=(str(n)+',')
    msg=msg[:-1]+'\n'
  outfile.writelines(msg)
  outfile.close()



if __name__=='__main__':
  inpPlace = sys.argv[sys.argv.index('-i')+1]
  outPlace = sys.argv[sys.argv.index('-o')+1] if '-o' in sys.argv else None
  run(inpPlace,outPlace)
