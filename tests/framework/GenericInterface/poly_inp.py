import numpy as np

def eval(x,y):
  dat=[]
  c = 0
  for i in [0.3,0.5,0.7,1.0]:
    for j in [1.3,1.5,1.7,2.0]:
      c+=1
      dat.append([c,i,j,x,y,(i-x)*(j-y)])
  return dat

def run(xin,yin,out):
  inx = file(xin,'r')
  iny = file(yin,'r')
  for line in inx:
    if line.startswith('x ='):
      x=float(line.split('=')[1])
      break
  for line in iny:
    if line.startswith('y ='):
      y=float(line.split('=')[1])
      break

  dat = eval(x,y)

  outf = file(out+'.csv','w')
  outf.writelines('step,i,j,x,y,poly\n')
  for e in dat:
    outf.writelines(','.join(str(i) for i in e)+'\n')
  outf.close()

if __name__=='__main__':
  import sys
  args = sys.argv
  inp1 = args[args.index('-i')+1] if '-i' in args else None
  inp2 = args[args.index('-a')+1] if '-a' in args else None
  out  = args[args.index('-o')+1] if '-o' in args else None
  run(inp1,inp2,out)
