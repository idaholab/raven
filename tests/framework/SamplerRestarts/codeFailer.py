import sys

infile = sys.argv[1]

for line in file(infile,'r'):
  if line.startswith('x ='): x=float(line.split('=')[1])
  if line.startswith('y ='): y=float(line.split('=')[1])
  if line.startswith('out ='): out=line.split('=')[1].strip()

# generate fails roughly half the time.
#if x+y>0: raise RuntimeError('Answer is bigger than 0.  Just a test error.')

outfile = file(out+'.csv','w')
outfile.writelines('x,y,ans\n')
outfile.writelines(','.join([str(x),str(y),str(x+y)]))
outfile.close()
