import numpy as np

def R(m,r,C,rho,v,ang,g,sy=0,dt=0.0001,tol=1e-12,verbose=False,retpts=True):
#def R(vrs,dt=0.001,tol=1e-12,verbose=False,retpts=True):
  D = rho*C*0.5*(np.pi*r*r)
  if C==0:D=0
  ang*=np.pi/180.
  t = 0
  sx = 0.
  #sy taken as input parameter
  vx = v*np.cos(ang)
  vy = v*np.cos(ang)
  nores = vx/g*(vy+np.sqrt(vy*vy+2*g*sy))
  if verbose: print 'Expected no-resistance value:',nores
  converged = False
  pos=[]
  while not converged:
    old,new = takeAStep(t,dt,vx,vy,sx,sy,D,m,g,verbose=verbose)
    if new[2]>0:
      if retpts: pos.append(old)
      t=new[0]
      sx=new[1]
      sy=new[2]
      vx=new[3]
      vy=new[4]
    else:
      if dt>=tol:
        dt*=0.01
      else:
        converged=True
        if retpts:
          pos.append(old)
          pos.append(np.array(new))
  if retpts: pos=np.array(pos)
  #if verbose:
  #  print 'END STATS'
  #  print '  Range:',pos[-1][1]
  #  print '  Max height:',np.max(pos[:,2])
  #  print '  Time of Flight:',pos[-1][0]
  if verbose: print ''
  if retpts:
    return pos,new[1]
  return new[1]

def takeAStep(t,dt,vx,vy,sx,sy,D,m,g,verbose=True):
  v = np.sqrt(vx*vx + vy*vy)
  ax = -D/m*v*vx
  ay = -g-D/m*v*vy
  old = [t,sx,sy,vx,vy]
  vx += ax*dt
  vy += ay*dt
  sx += vx*dt + 0.5*ax*dt*dt
  sy += vy*dt + 0.5*ay*dt*dt
  if verbose:
    print '  At %1.2e, %1.2e, t,dt=%1.2e, %1.2e' %(sx,sy,t,dt),'\r',
  t += dt
  new = [t,sx,sy,vx,vy]
  return old,new

def run(self,Input): #for Raven
  self.ans = R(self.m,self.r,self.c,self.d,self.v,self.a,self.g,sy=self.y,retpts=False)

def parseInputFile(infileName,outfileName='screen'):
  infile = file(infileName,'r')
  m=r=C=rho=v=ang=g=y=dt=tol=None
  for line in infile:
    if   line.startswith('m ='  ): m   = float(line.split('=')[1].strip())
    elif line.startswith('r ='  ): r   = float(line.split('=')[1].strip())
    elif line.startswith('C ='  ): C   = float(line.split('=')[1].strip())
    elif line.startswith('rho ='): rho = float(line.split('=')[1].strip())
    elif line.startswith('v ='  ): v   = float(line.split('=')[1].strip())
    elif line.startswith('ang ='): ang = float(line.split('=')[1].strip())
    elif line.startswith('g ='  ): g   = float(line.split('=')[1].strip())
    elif line.startswith('y ='  ): y   = float(line.split('=')[1].strip())
    elif line.startswith('dt =' ): dt  = float(line.split('=')[1].strip())
    elif line.startswith('tol ='): tol = float(line.split('=')[1].strip())
  infile.close()
  if dt==None:dt=1e-4
  if tol==None:tol=1e-10
  if None in [m,r,C,rho,v,ang,g,y]:
    raise IOError('Some values were not specified in input!')
  hist,sol = R(m,r,C,rho,v,ang,g,sy=y,dt=dt,tol=tol,retpts=True)
  if outfileName=='screen':print('Range:', sol)
  else:
    outfile = file(outfileName,'w')
    outfile.writelines('t,sx,sy,vx,vy\n')
    for hs in hist:
      msg=''
      for h in hs:
        msg+=str(h)+','
      msg=msg[:-1]+'\n'
      outfile.writelines(msg)
    outfile.close()

if __name__=="__main__":
  #res = R(0.145,0.0336,0.5,1.2,50.,45.,9.81,sy=1,verbose=True,retpts=False)
  #print 'Range:',res
  import sys
  inp = sys.argv[sys.argv.index('-i')+1]
  out = sys.argv[sys.argv.index('-o')+1] if '-o' in sys.argv else 'screen'
  parseInputFile(inp,out)
