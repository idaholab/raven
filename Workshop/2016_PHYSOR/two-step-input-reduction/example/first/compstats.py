import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xml.etree.ElementTree as ET

cases = [1,2,5,9,16,25]
bcases = cases+[30]
cases_pct=[0.6,0.8,0.9,0.95,0.99,0.999]

mc_err = 1.5

def loadValues(infile):
  for line in infile:
    sp = line.split(',')
    if sp[0] == 'variance':
      variance = float(sp[1])
    elif sp[0].startswith('expected'):
      mean = float(sp[1])
  return mean,np.sqrt(variance)

def err(got,correct):
  return abs(got-correct)/correct

#load original
orig = loadValues(file('basic_stats_orig.csv','r'))
oerr = [[0.01*orig[0],0.01*orig[1]]]*4

#load pca reduction
pca = loadValues(file('basic_stats.csv','r'))
pcaerr = [[pca[0]*0.01,pca[1]*0.01]]*5

caseStats = []
errors = []
errors2 = []
bars = []
#load cases
for case in cases:
  mean,var = loadValues(file('basic_stats_2_'+str(case)+'.csv','r'))
  caseStats.append((mean,var))
  errors.append(( err(mean,orig[0]), err(var,orig[1]) ))
  errors2.append(( err(mean,pca[0]), err(var,pca[1]) ))
  ebars = [0.01*mc_err*mean, 0.01*mc_err*var]
  bars.append( ebars )

bcaseStats = []
berrors = []
berrors2 = []
bbars = []
#load pca-only cases
for case in bcases:
  mean,var = loadValues(file('../pca_mc_run/first/basic_stats_'+str(case)+'.csv','r'))
  bcaseStats.append((mean,var))
  berrors.append(( err(mean,orig[0]), err(var,orig[1]) ))
  berrors2.append(( err(mean,pca[0]), err(var,pca[1]) ))
  ebars = [0.01*mc_err*mean, 0.01*mc_err*var]
  #if ebars[0] < 0: ebars[0]=0
  #if ebars[1] < 0: ebars[1]=0
  bbars.append( ebars )


#plot
colOrig = 'k'
colR2 = 'g'
colR1 = '#0000ff' #blue
colPCA = 'r'
colAS = 'r'

colSC = '#00b000'##ff0000' #red
colMC = '#0000ff' #blue
col10 = 'm'#'#00ff00' #green

lw = 3

x = [0,30]
y = [-0.5,0.5]

er1=list(b[0] for b in bars)
er2=list(b[1] for b in bars)
ber1=list(b[0] for b in bbars)
ber2=list(b[1] for b in bbars)

title = 'Input Reduction Convergence'
#mean values
print 'plotting mean'
plt.figure()
plt.title(title+', Mean')
plt.xlabel('Terms Retained')
plt.ylabel('Value')
plt.errorbar([0,10,20,30],[orig[0]]*4,yerr=[o[0] for o in oerr],color=colOrig,linewidth=lw,label='original')
plt.errorbar([0,7,15,22,30],[pca[0]]*5,yerr=[o[0] for o in pcaerr],color=colR1,linewidth=lw,label='pca-reduced 50')
plt.errorbar(cases,list(e[0] for e in caseStats),marker='.',markersize=10,linewidth=lw,color=colR2,yerr=er1,label='twice-reduced')
plt.errorbar(bcases,list(e[0] for e in bcaseStats),marker='x',markersize=10,linewidth=lw,color=colPCA,yerr=ber1,label='pca-only')
plt.legend(loc=4)
plt.savefig('mean.pdf')

#variance values
print 'plotting variance'
plt.figure()
plt.title(title+', Std. Dev.')
plt.xlabel('Terms Retained')
plt.ylabel('Value')
plt.errorbar([0,10,20,30],[orig[1]]*4,yerr=[o[1] for o in oerr],color=colOrig,linewidth=lw,label='original')
plt.errorbar([0,7,15,22,30],[pca[1]]*5,yerr=[o[1] for o in pcaerr],color=colR1,linewidth=lw,label='pca-reduced 50')
plt.errorbar(cases,list(e[1] for e in caseStats),marker='.',markersize=10,color=colR2,linewidth=lw,yerr=er2,label='twice-reduced')
plt.errorbar(bcases,list(e[1] for e in bcaseStats),marker='x',markersize=10,color=colPCA,linewidth=lw,yerr=ber2,label='pca-only')
plt.legend(loc=4)
plt.savefig('var.pdf')

print 'plotting mean errors'
plt.figure()
plt.title(title+', Error in Mean')
plt.xlabel('Percent Variance Retained')
plt.ylabel('Relative Error')
plt.plot(x,[err(pca[0],orig[0])]*2,'-',color=colR1,linewidth=lw,label='pca-reduced 50')
plt.plot(cases,list(e[0] for e in errors),marker='.',color=colR2,markersize=10,linewidth=lw,label='twice-reduced')
plt.plot(bcases,list(e[0] for e in berrors),marker='x',color=colPCA,markersize=10,linewidth=lw,label='pca-only')
plt.yscale('log')
plt.legend(loc=1)
#plt.axis([0,30,1e-5,1e0])
plt.savefig('mean_err.pdf')

print 'plotting variance errors'
plt.figure()
plt.title(title+', Error in Std. Dev.')
plt.xlabel('Percent Variance Retained')
plt.ylabel('Relative Error')
plt.plot(x,[err(pca[1],orig[1])]*2,'-',color=colR1,linewidth=lw,label='pca-reduced 50')
plt.plot(cases,list(e[1] for e in errors),marker='.',color=colR2,linewidth=lw,markersize=10,label='twice-reduced')
plt.plot(bcases,list(e[1] for e in berrors),marker='x',color=colPCA,linewidth=lw,markersize=10,label='pca-only')
plt.yscale('log')
plt.legend(loc=3)
#plt.axis([0,30,1e-5,1e0])
plt.savefig('var_err.pdf')


#now do convergence of MC and SCgPC for the 9-variable latent case
# benchmark value is in orig, oerr

#collect MC values
mcfile = file('csv_db_2_9.csv','r')
means = []
vars = []
stdv = []
runs = []
t1 = 0
t2 = 0
errm = []
errd = []
errv = []
for l,line in enumerate(mcfile):
  if l==0: continue
  ans = float(line.split(',')[-1])
  t1 += ans
  t2 += ans*ans
  mean = t1/float(l)
  means.append(mean)
  second = t2/float(l)
  stdv.append(np.sqrt(second-mean*mean))
  vars.append(second - mean*mean)
  errm.append(1.0/np.sqrt(l)*mc_err*mean)
  errd.append(1.0/np.sqrt(l)*mc_err*stdv[-1])
  errv.append(max( -second/np.sqrt(l) + mean*mean*(2./np.sqrt(l) - 1./float(l)),second/np.sqrt(l) - mean*mean*(2./np.sqrt(l) - 1./float(l))))
  runs.append(l)

#collect SC
scruns=[]
scmeans=[]
scstdv=[]
scvars=[]
for case in ['td_1_dump.xml','td_2_dump.xml','td_3_dump.xml']:
  tree = ET.parse(file(case,'r'))
  root = tree.getroot()
  scruns.append(int(root.find('response').find('numRuns').text))
  scmeans.append(float(root.find('response').find('mean').text))
  scvars.append(float(root.find('response').find('variance').text))
  scstdv.append(np.sqrt(scvars[-1]))

#load adaptive sobol
ascases = [10,25,50,75,100,150,200,300,400,500,600,700,800,900,1000]
asmean=[]
asstdv=[]
asruns=[]
aserrm=[]
aserrd=[]
for case in ascases:
  asruns.append(int(ET.parse(file('adsob_'+str(case)+'.xml','r')).find('response').find('numRuns').text))
  asmean.append(float(ET.parse(file('adsob_'+str(case)+'.xml','r')).find('response').find('mean').text))
  asstdv.append(np.sqrt(float(ET.parse(file('adsob_'+str(case)+'.xml','r')).find('response').find('variance').text)))
  aserrm.append(err(asmean[-1],orig[0]))
  aserrd.append(err(asstdv[-1],orig[1]))
#print 'adaptsob:'
#for i in range(len(asruns)):
#  print asruns[i],asmean[i],asstdv[i]

ninth = cases.index(9)
mean9,sd9 = caseStats[ninth]
bars9 = bars[ninth]



title = 'Two Stage Reduction to 9 Variables'

tVar = ', 9v'
tMC = 'MC'+tVar
tSC = 'Sparse Grid'+tVar
tAS = 'Adaptive Sobol, 50v'
t10k = ' (10k)'

print 'plotting 9-variable Monte Carlo mean values'
plt.figure()
plt.title(title+', Mean, Zoomed')
plt.xlabel('Number of Runs')
plt.ylabel('Mean Value')
plt.errorbar(runs,means,yerr=errm,color=colMC,alpha=0.1)
plt.plot(runs,means,color=colMC,label=tMC)
plt.errorbar([0,200,1000,1200],[mean9]*4,yerr=[bars9[0]]*4,color=col10,linewidth=3,label=tMC+t10k)
plt.plot(scruns,scmeans,linewidth=3,marker='o',color=colSC,label=tSC)
plt.plot(asruns,asmean,marker='v',markersize=10,linewidth=lw,color=colAS,label=tAS)
plt.axis([0,600,0.08,0.1])
plt.legend()
plt.savefig('mc_vs_sc_mean.pdf')

plt.figure()
plt.title(title+', Mean')
plt.xlabel('Number of Runs')
plt.ylabel('Mean Value')
plt.errorbar(runs,means,yerr=errm,color=colMC,alpha=0.02)
plt.plot(runs,means,color=colMC,label=tMC)
plt.errorbar([0,3000,7000,10000],[mean9]*4,yerr=[bars9[0]]*4,color=col10,linewidth=3,label=tMC+t10k)
plt.plot(scruns,scmeans,linewidth=3,marker='o',color=colSC,label=tSC)
plt.plot(asruns,asmean,marker='v',markersize=10,linewidth=lw,color=colAS,label=tAS)
plt.axis([0,10000,0.08,0.1])
plt.legend()
plt.savefig('mc_vs_sc_mean_wide.pdf')

print 'plotting 9-variable Monte Carlo variance values'
plt.figure()
plt.title(title+', Std. Dev., Zoomed')
plt.xlabel('Number of Runs')
plt.ylabel('Variance Value')
plt.errorbar(runs,stdv,yerr=errd,alpha=0.1,color=colMC)
plt.errorbar(runs,stdv,label=tMC,color=colMC)
plt.errorbar([0,200,1000,1200],[sd9]*4,yerr=[bars9[1]]*4,linewidth=3,label=tMC+t10k,color=col10)
plt.plot(scruns,scstdv,linewidth=3,marker='o',label=tSC,color=colSC)
plt.axis([0,1200,0.0046,0.0054])
plt.legend(loc=4)
plt.savefig('mc_vs_sc_var.pdf')

plt.figure()
plt.title(title+', Std. Dev.')
plt.xlabel('Number of Runs')
plt.ylabel('Variance Value')
plt.errorbar(runs,stdv,yerr=errd,alpha=0.02,color=colMC)
plt.errorbar(runs,stdv,label=tMC,color=colMC)
plt.errorbar([0,3000,6000,10000],[sd9]*4,yerr=[bars9[1]]*4,linewidth=3,label=tMC+t10k,color=col10)
plt.plot(scruns,scstdv,linewidth=3,marker='o',label=tSC,color=colSC)
plt.axis([0,10000,0.0046,0.0054])
plt.legend(loc=4)
plt.savefig('mc_vs_sc_var_wide.pdf')

print 'plotting Adaptive Sobol versus sobol method'
plt.figure()
plt.title(title+', Std. Dev., Adaptive Sobol')
plt.xlabel('Number of Runs')
plt.ylabel('Variance Value')
plt.errorbar(runs,stdv,label=tMC,color=colMC)
plt.errorbar([0,400,800,1200],[orig[1]]*4,yerr=[o[1] for o in oerr],linewidth=3,color='k',label='MC, 308v'+t10k)
plt.errorbar([0,300,600,900,1200],[pca[1]]*5,yerr=[o[1] for o in pcaerr],linewidth=3,label='MC, 50v'+t10k)
plt.plot(scruns,scstdv,linewidth=3,marker='o',label=tSC,color=colSC)
plt.plot(asruns,asstdv,marker='v',markersize=10,linewidth=lw,color=colAS,label=tAS)
plt.axis([0,1200,0.0046,0.0054])
plt.legend(loc=4)
plt.savefig('mc_vs_sc_var_adsob.pdf')


plt.show() #
