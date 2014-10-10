import re
#  author  : nieljw   
#  modified: alfoa

class relapdata:
  '''   class that parses output of relap5 output file and reads in trip, minor block and write a csv file ''' 
  def __init__(self,filen):
    self.lines=open(filen,"r").readlines()
    self.trips=self.returntrip(self.lines)
    self.minordata=self.getminor(self.lines)
    self.EndTime=self.gettime(self.lines)
    self.readraven()

  def gettime(self,lines):
    for i in lines:
      if re.match('^\s*Final time=',i): 
        time=i.split()[2]
        return time

  def returntrip(self,lines):
    triparray=[]
    for i in range(len(lines)):
      if re.match('^\s*0Trip\s*number',lines[i]):
        triparray=[]
        i=i+1
        while not (re.match('^0System|^0\s*Total',lines[i])):
          temp1 = lines[i].split();
          for j in range(len(temp1)/2):
            if (float(temp1[2*j+1])>-1.000):
              triparray.append({temp1[2*j]:temp1[2*j+1]});                     
          i=i+1;         
    return triparray;   

  def readminorblock(self,lines,i):
    '''   reads in a block of minor edit data and returns a dictionary of lists  '''
    minordict={}
    edit_keys=[]
    flagg1 = 0
    flagg2 = 0
    block_count=0
    while(flagg1==0 & flagg2==0):
      if flagg1==0:
        tempkeys=[]
        temp1 = re.split('\s{2,}|\n',lines[i])
        temp2 = re.split('\s{2,}|\n',lines[i+1])
        temp1.pop()
        temp2.pop()
        temp2.pop(0)
        temparray=[]
        for j in range(len(temp1)):
          tempkeys.append(temp1[j]+'_'+temp2[j])
          edit_keys.append(temp1[j]+'_'+temp2[j])
          temparray.append([]);     #   allocates array for data block
        i=i+4
        while not re.match('^\s*1 time|^1RELAP5|^\s*\n|^\s*1RELAP5|^\s*MINOR EDIT',lines[i]):
          tempdata=lines[i].split()
          for k in range(len(temparray)): temparray[k].append(tempdata[k])
          i=i+1
          if re.match('^\s*1 time|^\s*1\s*R5|^\s*\n|^1RELAP5',lines[i]): break 
        for l in range(len(tempkeys)): minordict.update({tempkeys[l]:temparray[l]})
        if re.match('^\s*1\s*R5|^\s*\n|^\s*1RELAP5|^\s*MINOR EDIT',lines[i]):
          flagg2=1
          flagg1=1
        elif re.match('^\s*1 time',lines[i]):
          block_count=block_count+1
          flagg=1
    return minordict

  def getminor(self,lines):
    '''    looks for key word MINOR EDIT for reading minor edit block
     and calls readminor block to read in the block of minor edit data '''
    count  = 0
    for i in range(len(lines)):
      if re.match('^MINOR EDIT',lines[i]):
        j=i+1
        count=count+1
        tempdict=self.readminorblock(self.lines,j)
        if (count==1): minordict=tempdict;
        else:
          for k in minordict.keys():
            for l in tempdict.get(k):
              minordict[k].append(l)
    return minordict 

  def readraven(self):
    flagg=0
    self.ravendata={}
    for i in range(len(self.lines)):
      if re.search('RAVEN',self.lines[i]):
        i=i+1
        while flagg==0: 
          if re.search('RAVEN',self.lines[i]): flagg=1
          else: self.ravendata[self.lines[i].split()[1].replace("*","")]=self.lines[i].split()[3]
          i=i+1
    return

  def write_csv(self,filen):
    '''   writes the csv file from minor edit data '''
    IOcsvfile=open(filen,'w')
    for i in range(len(self.minordata.keys())): IOcsvfile.write('%s,' %(self.minordata.keys()[i]))
    for j in range(len(self.ravendata.keys())):
      IOcsvfile.write('%s' %(self.ravendata.keys()[j]))
      if j+1<len(self.ravendata.keys()): IOcsvfile.write(',')
    IOcsvfile.write('\n')
    for i in range(len(self.minordata.get(self.minordata.keys()[0]))):
      for j in range(len(self.minordata.keys())): IOcsvfile.write('%s,' %(self.minordata.get(self.minordata.keys()[j])[i]))
      for k in range(len(self.ravendata.keys())):
        IOcsvfile.write('%s' %(self.ravendata[self.ravendata.keys()[k]]))
        if k+1<len(self.ravendata.keys()): IOcsvfile.write(',')
      IOcsvfile.write('\n')

