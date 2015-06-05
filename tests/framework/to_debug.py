import os
from os.path import isfile,join

allfiles=[f for f in os.listdir(os.getcwd()) if isfile(join(os.getcwd(),f)) ]
files=[]
for f in allfiles:
  if f.endswith('xml'):
    files.append(f)

casesT=['debug="True"',
       'debug="true"',
       "debug='True'",
       "debug='true'",
       'debug = "True"',
       'debug = "true"',
       "debug = 'True'",
       "debug = 'true'"
       ]
casesF=['debug="False"',
       'debug="false"',
       "debug='False'",
       "debug='false'",
       'debug = "False"',
       'debug = "false"',
       "debug = 'False'",
       "debug = 'false'"]

for fileName in files:
  new=file(fileName+'_mod','w')
  old=file(fileName,'r')
  for line in old:
    if 'debug' in line and 'verbosity' not in line:
      for case in casesT:
        if case in line:
          print fileName
          a=line.split(case)
          newline=a[0]+"verbosity='debug'"+a[1]
      for case in casesF:
        if case in line:
          a=line.split(case)
          newline=''.join(a)
    else: newline=line
    new.writelines(newline)
  new.close()
  os.system('mv '+fileName+'_mod '+fileName)

