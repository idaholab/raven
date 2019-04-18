# Copyright 2017 University of Rome La Sapienza
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
  MELCOR FORMAT SUPPORT MODULE
  Created on March 28, 2017
  @authors: 
           Paolo Balestra (University of Rome La Sapienza)
           Matteo D'Onorio (University of Rome La Sapienza)
"""

def MCRBin(fileDir, VarSrch):
  """
    This method is called to collect the variables to be used in the postprocess
    @ In, fileDirectory, string, the file directory. This is the directory of the MELCOR plot file
    @ In, variableSearch, list, list of variables to be collected
    @ Out, Data, tuple (numpy.ndarray,numpy.ndarray,numpy.ndarray), this contains the extracted data for each declare variable
  """
  from struct import unpack
  import numpy as np
  
  HdrList = []
  BlkLenBef = []
  BlkLenAft = []
  DataPos=[] 
  cntr = 0
  
  with open(fileDir, 'rb') as bf:
    while True:
      BlkLenBefSlave = bf.read(4)
      if not BlkLenBefSlave:
        break
      BlkLenBef.append(unpack('I', BlkLenBefSlave)[0])
      if BlkLenBef[cntr] == 4:
        HdrList.append(str(unpack('4s', bf.read(4))[0]))
      elif HdrList[cntr - 1] == 'TITL':
        probemTitle=str(unpack('%d' % BlkLenBef[cntr] + 's', bf.read(BlkLenBef[cntr]))[0])
        HdrList.append([])
      elif HdrList[cntr - 1] == 'KEY ':
        VarName=unpack('2I', bf.read(8))
        HdrList.append([])
      elif HdrList[cntr - 2] == 'KEY ':
        VarNam=[str(i) for i in unpack('24s' * VarName[0], bf.read(BlkLenBef[cntr]))]
        HdrList.append([])
      elif HdrList[cntr - 3] == 'KEY ':
        VarPos=unpack('%d' % VarName[0] + 'I', bf.read(BlkLenBef[cntr]))
        HdrList.append([])
      elif HdrList[cntr - 4] == 'KEY ':
        VarUdm=[str(i) for i in unpack('16s' * VarName[0], bf.read(BlkLenBef[cntr]))]
        HdrList.append([])
      elif HdrList[cntr - 5] == 'KEY ':
        VarNum=unpack('%d' % VarName[1] + 'I', bf.read(BlkLenBef[cntr]))
        VarNameFull=[]
        VarUdmFull=[]
        NamCntr=0
        VarPos = VarPos + (VarName[1]+1,)
        VarSrchPos=[0]
        for i,Num in enumerate(VarNum):
          if Num ==0:
            VarNameFull.append(VarNam[NamCntr].strip())
            VarUdmFull.append(VarUdm[NamCntr].strip())
            NamCntr+=1
          else:
            if i+1 < VarPos[NamCntr+1]:
              VarNameFull.append(VarNam[NamCntr].strip()+'_%d' %Num)
              VarUdmFull.append(VarUdm[NamCntr].strip())
            else:
              NamCntr+=1
              VarNameFull.append(VarNam[NamCntr].strip()+'_%d' %Num)
              VarUdmFull.append(VarUdm[NamCntr].strip())
        VarNameFull=['TIME','CPU','DT','UNKN03']+VarNameFull
        VarUdmFull=['sec','','','']+VarUdmFull
        for Nam in VarSrch:
          VarSrchPos.append(VarNameFull.index(Nam.strip()))
        VarUdmFull=[VarUdmFull[i] for i in VarSrchPos]
        SwapPosVarSrch=sorted(range(len(VarSrchPos)), key=lambda k: VarSrchPos[k])
        SwapPosVarSrch=sorted(range(len(SwapPosVarSrch)), key=lambda k: SwapPosVarSrch[k])
        VarSrchPos.sort()
        VarSrchPos.append(VarName[1]+4)
        HdrList.append([])
      elif HdrList[cntr - 1] == '.TR/':
        DataPos.append(bf.tell())
        bf.seek(BlkLenBef[cntr], 1)
        HdrList.append([])
      else:
        HdrList.append([])
      BlkLenAft.append(unpack('I', bf.read(4))[0])
      cntr +=1
  
  data=np.empty([len(DataPos), len(VarSrch)+1])*np.nan
  with open(fileDir, 'rb') as bf:
    for i,Pos in enumerate(DataPos):
      bf.seek(Pos, 0)
      for j in range(len(VarSrchPos)-1):
        data[i,j]=unpack('f', bf.read(4))[0]
        bf.seek((VarSrchPos[j+1]-VarSrchPos[j])*4-4, 1)
  data=data[:,SwapPosVarSrch]
  return data[:,0],data[:,1:],VarUdmFull[1:]
