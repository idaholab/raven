"""
  MELCOR FORMAT SUPPORT MODULE
  Created on March 28, 2017
  Last update on October 14, 2022
  @authors:
           Matteo D'Onorio (University of Rome La Sapienza)
           Paolo Balestra  (University of Rome La Sapienza)
"""
# ===============================================================================
def MCRBin(fileDir, VarSrch):
  """
    This method is called to collect the variables to be used in the postprocess
    @ In, fileDir, string, the file directory. This is the directory of the MELCOR plot file
    @ In, VarSrch, list, list of variables to be collected
    @ Out, Data, tuple (numpy.ndarray,numpy.ndarray,numpy.ndarray), this contains the extracted data for each declared variable
                                                                    from MELCOR output file. The first variables indicates the
                                                                    time arrays, the second one indicates the arryas of the melcor
                                                                    output variables
  """
  from struct import unpack
  import numpy as np
  from collections import Counter
  HdrList = []
  BlkLenBef = []
  BlkLenAft = []
  DataPos=[]
  cntr = 0
  Var_dict = {}
  with open(fileDir, 'rb') as bf:
    while True:
      BlkLenBefSlave = bf.read(4)
      if not BlkLenBefSlave:
        break
      BlkLenBef.append(unpack('I', BlkLenBefSlave)[0])
      if BlkLenBef[cntr] == 4:
        HdrList.append(str(unpack('4s', bf.read(4))[0], 'utf-8'))
      elif HdrList[cntr - 1] == 'TITL':
        probemTitle=str(unpack('%d' % BlkLenBef[cntr] + 's', bf.read(BlkLenBef[cntr]))[0], 'utf-8')
        HdrList.append([])
      elif HdrList[cntr - 1] == 'KEY ':
        VarName=unpack('2I', bf.read(8))
        HdrList.append([])
      elif HdrList[cntr - 2] == 'KEY ':
        a = BlkLenBef[-1]/VarName[0]
        stringa=str(int(a))+"s"
        VarNam=[str(i, 'utf-8') for i in unpack(stringa * VarName[0], bf.read(BlkLenBef[cntr]))]
        HdrList.append([])
      elif HdrList[cntr - 3] == 'KEY ':
        VarPos=unpack('%d' % VarName[0] + 'I', bf.read(BlkLenBef[cntr]))
        HdrList.append([])
      elif HdrList[cntr - 4] == 'KEY ':
        VarUdm=[str(i, 'utf-8') for i in unpack('16s' * VarName[0], bf.read(BlkLenBef[cntr]))]
        HdrList.append([])
      elif HdrList[cntr - 5] == 'KEY ':
        VarNum=unpack('%d' % VarName[1] + 'I', bf.read(BlkLenBef[cntr]))
        VarNameFull=[]
        VarUdmFull=[]
        NamCntr=0
        VarPos = VarPos + (VarName[1]+1,)
        VarSrchPos=[0]
        itm_x_Var = []
        for k in range(0,len(VarNam)):
          itm_x_Var.append(VarPos[k+1]-VarPos[k])
        if len(itm_x_Var) != len(VarNam):
          print("MelcorTools: Number of variables different from number of items of offset array")
          print(itm_x_Var)
          print(len(VarNam))
          break
        Items_Tot = sum(itm_x_Var)
        if Items_Tot != len(VarNum):
          print("MelcorTools: The Sum of the items to be associated with each variable is different from the sum of all items id VarNum")
        VarNum_Cntr =0
        Var_dict = {}
        for i,Var in enumerate(VarNam):
          NumOfItems = itm_x_Var[i]
          end = VarNum_Cntr + NumOfItems
          Var_dict[Var] = list(VarNum[VarNum_Cntr:end])
          VarNum_Cntr = VarNum_Cntr+NumOfItems
        for key in Var_dict.keys():
          for element in Var_dict[key]:
            if element == 0:
              VarNameFull.append(str(key).strip())
            else:
              VarNameFull.append(key.strip()+'_%d' %element)
        for i,item in enumerate(itm_x_Var):
          for k in range(0,item):
            VarUdmFull.append(VarUdm[i].strip())
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
