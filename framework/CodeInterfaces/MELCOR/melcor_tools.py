# Copyright 2017 University of Rome La Sapienza and Battelle Energy Alliance, LLC
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
''' MELOCR FORMAT SUPPORT MOULE
  Created on July 4, 2018
  @author: Paolo Balestra (University of Rome La Sapienza)
           Matteo D'Onorio (University of Rome La Sapienza)
'''

#===============================================================================
# MELCOR bin file Reader
#===============================================================================

def MCR_bin(file_dir, Var_srch):
    from struct import unpack
    import numpy as np

    Hdr_list = [] # List of the header blocks
    blk_len_b = [] # List of the blocks length (before the block)
    blk_len_e = [] # List of the blocks length (after the block)

    data_Pos=[] # Byte position of the time step Blocks
    cntr = 0

    dbg=False # Debug flag

    with open(file_dir, 'rb') as bf:
        while True:
            #===================================================================
            # Beginning block length (Byte)
            #===================================================================
            blk_len_b_slave = bf.read(4) # Slave variable to check eof
            #------------------- End of file checking at the iteration beginning
            if not blk_len_b_slave:
                break # Stop iteration exit while
            blk_len_b.append(unpack('I', blk_len_b_slave)[0])
            #---------------------------------------------------- Debug printing
            if dbg: print("BLOCK %02d ----------------------------------------------" % cntr)
            if dbg: print(blk_len_b[cntr])

            if blk_len_b[cntr] == 4:
                #===============================================================
                # Header Block, 4 char that describe the following block content
                #===============================================================
                Hdr_list.append(str(unpack('4s', bf.read(4))[0]))
                #---------- Empty element to maintain the correct list numbering
                if dbg: print(Hdr_list[cntr])

            elif Hdr_list[cntr - 1] == 'TITL':
                #===============================================================
                # Problem Title`
                #===============================================================
                prob_title=str(unpack('%d' % blk_len_b[cntr] + 's', bf.read(blk_len_b[cntr]))[0])
                #-------------- Empty element to maintain the correct list numbering
                Hdr_list.append([])
                #------------------------------------------------ Debug printing
                if dbg: print(prob_title)

            elif Hdr_list[cntr - 1] == 'KEY ':
                #===============================================================
                # Number of names and variables
                #===============================================================
                Var_N=unpack('2I', bf.read(8))
                #---------- Empty element to maintain the correct list numbering
                Hdr_list.append([])

            elif Hdr_list[cntr - 2] == 'KEY ':
                #===============================================================
                # Variables names #Var_N[0]x24 char
                #===============================================================
                Var_Nam=[str(i) for i in unpack('24s' * Var_N[0], bf.read(blk_len_b[cntr]))]
                #---------- Empty element to maintain the correct list numbering
                Hdr_list.append([])

            elif Hdr_list[cntr - 3] == 'KEY ':
                #===============================================================
                # Variables position #Var_N[0]x24 char
                #===============================================================
                Var_Pos=unpack('%d' % Var_N[0] + 'I', bf.read(blk_len_b[cntr]))
                #---------- Empty element to maintain the correct list numbering
                Hdr_list.append([])

            elif Hdr_list[cntr - 4] == 'KEY ':
                #===============================================================
                # Variables udm #Var_N[0]x16 char
                #===============================================================
                Var_Udm=[str(i) for i in unpack('16s' * Var_N[0], bf.read(blk_len_b[cntr]))]
                #---------- Empty element to maintain the correct list numbering
                Hdr_list.append([])

            elif Hdr_list[cntr - 5] == 'KEY ':
                #===============================================================
                # Variables number #Var_N[1]x1 integer
                #===============================================================
                Var_Num=unpack('%d' % Var_N[1] + 'I', bf.read(blk_len_b[cntr]))
                #------------------------------------ Full list of the variables
                Var_Nam_full=[]
                Var_Udm_full=[]
                Nam_cntr=0
                #--------- Put at the end the final position to iterate with n+1
                Var_Pos = Var_Pos + (Var_N[1]+1,)
                Var_srch_pos=[0]
                #-------------------------------- Iterate to generate full lists
                for i,Num in enumerate(Var_Num):
                    if Num ==0:
                        Var_Nam_full.append(Var_Nam[Nam_cntr].strip())
                        Var_Udm_full.append(Var_Udm[Nam_cntr].strip())
                        Nam_cntr+=1
                    else:
                        if i+1 < Var_Pos[Nam_cntr+1]:
                            Var_Nam_full.append(Var_Nam[Nam_cntr].strip()+'_%d' %Num)
                            Var_Udm_full.append(Var_Udm[Nam_cntr].strip())
                        else:
                            Nam_cntr+=1
                            Var_Nam_full.append(Var_Nam[Nam_cntr].strip()+'_%d' %Num)
                            Var_Udm_full.append(Var_Udm[Nam_cntr].strip())
                #---------------- Add first four variables (name not inthe data)
                Var_Nam_full=['TIME','CPU','DT','UNKN03']+Var_Nam_full
                Var_Udm_full=['sec','','','']+Var_Udm_full
                #------------------------------------- Search variables position
                for Nam in Var_srch:
                    Var_srch_pos.append(Var_Nam_full.index(Nam.strip()))
                #- Sort the variable position in the block and store the swap ID
                Var_Udm_full=[Var_Udm_full[i] for i in Var_srch_pos]
                Var_srch_pos_swap=sorted(range(len(Var_srch_pos)), key=lambda k: Var_srch_pos[k])
                Var_srch_pos_swap=sorted(range(len(Var_srch_pos_swap)), key=lambda k: Var_srch_pos_swap[k])
                Var_srch_pos.sort()
                Var_srch_pos.append(Var_N[1]+4)

                #---------- Empty element to maintain the correct list numbering
                Hdr_list.append([])

            elif Hdr_list[cntr - 1] == '.TR/':
                #===============================================================
                # Hdr_list for each Time step
                #===============================================================
                data_Pos.append(bf.tell())
                bf.seek(blk_len_b[cntr], 1)
                #---------- Empty element to maintain the correct list numbering
                Hdr_list.append([])

            else:
                #===============================================================
                # Skip all the unknown blocks
                # bf.seek(blk_len_b[cntr], 1)
                #===============================================================
                #---------- Empty element to maintain the correct list numbering
                Hdr_list.append([])

            #===================================================================
            # Ending block length (Byte)
            #===================================================================
            blk_len_e.append(unpack('I', bf.read(4))[0])
            #---------------------------------------------------- Debug printing
            if dbg: print(blk_len_e[cntr])
            if dbg: print("BLOCK %02d ----------------------------------------------\n" % cntr)

            #---------------------------------------------------- Update Counter
            cntr +=1

    #===========================================================================
    # Second Reading of the file for data extraction
    #===========================================================================
    data=np.empty([len(data_Pos), len(Var_srch)+1])*np.nan # array of the data
    with open(file_dir, 'rb') as bf:
        for i,Pos in enumerate(data_Pos):
            bf.seek(Pos, 0)
            for j in range(len(Var_srch_pos)-1):
                data[i,j]=unpack('f', bf.read(4))[0]
                bf.seek((Var_srch_pos[j+1]-Var_srch_pos[j])*4-4, 1)
            #---------------------------------------------------- Debug printing
            if dbg: print("DATA BLOCK %02d -----------------------------------------\n" % i)
            if dbg: print(data[i,:])
            if dbg: print("DATA BLOCK %02d -----------------------------------------\n" % i)
    #------------------------------------ Swap the data to the original position
    data=data[:,Var_srch_pos_swap]

    return data[:,0],data[:,1:],Var_Udm_full[1:]

#===============================================================================
