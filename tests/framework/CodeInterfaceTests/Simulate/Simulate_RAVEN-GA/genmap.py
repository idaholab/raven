'''
This is an external sAript to generate Aore map and additional FUEL.NEW Aard 
for RAVEN-SIMULATE interface test on GA framework.
Edit log:
07/05/2022: khnguy22 NCSU - Creation
'''
import os
import sys
import numpy
import random
# no fixed group in case of RAVEN testing
# gen map for problem without Fuel/LAB card

# input is 35 random number for a map of 35 locations
# # map to selected
# select_loc=[] 
# for i in range (1,21):
#    select_loc.append(random.choice(range(5)))
# select_loc.append(5)
# for i in range (1,5):
#    select_loc.append(random.choice(range(5)))
# select_loc.append(5)
# select_loc.append(5)
# select_loc.append(random.choice(range(5)))
# select_loc.append(random.choice(range(5)))
# select_loc.append(5)
# select_loc.append(5)
# select_loc.append(5)
# select_loc.append(5)
# select_loc.append(5)
# select_loc.append(5)
# pool to select
genome_key={
   'FA1': {'gene_group': 2.0, 'type': 2, 'serial': 'A300', 'name': '2.0_w/o', 'map':          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]},
   'FA2': {'gene_group': 2.5, 'type': 3, 'serial': 'B300', 'name': '2.5_w/o_no_bp', 'map':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]},
   'FA3': {'gene_group': 3.2, 'type': 5, 'serial': 'C300', 'name': '3.2_w/o_no_bp', 'map':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]},
   'FA4': {'gene_group': 2.5, 'type': 4, 'serial': 'D300', 'name': '2.5_w/o_with_bp', 'map':  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]},
   'FA5': {'gene_group': 3.2, 'type': 6, 'serial': 'E300', 'name': '3.2_w/o_with_bp', 'map':  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]},
   'REF': {'type': 1, 'gene_group': 'reflector', 'serial': 'none', 'name': 'reflector', 'map':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]}}
core_map={
   8:{8:0, 9:1, 10:3, 11:6,   12:10,  13:15,  14:21  , 15:27,  16:32},
   9:{8:1, 9:2, 10:4, 11:7,   12:11,  13:16,  14:22  , 15:28,  16:33},
  10:{8:3, 9:4, 10:5, 11:8,   12:12,  13:17,  14:23  , 15:29,  16:34},
  11:{8:6, 9:7, 10:8, 11:9,   12:13,  13:18,  14:24  , 15:30,  16:None},
  12:{8:10,9:11,10:12,11:13,  12:14,  13:19,  14:25  , 15:31,  16:None},
  13:{8:15,9:16,10:17,11:18,  12:19,  13:20,  14:26  , 15:None,16:None},
  14:{8:21,9:22,10:23,11:24,  12:25,  13:26,  14:None, 15:None,16:None},
  15:{8:27,9:28,10:29,11:30,  12:31,  13:None,14:None, 15:None,16:None},
  16:{8:32,9:33,10:34,11:None,12:None,13:None,14:None, 15:None,16:None}} 
## function list

def mapping_genome(list_input, genome_key):
   """
   Returns a list of the genes from input index
   """
   gene_list = []
   keys = list(genome_key.keys())
   for index in list_input:
      gene_list.append(keys[index])
   return gene_list
           
def serial_loading_pattern(core_map,genome_key, gene_list):
    """
    Writes a core loading pattern using serial numbers as the loading
    pattern designator.
    Written by Brian Andersen 4/5/2020
    """
    biggest_number = 0
    for gene in genome_key:
        if gene == "additional_information" or gene == 'symmetry_list':
            pass
        else:
            if genome_key[gene]['type'] > biggest_number:
                biggest_number = genome_key[gene]['type']
    
    number_spaces = len(str(biggest_number)) + 1
    problem_map = core_map
    row_count = 1
    loading_pattern = ""
    for row in range(25):    #I doubt a problem will ever be larger than a 25 by 25 core
        if row in problem_map:                 #it will just work
            loading_pattern += f"'FUE.TYP'  {row_count},"
            for col in range(25):#so I hard coded the value because if I did this algorithm right  
                if col in problem_map[row]:
                    if not problem_map[row][col]:
                        if type(problem_map[row][col]) == int:
                            gene_number = problem_map[row][col]
                            gene = gene_list[gene_number]
                            value = genome_key[gene]['type']
                            str_ = f"{value}"
                            loading_pattern += f"{str_.rjust(number_spaces)}"
                        else:
                            loading_pattern += f"{'0'.rjust(number_spaces)}"
                    else:
                        gene_number = problem_map[row][col]
                        gene = gene_list[gene_number]
                        value = genome_key[gene]['type']
                        str_ = f"{value}"
                        loading_pattern += f"{str_.rjust(number_spaces)}"
            loading_pattern += "/\n"
            row_count += 1
    loading_pattern += "\n"
    return loading_pattern
def evaluate(self):
   tmp=list(range(35))
   tmp[0:35]=self.loc1,self.loc2,self.loc3,self.loc4,self.loc5,self.loc6,self.loc7,\
                    self.loc8,self.loc9,self.loc10,self.loc11,self.loc12,self.loc13,self.loc14,\
                    self.loc15,self.loc16,self.loc17,self.loc18,self.loc19,self.loc20,5,\
                    self.loc22,self.loc23,self.loc24,self.loc25,5,5,self.loc28,\
                    self.loc29,5,5,5,5,5,5
   select_loc=[int(ii) for ii in tmp]
   #print(select_loc)
   gene_list=mapping_genome(select_loc, genome_key)
   loading_pattern = serial_loading_pattern(core_map,genome_key, gene_list)
   return loading_pattern

# if __name__ == '__main__':
#    select_loc=list(range(35))
#    select_loc[0:35]=0,0,0,4,0,0,2,0,0,3,1,1,4,1,0,0,0,1,2,1,5,2,2,3,2,5,5,4,0,5,5,5,5,5,5
#    gene_list=mapping_genome(select_loc, genome_key)
#    loading_pattern = serial_loading_pattern(core_map,genome_key, gene_list)
#    print(loading_pattern)