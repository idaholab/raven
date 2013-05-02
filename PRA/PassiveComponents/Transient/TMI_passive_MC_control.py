import sys
import math
import distribution1D
import raventools

# initialize distribution container
distcont  = distribution1D.DistributionContainer.Instance()


def initial_function(monitored, controlled, auxiliary):

    return

def control_function(monitored, controlled, auxiliary):
  
  weekPerYear = 52

  lengthCycle1      = 40 # in years
  temperatureCycle1 = 500 # K
  pressureCycle1    = 15.1 # MPa
  theta1C           = 8.3081
  theta1R           = 6.1866
  
  lengthCycle2      = 20 # in years
  temperatureCycle2 = 520 # K
  pressureCycle2    = 15.1 # MPa
  theta2C           = 8.3074
  theta2R           = 6.1865
  	
  a_c = 0.0002 # in m
  a_d = 0.0002 # in m
  
  numberWeeks = (lengthCycle1+lengthCycle2)*weekPerYear
  
  random_n_1 = distcont.random()  # transtion S to M
  random_n_2 = distcont.random()  # transtion M to C
  random_n_3 = distcont.random()  # transtion M to D
  random_n_4 = distcont.random()  # repair in M
  random_n_5 = distcont.random()  # repair in C
  random_n_6 = distcont.random()  # repair in D
  random_n_7 = distcont.random()  # repair in L
  
  state = 1;
  stateScdfProgress  = 0
  stateMCcdfProgress = 0
  stateMDcdfProgress = 0
  repairMcdfProgress = 0
  repairCcdfProgress = 0
  
  writer = csv.writer(open("debug.csv", "w"))
  c.writerow(["Name","Address","Telephone","Fax","E-mail","Others"])
  
	for i in range numberWeeks:
    
    timeInYear   = i/52.1775;
    timeInYear_1 = (i-1)/52.1775;
    
    if i < (firstCycleLength*weekPerYear):
      
      ######################
      #### First Cycle #####
      ###################### 
      
      etag1 = 25248/temperatureCycle1 - 0.12*pressureCycle1 -36.4
      
      #### State 1 ####
      if state==1:
        stateScdfProgress = stateScdfProgress + (distcont.Pdf('sojTimeS',timeInYear)+distcont.Pdf('sojTimeS',timeInYear_1))*(timeInYear-timeInYear_1)/2
        if stateScdfProgress>random_n_1:
          state=2
          stateMstartTime = i/52.1775
          stateScdfProgress=0
          
      #### State 2 ####
      if state==2:
        stateMCcdfProgress = stateMCcdfProgress + (distcont.Pdf('sojTimeMC',timeInYear-stateMstartTime)+distcont.Pdf('sojTimeMC',timeInYear_1-stateMstartTime))*(timeInYear-timeInYear_1)/2
        stateMDcdfProgress = stateMDcdfProgress + (distcont.Pdf('sojTimeMD',timeInYear-stateMstartTime)+distcont.Pdf('sojTimeMD',timeInYear_1-stateMstartTime))*(timeInYear-timeInYear_1)/2
        repairMcdfProgress = repairMcdfProgress + (distcont.Pdf('repairM',timeInYear-stateMstartTime)  +distcont.Pdf('repairM',timeInYear_1-stateMstartTime))  *(timeInYear-timeInYear_1)/2
        
        if stateMCcdfProgress>random_n_2:
          state=3
          stateCstartTime = i/52.1775
          sojTimeM = stateMstartTime - stateCstartTime
          stateMCcdfProgress = 0
          stateMDcdfProgress = 0
          repairMcdfProgress = 0          

        if stateMDcdfProgress>random_n_3:
          state=4    
          stateDstartTime = i/52.1775
          sojTimeM = stateMstartTime - stateCstartTime
          stateMCcdfProgress = 0
          stateMDcdfProgress = 0
          repairMcdfProgress = 0  
                    
        if repairMcdfProgress > random_n_4:
          state=1
          stateMCcdfProgress = 0
          stateMDcdfProgress = 0
          repairMcdfProgress = 0
      
      #### State 3 ####    
      if state = 3:
        repairCcdfProgress = repairCcdfProgress + (distcont.Pdf('repairC',timeInYear-stateCstartTime)+distcont.Pdf('repairC',timeInYear_1-stateCstartTime)) *(timeInYear-timeInYear_1)/2
        t_C = sojTimeM
        y = stateCstartTime - i/52.1775
        argument = (t_C+y)/t_C
        a = a_c * math.pow(argument,theta1C)
        failurePressure = pressureRupture(a,1)
        
        if repairCcdfProgress > random_n_5:
          state = 1
          
      #### State 4 ####    
      if state = 4:
        sojTimeD = sojTimeM * 1.34
        repairDcdfProgress = repairCcdfProgress + (distcont.Pdf('repairD',timeInYear-stateCstartTime)+distcont.Pdf('repairD',timeInYear_1-stateCstartTime)) *(timeInYear-timeInYear_1)/2
        
        if (i/52.1775-stateCstartTime)>sojTimeD:
          state = 5
          stateLstartTime = i/52.1775
          sojTimeD = stateDstartTime - i/52.1775
        
        if repairDcdfProgress > random_n_6:
          state = 1
          repairDcdfProgress = 0
          
      #### State 5 ####    
      if state = 5:
        repairLcdfProgress = repairLcdfProgress + (distcont.Pdf('repairL',timeInYear-stateLstartTime)+distcont.Pdf('repairL',timeInYear_1-stateLstartTime)) *(timeInYear-timeInYear_1)/2
        t_D = sojTimeD + sojTimeM
        y = stateDstartTime - i/52.1775
        argument = (t_C+y)/t_C
        a = a_d * math.pow(argument,theta1C)
        failurePressure = pressureRupture(a,1)
        
        if repairCcdfProgress > random_n_5:
          state = 1  
          repairLcdfProgress = 0
          failurePressure = 0
             
    else:
      
      #######################
      #### Second Cycle #####
      #######################

      etag1 = 25248/temperatureCycle1 - 0.12*pressureCycle1 -36.4
      
      #### State 1 ####
      if state==1:
        stateScdfProgress = stateScdfProgress + (distcont.Pdf('sojTimeS',timeInYear)+distcont.Pdf('sojTimeS',timeInYear_1))*(timeInYear-timeInYear_1)/2
        if stateScdfProgress>random_n_1:
          state=2
          stateMstartTime = i/52.1775
          stateScdfProgress=0
          
      #### State 2 ####
      if state==2:
        stateMCcdfProgress = stateMCcdfProgress + (distcont.Pdf('sojTimeMC',timeInYear-stateMstartTime)+distcont.Pdf('sojTimeMC',timeInYear_1-stateMstartTime))*(timeInYear-timeInYear_1)/2
        stateMDcdfProgress = stateMDcdfProgress + (distcont.Pdf('sojTimeMD',timeInYear-stateMstartTime)+distcont.Pdf('sojTimeMD',timeInYear_1-stateMstartTime))*(timeInYear-timeInYear_1)/2
        repairMcdfProgress = repairMcdfProgress + (distcont.Pdf('repairM',timeInYear-stateMstartTime)  +distcont.Pdf('repairM',timeInYear_1-stateMstartTime))  *(timeInYear-timeInYear_1)/2
        
        if stateMCcdfProgress>random_n_2:
          state=3
          stateCstartTime = i/52.1775
          sojTimeM = stateMstartTime - stateCstartTime
          stateMCcdfProgress = 0
          stateMDcdfProgress = 0
          repairMcdfProgress = 0          

        if stateMDcdfProgress>random_n_3:
          state=4    
          stateDstartTime = i/52.1775
          sojTimeM = stateMstartTime - stateCstartTime
          stateMCcdfProgress = 0
          stateMDcdfProgress = 0
          repairMcdfProgress = 0  
                    
        if repairMcdfProgress > random_n_4:
          state=1
          stateMCcdfProgress = 0
          stateMDcdfProgress = 0
          repairMcdfProgress = 0
      
      #### State 3 ####    
      if state = 3:
        repairCcdfProgress = repairCcdfProgress + (distcont.Pdf('repairC',timeInYear-stateCstartTime)+distcont.Pdf('repairC',timeInYear_1-stateCstartTime)) *(timeInYear-timeInYear_1)/2
        t_C = sojTimeM
        y = stateCstartTime - i/52.1775
        argument = (t_C+y)/t_C
        a = a_c * math.pow(argument,theta1C)
        failurePressure = pressureRupture(a,1)
        
        if repairCcdfProgress > random_n_5:
          state = 1
          
      #### State 4 ####    
      if state = 4:
        sojTimeD = sojTimeM * 1.34
        repairDcdfProgress = repairCcdfProgress + (distcont.Pdf('repairD',timeInYear-stateCstartTime)+distcont.Pdf('repairD',timeInYear_1-stateCstartTime)) *(timeInYear-timeInYear_1)/2
        
        if (i/52.1775-stateCstartTime)>sojTimeD:
          state = 5
          stateLstartTime = i/52.1775
          sojTimeD = stateDstartTime - i/52.1775
        
        if repairDcdfProgress > random_n_6:
          state = 1
          repairDcdfProgress = 0
          
      #### State 5 ####    
      if state = 5:
        repairLcdfProgress = repairLcdfProgress + (distcont.Pdf('repairL',timeInYear-stateLstartTime)+distcont.Pdf('repairL',timeInYear_1-stateLstartTime)) *(timeInYear-timeInYear_1)/2
        t_D = sojTimeD + sojTimeM
        y = stateDstartTime - i/52.1775
        argument = (t_C+y)/t_C
        a = a_d * math.pow(argument,theta1C)
        failurePressure = pressureRupture(a,1)
        
        if repairCcdfProgress > random_n_5:
          state = 1  
          repairLcdfProgress = 0
          failurePressure = 0
                
  return 

