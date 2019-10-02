****************************************************************
*  this  deck  contains proprietary information *
*  do not   disseminate any   part  of this  deck  without  the   *
*  written  approval of the   eg&g  idaho severe   accident *
*  analysis manager  (r.   j. dallman).   *
****************************************************************
********************************************************
*  RELAP5-3D   three channel  vessel   model for   surry steady   state
*  Developed by   P. Bayless     -     1987
*  Updated by  C. Parisi   (INL) -  March 2016
*
=  surry 3-channel   vessel
*
*  PROBLEM  TYPE  AND   OPTION   100-199
*
0000100  new   stdy-st
0000101  run
0000102  british  si
*  0000105  10.   15.   removed
*
*
*  /  trip  /  /  CHF  /   /  end cntrl  /
0000107  1  1  1
*
0000110  hydrogen helium   krypton  xenon nitrogen
0000115  0.0   0.0   0.0   0.0   1.0
*
*  TIME  STEP  CONTROL  200-299
*  time  dt min   dt max   ssdtt min.ed.(plt)   maj.ed (out)   restart
0000201  1000.0   1.e-7 0.06  00019 75 750   2500
*
****************************************************************
*  minor edits
****************************************************************
0000301  tempf 108010000
0000302  tempf 111100000
0000303  tempf 112100000
0000304  tempf 113100000
0000305  tempf 151010000
0000306  tempf 161010000
0000307  tempf 171010000
0000308  tempf 172010000
0000309  tempf 154010000
0000310  tempf 164010000
0000311  tempf 174010000
0000312  tempf 190010000
0000313  p  108010000
0000314  p  151010000
0000315  p  161010000
0000316  p  171010000
0000317  p  172010000
0000318  mflowj   102040000
0000319  mflowj   108010000
0000320  mflowj   108020000
0000321  mflowj   108030000
0000322  mflowj   108040000
0000323  mflowj   151010000
0000324  mflowj   161010000
0000325  mflowj   171010000
0000326  mflowj   172040000
0000327  mflowj   172050000
0000328  mflowj   172060000
0000329  mflowj   102030000
0000330  mflowj   102040000
0000335  cntrlvar 111
0000336  cntrlvar 112
0000337  cntrlvar 113
0000338  cntrlvar 115
0000350  tempf 220010000   *  loop  a  cl temp
0000351  tempf 200010000   *  loop  a  hl temp
0000352  cntrlvar 4  *  rcp   speed
0000353  p  282010000   *  sga   pressure
0000354  p  440010000   *  press.   pressure
0000355  mflowj   267000000   *  sga   feedwater   flow
0000356  mflowj   285000000   *  sga   steam flow
0000357  cntrlvar 71 *  sga   control  valve
0000358  cntrlvar 22 *  sga   mass  error
0000359  cntrlvar 95 *  pzr   volume   error
0000360  tempf 320010000   *  loop  b  cl temp
0000361  tempf 300010000   *  loop  b  hl temp
0000363  p  382010000   *  sgb   pressure
0000365  mflowj   367000000   *  sgb   feedwater   flow
0000366  mflowj   385000000   *  sgb   steam flow
0000367  cntrlvar 73 *  sgb   control  valve
0000368  cntrlvar 42 *  sgb   mass  error
0000370  tempf 420010000   *  loop  c  cl temp
0000371  tempf 400010000   *  loop  c  hl temp
0000373  p  482010000   *  sgc   pressure
0000375  mflowj   467000000   *  sgc   feedwater   flow
0000376  mflowj   485000000   *  sgc   steam flow
0000377  cntrlvar 75 *  sgc   control  valve
0000378  cntrlvar 62 *  sgc   mass  error
0000379  cntrlvar 440   *  sga   level
0000380  cntrlvar 441   *  sgb   level
0000381  cntrlvar 442   *  sgc   level
0000382  cntrlvar 430   *  prz   level
0000383  httemp   111101012   *  core  clad  temp
0000384  httemp   112101012   *  core  clad  temp
0000385  httemp   113101012   *  core  clad  temp
0000386  httemp   111100912   *  core  clad  temp
0000387  httemp   112100912   *  core  clad  temp
0000388  httemp   113100912   *  core  clad  temp
0000389  httemp   111100812   *  core  clad  temp
0000390  httemp   112100812   *  core  clad  temp
0000391  httemp   113100812   *  core  clad  temp
0000392  cntrlvar 497   *    MAX CORE TEMP
0000393  cntrlvar 498   *    END OF CALC LOGIC VARIABLE
*
20800001 timeof   527   *  ACCU1 avail
20800002 timeof   556   *  ACCU2 avail
20800003 timeof   551   *  LPI1 start
20800004 timeof   552   *  LPI2 start
20800005 timeof   558   *  LPI1 Fail
20800006 timeof   559   *  LPI2 Fail
20800007 timeof   566   *  LPI1 Rec start
20800008 timeof   567   *  LPI2 Rec Start
20800009 timeof   571   *  LPI1 Rec Fail
20800010 timeof   572   *  LPI2 Rec Fail
20800011 timeof   565   *  RWST EMPTY
*
****************************************************************
*  trips
****************************************************************
*  VARIABLE TRIPS 401/599  1/1000         206NNNN0
*
0000401  cntrlvar 497   ge null  0  1467.0   l  -1.0  *end calculation
0000501  p  440010000   gt null  0  2280. n  * PRZ PORV Set point close
0000502  p  440010000   gt null  0  2350. n  * PRZ PORV Set point open
0000503  p  440010000   gt null  0  2375. n  * PRZ SV Set point close
0000504  p  440010000   gt null  0  2575. n  * PRZ SV Set point open
0000505  time  0  lt null  0  1.0e+6   n  * MSIV open for imposing conditions on empty SGs
0000506  p  284010000   gt null  0  1000. n  * SG A PORV Set point close
0000507  p  284010000   gt null  0  1050. n  * SG A PORV Set point open
0000508  p  384010000   gt null  0  1000. n  * SG B PORV Set point close
0000509  p  384010000   gt null  0  1050. n  * SG B PORV Set point open
0000510  p  484010000   gt null  0  1000. n  * SG C PORV Set point close
0000511  p  484010000   gt null  0  1050. n  * SG C PORV Set point open
0000512  p  284010000   gt null  0  1092. n  * SG A SV Set point close
0000513  p  284010000   gt null  0  1184. n  * SG A SV Set point open
0000514  p  384010000   gt null  0  1092. n  * SG B SV Set point close
0000515  p  384010000   gt null  0  1184. n  * SG B SV Set point open
0000516  p  484010000   gt null  0  1092. n  * SG C SV Set point close
0000517  p  484010000   gt null  0  1184. n  * SG C SV Set point open
0000518  time  0  gt null  0  1.0e+6   l  * pmp control  "(pmp trip off, Power supplied to the motor)"
0000519  time  0  gt null  0  0.0   l  * pmp control  (it has precedence over the previous one)
0000522  time  0  gt null  0  1.0e+6   l  -1.0  * SCRAM SIGNAL
0000523  time  0  gt null  0  1.0e+6   n  -1.0  * TD AFW SIGNAL   - Auto initiation at FULL Flow
0000524  time  0  gt null  0  1.0e+6   n  -1.0  * TD AFW SIGNAL   - Operator Control   of the SG Level
0000525  cntrlvar 437   ge null  0  0.0   l  -1.0  * STOP TD AFW for ECST depletion
0000526  time  0  gt null  0  1.0e+6   l  -1.0  * TD AFW EQ DAMAGE TRIP
0000527  time  0  gt null  0  1.0e+6   l  -1.0  * ACCUMULATOR1 " ISOLATION TRIP [OFF, accumulator is isolated]"
0000528  time  0  gt null  0  1.0e+6   l  -1.0  * MCP Leakage (21 gpm) ON/OFF
0000529  time  0  gt null  0  1.0e+6   n  -1.0  * SG COOL DOWN ON
0000530  p  284010000   lt null  0  134.7 l  -1.0  * SG COOL DOWN OFF
0000531  p  384010000   lt null  0  134.7 l  -1.0  * SG COOL DOWN OFF
0000532  p  484010000   lt null  0  134.7 l  -1.0  * SG COOL DOWN OFF
0000533  time  0  gt null  0  1.0e+6   l  -1.0  * MCP Leakage (@ 21 gpm) ON/OFF
0000534  time  0  gt null  0  1.0e+6   l  -1.0  * MCP Leakage (@ 183 gpm) ON/OFF
0000535  time  0  gt null  0  1.0e+6   l  -1.0  * MCP Leakage (@ 183 gpm) ON/OFF
0000536  time  0  gt null  0  1.0e+6   l  -1.0  * MCP Leakage SHUT TRIP
0000537  time  0  lt null  0  1.0e+6   n  0.0   * SHUT OFF PORV BECAUSE OF BATTERIES OUT
0000538  time  0  gt null  0  1.0e+6   n  -1.0  * TURN ON THE KERR PUMP FOR EMERGENCY INJECTION!
0000539  p  222010000   lt null  0  290.1 n  -1.0  * PRESSURE CONDITION FOR KERR PUMP OPERATION
0000540  time  0  gt null  0  1.0e+6   l  -1.0  * BATTERIES DAMAGE FOR FLOODING
0000541  time  0  lt null  0  1.0e+6   n  -1.0  * TIME FOR RECOVERY ON SG PORV and AFW BLACKRUN
0000542  time  0  gt timeof   526   1.80E+03 n  -1.0  * EMERGENCY PS DEPRESSURZATION - TIME OF ACTION
0000543  p  440010000   gt null  0  167.0 n  -1.0  * EMERGENCY PS DEPRESSURZATION - Set point close
0000544  p  440010000   gt null  0  217.0 n  -1.0  * EMERGENCY PS DEPRESSURZATION - Set point open
0000545  voidf 250010000   lt null  0  1.00E-04 l  -1.0  * CONDITION FOR ACCUMULATOR ISOLATION
0000546  voidf 350010000   lt null  0  1.00E-04 l  -1.0  * CONDITION FOR ACCUMULATOR ISOLATION
0000547  voidf 450010000   lt null  0  1.00E-04 l  -1.0  * CONDITION FOR ACCUMULATOR ISOLATION
0000548  time  0  lt null  0  1.0e+6   n  0.0   * LBLOCA signal
0000549  p  440010000   lt null  0  1840.0   l  -1.0  * Low Pressure signal on PRZ     >> REACTOR TRIP
0000550  p  440010000   lt null  0  1789.7   l  -1.0  * Low-low Pressure signal on PRZ    >> SAFETY INJECTION SIGNAL
*
0000551  time  0  gt timeof   550   1.00E+06 n  -1.0  * Startup of LPIS #1
0000552  time  0  gt timeof   550   1.00E+06 n  -1.0  * Startup of LPIS #2
0000553  time  0  gt timeof   550   1.00E+06 n  -1.0  * Startup of HPIS #1
0000554  time  0  gt timeof   550   1.00E+06 n  -1.0  * Startup of HPIS #2
0000555  time  0  gt timeof   550   1.00E+06 n  -1.0  * Startup of HPIS #3
0000556  time  0  gt null  0  1.0e+6   l  -1.0  * Connect the ACCUMULATOR2 when Transient Starts
0000557  time  0  gt null  0  1.0e+6   l  -1.0  * Connect the ACCUMULATOR3 when Transient Starts
0000558  time  0  lt timeof   551   1.00E+06  n  0.0   * Failure to Run LPIS #1
0000559  time  0  lt timeof   552   1.00E+06  n  0.0   * Failure to Run LPIS #2
0000560  time  0  lt timeof   553   1.00E+06  n  0.0   * Failure to Run HPIS #1
0000561  time  0  lt timeof   554   1.00E+06  n  0.0   * Failure to Run HPIS #2
0000562  time  0  lt timeof   555   1.00E+06  n  0.0   * Failure to Run HPIS #3
*
0000563  time  0  gt null  0  1.50E+03 l  -1.0  * stop AFW
0000564  time  0  gt null  0  1.00E+06 l  -1.0  * CONTAIMNET SPRAY ON
0000565  cntrlvar 489   ge null  0  0.0   l  -1.0  * STOP ECCS because RWST depletion
*
0000566  time  0  gt timeof   565   1.00E+06 n  -1.0  * Startup of LPIS #1 - DeltaT4 for LPI1 Start Time] - REC
0000567  time  0  gt timeof   565   1.00E+06 n  -1.0  * Startup of LPIS #2 - DeltaT4 for LPI2 Start Time] - REC
0000568  time  0  gt timeof   565   1.00E+06 n  -1.0  * Startup of HPIS #1 - REC
0000569  time  0  gt timeof   565   1.00E+06 n  -1.0  * Startup of HPIS #2 - REC
0000570  time  0  gt timeof   565   1.00E+06 n  -1.0  * Startup of HPIS #3 - REC
*
0000571  time  0  gt timeof   566   1.00E+06 n  -1.0   * Failure to Run LPIS #1 - [DeltaT5 for LPI1 Run Time] - REC
0000572  time  0  gt timeof   567   1.00E+06 n  -1.0   * Failure to Run LPIS #2 - [DeltaT5 for LPI2 Run Time] - REC
0000573  time  0  gt timeof   568   1.00E+06 n  -1.0   * Failure to Run HPIS #1 - REC
0000574  time  0  gt timeof   569   1.00E+06 n  -1.0   * Failure to Run HPIS #2 - REC
0000575  time  0  gt timeof   570   1.00E+06 n  -1.0   * Failure to Run HPIS #3 - REC
*
0000576  time  0  gt timeof   558   1.00E+06 n  -1.0  * Repair Time DT3 of LPIS #1 - INJ
0000577  time  0  gt timeof   559   1.00E+06 n  -1.0  * Repair Time DT3 of LPIS #2 - INJ
*
0000578  cntrlvar 499  lt timeof  565  0.0  n  -1.0  * condition for  LPI1 Run Time INJ mode less LPR switch time
0000579  cntrlvar 500  lt timeof  565  0.0  n  -1.0  * condition for  LPI2 Run Time INJ mode less LPR switch time
*
0000580  time  0  gt timeof   571   0.600E+3 n  -1.0  * Repair Time DT6 of LPIS #1 - REC
0000581  time  0  gt timeof   572   0.600E+3 n  -1.0  * Repair Time DT6 of LPIS #2 - REC
*
0000582  cntrlvar 503  lt timeof   566   0.0 n  -1.0  * Condition for Case LPIS1-REC, Case C,D
0000583  cntrlvar 504  lt timeof   566   0.0 n  -1.0  * Condition for Case LPIS2-REC, Case C,D
*
0000584  time  0  gt timeof   576   0.600E+3 n  -1.0  * [DeltaT5 for LPI1 Run Time] - REC
0000585  time  0  gt timeof   577   0.600E+3 n  -1.0  * [DeltaT5 for LPI2 Run Time] - REC
*
0000586  time  0  gt timeof   584   0.600E+3 n  -1.0  * Repair Time DT6 of LPIS #1 - REC
0000587  time  0  gt timeof   585   0.600E+3 n  -1.0  * Repair Time DT6 of LPIS #2 - REC
*
0000588  cntrlvar 503  lt timeof   565   0.0 n  -1.0  * Condition for Case LPIS1-REC, Case B
0000589  cntrlvar 504  lt timeof   565   0.0 n  -1.0  * Condition for Case LPIS2-REC, Case B
*
****************************************************************
*l trip  trip  1  rel   trip  2  l  timeof   comments
*
0000601  540   or -537  l  -1.0  *  BATTERIES OUT  {Batteries exausted or flooded}
0000602  601   and   541   n  -1.0  *  TRUE BETWEEN BATTERIES FAILURE      AND RECOVERY TIME
0000603  529   and   -602  n  -1.0  *  SG COOLDOWN for NORMAL PROCEDURE AND for AFTER BATTERY FAILURE & RECOVERY ACTIONS
0000604  601   or 526   n  -1.0  *  MAIN PRZ PORV OFF for Batteries OUT OR AFW FAILURE FOR EQ
0000605  -601  and   542   n  -1.0  *  SECONDARY PRZ PORV ON for Batteries OK AND TIME OF ACTION AFTER AFW FAILURE FOR EQ
*
*
0000611  501   and   -604  n  -1.0  *  PRZ   PORV OFF
0000612  502   and   -604  n  -1.0  *  PRZ   PORV OFF
0000616  506   and   -601  n  -1.0  *  SGA   PORV OFF
0000617  507   and   -601  n  -1.0  *  SGA   PORV OFF
0000618  508   and   -601  n  -1.0  *  SGB   PORV OFF
0000619  509   and   -601  n  -1.0  *  SGB   PORV OFF
0000620  510   and   -601  n  -1.0  *  SGC   PORV OFF
0000621  511   and   -601  n  -1.0  *  SGC   PORV OFF
0000622  543   and   605   n  -1.0  *  SECONDARY PRZ  PORV ON
0000623  544   and   605   n  -1.0  *  SECONDARY PRZ  PORV ON
*
0000641  527   and   -545  n  -1.0  *  ACCUMULATOR OFF
0000642  556   and   -546  n  -1.0  *  ACCUMULATOR OFF
0000643  557   and   -547  n  -1.0  *  ACCUMULATOR OFF
*
0000600  650
*
0000650  401   or 401   n  -1.0  *
*
0000701  611   and   702   n  *press.  porv  open
0000702  612   or 701   n  *press.  porv
0000703  -702  or -611  n  *press.  porv  closed
0000704  503   and   705   n  *press.  safety   open
0000705  504   or 704   n  *press.  safety
0000706  -705  or -503  n  *press.  safety   closed
0000707  616   and   708   n  *sga  porv  open
0000708  617   or 707   n  *sga  porv
0000709  -708  or -616  n  *sga  porv  closed
0000710  618   and   711   n  *sgb  porv  open
0000711  619   or 710   n  *sgb  porv
0000712  -711  or -618  n  *sgb  porv  closed
0000713  620   and   714   n  *sgc  porv  open
0000714  621   or 713   n  *sgc  porv
0000715  -714  or -620  n  *sgc  porv  closed
0000716  512   and   717   n  *sga  safety   open
0000717  513   or 716   n  *sga  safety
0000718  -717  or -512  n  *sga  safety   closed
0000719  514   and   720   n  *sgb  safety   open
0000720  515   or 719   n  *sgb  safety
0000721  -720  or -514  n  *sgb  safety   closed
0000722  516   and   723   n  *sgc  safety   open
0000723  517   or 722   n  *sgc  safety
0000724  -723  or -516  n  *sgc  safety   closed
0000725  525   or 526   n  -1.0  *AFW  STOP  Conditions  (EQ damage OR ECST empty)
0000726  601   or 725   n  -1.0  *AFW  STOP  Conditions  (EQ damage OR ECST empty or Flooding or Depletion)
0000727  726   or 524   n  -1.0  *AFW  Auto initiation   OFF Conditions (EQ damage OR ECST empty OR Flooding/Depletion OR SG Level Regulation ON)
0000728  523   and   -727  n  -1.0  *AFW  Auto initiation   at full flow   COMMAND  [OFF when the Operator Controlled SG Level is ON or when EQ or ECST empty or Flooding/Depletion]
0000729  726   and   541   n  -1.0
0000730  524   and   -729  n  -1.0  *AFW  SG Level Control  COMMAND  [OFF when EQ or ECST empty OR BATTERY FLOODED/DEPLETED]
0000731  523   and   -729  n  -1.0  *AFW  "Steam outlet [ON when Auto initiation OR when recovery actions, OFF when EQ damage OR ECST empty OR batteries flooded/depleted.]"
0000732  603   and   -530  n  -1.0  * SG1 COOL DOWN [OFF if pressure below of 134 psia]
0000733  603   and   -531  n  -1.0  * SG2 COOL DOWN [OFF if pressure below of 134 psia]
0000734  603   and   -532  n  -1.0  * SG3 COOL DOWN [OFF if pressure below of 134 psia]
*
0000740  528   and   533   n  -1.0  *  SEAL LOCA @ 21 gpm
0000741  534   and   535   n  -1.0  *  MCP Leakage (@ 183 gpm) ON/OFF
0000742  740   or 741   n  -1.0  *  SEAL LOCA
*
0000743  538   and   539   n  -1.0  *  KERR PUMP OPERATION LOGIC
*
0000751  622   and   752   n  *press.  porv  open
0000752  623   or 751   n  *press.  porv
0000753  -752  or -622  n  *press.  porv  closed
*
0000754  548   and   548   n  0.0   *LBLOCA  connection valve
0000755  -548  and   -548  n  -1.0  *LBLOCA  BREAK OPENING
*
* ACTUATION LOGIC FOR LPIS 1 & 2 - INJECTION PHASE
*
0000781  -558  and   551   n  -1.0
0000782  -559  and   552   n  -1.0
0000783  -565  and   553   n  -1.0
0000784  -565  and   554   n  -1.0
0000785  -565  and   555   n  -1.0
*
0000786  781   or    576   n  -1.0  * LPIS #1 start  condition [LPIrun or after repair]
0000787  782   or    576   n  -1.0  * LPIS #2 start  condition [LPIrun or after repair]
0000788  783   and   560   n  -1.0  * HPIS #1 start
0000789  784   and   561   n  -1.0  * HPIS #2 start
0000790  785   and   562   n  -1.0  * HPIS #3 start
*
0000796  786   and   -565  n  -1.0  * LPIS #1 start - INJ  condition [run until RWST empty]
0000797  787   and   -565  n  -1.0  * LPIS #2 start - INJ  condition [run until RWST empty]
*
* ACTUATION LOGIC FOR LPIS 1 & 2 - RECIRCULATION PHASE
*
* CASE A
*
0000791  566   and   -571  n  -1.0  * LPIS #1 start - REC condition [time > than RWST empty + switch DT4 & Time < DT5]
0000792  567   and   -572  n  -1.0  * LPIS #2 start - REC condition [time > than RWST empty + switch DT4 & Time < DT5]
0000793  568   and    573  n  -1.0  * HPIS #1 start - REC
0000794  569   and    574  n  -1.0  * HPIS #2 start - REC
0000795  570   and    575  n  -1.0  * HPIS #3 start - REC
*
0000798  791   or     580  n  -1.0  * LPIS #1 start - REC [time = 791 condition or time > DT6]
0000799  792   or     581  n  -1.0  * LPIS #2 start - REC [time = 792 condition or time > DT6]
*
0000651  798   and   -578  n  -1.0  * LPIS #1 start - REC  [time = 798 condition and condition for case A]
0000652  799   and   -579  n  -1.0  * LPIS #2 start - REC  [time = 799 condition and condition for case A]
*
0000677  651   and    682  n  -1.0  * LPIS #1 start - REC  [time = 651 condition and condition for case A]
0000678  652   and    683  n  -1.0  * LPIS #2 start - REC  [time = 652 condition and condition for case A]
*
*
* CASES B
*
0000653  566   and   -571  n  -1.0  * LPIS #1 start - REC  [condition Time>DT4 and Time < DT5]
0000654  567   and   -572  n  -1.0  * LPIS #2 start - REC  [condition Time>DT4 and Time < DT5]
*
0000655  653   or     580  n  -1.0  * LPIS #1 start - REC  [condition 653 and Time > DT6]
0000656  654   or     581  n  -1.0  * LPIS #2 start - REC  [condition 654 and Time > DT6]
*
0000657  655   and    578  n  -1.0  * LPIS #1 start - REC  [condition 655 and DT2 < Tswitch]
0000658  656   and    578  n  -1.0  * LPIS #2 start - REC  [condition 656 and DT2 < Tswitch]
*
0000659  657   and   -682  n  -1.0  * LPIS #1 start - REC [condition 657 and condition for B]
0000660  658   and   -683  n  -1.0  * LPIS #2 start - REC [condition 658 and condition for B]
*
*
* CASE C
*
*
0000661   576   and   -584  n  -1.0  * LPIS #1 start - REC [condition Time>DT3 and Time < DT5]
0000662   577   and   -585  n  -1.0  * LPIS #2 start - REC [condition Time>DT3 and Time < DT5]
*
0000663   661   or     586  n  -1.0  * LPIS #1 start - REC [condition 661 or  time > DT6]
0000664   662   or     587  n  -1.0  * LPIS #2 start - REC [condition 662 or  time > DT6]
*
0000665   663   and    578  n  -1.0  * LPIS #1 start - REC [condition 663 and DT2 < Tswitch]
0000666   664   and    578  n  -1.0  * LPIS #2 start - REC [condition 664 and DT2 < Tswitch]
*
0000667   665   and    680  n  -1.0  * LPIS #1 start - REC [condition 665 and condition for C]
0000668   666   and    680  n  -1.0  * LPIS #2 start - REC [condition 666 and condition for C]
*
* CASE D
*
*
0000669   566   and   -571  n  -1.0  * LPIS #1 start - REC [condition Time>DT4 and Time < DT5]
0000670   567   and   -572  n  -1.0  * LPIS #2 start - REC [condition Time>DT4 and Time < DT5]
*
0000671   669   or     580  n  -1.0  * LPIS #1 start - REC [condition 669 or  time > DT6]
0000672   670   or     581  n  -1.0  * LPIS #2 start - REC [condition 670 or  time > DT6]
*
0000673   671   and    578  n  -1.0  * LPIS #1 start - REC [condition 663 and DT2 < Tswitch]
0000674   672   and    578  n  -1.0  * LPIS #2 start - REC [condition 664 and DT2 < Tswitch]
*
0000675   673   and   -680  n  -1.0  * LPIS #1 start - REC [condition 665 and condition for D]
0000676   674   and   -680  n  -1.0  * LPIS #2 start - REC [condition 666 and condition for D]
*
*
*
* RUNNING LPIS - REC
*
*
0000685  677   or    659  n  -1.0  * LPIS #1 start - REC [run if A or B]
0000686  678   or    660  n  -1.0  * LPIS #2 start - REC [run if A or B]
*
0000687  685   or    667  n  -1.0  * LPIS #1 start - REC [run if A,B or C]
0000688  686   or    668  n  -1.0  * LPIS #2 start - REC [run if A,B or C]
*
0000689  687   or    675  n  -1.0  * LPIS #1 start - REC [run if A,B,C or D]
0000690  688   or    676  n  -1.0  * LPIS #2 start - REC [run if A,B,C or D]
*
*
*
* GENERAL CONDITION FOR CASE C, D
*
0000680   -582   and    -582  n  -1.0  * condition for C
0000681   -583   and    -583  n  -1.0  * condition for C
*
* GENERAL CONDITION FOR CASE B
*
0000682   -588   and    -588  n  -1.0  * condition for B
0000683   -589   and    -589  n  -1.0  * condition for B
*
*
*  HYDRODYNAMIC   COMPONENTS  -  CCCXXNN
*
***********************************************************
*  the   100   series   cards are   the   reactor  vessel   *
***********************************************************
*  components  100-104  model the   downcomer   *
************************************************
*  component   100   models   the   upper annulus  *
********************************************
1000000  upprann  snglvol
********************************************
1000101  0.0   5.4355   104.1901 0.0   90.0  5.4355
1000102  0.00015  1.1526   00
1000200  3  2286.75  543.75
********************************************
*  1000000  upprann  pipe
********************************************
*  1000001  2
*  1000101  0.0   2
*  1000301  2.1022   1
*  1000302  3.3333   2
*  1000401  57.6325  1
*  1000402  46.5576  2
*  1000601  90.0  2
*  1000701  2.1022   1
*  1000702  3.3333   2
*  1000801  0.00015  1.4275   1
*  1000802  0.00015  0.9793   2
*  1000901  0.0   0.0   1
*  1001001  00 2
*  1001101  00000 1
*  1001201  3  2287.2   543.45   0.0   0.0   0.0   1
*  1001202  3  2286.3   544.03   0.0   0.0   0.0   2
*  1001300  0
*  1001301  0.11362  0.11362  0.0   1
*************************************************
*  component   102   models   the   inlet annulus  *
*************************************************
1020000  inlt1 branch
********************************************
1020001  6  0
1020101  0.0   2.8685   63.0505  0.0   90.0  2.8685
1020102  0.00015  1.4987   00
1020200  3  2288.0   542.98
1021101  222010000   102010003   6.4625   2.0   0.04  00001
1022101  322010000   102010003   6.4625   2.0   0.04  00001
1023101  422010000   102010003   6.4625   2.0   0.04  00001
1024101  102000000   104000000   0.0   0.0   0.0   00000
1025101  102010000   100000000   0.0   0.0   0.0   00000
1026101  102010000   172010000   0.07580  0.7   0.7   00003
1021201  31.0  31.0  0.0
1022201  31.0  31.0  0.0
1023201  31.0  31.0  0.0
1024201  26.701   26.701   0.0
1025201  0.072 0.072 0.0
1026201  32.762   32.762   0.0
*************************************************
*  7020000  inlt2 branch
********************************************
*  7020001  4  0
*  7020101  0.0   2.8685   21.0168  0.0   90.0  2.8685
*  7020102  0.00015  1.4987   00
*  7020200  3  2288.0   542.98
*  7021101  322010000   702010003   6.4625   2.0   0.04  00001
*  7022101  702000000   704000000   0.0   0.0   0.0   00000
*  7023101  702010000   100000000   0.0   0.0   0.0   00000
*  7024101  702010000   172010000   0.02527  0.7   0.7   00003
*  7021201  31.0  31.0  0.0
*  7022201  26.701   26.701   0.0
*  7023201  0.072 0.072 0.0
*  7024201  32.762   32.762   0.0
*************************************************
*  8020000  intl3 branch
********************************************
*  8020001  4  0
*  8020101  0.0   2.8685   21.0168  0.0   90.0  2.8685
*  8020102  0.00015  1.4987   00
*  8020200  3  2288.0   542.98
*  8021101  422010000   802010003   6.4625   2.0   0.04  00001
*  8022101  802000000   804000000   0.0   0.0   0.0   00000
*  8023101  802010000   100000000   0.0   0.0   0.0   00000
*  8024101  802010000   172010000   0.02527  0.7   0.7   00003
*  8021201  31.0  31.0  0.0
*  8022201  26.701   26.701   0.0
*  8023201  0.072 0.072 0.0
*  8024201  32.762   32.762   0.0
********************************************
*  component   104   models   the   downcomer   *
********************************************
1040000  downcm1  pipe
********************************************
1040001  6
1040101  0.0   6
1040301  3.0513   1
1040302  2.40  4
1040303  4.8   5
1040304  4.327 6
1040401  75.0398  1
1040402  53.1455  4
1040403  106.29   5
1040404  120.0449 6
1040601  -90.0 6
1040701  -3.0513  1
1040702  -2.40 4
1040703  -4.8  5
1040704  -4.327   6
1040801  0.00015  0.8071   1
1040802  0.00015  0.5778   4
1040803  0.00015  0.5778   5
1040804  0.00015  1.0744   6
1040901  0.0   0.0   5
1041001  00 6
1041101  00000 5
1041201  3  2286.0   542.98   0.0   0.0   0.0   2
1041202  3  2286.6   542.98   0.0   0.0   0.0   3
1041203  3  2287.2   542.98   0.0   0.0   0.0   4
1041204  3  2287.5   542.99   0.0   0.0   0.0   5
1041205  3  2290.5   543.00   0.0   0.0   0.0   6
1041300  0
1041301  26.504   26.504   0.0   5
*************************************************
*  7040000  downcm2  pipe
********************************************
*  7040001  7
*  7040101  0.0   7
*  7040301  3.0513   1
*  7040302  2.40  6
*  7040303  4.327 7
*  7040401  25.013   1
*  7040402  17.715   6
*  7040403  40.015   7
*  7040601  -90.0 7
*  7040701  -3.0513  1
*  7040702  -2.40 6
*  7040703  -4.327   7
*  7040801  0.00015  0.8071   1
*  7040802  0.00015  0.5778   6
*  7040803  0.00015  1.0744   7
*  7040901  0.0   0.0   6
*  7041001  00 7
*  7041101  00000 6
*  7041201  3  2286.0   542.98   0.0   0.0   0.0   2
*  7041202  3  2286.6   542.98   0.0   0.0   0.0   3
*  7041203  3  2287.2   542.98   0.0   0.0   0.0   4
*  7041204  3  2287.7   542.99   0.0   0.0   0.0   5
*  7041205  3  2288.3   542.99   0.0   0.0   0.0   6
*  7041206  3  2290.5   543.00   0.0   0.0   0.0   7
*  7041300  0
*  7041301  26.504   26.504   0.0   6
*************************************************
*  8040000  downcm3  pipe
********************************************
*  8040001  7
*  8040101  0.0   7
*  8040301  3.0513   1
*  8040302  2.40  6
*  8040303  4.327 7
*  8040401  25.013   1
*  8040402  17.715   6
*  8040403  40.015   7
*  8040601  -90.0 7
*  8040701  -3.0513  1
*  8040702  -2.40 6
*  8040703  -4.327   7
*  8040801  0.00015  0.8071   1
*  8040802  0.00015  0.5778   6
*  8040803  0.00015  1.0744   7
*  8040901  0.0   0.0   6
*  8041001  00 7
*  8041101  00000 6
*  8041201  3  2286.0   542.98   0.0   0.0   0.0   2
*  8041202  3  2286.6   542.98   0.0   0.0   0.0   3
*  8041203  3  2287.2   542.98   0.0   0.0   0.0   4
*  8041204  3  2287.7   542.99   0.0   0.0   0.0   5
*  8041205  3  2288.3   542.99   0.0   0.0   0.0   6
*  8041206  3  2290.5   543.00   0.0   0.0   0.0   7
*  8041300  0
*  8041301  26.504   26.504   0.0   6
*************************************************
*  8100000  dcjun mtpljun
*  8100001  14 0
*  8100011  104010004   704010003   0.0   0.0   0.0   00000000 1.0   1.0   1.0
*  8100012  10000 10000 0  7
*  8100021  704010004   804010003   0.0   0.0   0.0   00000000 1.0   1.0   1.0
*  8100022  10000 10000 0  14
*  8101011  0.0   0.0   14
*
*************************************************
*  component   106   models   the   lower head  *
*************************************************
1060000  lowerhd  branch
********************************************
1060001  2  0
1060101  0.0   5.7267   436.5539 0.0   90.0  5.7267
1060102  0.00015  3.4957   00
1060200  3  2294.4   543.02
1061101  104010000   106010000   0.0   0.0   0.0   00000
1062101  106010000   108000000   46.237   0.0   0.0   00100
1061201  21.154   21.154   0.0
1062201  8.4252   8.4252   0.0
*************************************************
*  component   108   models   the   lower plenum   *
*************************************************
1080000  lowplnm  branch
********************************************
1080001  4  0
1080101  0.0   4.327 301.4 0.0   90.0  4.327
1080102  0.00015  2.872 00
1080200  3  2294.4   543.02
1081101  108010000   111000000   0.0   8.0   8.0   00100
1082101  108010000   112000000   0.0   8.0   8.0   00100
1083101  108010000   113000000   0.0   8.0   8.0   00100
1084101  108010000   118000000   1.0   8.0   8.0   00100
1081201  13.4  13.4  0.0
1082201  13.4  13.4  0.0
1083201  13.4  13.4  0.0
1084201  8.4   8.4   0.0
*************************************************
*  component   111   models   the   center   core  channel  *
********************************************
1110000  cencore  pipe
********************************************
1110001  10
1110101  6.825 10
1110301  1.200 10
1110401  0.0   10
1110601  90.0  10
1110701  1.200 10
1110801  0.000005 0.04335  10
1110901  0.0   0.0   1
1110902  0.22  0.22  2
1110903  0.0   0.0   3
1110904  0.22  0.22  6
1110905  0.0   0.0   7
1110906  0.22  0.22  8
1110907  0.0   0.0   9
1111001  00 10
1111101  00000 9
1111201  3  2275.0   577.0 0.0   0.0   0.0   10
1111300  0
1111301  14.0  14.0  0.0   9
*************************************************
*  component   112   models   the   middle   core  channel  *
********************************************
1120000  midcore  pipe
********************************************
1120001  10
1120101  26.208   10
1120301  1.200 10
1120401  0.0   10
1120601  90.0  10
1120701  1.200 10
1120801  0.000005 0.04335  10
1120901  0.0   0.0   1
1120902  0.22  0.22  2
1120903  0.0   0.0   3
1120904  0.22  0.22  6
1120905  0.0   0.0   7
1120906  0.22  0.22  8
1120907  0.0   0.0   9
1121001  00 10
1121101  00000 9
1121201  3  2275.0   577.0 0.0   0.0   0.0   10
1121300  0
1121301  14.0  14.0  0.0   9
************************************************
*  component   113   models   the   outer core  channel  *
********************************************
1130000  outcore  pipe
********************************************
1130001  10
1130101  9.828 10
1130301  1.200 10
1130401  0.0   10
1130601  90.0  10
1130701  1.200 10
1130801  0.000005 0.04142  10
1130901  0.0   0.0   1
1130902  0.22  0.22  2
1130903  0.0   0.0   3
1130904  0.22  0.22  6
1130905  0.0   0.0   7
1130906  0.22  0.22  8
1130907  0.0   0.0   9
1131001  00 10
1131101  00000 9
1131201  3  2275.0   577.0 0.0   0.0   0.0   10
1131300  0
1131301  14.0  14.0  0.0   9
************************************************
*  components  120-129  connect  the   center   and   middle   core  channels *
********************************************
1200000  midcenj1 sngljun
********************************************
1200101  112010004   111010003   5.981 32.04 32.04 00003
1200201  0  0.0   0.0   0.0
************************************************
1210000  midcenj2 sngljun
********************************************
1210101  112020004   111020003   5.981 32.04 32.04 00003
1210201  0  0.0   0.0   0.0
************************************************
1220000  midcenj3 sngljun
********************************************
1220101  112030004   111030003   5.981 32.04 32.04 00003
1220201  0  0.0   0.0   0.0
************************************************
1230000  midcenj4 sngljun
********************************************
1230101  112040004   111040003   5.981 32.04 32.04 00003
1230201  0  0.0   0.0   0.0
************************************************
1240000  midcenj5 sngljun
********************************************
1240101  112050004   111050003   5.981 32.04 32.04 00003
1240201  0  0.0   0.0   0.0
************************************************
1250000  midcenj6 sngljun
********************************************
1250101  112060004   111060003   5.981 32.04 32.04 00003
1250201  0  0.0   0.0   0.0
************************************************
1260000  midcenj7 sngljun
********************************************
1260101  112070004   111070003   5.981 32.04 32.04 00003
1260201  0  0.0   0.0   0.0
************************************************
1270000  midcenj8 sngljun
********************************************
1270101  112080004   111080003   5.981 32.04 32.04 00003
1270201  0  0.0   0.0   0.0
************************************************
1280000  midcenj9 sngljun
********************************************
1280101  112090004   111090003   5.981 32.04 32.04 00003
1280201  0  0.0   0.0   0.0
************************************************
1290000  midcenjt sngljun
********************************************
1290101  112100004   111100003   5.981 32.04 32.04 00003
1290201  0  0.0   0.0   0.0
************************************************
*  components  130-139  connect  the   middle   and   outer core  channels *
********************************************
1300000  outmidj1 sngljun
********************************************
1300101  113010004   112010003   11.107   16.02 16.02 00003
1300201  0  0.0   0.0   0.0
************************************************
1310000  outmidj2 sngljun
********************************************
1310101  113020004   112020003   11.107   16.02 16.02 00003
1310201  0  0.0   0.0   0.0
************************************************
1320000  outmidj3 sngljun
********************************************
1320101  113030004   112030003   11.107   16.02 16.02 00003
1320201  0  0.0   0.0   0.0
************************************************
1330000  outmidj4 sngljun
********************************************
1330101  113040004   112040003   11.107   16.02 16.02 00003
1330201  0  0.0   0.0   0.0
************************************************
1340000  outmidj5 sngljun
********************************************
1340101  113050004   112050003   11.107   16.02 16.02 00003
1340201  0  0.0   0.0   0.0
************************************************
1350000  outmidj6 sngljun
********************************************
1350101  113060004   112060003   11.107   16.02 16.02 00003
1350201  0  0.0   0.0   0.0
************************************************
1360000  outmidj7 sngljun
********************************************
1360101  113070004   112070003   11.107   16.02 16.02 00003
1360201  0  0.0   0.0   0.0
************************************************
1370000  outmidj8 sngljun
********************************************
1370101  113080004   112080003   11.107   16.02 16.02 00003
1370201  0  0.0   0.0   0.0
************************************************
1380000  outmidj9 sngljun
********************************************
1380101  113090004   112090003   11.107   16.02 16.02 00003
1380201  0  0.0   0.0   0.0
************************************************
1390000  outmidjt sngljun
********************************************
1390101  113100004   112100003   11.107   16.02 16.02 00003
1390201  0  0.0   0.0   0.0
*************************************************
*  component   118   models   the   core  bypass   *
********************************************
1180000  corbyps  pipe
********************************************
1180001  5
1180101  15.850   5
1180201  1.0   4
1180301  2.400 5
1180401  0.0   5
1180601  90.0  5
1180701  2.400 5
1180801  0.00015  0.8194   5
1180901  0. 0.0   4
1181001  00 5
1181101  00100 4
1181201  3  2276.0   543.2 0.0   0.0   0.0   5
1181300  0
1181301  0.8124   0.8124   0.0   4
*******************************************************
*  components  151   through  174   model the   upper plenum   volumes  *
*************************************************
*  components  151-154  model the   center   channel  of the   upper plenum
********************************************
1510000  cenupv1  branch
********************************************
1510001  2  0
1510101  0.0   3.0513   39.2  0.0   90.0  3.0513
1510102  0.00015  2.88  00
1510200  3  2250.0   606.0
1511101  111010000   151000000   0.0   8.0   8.0   00100
1512101  161010004   151010003   60.28 5.0   5.0   00003
1511201  15.0  15.0  0.0
1512201  0.0   0.0   0.0
********************************************
1520000  cenupv2  branch
********************************************
1520001  3  0
1520101  0.0   2.8685   26.91 0.0   90.0  2.8685
1520102  0.00015  1.52  00
1520200  3  2250.0   606.0
1521101  151010000   152000000   0.0   0.0   0.0   00000
1522101  162010004   152010003   56.66 5.0   5.0   00003
1523101  181010000   152010003   0.0   0.0   0.0   00101
1521201  8.0   8.0   0.0
1522201  -10.0 -10.0 0.0
1523201  0.0   0.0   0.0
********************************************
1530000  cenupv3  branch
********************************************
1530001  2  0
1530101  8.70  2.1022   0.0   0.0   90.0  2.1022
1530102  0.00015  1.04  00
1530200  3  2250.0   606.0
1531101  152010000   153000000   0.0   0.0   0.0   00000
1532101  163010004   153010003   41.53 5.0   5.0   00003
1531201  0.0   0.0   0.0
1532201  0.0   0.0   0.0
********************************************
1540000  cenupv4  branch
********************************************
1540001  1  0
1540101  8.70  2.9585   0.0   0.0   90.0  2.9585
1540102  0.00015  1.04  00
1540200  3  2250.0   606.0
1541101  153010000   154000000   0.0   0.0   0.0   00000
1541201  0.0   0.0   0.0
*************************************************
*  components  161-164  model the   middle   channel  of the   upper plenum
********************************************
1610000  midupv1  branch
********************************************
1610001  2  0
1610101  0.0   3.0513   151.1 0.0   90.0  3.0513
1610102  0.00015  3.03  00
1610200  3  2250.0   606.0
1611101  112010000   161000000   0.0   8.0   8.0   00100
1612101  171010004   161010003   111.9 7.0   7.0   00003
1611201  15.0  15.0  0.0
1612201  0.0   0.0   0.0
********************************************
1620000  midupv2  branch
********************************************
1620001  3  0
1620101  0.0   2.8685   103.13   0.0   90.0  2.8685
1620102  0.00015  1.55  00
1620200  3  2250.0   606.0
1621101  161010000   162000000   0.0   0.0   0.0   00000
1622101  172010004   162010003   105.2 7.0   7.0   00003
1623101  182010000   162010003   0.0   0.0   0.0   00101
1621201  8.0   8.0   0.0
1622201  -10.0 -10.0 0.0
1623201  0.0   0.0   0.0
********************************************
1630000  midupv3  branch
********************************************
1630001  2  0
1630101  33.22 2.1022   0.0   0.0   90.0  2.1022
1630102  0.00015  1.03  00
1630200  3  2250.0   606.0
1631101  162010000   163000000   0.0   0.0   0.0   00000
1632101  173010004   163010003   77.12 7.0   7.0   00003
1631201  0.0   0.0   0.0
1632201  0.0   0.0   0.0
********************************************
1640000  midupv4  branch
********************************************
1640001  1  0
1640101  33.22 2.9585   0.0   0.0   90.0  2.9585
1640102  0.00015  1.03  00
1640200  3  2250.0   606.0
1641101  163010000   164000000   0.0   0.0   0.0   00000
1641201  0.0   0.0   0.0
*************************************************
*  components  171-174  model the   outer channel  of the   upper plenum
********************************************
1710000  outupv1  branch
********************************************
1710001  2  0
1710101  0.0   3.0513   110.8 0.0   90.0  3.0513
1710102  0.00015  2.66  00
1710200  3  2250.0   606.0
1711101  113010000   171000000   0.0   8.0   8.0   00100
1712101  118010000   171000000   0.0   8.0   8.0   00100
1711201  15.0  15.0  0.0
1712201  0.8   0.8   0.0
********************************************
1720000  outupv2  branch
********************************************
1720001  5  0
1720101  0.0   2.8685   98.64 0.0   90.0  2.8685
1720102  0.00015  2.12  00
1720200  3  2250.0   606.0
1721101  171010000   172000000   0.0   0.0   0.0   00000
1723101  183010000   172010003   0.0   0.0   0.0   00101
1724101  172010003   200000000   0.0   0.2   1.0   00002
1725101  172010003   300000000   0.0   0.2   1.0   00002
1726101  172010003   400000000   0.0   0.2   1.0   00002
1721201  8.0   8.0   0.0
1723201  0.0   0.0   0.0
1724201  47.6  47.6  0.0
1725201  47.6  47.6  0.0
1726201  47.6  47.6  0.0
********************************************
1730000  outupv3  branch
********************************************
1730001  1  0
1730101  33.78 2.1022   0.0   0.0   90.0  2.1022
1730102  0.00015  1.93  00
1730200  3  2250.0   606.0
1731101  172010000   173000000   0.0   0.0   0.0   00000
1731201  0.0   0.0   0.0
********************************************
1740000  outupv4  branch
********************************************
1740001  1  0
1740101  33.78 2.9585   0.0   0.0   90.0  2.9585
1740102  0.00015  1.93  00
1740200  3  2250.0   606.0
1741101  173010000   174000000   0.0   0.0   0.0   00000
1741201  0.0   0.0   0.0
*************************************************
*  components  181   through  183   model the   control  assembly housings *
*************************************************
*  component   181   models   the   housings in the   center   channel  *
********************************************
1810000  cenhsing snglvol
********************************************
1810101  0.0   6.86975  19.62 0.0   -90.0 -6.86975
1810102  0.00015  0.0492   00
1810200  3  2250.0   550.0
********************************************
*  component   182   models   the   housings in the   middle   channel  *
********************************************
1820000  midhsing snglvol
********************************************
1820101  0.0   6.86975  78.48 0.0   -90.0 -6.86975
1820102  0.00015  0.0492   00
1820200  3  2250.0   550.0
********************************************
*  component   183   models   the   housings in the   outer channel  *
********************************************
1830000  outhsing snglvol
********************************************
1830101  0.0   6.86975  17.44 0.0   -90.0 -6.86975
1830102  0.00015  0.0492   00
1830200  3  2250.0   550.0
*************************************************
*  component   190   models   the   upper head  *
*************************************************
1900000  upprhead branch
********************************************
1900001  4  0
1900101  0.0   6.0391   500.  0.0   90.0  6.0391
1900102  0.00015  3.50  00
1900200  3  2250.0   543.0
1901101  100010000   190000000   0.01674  0.0   0.0   00100
1902101  190000000   181000000   0.1215   0.0   0.0   00100
1903101  190000000   182000000   0.486 0.0   0.0   00100
1904101  190000000   183000000   0.108 0.0   0.0   00100
1901201  1.0   1.0   0.0
1902201  1.0   1.0   0.0
1903201  1.0   1.0   0.0
1904201  1.0   1.0   0.0
*************************************************
*************************************************
****************************************************************
*  primary  coolant  loop  components
****************************************************************
*  coolant  loop  a
****************************************************************
2000000  hla-vess pipe
2000001  2
2000101  4.587 2
2000301  0.0   2
2000401  23.330   2
2000601  0.0   2
2000701  0.0   2
2000801  .00015   0.0   2
2001001  00 2
2001101  00000 1
2001201  0  2234.9322   611.84403   1053.3231   0.0   0.0   1
2001202  0  2234.8026   611.84456   1053.3323   0.0   0.0   2
2001300  0
2001301  47.655407   47.655407   0.0   1
****************************************************************
2020000  hla-pzr  branch
2020001  1  0
2020101  4.587 0.0   40.082
2020102  0.0   0.0   0.0
2020103  .00015   0.0   00
2020200  0  2234.6309   611.84546   1053.3444   0.0
2021101  200010000   202000000   4.587 0.0   0.0   00000
2021201  47.655530   47.655530   0.0
****************************************************************
2030000  hla-stop sngljun
2030101  202010000   204000000   4.167 0.219 0.219 00000
2030201  0  52.459020   52.459020   0.0
****************************************************************
2040000  hla-sgin snglvol
2040101  0.0   6.292 38.14
2040102  0.0   14.79 1.606
2040103  .00015   0.0   00
2040200  0  2232.3070   611.84575   1053.5092   0.0
****************************************************************
2060000  sga-inpl branch
2060001  2  0
2060101  0.0   7.197 157.33
2060102  0.0   51.70 5.648
2060103  .00015   0.0   00
2060200  0  2240.2858   611.84038   1052.9441   0.0
2061101  204010000   206000000   5.241 0.0   0.0   00000
2062101  206010000   208000000   10.95 0.0   0.0   00000
2061201  41.710353   41.710353   0.0
2062201  19.961439   19.961439   0.0
****************************************************************
2080000  sga-tube pipe
2080001  8
2080101  10.95 8
2080301  8.49  3
2080302  9.91  5
2080303  8.49  8
2080401  0.0   8
2080601  90.0  3
2080602  58.95 4
2080603  -58.95   5
2080604  -90.0 8
2080701  8.49  4
2080702  -8.49 8
2080801  .00015   0.0646   8
2080901  0.0   0.0   3
2080902  0.048 0.048 4
2080903  0.0   0.0   7
2081001  00 8
2081101  00000 7
2081201  0  2232.1742   592.38674   1053.5186   0.0   0.0   1
2081202  0  2220.7340   577.14527   1054.3319   0.0   0.0   2
2081203  0  2209.4236   565.11849   1055.1394   0.0   0.0   3
2081204  0  2197.4887   554.54911   1055.9951   0.0   0.0   4
2081205  0  2187.5896   545.98591   1056.7077   0.0   0.0   5
2081206  0  2181.3061   539.51890   1057.1615   0.0   0.0   6
2081207  0  2175.8004   534.04688   1057.5599   0.0   0.0   7
2081208  0  2170.3601   529.37052   1057.9267   0.0   0.0   8
2081300  0
2081301  19.418115   19.418115   0.0   1
2081302  19.035939   19.035939   0.0   2
2081303  18.753128   18.753128   0.0   3
2081304  18.519150   18.519150   0.0   4
2081305  18.336644   18.336644   0.0   5
2081306  18.205045   18.205045   0.0   6
2081307  18.095997   18.095997   0.0   7
****************************************************************
2100000  sga-outp branch
2100001  2  0
2100101  0.0   7.197 157.33
2100102  0.0   -51.70   -5.648
2100103  .00015   0.0   00
2100200  0  2169.8338   529.33070   1057.9621   0.0
2101101  208010000   210000000   10.95 0.0   0.0   00000
2102101  210010000   212000000   5.241 0.0   0.0   00000
2101201  18.004118   18.004118   0.0
2102201  37.616349   37.616349   0.0
****************************************************************
2120000  rcpa-suc pipe
2120001  5
2120101  5.241 4
2120102  5.002 5
2120301  2.676 1
2120302  4.971 2
2120303  6.709 3
2120304  2.365 4
2120305  4.271 5
2120401  0.0   5
2120601  -67.04   1
2120602  -90.  2
2120603  -39.53   3
2120604  0.0   4
2120605  39.53 5
2120701  -2.464   1
2120702  -4.971   2
2120703  -4.271   3
2120704  0.0   4
2120705  4.271 5
2120801  .00015   0.0   5
2120901  .1136 .1136 1
2120902  .0803 .0803 3
2120903  .1463 .1463 4
2121001  00 5
2121101  00000 4
2121201  0  2164.3357   529.32635   1058.3135   0.0   0.0   1
2121202  0  2164.6742   529.31900   1058.2925   0.0   0.0   2
2121203  0  2165.5188   529.30910   1058.2400   0.0   0.0   3
2121204  0  2165.5676   529.30559   1058.2369   0.0   0.0   4
2121205  0  2162.9412   529.29941   1058.4003   0.0   0.0   5
2121300  0
2121301  37.618058   37.618058   0.0   1
2121302  37.617964   37.617964   0.0   2
2121303  37.617717   37.617717   0.0   3
2121304  39.415114   39.415114   0.0   4
****************************************************************
*  MCP   SEALS
****************************************************************
2130000  sealA valve          * standard choking model
2130101  216000000   215000000   8.1121E-04  0.0   0.0   00000100 1.0   1.0   1.0
2130201  1  0.0   0.0   0.0
2130300  mtrvlv
2130301  742   536   0.115420 0.0
****************************************************************
2150000  contA tmdpvol
2150101  1.0e6 .0 1.0e+06
2150102  .0 .0 .0
2150103  .0 .0 0000010
2150200  102
2150201  0.0   14.7  1.0
*
****************************************************************
2140000  rcpa  pump
2140101  0.0   5.829 56.
2140102  0.0   90.   5.829
2140103  0
2140108  212010000   5.002 0.0   0.0   00000
2140109  216000000   4.1247   0.0   0.0   00000
2140200  0  2242.4555   529.43232   1052.7907   0.0
****************************************************************
2140201  0  39.415978   39.415978   0.0
2140202  0  47.775937   47.775937   0.0
2140301  0  0  0  -1 0  518   0
2140302  1170.0000   1.1571279   88500.   280.
2140303  28015.   70000.   47.31 .0
2140304  0.0   280.15   28.015   0.0
***************************************************************
*  pump  speed controller  for   steady   state to produce  desired  *
*  mass  flow. to "disable,"  change   w5 on 2140301  to -1 and   *
*  delete   the   21461xx  cards.   *
***************************************************************
2146100  519   cntrlvar 4
2146101  -1.0  1170.0
2146102  0.0   0.0
2146103  1500. 1500.
**************************************************************
*  head  data  for   be/em pump--single   "phase," type  93a   *
*  ref;  relap4/mod5(1) built-in &  "wcap-8302,"   fig   2-18  *
**************************************************************
2141101  1  1  0.0   1.8   0.15  1.7   0.22  1.65  0.3   1.5   0.4   1.4
2141102  0.5   1.35  0.62  1.3   0.75  1.31  0.87  1.22  1.0   1.0
2141201  1  2  0.0   -1.55 0.15  -1.2  0.3   -0.85 0.53  -0.35 0.65  0
2141202  0.80  0.37  1.0   1.0
2141301  1  3  -1.0  4.2   -0.80 3.65  -0.69 3.30  -0.5  2.80  -0.30 2.3
2141302  -0.17 2.05  -0.08 1.85  0.0   1.80
2141401  1  4  -1.0  4.20  -0.80 3.60  -0.69 3.20  -0.58 2.85  -0.46 2.6
2141402  -0.3  2.20  -0.18 1.92  0.0   1.65
2141501  1  5  0.0   -0.16 0.10  -0.12 0.2   -0.06 0.28  0.00  0.40  0.09
2141502  0.60  0.31  0.70  0.42  0.8   0.50  0.88  0.54  1.00  .59
2141601  1  6  0.0   1.65  0.37  0.80  0.43  0.74  0.50  0.68  0.58  0.64
2141602  0.64  0.62  0.70  0.61  1.00  0.59
2141701  1  7  -1.0  0.0   0.0   0.0
2141801  1  8  -1.0  0.0   0.0   0.0
**************************************************************
*  torque   data  for   be/em pump--single   "phase," type  93a   *
*  ref;  relap4/mod5(1) built-in &  "wcap-8302,"   fig   2-19  *
**************************************************************
2141901  2  1  0.0   0.95  0.5   0.98  1.00  1.00
2142001  2  2  0.0   -1.4  0.15  -1.0  0.4   -0.31 0.74  0.4   1.0   1.0
2142101  2  3  -1.0  2.98  -0.78 2.40  -0.50 1.65  -0.38 1.35  -.27  1.2
2142102  -0.16 1.0   -0.07 0.97  0.0   0.95
2142201  2  4  -1.0  2.98  -0.77 2.4   -0.55 2.0   -0.4  1.75  -.3   1.65
2142202  -0.2  1.55  -0.1  1.52  0.0   1.5
2142301  2  5  0.0   -1.0  0.25  -0.6  0.4   -0.37 0.5   -0.25 0.6   -0.16
2142302  0.8   -0.01 1.0   0.11
2142401  2  6  0.0   1.5   0.6   0.61  0.8   0.35  1.0   0.11
2142501  2  7  -1.0  0.0   0.0   0.0
2142601  2  8  -1.0  0.0   0.0   0.0
**************************************************************
*  pump  head  multiplier  data  --m3(a)  *
*  ref;  srd-113-76  "interim,"  vol   "1,"  page  199   *
**************************************************************
2143001  0  0. 0. .1 0. .15   .05   .24   .8 .3 .96   0.4   0.98  0.6   0.97
2143002  .8 .9 .9 .8 .96   .5 1.0   0.
**************************************************************
*  pump  torque   multiplier  data--n(a)  *
*  ref;  letter   pml-267-74. *
**************************************************************
2143101  0  0. 0. 0.1   0. 0.15  0.05  0.24  0.56  0.8   0.56  0.96  0.45
+  1.0   0.0
**************************************************************
*  two   phase difference  curves   *
*  ref;  relap4/mod5(1) built-in from  *
*  "geg-1-74," table 2  revised  *
**************************************************************
2144101  1  1  0.0   0.0   0.1   0.83  0.2   1.09  0.5   1.02  0.70  1.01
2144102  0.9   0.94  1.0   1.0
2144201  1  2  0.0   0.0   0.1   -0.04 0.2   0.0   0.3   0.1   0.4   0.21
2144202  0.8   0.67  0.9   0.80  1.0   1.0
2144301  1  3  -1.0  -1.16 -0.9  -1.24 -0.8  -1.77 -0.70 -2.36 -0.6  -2.79
2144302  -0.5  -2.91 -0.4  -2.67 -0.25 -1.69 -0.10 -0.50 0.0   0.0
2144401  1  4  -1.0  -1.16 -0.9  -0.78 -0.8  -0.5  -0.7  -0.31 -0.6  -0.17
2144402  -0.5  -0.08 -0.35 0.0   -0.2  0.05  -0.1  0.08  0.0   0.11
2144501  1  5  0.0   0.0   0.2   -0.34 0.4   -0.65 0.6   -0.93 0.8   -1.19
2144502  1.0   -1.47
2144601  1  6  0.0   0.11  0.1   0.13  0.25  0.15  0.4   0.13  0.5   0.07
2144602  0.6   -0.04 0.7   -0.23 0.8   -0.51 0.9   -0.91 1.0   -1.47
2144701  1  7  -1.0  0.0   0.0   0.0
2144801  1  8  -1.0  0.0   0.0   0.0
**************************************************************
*  torque   difference  curves   *
*  *
*  torque   difference  curves   input uses  single   "phase," type  93a   *
*  data  to conform  to torque   multiplier  derivation  from  *
*  semiscale.must use   this  until fully degraded 2-phase  *
*  torque   data  available   for   93-a  pump. *
**************************************************************
2144901  2  1  0.0   0.95  0.5   0.98  1.00  1.00
2145001  2  2  0.0   -1.4  0.15  -1.0  0.4   -0.31 0.74  0.4   1.0   1.0
2145101  2  3  -1.0  2.98  -0.78 2.40  -0.50 1.65  -0.38 1.35  -.27  1.2
2145102  -0.16 1.0   -0.07 0.97  0.0   0.95
2145201  2  4  -1.0  2.98  -0.77 2.4   -0.55 2.0   -0.4  1.75  -.3   1.65
2145202  -0.2  1.55  -0.1  1.52  0.0   1.5
2145301  2  5  0.0   -1.0  0.25  -0.6  0.4   -0.37 0.5   -0.25 0.6   -0.16
2145302  0.8   -0.01 1.0   0.11
2145401  2  6  0.0   1.5   0.6   0.61  0.8   0.35  1.0   0.11
2145501  2  7  -1.0  0.0   0.0   0.0
2145601  2  8  -1.0  0.0   0.0   0.0
****************************************************************
2160000  cla-rcp1 pipe
2160001  2
2160101  0.0   2
2160301  3.293 2
2160401  15.662   2
2160601  0.0   2
2160701  0.0   2
2160801  0.00015  0.0   2
2160901  0.0   0.0   1
2161001  0000000  2
2161101  00000000 1
2161201  000   2310.2434   529.42862   1048.0217   0.0   0.0   2
2161300  0
2161301  41.41015 41.41015 0.0   1
****************************************************************
2170000  cla-stop sngljun
2170101  216010000   218000000   4.167 0.219 0.219 00000
2170201  0  47.264901   47.264901   0.0
****************************************************************
2180000  cla-ecc1 pipe
2180001  2
2180101  0.0   2
2180301  4.043 2
2180401  18.7555  2
2180601  0.0   2
2180701  0.0   2
2180801  0.00015  0.0   2
2180901  0.0   0.0   1
2181001  0000000  2
2181101  00000000 1
2181201  000   2310.2434   529.42862   1048.0217   0.0   0.0   2
2181300  0
2181301  42.45583 42.45583 0.0   1
****************************************************************
2200000  cla-ecc2 branch
2200001  2  0
2200101  4.125 4.043 0.0
2200102  0.0   0.0   0.0
2200103  .00015   0.0   00
2200200  0  2304.5001   529.40911   1048.4551   0.0
2201101  218010000   220000000   0.0   0.0   0.0   00000
2202101  220010000   221000000   0.0   0.0   0.0   00000
2201201  47.747364   47.747364   0.0
2202201  47.747364   47.747364   0
****************************************************************
2210000  cla-vex1 snglvol
2210101  4.125 4.043 0.0
2210102  0.0   0.000 0
2210103  0.00015  0.0   00
2210200  0  2304.5001   529.40911   1048.4551   0.0
****************************************************************
2190000  un1   sngljun
2190101  221010000   222000000   4.125 0.0   0.0   00000
2190201  0  47.264901   47.264901   0.0
****************************************************************
*  2220000  cla-vess pipe
*  2220001  2
*  2220101  0.0   2
*  2220301  3.020 2
*  2220401  14.7335  2
*  2220601  0.0   2
*  2220701  0.0   2
*  2220801  0.00015  0.0   2
*  2220901  0.0   0.0   1
*  2221001  0000000  2
*  2221101  00000000 1
*  2221201  000   2306.378 529.40117   1048.3254   0.0   0.0   2
*  2221300  0
*  2221301  40.37042 40.37042 0.0   1
****************************************************************
2220000  cla-vess snglvol
2220101  0.0   6.04  29.467
2220102  0.0   0.0   0.0
2220103  .00015   0.0   00
2220200  0  2306.378 529.40117   1048.3254   0.0
****************************************************************
*
*  ECCS
*
*  ACCUMULATOR
****************************************************************
2500000  accuA accum
2500101  0.0   15.15 1450.0
2500102  0.0   -90.0 -15.15
2500103  0.00015  0.0   0000000  0
2500200  580.0 105.0 0.0
2501101  220000000   0.601 5.65  5.65  00000000
2502200  1000.0   0.0   104.38   44.0  0.229
+  0  0.0   0.0   641
*
*
*  steam generator   a  secondary   side  components
*
****************************************************************
2700000  sga-dc1  snglvol
2700101  0.0   6.936 700.
2700102  0.0   -90.  -6.936
2700103  .00015   0.0   00
2700200  0  830.89101   502.28620   1116.1629   .50582284
****************************************************************
2720000  sga-dc2  branch
2720001  2  0
2720101  0.0   3.917 340.
2720102  0.0   -90.  -3.917
2720103  .00015   0.0   00
2720200  0  832.13033   495.84672   1114.4162   0.0
2721101  270010000   272000000   0.0   0.0   0.0   00000
2722101  272010000   274000000   0.0   0.12  0.64  00000
2721201  -2.5831-3   -5.837087   0.0
2722201  7.7543662   7.7543662   0.0
****************************************************************
2740000  sga-dc3  pipe
2740001  4
2740101  0.0   4
2740301  11.02 1
2740302  8.49  3
2740303  6.74  4
2740401  191.8 1
2740402  60.19 3
2740403  47.79 4
2740601  -90.  4
2740701  -11.02   1
2740702  -8.49 3
2740703  -6.74 4
2740801  .00015   0.43  4
2741001  00 4
2741101  00000 3
2741201  0  834.24330   495.75817   1114.3737   0.0   0.0   1
2741202  0  835.59224   495.78343   1114.3466   0.0   0.0   2
2741203  0  837.87411   495.80544   1114.3009   0.0   0.0   3
2741204  0  839.92136   495.80840   1114.2599   0.0   0.0   4
2741300  0
2741301  19.038383   19.038383   0.0   1
2741302  19.039778   19.039778   0.0   2
2741303  19.040957   19.040957   0.0   3
****************************************************************
2750000  sga-boli sngljun
2750101  274010000   276000000   37.79 46.8  46.8  00000
2750201  0  3.5722534   3.8225232   0.0
****************************************************************
2760000  sga-boil pipe
2760001  5
2760101  0.0   5
2760301  6.74  1
2760302  8.49  3
2760303  11.02 4
2760304  3.917 5
2760401  369.98   1
2760402  466.04   3
2760403  658.2 4
2760404  260.  5
2760601  +90.  5
2760701  6.74  1
2760702  8.49  3
2760703  11.02 4
2760704  3.917 5
2760801  .00015   0.140 5
2760901  16.9  16.9  1
2760902  6.0   6.0   3
2760903  0.0   0.0   4
2761001  00 5
2761101  00000 4
2761201  0  838.75235   509.75315   1114.2699   .34276726   0.0   1
2761202  0  836.40283   512.35263   1114.3137   .55862754   0.0   2
2761203  0  834.61317   512.55191   1114.3483   .66937467   0.0   3
2761204  0  832.80432   512.28988   1114.3973   .76660522   0.0   4
2761205  0  831.91737   512.14272   1114.4181   .77982352   0.0   5
2761300  0
2761301  3.6880988   5.6155472   0.0   1
2761302  5.2851929   8.0286252   0.0   2
2761303  6.7647608   10.423870   0.0   3
2761304  8.3970469   11.517427   0.0   4
****************************************************************
2780000  sga-swrl separatr
2780001  3  0
2780101  0.0   6.936 350.
2780102  0.0   90.   6.936
2780103  .00015   4.67  00
2780200  0  830.83376   512.07459   1114.3647   .53829806
2781101  278010000   280000000   0.0   1.93  0.0   00000 0.02
2782101  278000000   272000000   0.0   0.0   9.67  00000 0.80
2783101  276010000   278000000   0.0   0.967 0.967 00000
2781201  -9.669464   14.676371   0.0
2782201  5.6190925   -2.963842   0.0
2783201  14.736148   18.799016   0.0
****************************************************************
2800000  sga-dryr branch
2800001  2  0
2800101  0.0   9.470 1450.
2800102  0.0   90.   9.470
2800103  .00015   0.0   00
2800200  0  830.26239   511.86773   1114.4516   .99999995
2801101  280010000   282000000   0.0   0.0   0.0   00000
2802101  280000000   270000000   0.0   0.0   0.0   00000
2801201  5.9512255   6.3173370   0.0
2802201  8.1162296   3.00563-3   0.0
****************************************************************
2820000  sga-dome snglvol
2820101  0.0   5.107 467.96
2820102  0.0   90.   5.107
2820103  .00015   0.0   00
2820200  0  830.16123   511.85091   1114.4887   .99999970
****************************************************************
2840000  sga-stml branch
2840001  1  0
2840101  4.5869   0.0   824.2
2840102  0.0   -12.98   -40.35
2840103  .00015   0.0   00
2840200  0  825.93722   511.14924   1114.4480   .99998780
2841101  282010000   284000000   0.0   0.5   1.0   00000
2841201  89.848360   115.13716   0.0
****************************************************************
2850000  sga-msiv valve
2850101  284010000   286000000   4.5869   0.0   0.0   00100
2850201  0  102.20364   115.59958   0.0
2850300  trpvlv
2850301  505
****************************************************************
*  following   cards replace  the   msiv  with  a  steam control  valve used
*  to regulate secondary   pressure in response to cold  leg   temp. error
****************************************************************
2850300  srvvlv
2850301  71
****************************************************************
2870000  sga-porv valve
2870101  284010000   288000000   0.04565  0.0   0.0   00100
2870201  0  0.0   0.0   0.0
2870300  mtrvlv
2870301  707   709   0.556 0.0
****************************************************************
2890000  sga-sfty valve
2890101  284010000   290000000   0.39985  0.0   0.0   00100
2890201  0  0.0   0.0   0.0
2890300  mtrvlv
2890301  716   718   20.0  0.0
****************************************************************
2860000  sga-htdv tmdpvol
2860101  1.0e6 .0 1.0e+06
2860102  .0 .0 .0
2860103  .0 .0 0000010
2860200  102
2860201  0.0   785.0 1.0
****************************************************************
2880000  sga-ptdv tmdpvol
2880101  1.0e6 .0 1.0e+06
2880102  .0 .0 .0
2880103  .0 .0 0000010
2880200  102
2880201  0.0   14.7  1.0
****************************************************************
2900000  sga-stdv tmdpvol
2900101  1.0e6 .0 1.0e+06
2900102  .0 .0 .0
2900103  .0 .0 0000010
2900200  102
2900201  0.0   14.7  1.0
****************************************************************
2670000  sga-mfwj tmdpjun
2670101  266000000   272000000   3.54
2670200  1  0  cntrlvar 25
2670201  0.0   0.0   0.0   0.0
2670202  10000.   10000.   0.0   0.0
****************************************************************
2660000  sga-ftdv tmdpvol
2660101  1.0e6 .0 1.0e+06
2660102  .0 .0 .0
2660103  .0 .0 0000010
2660200  103
2660201  0.0   785.  443.0
****************************************************************
*  AUX FW   *
****************************************************************
2680000  sga-aft  tmdpvol
2680101  1.0e6 .0 1.0e+06
2680102  .0 .0 .0
2680103  .0 .0 0000010
2680200  103
2680201  0.0   785.  120.0
*
*  MAX   Injection
****************************************************************
2690000  sga-afwM tmdpjun
2690101  268000000   272000000   3.54
2690200  1  728
2690201  -1.0  0.0   0.0   0.0
2690202  0.0   32.1  0.0   0.0
2690203  1.00E+05 32.1  0.0   0.0
*
*  Operator    Controlling Level
****************************************************************
2650000  sga-afwC tmdpjun
2650101  268000000   272000000   3.54
2650200  1  730   cntrlvar 472
2650201  -1.0  0.0   0.0   0.0
2650202  0.0   0.0   0.0   0.0
2650203  32.1  32.1  0.0   0.0
2650204  1000.0   32.1  0.0   0.0
****************************************************************
*  Turbine  Driven   AFW   Steam
****************************************************************
2910000  tdafwva  valve
2910101  284010000   288000000   5.4188E-04  0.0   0.0   00100
2910201  1  0.0   0.0   0.0
2910300  trpvlv
2910301  731
****************************************************************
****************************************************************
*  coolant  loop  b
****************************************************************
3000000  hlb-vess pipe
3000001  2
3000101  4.587 2
3000301  0.0   2
3000401  23.330   2
3000601  0.0   2
3000701  0.0   2
3000801  .00015   0.0   2
3001001  00 2
3001101  00000 1
3001201  0  2234.9322   611.84403   1053.3231   0.0   0.0   1
3001202  0  2234.8026   611.84456   1053.3323   0.0   0.0   2
3001300  0
3001301  47.655407   47.655407   0.0   1
****************************************************************
3020000  hlb-pzr  branch
3020001  1  0
3020101  4.587 0.0   40.082
3020102  0.0   0.0   0.0
3020103  .00015   0.0   00
3020200  0  2234.6309   611.84546   1053.3444   0.0
3021101  300010000   302000000   4.587 0.0   0.0   00000
3021201  47.655530   47.655530   0.0
****************************************************************
3030000  hlb-stop sngljun
3030101  302010000   304000000   4.167 0.219 0.219 00000
3030201  0  52.459020   52.459020   0.0
****************************************************************
3040000  hlb-sgin snglvol
3040101  0.0   6.292 38.14
3040102  0.0   14.79 1.606
3040103  .00015   0.0   00
3040200  0  2232.3070   611.84575   1053.5092   0.0
****************************************************************
3060000  sgb-inpl branch
3060001  2  0
3060101  0.0   7.197 157.33
3060102  0.0   51.70 5.648
3060103  .00015   0.0   00
3060200  0  2240.2858   611.84038   1052.9441   0.0
3061101  304010000   306000000   5.241 0.0   0.0   00000
3062101  306010000   308000000   10.95 0.0   0.0   00000
3061201  41.710353   41.710353   0.0
3062201  19.961439   19.961439   0.0
****************************************************************
3080000  sgb-tube pipe
3080001  8
3080101  10.95 8
3080301  8.49  3
3080302  9.91  5
3080303  8.49  8
3080401  0.0   8
3080601  90.0  3
3080602  58.95 4
3080603  -58.95   5
3080604  -90.0 8
3080701  8.49  4
3080702  -8.49 8
3080801  .00015   0.0646   8
3080901  0.0   0.0   3
3080902  0.048 0.048 4
3080903  0.0   0.0   7
3081001  00 8
3081101  00000 7
3081201  0  2232.1742   592.38674   1053.5186   0.0   0.0   1
3081202  0  2220.7340   577.14527   1054.3319   0.0   0.0   2
3081203  0  2209.4236   565.11849   1055.1394   0.0   0.0   3
3081204  0  2197.4887   554.54911   1055.9951   0.0   0.0   4
3081205  0  2187.5896   545.98591   1056.7077   0.0   0.0   5
3081206  0  2181.3061   539.51890   1057.1615   0.0   0.0   6
3081207  0  2175.8004   534.04688   1057.5599   0.0   0.0   7
3081208  0  2170.3601   529.37052   1057.9267   0.0   0.0   8
3081300  0
3081301  19.418115   19.418115   0.0   1
3081302  19.035939   19.035939   0.0   2
3081303  18.753128   18.753128   0.0   3
3081304  18.519150   18.519150   0.0   4
3081305  18.336644   18.336644   0.0   5
3081306  18.205045   18.205045   0.0   6
3081307  18.095997   18.095997   0.0   7
****************************************************************
3100000  sgb-outp branch
3100001  2  0
3100101  0.0   7.197 157.33
3100102  0.0   -51.70   -5.648
3100103  .00015   0.0   00
3100200  0  2169.8338   529.33070   1057.9621   0.0
3101101  308010000   310000000   10.95 0.0   0.0   00000
3102101  310010000   312000000   5.241 0.0   0.0   00000
3101201  18.004118   18.004118   0.0
3102201  37.616349   37.616349   0.0
****************************************************************
3120000  rcpb-suc pipe
3120001  5
3120101  5.241 4
3120102  5.002 5
3120301  2.676 1
3120302  4.971 2
3120303  6.709 3
3120304  2.365 4
3120305  4.271 5
3120401  0.0   5
3120601  -67.04   1
3120602  -90.  2
3120603  -39.53   3
3120604  0.0   4
3120605  39.53 5
3120701  -2.464   1
3120702  -4.971   2
3120703  -4.271   3
3120704  0.0   4
3120705  4.271 5
3120801  .00015   0.0   5
3120901  .1136 .1136 1
3120902  .0803 .0803 3
3120903  .1463 .1463 4
3121001  00 5
3121101  00000 4
3121201  0  2164.3357   529.32635   1058.3135   0.0   0.0   1
3121202  0  2164.6742   529.31900   1058.2925   0.0   0.0   2
3121203  0  2165.5188   529.30910   1058.2400   0.0   0.0   3
3121204  0  2165.5676   529.30559   1058.2369   0.0   0.0   4
3121205  0  2162.9412   529.29941   1058.4003   0.0   0.0   5
3121300  0
3121301  37.618058   37.618058   0.0   1
3121302  37.617964   37.617964   0.0   2
3121303  37.617717   37.617717   0.0   3
3121304  39.415114   39.415114   0.0   4
****************************************************************
*  MCP   SEALS
****************************************************************
3130000  sealB valve          * standard choking model
3130101  316000000   315000000   8.1121E-04  0.0   0.0   00000100 1.0   1.0   1.0
3130201  1  0.0   0.0   0.0
3130300  mtrvlv
3130301  742   536   0.115420 0.0
****************************************************************
3150000  contB tmdpvol
3150101  1.0e6 .0 1.0e+06
3150102  .0 .0 .0
3150103  .0 .0 0000010
3150200  102
3150201  0.0   14.7  1.0
*
****************************************************************
3140000  rcpb  pump
3140101  0.0   5.829 56.
3140102  0.0   90.   5.829
3140103  0
3140108  312010000   5.002 0.0   0.0   00000
3140109  316000000   4.1247   0.0   0.0   00000
3140200  0  2242.4555   529.43232   1052.7907   0.0
****************************************************************
3140201  0  39.415978   39.415978   0.0
3140202  0  47.775937   47.775937   0.0
3140301  214   214   214   -1 214   518   0
3140302  1170.0000   1.1571279   88500.   280.
3140303  28015.   70000.   47.31 .0
3140304  0.0   280.15   28.015   0.0
****************************************************************
3160000  cla-rcp2 pipe
3160001  2
3160101  0.0   2
3160301  3.293 2
3160401  15.662   2
3160601  0.0   2
3160701  0.0   2
3160801  0.00015  0.0   2
3160901  0.0   0.0   1
3161001  0000000  2
3161101  00000000 1
3161201  000   2310.2434   529.42862   1048.0217   0.0   0.0   2
3161300  0
3161301  41.41015 41.41015 0.0   1
****************************************************************
3170000  cla-sto2 sngljun
3170101  316010000   318000000   4.167 0.219 0.219 00000
3170201  0  47.264901   47.264901   0.0
****************************************************************
3180000  cla-ecc2 pipe
3180001  2
3180101  0.0   2
3180301  4.043 2
3180401  18.7555  2
3180601  0.0   2
3180701  0.0   2
3180801  0.00015  0.0   2
3180901  0.0   0.0   1
3181001  0000000  2
3181101  00000000 1
3181201  000   2310.2434   529.42862   1048.0217   0.0   0.0   2
3181300  0
3181301  42.45583 42.45583 0.0   1
****************************************************************
3200000  cla-ecc2 branch
3200001  2  0
3200101  4.125 4.043 0.0
3200102  0.0   0.0   0.0
3200103  0.00015  0.0   00
3200200  0  2304.5001   529.40911   1048.4551   0.0
3201101  318010000   320000000   0.0   0.0   0.0   00000
3202101  320010000   321000000   0.0   0.0   0.0   00000
3201201  47.747364   47.747364   0.0
3202201  47.747364   47.747364   0
****************************************************************
3210000  cla-vex2 snglvol
3210101  4.125 4.043 0.0
3210102  0.0   0.000 0
3210103  0.00015  0.0   00
3210200  0  2304.5001   529.40911   1048.4551   0.0
****************************************************************
3190000  un2   sngljun
3190101  321010000   322000000   4.125 0.0   0.0   00000
3190201  0  47.264901   47.264901   0.0
****************************************************************
*  3220000  cla-vess pipe
*  3220001  2
*  3220101  0.0   2
*  3220301  3.020 2
*  3220401  14.7335  2
*  3220601  0.0   2
*  3220701  0.0   2
*  3220801  0.00015  0.0   2
*  3220901  0.0   0.0   1
*  3221001  0000000  2
*  3221101  00000000 1
*  3221201  000   2306.378 529.40117   1048.3254   0.0   0.0   2
*  3221300  0
*  3221301  40.37042 40.37042 0.0   1
****************************************************************
3220000  cla-vess snglvol
3220101  0.0   6.04  29.467
3220102  0.0   0.0   0.0
3220103  .00015   0.0   00
3220200  0  2306.378 529.40117   1048.3254   0.0
****************************************************************
*  ECCS
*
*  ACCUMULATOR
****************************************************************
3500000  accuB accum
3500101  0.0   15.15 1450.0
3500102  0.0   -90.0 -15.15
3500103  0.00015  0.0   0000000  0
3500200  580.0 105.0 0.0
3501101  320000000   0.601 5.65  5.65  00000000
3502200  1000.0   0.0   104.38   44.0  0.229
+  0  0.0   0.0   642
*
****************************************************************
*
*  steam generator   b  secondary   side  components
*
****************************************************************
3700000  sgb-dc1  snglvol
3700101  0.0   6.936 700.
3700102  0.0   -90.  -6.936
3700103  .00015   0.0   00
3700200  0  830.89101   502.28620   1116.1629   .50582284
****************************************************************
3720000  sgb-dc2  branch
3720001  2  0
3720101  0.0   3.917 340.
3720102  0.0   -90.  -3.917
3720103  .00015   0.0   00
3720200  0  832.13033   495.84672   1114.4162   0.0
3721101  370010000   372000000   0.0   0.0   0.0   00000
3722101  372010000   374000000   0.0   0.12  0.64  00000
3721201  -2.5831-3   -5.837087   0.0
3722201  7.7543662   7.7543662   0.0
****************************************************************
3740000  sgb-dc3  pipe
3740001  4
3740101  0.0   4
3740301  11.02 1
3740302  8.49  3
3740303  6.74  4
3740401  191.8 1
3740402  60.19 3
3740403  47.79 4
3740601  -90.  4
3740701  -11.02   1
3740702  -8.49 3
3740703  -6.74 4
3740801  .00015   0.43  4
3741001  00 4
3741101  00000 3
3741201  0  834.24330   495.75817   1114.3737   0.0   0.0   1
3741202  0  835.59224   495.78343   1114.3466   0.0   0.0   2
3741203  0  837.87411   495.80544   1114.3009   0.0   0.0   3
3741204  0  839.92136   495.80840   1114.2599   0.0   0.0   4
3741300  0
3741301  19.038383   19.038383   0.0   1
3741302  19.039778   19.039778   0.0   2
3741303  19.040957   19.040957   0.0   3
****************************************************************
3750000  sgb-boli sngljun
3750101  374010000   376000000   37.79 46.8  46.8  00000
3750201  0  3.5722534   3.8225232   0.0
****************************************************************
3760000  sgb-boil pipe
3760001  5
3760101  0.0   5
3760301  6.74  1
3760302  8.49  3
3760303  11.02 4
3760304  3.917 5
3760401  369.98   1
3760402  466.04   3
3760403  658.2 4
3760404  260.  5
3760601  +90.  5
3760701  6.74  1
3760702  8.49  3
3760703  11.02 4
3760704  3.917 5
3760801  .00015   0.140 5
3760901  16.9  16.9  1
3760902  6.0   6.0   3
3760903  0.0   0.0   4
3761001  00 5
3761101  00000 4
3761201  0  838.75235   509.75315   1114.2699   .34276726   0.0   1
3761202  0  836.40283   512.35263   1114.3137   .55862754   0.0   2
3761203  0  834.61317   512.55191   1114.3483   .66937467   0.0   3
3761204  0  832.80432   512.28988   1114.3973   .76660522   0.0   4
3761205  0  831.91737   512.14272   1114.4181   .77982352   0.0   5
3761300  0
3761301  3.6880988   5.6155472   0.0   1
3761302  5.2851929   8.0286252   0.0   2
3761303  6.7647608   10.423870   0.0   3
3761304  8.3970469   11.517427   0.0   4
****************************************************************
3780000  sgb-swrl separatr
3780001  3  0
3780101  0.0   6.936 350.
3780102  0.0   90.   6.936
3780103  .00015   4.67  00
3780200  0  830.83376   512.07459   1114.3647   .53829806
3781101  378010000   380000000   0.0   1.93  0.0   00000 0.02
3782101  378000000   372000000   0.0   0.0   9.67  00000 0.80
3783101  376010000   378000000   0.0   0.967 0.967 00000
3781201  -9.669464   14.676371   0.0
3782201  5.6190925   -2.963842   0.0
3783201  14.736148   18.799016   0.0
****************************************************************
3800000  sgb-dryr branch
3800001  2  0
3800101  0.0   9.470 1450.
3800102  0.0   90.   9.470
3800103  .00015   0.0   00
3800200  0  830.26239   511.86773   1114.4516   .99999995
3801101  380010000   382000000   0.0   0.0   0.0   00000
3802101  380000000   370000000   0.0   0.0   0.0   00000
3801201  5.9512255   6.3173370   0.0
3802201  8.1162296   3.00563-3   0.0
****************************************************************
3820000  sgb-dome snglvol
3820101  0.0   5.107 467.96
3820102  0.0   90.   5.107
3820103  .00015   0.0   00
3820200  0  830.16123   511.85091   1114.4887   .99999970
****************************************************************
3840000  sgb-stml branch
3840001  1  0
3840101  4.5869   0.0   824.2
3840102  0.0   -12.98   -40.35
3840103  .00015   0.0   00
3840200  0  825.93722   511.14924   1114.4480   .99998780
3841101  382010000   384000000   0.0   0.5   1.0   00000
3841201  89.848360   115.13716   0.0
****************************************************************
3850000  sgb-msiv valve
3850101  384010000   386000000   4.5869   0.0   0.0   00100
3850201  0  102.20364   115.59958   0.0
3850300  trpvlv
3850301  505
****************************************************************
*  following   cards replace  the   msiv  with  a  steam control  valve used
*  to regulate secondary   pressure in response to cold  leg   temp. error
****************************************************************
3850300  srvvlv
3850301  73
****************************************************************
3870000  sgb-porv valve
3870101  384010000   388000000   0.04565  0.0   0.0   00100
3870201  0  0.0   0.0   0.0
3870300  mtrvlv
3870301  710   712   0.556 0.0
****************************************************************
3890000  sgb-sfty valve
3890101  384010000   390000000   0.39985  0.0   0.0   00100
3890201  0  0.0   0.0   0.0
3890300  mtrvlv
3890301  719   721   20.0  0.0
****************************************************************
3860000  sgb-htdv tmdpvol
3860101  1.0e6 .0 1.0e+06
3860102  .0 .0 .0
3860103  .0 .0 0000010
3860200  102
3860201  0.0   785.0 1.0
****************************************************************
3880000  sgb-ptdv tmdpvol
3880101  1.0e6 .0 1.0e+06
3880102  .0 .0 .0
3880103  .0 .0 0000010
3880200  102
3880201  0.0   14.7  1.0
****************************************************************
3900000  sgb-stdv tmdpvol
3900101  1.0e6 .0 1.0e+06
3900102  .0 .0 .0
3900103  .0 .0 0000010
3900200  102
3900201  0.0   14.7  1.0
****************************************************************
3670000  sgb-mfwj tmdpjun
3670101  366000000   372000000   3.54
3670200  1  0  cntrlvar 45
3670201  0.0   0.0   0.0   0.0
3670202  10000.   10000.   0.0   0.0
****************************************************************
3660000  sgb-ftdv tmdpvol
3660101  1.0e6 .0 1.0e+06
3660102  .0 .0 .0
3660103  .0 .0 0000010
3660200  103
3660201  0.0   785.  443.0
****************************************************************
*  AUX FW   *
****************************************************************
3680000  sgb-aft  tmdpvol
3680101  1.0e6 .0 1.0e+06
3680102  .0 .0 .0
3680103  .0 .0 0000010
3680200  103
3680201  0.0   785.  120.0
*
*  MAX   Injection
****************************************************************
3690000  sgb-afwj tmdpjun
3690101  368000000   372000000   3.54
3690200  1  728
3690201  -1.0  0.0   0.0   0.0
3690202  0.0   32.1  0.0   0.0
3690203  1.00E+05 32.1  0.0   0.0
*
*  Operator    Controlling Level
****************************************************************
3650000  sgb-afwC tmdpjun
3650101  368000000   372000000   3.54
3650200  1  730   cntrlvar 474
3650201  -1.0  0.0   0.0   0.0
3650202  0.0   0.0   0.0   0.0
3650203  32.1  32.1  0.0   0.0
3650204  1000.0   32.1  0.0   0.0
****************************************************************
*  Turbine  Driven   AFW   Steam
****************************************************************
3910000  tdafwvb  valve
3910101  384010000   388000000   5.4188E-04  0.0   0.0   00100
3910201  1  0.0   0.0   0.0
3910300  trpvlv
3910301  731
****************************************************************
*
*
****************************************************************
****************************************************************
*  coolant  loop  c
****************************************************************
4000000  hlc-vess pipe
4000001  2
4000101  4.587 2
4000301  0.0   2
4000401  23.330   2
4000601  0.0   2
4000701  0.0   2
4000801  .00015   0.0   2
4001001  00 2
4001101  00000 1
4001201  0  2234.9322   611.84403   1053.3231   0.0   0.0   1
4001202  0  2234.8026   611.84456   1053.3323   0.0   0.0   2
4001300  0
4001301  47.655407   47.655407   0.0   1
****************************************************************
4020000  hlc-pzr  branch
4020001  2  0
4020101  4.587 0.0   40.082
4020102  0.0   0.0   0.0
4020103  .00015   0.0   00
4020200  0  2234.6309   611.84546   1053.3444   0.0
4021101  400010000   402000000   4.587 0.0   0.0   00000000
4022101  443010000   402000000   .5458 1.0   0.5   00002
4021201  47.655530   47.655530   0.0
4022201  0.007233590 0.00723359  0.0
****************************************************************
4030000  hlc-stop sngljun
4030101  402010000   404000000   4.167 0.219 0.219 00000
4030201  0  52.459020   52.459020   0.0
****************************************************************
4040000  hlc-sgin snglvol
4040101  0.0   6.292 38.14
4040102  0.0   14.79 1.606
4040103  .00015   0.0   00
4040200  0  2232.3070   611.84575   1053.5092   0.0
****************************************************************
4060000  sgc-inpl branch
4060001  2  0
4060101  0.0   7.197 157.33
4060102  0.0   51.70 5.648
4060103  .00015   0.0   00
4060200  0  2240.2858   611.84038   1052.9441   0.0
4061101  404010000   406000000   5.241 0.0   0.0   00000
4062101  406010000   408000000   10.95 0.0   0.0   00000
4061201  41.710353   41.710353   0.0
4062201  19.961439   19.961439   0.0
****************************************************************
4080000  sgc-tube pipe
4080001  8
4080101  10.95 8
4080301  8.49  3
4080302  9.91  5
4080303  8.49  8
4080401  0.0   8
4080601  90.0  3
4080602  58.95 4
4080603  -58.95   5
4080604  -90.0 8
4080701  8.49  4
4080702  -8.49 8
4080801  .00015   0.0646   8
4080901  0.0   0.0   3
4080902  0.048 0.048 4
4080903  0.0   0.0   7
4081001  00 8
4081101  00000 7
4081201  0  2232.1742   592.38674   1053.5186   0.0   0.0   1
4081202  0  2220.7340   577.14527   1054.3319   0.0   0.0   2
4081203  0  2209.4236   565.11849   1055.1394   0.0   0.0   3
4081204  0  2197.4887   554.54911   1055.9951   0.0   0.0   4
4081205  0  2187.5896   545.98591   1056.7077   0.0   0.0   5
4081206  0  2181.3061   539.51890   1057.1615   0.0   0.0   6
4081207  0  2175.8004   534.04688   1057.5599   0.0   0.0   7
4081208  0  2170.3601   529.37052   1057.9267   0.0   0.0   8
4081300  0
4081301  19.418115   19.418115   0.0   1
4081302  19.035939   19.035939   0.0   2
4081303  18.753128   18.753128   0.0   3
4081304  18.519150   18.519150   0.0   4
4081305  18.336644   18.336644   0.0   5
4081306  18.205045   18.205045   0.0   6
4081307  18.095997   18.095997   0.0   7
****************************************************************
4100000  sgc-outp branch
4100001  2  0
4100101  0.0   7.197 157.33
4100102  0.0   -51.70   -5.648
4100103  .00015   0.0   00
4100200  0  2169.8338   529.33070   1057.9621   0.0
4101101  408010000   410000000   10.95 0.0   0.0   00000
4102101  410010000   412000000   5.241 0.0   0.0   00000
4101201  18.004118   18.004118   0.0
4102201  37.616349   37.616349   0.0
****************************************************************
4120000  rcpc-suc pipe
4120001  5
4120101  5.241 4
4120102  5.002 5
4120301  2.676 1
4120302  4.971 2
4120303  6.709 3
4120304  2.365 4
4120305  4.271 5
4120401  0.0   5
4120601  -67.04   1
4120602  -90.  2
4120603  -39.53   3
4120604  0.0   4
4120605  39.53 5
4120701  -2.464   1
4120702  -4.971   2
4120703  -4.271   3
4120704  0.0   4
4120705  4.271 5
4120801  .00015   0.0   5
4120901  .1136 .1136 1
4120902  .0803 .0803 3
4120903  .1463 .1463 4
4121001  00 5
4121101  00000 4
4121201  0  2164.3357   529.32635   1058.3135   0.0   0.0   1
4121202  0  2164.6742   529.31900   1058.2925   0.0   0.0   2
4121203  0  2165.5188   529.30910   1058.2400   0.0   0.0   3
4121204  0  2165.5676   529.30559   1058.2369   0.0   0.0   4
4121205  0  2162.9412   529.29941   1058.4003   0.0   0.0   5
4121300  0
4121301  37.618058   37.618058   0.0   1
4121302  37.617964   37.617964   0.0   2
4121303  37.617717   37.617717   0.0   3
4121304  39.415114   39.415114   0.0   4
****************************************************************
*  MCP   SEALS
****************************************************************
4130000  sealC valve          * standard choking model
4130101  416000000   415000000   8.1121E-04  0.0   0.0   00000100 1.0   1.0   1.0
4130201  1  0.0   0.0   0.0
4130300  mtrvlv
4130301  742   536   0.115420 0.0
****************************************************************
4150000  contC tmdpvol
4150101  1.0e6 .0 1.0e+06
4150102  .0 .0 .0
4150103  .0 .0 0000010
4150200  102
4150201  0.0   14.7  1.0
*
****************************************************************
4140000  rcpc  pump
4140101  0.0   5.829 56.
4140102  0.0   90.   5.829
4140103  0
4140108  412010000   5.002 0.0   0.0   00000
4140109  416000000   4.1247   0.0   0.0   00000
4140200  0  2242.4555   529.43232   1052.7907   0.0
****************************************************************
4140201  0  39.415978   39.415978   0.0
4140202  0  47.775937   47.775937   0.0
4140301  214   214   214   -1 214   518   0
4140302  1170.0000   1.1571279   88500.   280.
4140303  28015.   70000.   47.31 .0
4140304  0.0   280.15   28.015   0.0
****************************************************************
4160000  cla-rcp3 pipe
4160001  2
4160101  0.0   2
4160301  3.293 2
4160401  15.662   2
4160601  0.0   2
4160701  0.0   2
4160801  0.00015  0.0   2
4160901  0.0   0.0   1
4161001  0000000  2
4161101  00000000 1
4161201  000   2310.2434   529.42862   1048.0217   0.0   0.0   2
4161300  0
4161301  41.41015 41.41015 0.0   1
****************************************************************
4170000  cla-sto3 sngljun
4170101  416010000   418000000   4.167 0.219 0.219 00000
4170201  0  47.264901   47.264901   0.0
****************************************************************
4180000  cla-ecc3 pipe
4180001  2
4180101  0.0   2
4180301  4.043 2
4180401  18.7555  2
4180601  0.0   2
4180701  0.0   2
4180801  0.00015  0.0   2
4180901  0.0   0.0   1
4181001  0000000  2
4181101  00000000 1
4181201  000   2310.2434   529.42862   1048.0217   0.0   0.0   2
4181300  0
4181301  42.45583 42.45583 0.0   1
****************************************************************
4200000  cla-ecc3 branch
4200001  2  0
4200101  4.125 4.043 0.0
4200102  0.0   0.0   0.0
4200103  0.00015  0.0   00
4200200  0  2304.5001   529.40911   1048.4551   0.0
4201101  418010000   420000000   0.0   0.0   0.0   00000
4202101  420010000   421000000   0.0   0.0   0.0   00000
4201201  47.747364   47.747364   0.0
4202201  47.747364   47.747364   0
****************************************************************
4210000  cla-vex3 snglvol
4210101  4.125 4.043 0.0
4210102  0.0   0.000 0
4210103  0.00015  0.0   00
4210200  0  2304.5001   529.40911   1048.4551   0.0
****************************************************************
4220000  cla-ves3 pipe
4220001  2
4220101  0.0   2
4220301  3.020 2
4220401  14.7335  2
4220601  0.0   2
4220701  0.0   2
4220801  0.00015  0.0   2
4220901  0.0   0.0   1
4221001  0000000  2
4221101  00000000 1
4221201  000   2306.378 529.40117   1048.3254   0.0   0.0   2
4221300  0
4221301  40.37042 40.37042 0.0   1
****************************************************************
*  ECCS
*
*  ACCUMULATOR
****************************************************************
4500000  accuC accum
4500101  0.0   15.15 1450.0
4500102  0.0   -90.0 -15.15
4500103  0.00015  0.0   0000000  0
4500200  580.0 105.0 0.0
4501101  420000000   0.601 5.65  5.65  00000000
4502200  1000.0   0.0   104.38   44.0  0.229
+  0  0.0   0.0   643
*
****************************************************************
*
*  pressurizer components
*
****************************************************************
4430000  surgeln  pipe
4430001  3
4430101  0. 3
4430301  15.444   1
4430302  18.298   2
4430303  26.558   3
4430401  9.541 1
4430402  11.003   2
4430403  15.656   3
4430601  -54.61   1
4430602  0.0   2
4430603  -0.22 3
4430701  -12.59   1
4430702  0.0   2
4430703  -0.089   3
4430801  .00015   0.0   3
4430901  0.21  0.21  1
4430902  0.42  0.42  2
4431001  00 3
4431101  00000 2
4431201  0  2232.9187   670.82410   1053.4658   0.0   0.0   1
4431202  0  2234.6059   655.13667   1053.3462   0.0   0.0   2
4431203  0  2234.6184   645.16180   1053.3453   0.0   0.0   3
4431300  0
4431301  6.58013-3   6.58013-3   0.0   1
4431302  6.70063-3   6.70063-3   0.0   2
****************************************************************
4420000  pzr-srg  sngljun
4420101  441010000   443000000   0.7213   0.507 1.004 00000
4420201  0  5.50138-3   5.50138-3   0.0
****************************************************************
4400000  pzrsdom  branch
4400001  1  0
4400101  0.0   3.16  66.317
4400102  0.0   -90.  -3.16
4400103  .00015   0. 00
4400200  0  2225.0909   687.43448   1054.0065   .99999892
4401101  440010000   441000000   38.485   0.0   0.0   00000
4401201  .12420298   153.746-9   0.0
****************************************************************
4410000  prizer   pipe
4410001  7
4410101  0. 7
4410301  5.056 6
4410302  3.16  7
4410401  194.561  6
4410402  66.317   7
4410601  -90.0 7
4410701  -5.056   6
4410702  -3.16 7
4410801  .00015   0. 7
4411001  00 7
4411101  00000 6
4411201  0  2225.2695   687.45418   1054.0018   .99999285   0.0   1
4411202  0  2225.4893   687.47844   1053.9718   .99999950   0.0   2
4411203  0  2226.0457   687.82228   1054.2783   .38099251   0.0   3
4411204  0  2227.1461   687.67714   1053.8752   76.3470-6   0.0   4
4411205  0  2228.4535   687.81951   1053.7827   494.729-6   0.0   5
4411206  0  2229.7604   687.98159   1053.6899   202.060-6   0.0   6
4411207  0  2230.8227   687.59554   1053.6145   0.0   0.0   7
4411300  0
4411301  .14857803   -255.36-9   0.0   1
4411302  2.9585826   119.760-9   0.0   2
4411303  385.754-6   -2.483720   0.0   3
4411304  221.343-6   -.3294188   0.0   4
4411305  143.068-6   -.3328259   0.0   5
4411306  189.135-6   -.3658623   0.0   6
****************************************************************
4440000  pzrporv  valve
****************************************************************
4440101  440000000   449000000   .01811   0.0   0.0   00100
4440201  0  0.0   0.0   0.0
4440300  mtrvlv
4440301  701   703   5.0   0.0
****************************************************************
4450000  pzrsfty  valve
****************************************************************
4450101  440000000   449000000   .04120   0.0   0.0   00100
4450201  0  0.0   0.0   0.0
4450300  mtrvlv
4450301  704   706   5.0   0.0
****************************************************************
4460000  pzporv2  valve * For Primary  Side Depressurization      in case of loss of AFW
****************************************************************
4460101  440000000   449000000   .01811   0.0   0.0   00100
4460201  0  0.0   0.0   0.0
4460300  mtrvlv
4460301  751   753   5.0   0.0
****************************************************************
4490000  contain  snglvol
****************************************************************
4490101  0.0   200.0 1.0e6 0.0   0.0   0.0   0.0   0.0   10
4490200  4  14.7  80.0  1.0
****************************************************************
*  components  for   loop  steady   state calculations
****************************************************************
0940000  sysprjn  sngljun
0940101  095000000   440010000   0.0   0.0   0.0   00000
0940201  1  0.0   0.0   0
****************************************************************
0950000  syspres  tmdpvol
0950101  1.0e+6   .0 1.0e+06
0950102  .0 .0 .0
0950103  .0 .0 0000010
0950200  102
0950201  0.0   2250.1   1.0
****************************************************************
*
*  steam generator   c  secondary   side  components
*
****************************************************************
4700000  sgc-dc1  snglvol
4700101  0.0   6.936 700.
4700102  0.0   -90.  -6.936
4700103  .00015   0.0   00
4700200  0  830.89101   502.28620   1116.1629   .50582284
****************************************************************
4720000  sgc-dc2  branch
4720001  2  0
4720101  0.0   3.917 340.
4720102  0.0   -90.  -3.917
4720103  .00015   0.0   00
4720200  0  832.13033   495.84672   1114.4162   0.0
4721101  470010000   472000000   0.0   0.0   0.0   00000
4722101  472010000   474000000   0.0   0.12  0.64  00000
4721201  -2.5831-3   -5.837087   0.0
4722201  7.7543662   7.7543662   0.0
****************************************************************
4740000  sgc-dc3  pipe
4740001  4
4740101  0.0   4
4740301  11.02 1
4740302  8.49  3
4740303  6.74  4
4740401  191.8 1
4740402  60.19 3
4740403  47.79 4
4740601  -90.  4
4740701  -11.02   1
4740702  -8.49 3
4740703  -6.74 4
4740801  .00015   0.43  4
4741001  00 4
4741101  00000 3
4741201  0  834.24330   495.75817   1114.3737   0.0   0.0   1
4741202  0  835.59224   495.78343   1114.3466   0.0   0.0   2
4741203  0  837.87411   495.80544   1114.3009   0.0   0.0   3
4741204  0  839.92136   495.80840   1114.2599   0.0   0.0   4
4741300  0
4741301  19.038383   19.038383   0.0   1
4741302  19.039778   19.039778   0.0   2
4741303  19.040957   19.040957   0.0   3
****************************************************************
4750000  sgc-boli sngljun
4750101  474010000   476000000   37.79 46.8  46.8  00000
4750201  0  3.5722534   3.8225232   0.0
****************************************************************
4760000  sgc-boil pipe
4760001  5
4760101  0.0   5
4760301  6.74  1
4760302  8.49  3
4760303  11.02 4
4760304  3.917 5
4760401  369.98   1
4760402  466.04   3
4760403  658.2 4
4760404  260.  5
4760601  +90.  5
4760701  6.74  1
4760702  8.49  3
4760703  11.02 4
4760704  3.917 5
4760801  .00015   0.140 5
4760901  16.9  16.9  1
4760902  6.0   6.0   3
4760903  0.0   0.0   4
4761001  00 5
4761101  00000 4
4761201  0  838.75235   509.75315   1114.2699   .34276726   0.0   1
4761202  0  836.40283   512.35263   1114.3137   .55862754   0.0   2
4761203  0  834.61317   512.55191   1114.3483   .66937467   0.0   3
4761204  0  832.80432   512.28988   1114.3973   .76660522   0.0   4
4761205  0  831.91737   512.14272   1114.4181   .77982352   0.0   5
4761300  0
4761301  3.6880988   5.6155472   0.0   1
4761302  5.2851929   8.0286252   0.0   2
4761303  6.7647608   10.423870   0.0   3
4761304  8.3970469   11.517427   0.0   4
****************************************************************
4780000  sgc-swrl separatr
4780001  3  0
4780101  0.0   6.936 350.
4780102  0.0   90.   6.936
4780103  .00015   4.67  00
4780200  0  830.83376   512.07459   1114.3647   .53829806
4781101  478010000   480000000   0.0   1.93  0.0   00000 0.20
4782101  478000000   472000000   0.0   0.0   9.67  00000 0.80
4783101  476010000   478000000   0.0   0.967 0.967 00000
4781201  -9.669464   14.676371   0.0
4782201  5.6190925   -2.963842   0.0
4783201  14.736148   18.799016   0.0
****************************************************************
4800000  sgc-dryr branch
4800001  2  0
4800101  0.0   9.470 1450.
4800102  0.0   90.   9.470
4800103  .00015   0.0   00
4800200  0  830.26239   511.86773   1114.4516   .99999995
4801101  480010000   482000000   0.0   0.0   0.0   00000
4802101  480000000   470000000   0.0   0.0   0.0   00000
4801201  5.9512255   6.3173370   0.0
4802201  8.1162296   3.00563-3   0.0
****************************************************************
4820000  sgc-dome snglvol
4820101  0.0   5.107 467.96
4820102  0.0   90.   5.107
4820103  .00015   0.0   00
4820200  0  830.16123   511.85091   1114.4887   .99999970
****************************************************************
4840000  sgc-stml branch
4840001  1  0
4840101  4.5869   0.0   824.2
4840102  0.0   -12.98   -40.35
4840103  .00015   0.0   00
4840200  0  825.93722   511.14924   1114.4480   .99998780
4841101  482010000   484000000   0.0   0.5   1.0   00000
4841201  89.848360   115.13716   0.0
****************************************************************
4850000  sgc-msiv valve
4850101  484010000   486000000   4.5869   0.0   0.0   00100
4850201  0  102.20364   115.59958   0.0
4850300  trpvlv
4850301  505
****************************************************************
*  following   cards replace  the   msiv  with  a  steam control  valve used
*  to regulate secondary   pressure in response to cold  leg   temp. error
****************************************************************
4850300  srvvlv
4850301  75
****************************************************************
4870000  sgc-porv valve
4870101  484010000   488000000   0.04565  0.0   0.0   00100
4870201  0  0.0   0.0   0.0
4870300  mtrvlv
4870301  713   715   0.556 0.0
****************************************************************
4890000  sgc-sfty valve
4890101  484010000   490000000   0.39985  0.0   0.0   00100
4890201  0  0.0   0.0   0.0
4890300  mtrvlv
4890301  722   724   20.0  0.0
****************************************************************
4860000  sgc-htdv tmdpvol
4860101  1.0e6 .0 1.0e+06
4860102  .0 .0 .0
4860103  .0 .0 0000010
4860200  102
4860201  0.0   785.0 1.0
****************************************************************
4880000  sgc-ptdv tmdpvol
4880101  1.0e6 .0 1.0e+06
4880102  .0 .0 .0
4880103  .0 .0 0000010
4880200  102
4880201  0.0   14.7  1.0
****************************************************************
4900000  sgc-stdv tmdpvol
4900101  1.0e6 .0 1.0e+06
4900102  .0 .0 .0
4900103  .0 .0 0000010
4900200  102
4900201  0.0   14.7  1.0
****************************************************************
*  Turbine  Driven   AFW   Steam
****************************************************************
4910000  tdafwvc  valve
4910101  484010000   488000000   5.4188E-04  0.0   0.0   00100
4910201  1  0.0   0.0   0.0
4910300  trpvlv
4910301  731
****************************************************************
****************************************************************
4670000  sgc-mfwj tmdpjun
4670101  466000000   472000000   3.54
4670200  1  0  cntrlvar 65
4670201  0.0   0.0   0.0   0.0
4670202  10000.   10000.   0.0   0.0
****************************************************************
4660000  sgc-ftdv tmdpvol
4660101  1.0e6 .0 1.0e+06
4660102  .0 .0 .0
4660103  .0 .0 0000010
4660200  103
4660201  0.0   785.  443.0
****************************************************************
****************************************************************
*  AUX FW   *
****************************************************************
4680000  sgc-aft  tmdpvol
4680101  1.0e6 .0 1.0e+06
4680102  .0 .0 .0
4680103  .0 .0 0000010
4680200  103
4680201  0.0   785.  120.0
*
*  MAX   Injection
****************************************************************
4690000  sgc-afwj tmdpjun
4690101  468000000   472000000   3.54
4690200  1  728
4690201  -1.0  0.0   0.0   0.0
4690202  0.0   32.1  0.0   0.0
4690203  1.00E+05 32.1  0.0   0.0
*
*  Operator    Controlling Level
****************************************************************
4650000  sgC-afwC tmdpjun
4650101  468000000   472000000   3.54
4650200  1  730   cntrlvar 476
4650201  -1.0  0.0   0.0   0.0
4650202  0.0   0.0   0.0   0.0
4650203  32.1  32.1  0.0   0.0
4650204  1000.0   32.1  0.0   0.0
****************************************************************
** PORTABLE    DIESEL   PUMP  (KERR PUMP) FOR EMERGENCY  INJECTION
****************************************************************
****************************************************************
*  5000000  ker-vol  tmdpvol
*  5000101  1.0e6 .0 1.0e+06
*  5000102  .0 .0 .0
*  5000103  .0 .0 0000010
*  5000200  103
*  5000201  0.0   292.0 68.0
****************************************************************
*  5010000  ker-inj  tmdpjun
*  5010101  500000000   502000000   2.945E-01
*  5010200  1  743   cntrlvar 150
*  5010201  -1.0  0.0   0.0   0.0
*  5010202  0.0   9.02  0.0   0.0   * bottom of RPV
*  5010203  3.064 9.02  0.0   0.0   * BAF
*  5010204  6.722 9.02  0.0   0.0   * TAF
*  5010205  7.652 9.02  0.0   0.0   * bottom of HL
*  5010206  8.526 0.44  0.0   0.0   * top of HL
*  5010207  8.600 0.0   0.0   0.0
*  5010208  12.0  0.0   0.0   0.0   * top of RPV
****************************************************************
*  5020000  ker-pi1  branch
********************************************
*  5020001  2  0
*  5020101  2.945E-01   9.0   0.0   0.0   90.0  9.0
*  5020102  0.00015  0.0   0000000
*  5020200  3  18.0  68.0
*  5021101  502010000   503000000   0.0   0.0   1.00E+06 00001000
*  5022101  502010000   504000000   0.0   0.0   1.00E+06 00001000
*  5021201  0.0   0.0   0.0
*  5022201  0.0   0.0   0.0
****************************************************************
5030000  ker-piA  snglvol
********************************************
5030101  4.4179E-01  9.0   0.0
5030102  0.0   90.0  9.0
5030103  0.00015  0.0   0000000
5030200  3  18.0  68.0
****************************************************************
*  5040000  ker-pi3  snglvol
********************************************
*  5040101  1.4726E-01  9.0   0.0
*  5040102  0.0   90.0  9.0
*  5040103  0.00015  0.0   0000000
*  5040200  3  18.0  68.0
*************************************************
*
*
****************************************************************
5050000  chkA1 valve
********************************************
5050101  503010000   220000000   0.04909  0.0   0.0   00000100
5050201  1  0.0   0.0   0.0
5050300  chkvlv
5050301  0  1  18.0  0.0
*
****************************************************************
5060000  chkB1 valve
********************************************
5060101  503010000   320000000   0.04909  0.0   0.0   00000100
5060201  1  0.0   0.0   0.0
5060300  chkvlv
5060301  0  1  18.0  0.0
*
****************************************************************
5070000  chkC1 valve
********************************************
5070101  503010000   420000000   0.04909  0.0   0.0   00000100
5070201  1  0.0   0.0   0.0
5070300  chkvlv
5070301  0  1  18.0  0.0
*
****************************************************************
*  5080000  chkA2 valve
********************************************
*  5080101  504010000   200020001   0.04909  0.0   0.0   00000100
*  5080201  1  0.0   0.0   0.0
*  5080300  chkvlv
*  5080301  0  1  18.0  0.0
*
****************************************************************
*  5090000  chkB2 valve
********************************************
*  5090101  504010000   300020001   0.04909  0.0   0.0   00000100
*  5090201  1  0.0   0.0   0.0
*  5090300  chkvlv
*  5090301  0  1  18.0  0.0
*
****************************************************************
*  5100000  chkC2 valve
********************************************
*  5100101  504010000   400020001   0.04909  0.0   0.0   00000100
*  5100201  1  0.0   0.0   0.0
*  5100300  chkvlv
*  5100301  0  1  18.0  0.0
*
*  LPIS  SYSTEM   - CL INJECTION
*
****************************************************************
5110000  rwstsu   tmdpvol
5110101  1.0e+06  .0 1.0e+06
5110102  .0 .0 .0
5110103  .0 .0 0000010
5110200  103   550   cntrlvar 489
5110201  -2.00E+06   164.7 45.0
5110202  -1.0  164.7 45.0
5110203  0.0   164.7 45.0
5110204  24000.0  164.7 150.0
5110205  1.00E+08 164.7 150.0 *  3 days @ 400 Kg/s
****************************************************************
5120000  lpsi-A   tmdpjun
5120101  511000000   503000000   1.4726E-01
5120200  1  796   cntrlvar 492
5120201  -1.0  0.0   0.0   0.0
5120202  0.0   556.48   0.0   0.0
5120203  81.4  556.48   0.0   0.0
5120204  92.2  486.92   0.0   0.0
5120205  103.92   417.36   0.0   0.0
5120206  116.48   347.80   0.0   0.0
5120207  128.17   278.24   0.0   0.0
5120208  138.99   208.68   0.0   0.0
5120209  148.09   139.12   0.0   0.0
5120210  154.15   69.56 0.0   0.0
5120211  158.05   0.00  0.0   0.0
****************************************************************
5130000  lpsi-B   tmdpjun
5130101  511000000   503000000   1.4726E-01
5130200  1  797   cntrlvar 492
5130201  -1.0  0.0   0.0   0.0
5130202  0.0   556.48   0.0   0.0
5130203  81.4  556.48   0.0   0.0
5130204  92.2  486.92   0.0   0.0
5130205  103.92   417.36   0.0   0.0
5130206  116.48   347.80   0.0   0.0
5130207  128.17   278.24   0.0   0.0
5130208  138.99   208.68   0.0   0.0
5130209  148.09   139.12   0.0   0.0
5130210  154.15   69.56 0.0   0.0
5130211  158.05   0.00  0.0   0.0
****************************************************************
*
*  HPIS  SYSTEM   - CL INJECTION
*
****************************************************************
5150000  rwsts2   tmdpvol
5150101  1.0e+06  .0 1.0e+06
5150102  .0 .0 .0
5150103  .0 .0 0000010
5150200  103   550   cntrlvar 489
5150201  -2.00E+17   2749.7   45.0
5150202  -1.0  2749.7   45.0
5150203  0.0   2749.7   45.0
5150204  24000.0  2749.7   150.0
5150205  1.00E+08 2749.7   150.0 *  3 days @ 400 Kg/s
****************************************************************
5160000  hpsi-i1  tmdpjun
5160101  515000000   503000000   1.4726E-01
5160200  1  788   cntrlvar 493
5160201  -2.00E+17   0.00  0.0   0.0
5160202  -1.00E+08   0.00  0.0   0.0
5160203  0.0   85.56 0.0   0.0
5160204  692.8 76.52 0.0   0.0
5160205  1082.5   69.56 0.0   0.0
5160206  1418.08  62.60 0.0   0.0
5160207  1710.35  55.65 0.0   0.0
5160208  1948.50  48.69 0.0   0.0
5160209  2143.35  41.74 0.0   0.0
5160210  2294.90  34.78 0.0   0.0
5160211  2424.80  27.82 0.0   0.0
5160212  2511.40  20.87 0.0   0.0
5160213  2554.70  13.91 0.0   0.0
5160214  2556.87  0.00  0.0   0.0
*
****************************************************************
5170000  hpsi-i2  tmdpjun
5170101  515000000   503000000   1.4726E-01
5170200  1  789   cntrlvar 493
5170201  -1.0  0.0   0.0   0.0
5170202  0.0   85.56 0.0   0.0
5170203  692.8 76.52 0.0   0.0
5170204  1082.5   69.56 0.0   0.0
5170205  1418.08  62.60 0.0   0.0
5170206  1710.35  55.65 0.0   0.0
5170207  1948.50  48.69 0.0   0.0
5170208  2143.35  41.74 0.0   0.0
5170209  2294.90  34.78 0.0   0.0
5170210  2424.80  27.82 0.0   0.0
5170211  2511.40  20.87 0.0   0.0
5170212  2554.70  13.91 0.0   0.0
5170213  2556.87  0.00  0.0   0.0
*
****************************************************************
5180000  hpsi-i3  tmdpjun
5180101  515000000   503000000   1.4726E-01
5180200  1  790   cntrlvar 493
5180201  -1.0  0.0   0.0   0.0
5180202  0.0   85.56 0.0   0.0
5180203  692.8 76.52 0.0   0.0
5180204  1082.5   69.56 0.0   0.0
5180205  1418.08  62.60 0.0   0.0
5180206  1710.35  55.65 0.0   0.0
5180207  1948.50  48.69 0.0   0.0
5180208  2143.35  41.74 0.0   0.0
5180209  2294.90  34.78 0.0   0.0
5180210  2424.80  27.82 0.0   0.0
5180211  2511.40  20.87 0.0   0.0
5180212  2554.70  13.91 0.0   0.0
5180213  2556.87  0.00  0.0   0.0
****************************************************************
*  LPIS  SYSTEM   - CL INJECTION for RECIRCULATION
*
****************************************************************
5200000  rwstsu   tmdpvol
5200101  1.0e+06  .0 1.0e+06
5200102  .0 .0 .0
5200103  .0 .0 0000010
5200200  103   550   cntrlvar 489
5200201  -2.00E+06   164.7 45.0
5200202  -1.0  164.7 45.0
5200203  0.0   164.7 45.0
5200204  24000.0  164.70   150.0
5200205  1.00E+08 164.70   150.0 *  3 days @ 400 Kg/s
****************************************************************
5210000  lpsi-A   tmdpjun
5210101  520010000   503000000   1.4726E-01
5210200  1  689   cntrlvar 492
5210201  -1.0  0.00  0.0   0.0
5210202  0.0   556.48   0.0   0.0
5210203  81.4  556.48   0.0   0.0
5210204  92.2  486.92   0.0   0.0
5210205  103.92   417.36   0.0   0.0
5210206  116.48   347.80   0.0   0.0
5210207  128.17   278.24   0.0   0.0
5210208  138.99   208.68   0.0   0.0
5210209  148.09   139.12   0.0   0.0
5210210  154.15   69.56 0.0   0.0
5210211  158.05   0.00  0.0   0.0
****************************************************************
5220000  lpsi-B   tmdpjun
5220101  520010000   503000000   1.4726E-01
5220200  1  690   cntrlvar 492
5220201  -1.0  0.0   0.0   0.0
5220202  0.0   556.48   0.0   0.0
5220203  81.4  556.48   0.0   0.0
5220204  92.2  486.92   0.0   0.0
5220205  103.92   417.36   0.0   0.0
5220206  116.48   347.80   0.0   0.0
5220207  128.17   278.24   0.0   0.0
5220208  138.99   208.68   0.0   0.0
5220209  148.09   139.12   0.0   0.0
5220210  154.15   69.56 0.0   0.0
5220211  158.05   0.00  0.0   0.0
*
****************************************************************
*
*  HPIS  SYSTEM   - CL INJECTION for RECIRCULATION
*
****************************************************************
5230000  rwsts4   tmdpvol
5230101  1.0e+06  .0 1.0e+06
5230102  .0 .0 .0
5230103  .0 .0 0000010
5230200  103   550   cntrlvar 489
5230201  -2.00E+17   2749.7   45.0
5230202  -1.0  2749.7   45.0
5230203  0.0   2749.7   45.0
5230204  24000.0  2749.7   150.0
5230205  1.00E+08 2749.7   150.0 *  3 days @ 400 Kg/s
****************************************************************
5240000  hpsi-R1  tmdpjun
5240101  523000000   503000000   1.4726E-01
5240200  1  793   cntrlvar 493
5240201  -2.00E+17   0.00  0.0   0.0
5240202  -1.00E+08   0.00  0.0   0.0
5240203  0.0   85.56 0.0   0.0
5240204  692.8 76.52 0.0   0.0
5240205  1082.5   69.56 0.0   0.0
5240206  1418.08  62.60 0.0   0.0
5240207  1710.35  55.65 0.0   0.0
5240208  1948.50  48.69 0.0   0.0
5240209  2143.35  41.74 0.0   0.0
5240210  2294.90  34.78 0.0   0.0
5240211  2424.80  27.82 0.0   0.0
5240212  2511.40  20.87 0.0   0.0
5240213  2554.70  13.91 0.0   0.0
5240214  2556.87  0.00  0.0   0.0
*
****************************************************************
5250000  hpsi-R2  tmdpjun
5250101  523000000   503000000   1.4726E-01
5250200  1  794   cntrlvar 493
5250201  -1.0  0.0   0.0   0.0
5250202  0.0   85.56 0.0   0.0
5250203  692.8 76.52 0.0   0.0
5250204  1082.5   69.56 0.0   0.0
5250205  1418.08  62.60 0.0   0.0
5250206  1710.35  55.65 0.0   0.0
5250207  1948.50  48.69 0.0   0.0
5250208  2143.35  41.74 0.0   0.0
5250209  2294.90  34.78 0.0   0.0
5250210  2424.80  27.82 0.0   0.0
5250211  2511.40  20.87 0.0   0.0
5250212  2554.70  13.91 0.0   0.0
5250213  2556.87  0.00  0.0   0.0
*
****************************************************************
5260000  hpsi-R3  tmdpjun
5260101  523000000   503000000   1.4726E-01
5260200  1  795   cntrlvar 493
5260201  -1.0  0.0   0.0   0.0
5260202  0.0   85.56 0.0   0.0
5260203  692.8 76.52 0.0   0.0
5260204  1082.5   69.56 0.0   0.0
5260205  1418.08  62.60 0.0   0.0
5260206  1710.35  55.65 0.0   0.0
5260207  1948.50  48.69 0.0   0.0
5260208  2143.35  41.74 0.0   0.0
5260209  2294.90  34.78 0.0   0.0
5260210  2424.80  27.82 0.0   0.0
5260211  2511.40  20.87 0.0   0.0
5260212  2554.70  13.91 0.0   0.0
5260213  2556.87  0.00  0.0   0.0
****************************************************************
*
****************************************************************
*
*  LBLOCA   BREAK AREA
*
****************************************************************
5300000  connbrk  valve
5300101  421010000   422000000   4.125 0.0   0.0   00001000
5300201  0  47.264901   47.264901   0.0
5300300  trpvlv
5300301  754
****************************************************************
5310000  brk1  valve          * standard choking model & Abrupt Area Change
5310101  421010000   532000000   4.125 0.0   0.0   00000100 1.0   1.0   1.0
5310201  0  0.0   0.0   0.0
5310300  trpvlv
5310301  755
****************************************************************
5320000  cont1 tmdpvol
5320101  1.0e6 .0 1.0e+06
5320102  .0 .0 .0
5320103  .0 .0 0000010
5320200  102
5320201  0.0   14.7  1.0
****************************************************************
5330000  brk2  valve          * standard choking model & Abrupt Area Change
5330101  422000000   534000000   4.125 0.0   0.0   00000100 1.0   1.0   1.0
5330201  0  0.0   0.0   0.0
5330300  trpvlv
5330301  755
****************************************************************
5340000  cont2 tmdpvol
5340101  1.0e6 .0 1.0e+06
5340102  .0 .0 .0
5340103  .0 .0 0000010
5340200  102
5340201  0.0   14.7  1.0
*
*
*  SPRAY SYSTEMS  & CONTAINMENT
*
****************************************************************
5500000  rwst2 tmdpvol
5500101  1.0e6 .0 1.0e+06
5500102  .0 .0 .0
5500103  0  .0 0000010
5500200  103
5500201  0.0   14.7  45.0
****************************************************************
5510000  hpsi-i1  tmdpjun
5510101  550000000   552010000   1.4726E-01
5510200  1  564
5510201  -1.0  0.0   0.0   0.0
5510202  0.0   890.1 0.0   0.0
5510203  1.00E+06 890.1 0.0   0.0
****************************************************************
5520000  cont2 snglvol
5520101  0.0   185.0 1.73E+06 0.0   90.0  185.0
5520102  0.00015  0.0   00
5520200  104   10.3  74.5  0.0
********************************************
****************************************************************
*  this  deck  contains proprietary information *
*  do not   disseminate any   part  of this  deck  without  the   *
*  written  approval of the   eg&g  idaho severe   accident *
*  analysis manager  (r.   j. dallman).   *
****************************************************************
****************************************************************
*
*  HEAT  STRUCTURES
*
****************************************************************
******************************************************
******************************************************
*  heat  structures  100-190  model the   reactor  vessel   *
******************************************************
******************************************************
*  heat  structure   100   models   the   reactor  vessel   heads and   wall
********************************************
*  heat  structure   1001  upper head
11001000 1  9  3  0  6.5912   0
11001100 0  1
11001101 2  6.6042
11001102 6  7.1198
11001201 1  2
11001202 2  8
11001301 0.0   8
11001401 543.0 9
11001501 190010000   0  1  1  0.2716   1
11001601 0  0  0  1  0.2716   1
11001701 0  0.0   0.0   0.0   1
11001801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
********************************************
*  heat  structure   1002  upper vessel   wall
11002000 2  20 2  0  6.2161   0
11002100 0  1
11002101 2  6.2292
11002102 17 7.6823
11002201 1  2
11002202 2  19
11002301 0.0   19
11002401 543.0 20
11002501 190010000   0  1  1  2.458 1
11002502 100010000   0  1  1  2.473 2
11002601 0  0  0  1  2.458 1
11002602 0  0  0  1  2.473 2
11002701 0  0.0   0.0   0.0   2
11002801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
********************************************
*  heat  structure   1003  middle   vessel   wall
11003000 4  12 2  0  6.4792   0
11003100 0  1
11003101 2  6.4922
11003102 9  7.2422
11003201 1  2
11003202 2  11
11003301 0.0   11
11003401 543.0 12
11003501 100010000   0  1  1  0.8608   1
11003502 100010000   0  1  1  2.1022   2
11003503 102010000   0  1  1  2.8685   3
11003504 104010000   0  1  1  1.2533   4
11003601 0  0  0  1  0.8608   1
11003602 0  0  0  1  2.1022   2
11003603 0  0  0  1  2.8685   3
11003604 0  0  0  1  1.2533   4
11003701 0  0.0   0.0   0.0   4
11003801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
********************************************
*  heat  structure   1004  lower vessel   wall
11004000 7  11 2  0  6.5417   0
11004100 0  1
11004101 2  6.5547
11004102 8  7.2151
11004201 1  2
11004202 2  10
11004301 0.0   10
11004401 543.0 11
11004501 104010000   0  1  1  1.7980   1
11004502 104020000   10000 1  1  2.4   6
11004503 104050000   0  1  1  4.3270   7
11004601 0  0  0  1  1.7980   1
11004602 0  0  0  1  2.4   6
11004603 0  0  0  1  4.3270   7
11004701 0  0.0   0.0   0.0   7
11004801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   7
********************************************
*  heat  structure   1005  lower head
11005000 1  8  3  0  6.5912   0
11005100 0  1
11005101 2  6.6042
11005102 5  7.0208
11005201 1  2
11005202 2  7
11005301 0.0   7
11005401 543.0 8
11005501 106010000   0  1  1  0.4344   1
11005601 0  0  0  1  0.4344   1
11005701 0  0.0   0.0   0.0   1
11005801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
******************************************************
********************************************
*  heat  structure   1041  thermal  shield
11041000 6  3  2  0  5.9427   0
11041100 0  1
11041101 2  6.1666
11041201 1  2
11041301 0.0   2
11041401 543.0 3
11041501 104010000   0  1  1  1.798 1
11041502 104020000   10000 1  1  2.400 4
11041503 104050000   0  1  1  4.8   5
11041504 104060000   0  1  1  1.4828   6
11041601 0  0  0  1  1.798 1
11041602 0  0  0  1  2.400 4
11041603 0  0  0  1  4.8   5
11041604 0  0  0  1  1.4828   6
11041701 0  0.0   0.0   0.0   6
11041801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   6
11041901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   6
******************************************************
********************************************
*  heat  structure   1061  lower head  structures
11061000 1  3  2  0  0.0417   0
11061100 0  1
11061101 2  0.0625
11061201 1  2
11061301 0.0   2
11061401 543.0 3
11061501 0  0  0  1  3737. 1
11061601 106010000   0  1  1  3737. 1
11061701 0  0.0   0.0   0.0   1
11061901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
******************************************************
********************************************
*  heat  structure   1071  lower support  plate
11071000 1  3  1  0  0.0   0
11071100 0  1
11071101 2  0.2464
11071201 1  2
11071301 0.0   2
11071401 543.0 3
11071501 106010000   0  1  1  278.80   1
11071601 108010000   0  1  1  278.80   1
11071701 0  0.0   0.0   0.0   1
11071801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
11071901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
******************************************************
********************************************
*  heat  structure   1081  lower plenum   structures
11081000 1  2  1  0  0.0   0
11081100 0  1
11081101 1  0.0417
11081201 1  1
11081301 0.0   1
11081401 543.0 2
11081501 0  0  0  1  1269.36  1
11081601 108010000   0  1  1  1269.36  1
11081701 0  0.0   0.0   0.0   1
11081901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
******************************************************
*  =========   ==========
*  CORE  STRCUTURES
*  =========   ==========
******** Central  Core Channel   Heat  Strucuture
11111000 10 12 2  0  0.0   0
11111100 0  2
11111101 0.002177976 7  0.0003125   9  0.0010125   11
11111201 6  7  -8 9  -7 11
11111301 1.00  7  0.00  11
11111401 1520.6   2  1160.6   8  620.6 12
11111501 0  0  0  1  6120.0   10
11111601 111010000   10000 100   1  6120.0   10
11111701 101   22.598406   0.0   0.0   1
11111702 101   30.179807   0.0   0.0   2
11111703 101   33.095731   0.0   0.0   3
11111704 101   33.533119   0.0   0.0   4
11111705 101   33.095731   0.0   0.0   5
11111706 101   32.366750   0.0   0.0   6
11111707 101   31.491973   0.0   0.0   7
11111708 101   30.034011   0.0   0.0   8
11111709 101   27.263884   0.0   0.0   9
11111710 101   18.078725   0.0   0.0   10
11111901 5.246E-02   100.00   100.0 0  0  0.0   0.0   1.0   10
*
*
******** Middle   Core Channel   Heat  Strucuture
11121000 10 12 2  0  0.0   0
11121100 0  2
11121101 0.002177976 7  0.0003125   9  0.0010125   11
11121201 6  7  -8 9  -7 11
11121301 1.00  7  0.00  11
11121401 1520.6   2  1160.6   8  620.6 12
11121501 0  0  0  1  23500.8  10
11121601 112010000   10000 100   1  23500.8  10
11121701 101   77.877585   0.0   0.0   1
11121702 101   104.004259  0.0   0.0   2
11121703 101   114.052979  0.0   0.0   3
11121704 101   115.560287  0.0   0.0   4
11121705 101   114.052979  0.0   0.0   5
11121706 101   111.540799  0.0   0.0   6
11121707 101   108.526183  0.0   0.0   7
11121708 101   103.501823  0.0   0.0   8
11121709 101   93.955538   0.0   0.0   9
11121710 101   62.302068   0.0   0.0   10
11121901 5.246E-02   100.00   100.0 0  0  0.0   0.0   1.0   10
*
*
******** Peripheral  Core Channel   Heat  Strucuture
11131000 10 12 2  0  0.0   0
11131100 0  2
11131101 0.002177976 7  0.0003125   9  0.0010125   11
11131201 6  7  -8 9  -7 11
11131301 1.00  7  0.00  11
11131401 1520.6   2  1160.6   8  620.6 12
11131501 0  0  0  1  8812.8   10
11131601 113010000   10000 100   1  8812.8   10
11131701 101   21.138202   0.0   0.0   1
11131702 101   28.229727   0.0   0.0   2
11131703 101   30.957237   0.0   0.0   3
11131704 101   31.366364   0.0   0.0   4
11131705 101   30.957237   0.0   0.0   5
11131706 101   30.275360   0.0   0.0   6
11131707 101   29.457107   0.0   0.0   7
11131708 101   28.093352   0.0   0.0   8
11131709 101   25.502217   0.0   0.0   9
11131710 101   16.910561   0.0   0.0   10
11131901 5.246E-02   100.00   100.0 0  0  0.0   0.0   1.0   10
*
*
*
********************************************
*  heat  structure   1151  core  baffle
11151000 10 3  1  0  0.0   0
11151100 0  1
11151101 2  0.0833
11151201 1  2
11151301 0.0   2
11151401 543.0 3
11151501 113010000   10000 1  0  50.80 10
11151601 118010000   0  1  0  50.80 2
11151602 118020000   0  1  0  50.80 4
11151603 118030000   0  1  0  50.80 6
11151604 118040000   0  1  0  50.80 8
11151605 118050000   0  1  0  50.80 10
11151701 0  0.0   0.0   0.0   10
11151801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   10
11151901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   10
******************************************************
********************************************
*  heat  structure   1201  core  barrel
11201000 10 5  2  1  5.5781   0
11201100 0  1
11201101 4  5.7475
11201201 1  4
11201301 0.0   4
11201401 560.0 5
11201501 108010000   0  1  1  4.327 1
11201502 118010000   10000 1  1  2.400 6
11201503 171010000   0  1  1  3.0513   7
11201504 172010000   0  1  1  2.8685   8
11201505 173010000   0  1  1  2.1022   9
11201506 174010000   0  1  1  2.9585   10
11201601 104060000   0  1  1  4.327 1
11201602 104060000   -10000   1  1  2.400 6
11201603 104010000   0  1  1  3.0513   7
11201604 102010000   0  1  1  2.8685   8
11201605 100010000   0  1  1  2.1022   9
11201606 100010000   0  1  1  2.9585   10
11201701 0  0.0   0.0   0.0   10
11201801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   10
11201901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   10
******************************************************
********************************************
*  heat  structure   1501  center   channel  upper plenum   structures
11501000 4  2  1  0  0.0   0
11501100 0  1
11501101 1  0.0417
11501201 1  1
11501301 0.0   1
11501401 606.0 2
11501501 0  0  0  0  96.0  1
11501502 0  0  0  0  76.32 2
11501503 0  0  0  0  55.44 3
11501504 0  0  0  0  78.00 4
11501601 151010000   0  1  0  96.0  1
11501602 152010000   0  1  0  76.32 2
11501603 153010000   0  1  0  55.44 3
11501604 154010000   0  1  0  78.00 4
11501701 0  0.0   0.0   0.0   4
11501901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
******************************************************
********************************************
*  heat  structure   1601  middle   channel  upper plenum   structures
11601000 4  2  1  0  0.0   0
11601100 0  1
11601101 1  0.0417
11601201 1  1
11601301 0.0   1
11601401 606.0 2
11601501 0  0  0  0  367.  1
11601502 0  0  0  0  277.4 2
11601503 0  0  0  0  201.6 3
11601504 0  0  0  0  283.2 4
11601601 161010000   0  1  0  367.  1
11601602 162010000   0  1  0  277.4 2
11601603 163010000   0  1  0  201.6 3
11601604 164010000   0  1  0  283.2 4
11601701 0  0.0   0.0   0.0   4
11601901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
******************************************************
********************************************
*  heat  structure   1701  outer channel  upper plenum   structures
11701000 4  2  1  0  0.0   0
11701100 0  1
11701101 1  0.0417
11701201 1  1
11701301 0.0   1
11701401 606.0 2
11701501 0  0  0  0  137.  1
11701502 0  0  0  0  96.96 2
11701503 0  0  0  0  70.80 3
11701504 0  0  0  0  99.36 4
11701601 171010000   0  1  0  137.  1
11701602 172010000   0  1  0  96.96 2
11701603 173010000   0  1  0  70.80 3
11701604 174010000   0  1  0  99.36 4
11701701 0  0.0   0.0   0.0   4
11701901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
******************************************************
********************************************
*  heat  structure   1811  center   channel  control  housings
11811000 3  3  1  1  0.0   0
11811100 0  1
11811101 2  0.0052
11811201 1  2
11811301 0.0   2
11811401 570.0 3
11811501 181010000   0  1  0  41.31 1
11811502 181010000   0  1  0  40.99 2
11811503 181010000   0  1  0  57.69 3
11811601 152010000   0  1  0  41.31 1
11811602 153010000   0  1  0  40.99 2
11811603 154010000   0  1  0  57.69 3
11811701 0  0.0   0.0   0.0   3
11811801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
11811901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
******************************************************
********************************************
*  heat  structure   1821  middle   channel  control  housings
11821000 3  3  1  1  0.0   0
11821100 0  1
11821101 2  0.0052
11821201 1  2
11821301 0.0   2
11821401 570.0 3
11821501 182010000   0  1  0  165.2 1
11821502 182010000   0  1  0  164.0 2
11821503 182010000   0  1  0  230.8 3
11821601 162010000   0  1  0  165.2 1
11821602 163010000   0  1  0  164.0 2
11821603 164010000   0  1  0  230.8 3
11821701 0  0.0   0.0   0.0   3
11821801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
11821901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
******************************************************
********************************************
*  heat  structure   1831  outer channel  control  housings
11831000 3  3  1  1  0.0   0
11831100 0  1
11831101 2  0.0052
11831201 1  2
11831301 0.0   2
11831401 570.0 3
11831501 183010000   0  1  0  36.72 1
11831502 183010000   0  1  0  36.44 2
11831503 183010000   0  1  0  51.28 3
11831601 172010000   0  1  0  36.72 1
11831602 173010000   0  1  0  36.44 2
11831603 174010000   0  1  0  51.28 3
11831701 0  0.0   0.0   0.0   3
11831801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
11831901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
******************************************************
********************************************
*  heat  structure   1851  upper support  plate
11851000 3  6  1  1  0.0   0
11851100 0  1
11851101 5  0.3748
11851201 1  5
11851301 0.0   5
11851401 570.0 6
11851501 154010000   0  1  0  9.80  1
11851502 164010000   0  1  0  37.22 2
11851503 174010000   0  1  0  35.18 3
11851601 190010000   0  1  0  9.80  1
11851602 190010000   0  1  0  37.22 2
11851603 190010000   0  1  0  35.18 3
11851701 0  0.0   0.0   0.0   3
11851801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
11851901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
******************************************************
********************************************
*  heat  structure   1901  upper head  structures
11901000 1  2  1  0  0.0   0
11901100 0  1
11901101 1  0.0208
11901201 1  1
11901301 0.0   1
11901401 543.0 2
11901501 0  0  0  0  686.4 1
11901601 190010000   0  1  0  686.4 1
11901701 0  0.0   0.0   0.0   1
11901901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  coolant  loop  a  structures
****************************************************************
*  ht str   no.   2001  hot   leg   piping   *
12001000 4  4  2  1  1.208
12001100 0  1
12001101 3  1.474
12001201 1  3
12001301 0.0   3
12001400 0
12001401 605.8 4
12001501 200010000   10000 1  1  5.086 2
12001502 202010000   0  1  1  8.739 3
12001503 204010000   0  1  1  6.292 4
12001601 0  0  0  1  5.086 2
12001602 0  0  0  1  8.739 3
12001603 0  0  0  1  6.292 4
12001701 0  0  0  0  4
12001801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   2121  pump  suction  piping   *
12121000 5  4  2  1  1.292
12121100 0  1
12121101 3  1.531
12121201 1  3
12121301 0.0   3
12121400 0
12121401 543.  4
12121501 212010000   0  1  1  2.676 1
12121502 212020000   0  1  1  4.971 2
12121503 212030000   0  1  1  6.709 3
12121504 212040000   0  1  1  2.365 4
12121505 212050000   0  1  1  4.271 5
12121601 0  0  0  1  2.676 1
12121602 0  0  0  1  4.971 2
12121603 0  0  0  1  6.709 3
12121604 0  0  0  1  2.365 4
12121605 0  0  0  1  4.271 5
12121701 0  0  0  0  5
12121801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   5
****************************************************************
*  ht str   no.   2161  cold  leg   piping   *
12161000 7  4  2  1  1.146
12161100 0  1
12161101 3  1.398
12161201 1  3
12161301 0.0   3
12161400 0
12161401 543.  4
12161501 216010000   10000 1  1  3.293 2
12161502 218010000   10000 1  1  4.043 4
12161503 220010000   1000000  1  1  4.043 6
12161504 222010000   0  1  1  6.040 7
12161601 0  0  0  1  3.293 2
12161602 0  0  0  1  4.043 4
12161603 0  0  0  1  4.043 6
12161604 0  0  0  1  6.040 7
12161701 0  0  0  0  7
12161801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   7
****************************************************************
*  ht str   no.   2061  s. g. inlet plenum   *
12061000 1  6  3  1  5.21  0
12061100 0  1
12061101 1  5.223
12061102 4  5.73
12061201 1  1
12061202 2  5
12061301 0.0   5
12061400 0
12061401 605.8 6
12061501 206010000   0  1  1  0.25  1
12061601 0  0  0  1  0.25  1
12061701 0  0  0  0  1
12061801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   2062  s. g. partition   plate *
12062000 1  3  1  1  0.0   0
12062100 0  1
12062101 2  0.109
12062201 1  2
12062301 0.0   2
12062400 0
12062401 605.8 1
12062402 574.4 2
12062403 543.  3
12062501 206010000   0  1  1  43.03 1
12062601 210010000   0  1  1  43.03 1
12062701 0  0  0  0  1
12062801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
12062901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   2081  s. g. tubes *
12081000 8  6  2  1  0.0323   0
12081100 0  1
12081101 5  0.0363
12081201 3  5
12081301 0.0   5
12081400 0
12081401 516.1 6
12081501 208010000   10000 1  1  28373.6  3
12081502 208040000   10000 1  1  33119.2  5
12081503 208060000   10000 1  1  28373.6  8
12081601 276010000   10000 1  1  28373.6  3
12081602 276040000   0  1  1  33119.2  5
12081603 276030000   -10000   1  1  28373.6  8
12081701 0  0  0  0  8
12081801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   8
12081901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   8
****************************************************************
*  ht str   no.   2082  s. g. tubesheet   *
12082000 2  3  1  1  0.0   0
12082100 0  1
12082101 2  0.0473
12082201 2  2
12082301 0.0   2
12082400 0
12082401 574.4 3
12082501 208010000   70000 1  1  1188.3   2
12082601 0  0  0  1  1188.3   2
12082701 0  0  0  0  2
12082801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   2101  s. g. outlet   plenum   *
12101000 1  6  3  1  5.21  0
12101100 0  1
12101101 1  5.223
12101102 4  5.73
12101201 1  1
12101202 2  5
12101301 0.0   5
12101400 0
12101401 543.  6
12101501 210010000   0  1  1  .25   1
12101601 0  0  0  1  .25   1
12101701 0  0  0  0  1
12101801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   2701  s. g. upper shell *
12701000 3  4  2  1  7.02  0
12701100 0  1
12701101 3  7.322
12701201 1  3
12701301 0.0   3
12701400 0
12701401 516.1 4
12701501 272010000   0  1  1  2.18  1
12701502 270010000   0  1  1  6.936 2
12701503 280010000   0  1  1  9.47  3
12701601 0  0  0  1  2.18  1
12701602 0  0  0  1  6.936 2
12701603 0  0  0  1  9.47  3
12701701 0  0  0  0  3
12701801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
****************************************************************
*  ht str   no.   2721  s. g. middle   shell *
12721000 2  4  2  1  6.20  0
12721100 0  1
12721101 3  6.507
12721201 2  3
12721301 0.0   3
12721400 0
12721401 516.1 4
12721501 274010000   0  1  1  5.17  1
12721502 272010000   0  1  1  1.23  2
12721601 0  0  0  1  5.17  1
12721602 0  0  0  1  1.23  2
12721701 0  0  0  0  2
12721801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   2742  s. g. lower shell *
12742000 4  4  2  1  5.39  0
12742100 0  1
12742101 3  5.625
12742201 2  3
12742301 0.0   3
12742400 0
12742401 516.1 4
12742501 274040000   0  1  1  0.157 1
12742502 274030000   -10000   1  1  8.49  3
12742503 274010000   0  1  1  5.85  4
12742601 0  0  0  1  0.157 1
12742602 0  0  0  1  8.49  3
12742603 0  0  0  1  5.85  4
12742701 0  0  0  0  4
12742801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   2741  s. g. shell base  *
12741000 1  4  2  1  5.39  0
12741100 0  1
12741101 3  5.661
12741201 2  3
12741301 0.0   3
12741400 0
12741401 516.1 4
12741501 274040000   0  1  1  6.58  1
12741601 0  0  0  1  6.58  1
12741701 0  0  0  0  1
12741801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   2761  s. g. lower wrapper  *
12761000 4  3  2  1  5.125 0
12761100 0  1
12761101 2  5.156
12761201 2  2
12761301 0.0   2
12761400 0
12761401 516.1 3
12761501 276010000   0  1  1  5.57  1
12761502 276020000   10000 1  1  8.49  3
12761503 276040000   0  1  1  11.02 4
12761601 274040000   0  1  1  5.57  1
12761602 274030000   -10000   1  1  8.49  3
12761603 274010000   0  1  1  11.02 4
12761701 0  0  0  0  4
12761801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
12761901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   2781  s. g. upper wrapper/swirl  vanes *
12781000 2  3  2  1  4.67  0
12781100 0  1
12781101 2  4.701
12781201 2  2
12781301 0.0   2
12781400 0
12781401 516.1 3
12781501 276050000   0  1  1  11.751   1
12781502 272010000   0  1  1  20.808   2
12781601 278010000   0  1  1  11.751   1
12781602 270010000   0  1  1  20.808   2
12781701 0  0  0  0  2
12781801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
12781901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   2821  s. g. shell top   *
12821000 1  4  3  1  7.18  0
12821100 0  1
12821101 3  7.48
12821201 2  3
12821301 0.0   3
12821400 0
12821401 516.1 4
12821501 282010000   0  1  1  0.5   1
12821601 0  0  0  1  0.5   1
12821701 0  0  0  0  1
12821801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  coolant  loop  b  structures
****************************************************************
*  ht str   no.   3001  hot   leg   piping   *
13001000 4  4  2  1  1.208
13001100 0  1
13001101 3  1.474
13001201 1  3
13001301 0.0   3
13001400 0
13001401 605.8 4
13001501 300010000   10000 1  1  5.086 2
13001502 302010000   0  1  1  8.739 3
13001503 304010000   0  1  1  6.292 4
13001601 0  0  0  1  5.086 2
13001602 0  0  0  1  8.739 3
13001603 0  0  0  1  6.292 4
13001701 0  0  0  0  4
13001801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   3121  pump  suction  piping   *
13121000 5  4  2  1  1.292
13121100 0  1
13121101 3  1.531
13121201 1  3
13121301 0.0   3
13121400 0
13121401 543.  4
13121501 312010000   0  1  1  2.676 1
13121502 312020000   0  1  1  4.971 2
13121503 312030000   0  1  1  6.709 3
13121504 312040000   0  1  1  2.365 4
13121505 312050000   0  1  1  4.271 5
13121601 0  0  0  1  2.676 1
13121602 0  0  0  1  4.971 2
13121603 0  0  0  1  6.709 3
13121604 0  0  0  1  2.365 4
13121605 0  0  0  1  4.271 5
13121701 0  0  0  0  5
13121801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   5
****************************************************************
*  ht str   no.   3161  cold  leg   piping   *
13161000 7  4  2  1  1.146
13161100 0  1
13161101 3  1.398
13161201 1  3
13161301 0.0   3
13161400 0
13161401 543.  4
13161501 316010000   10000 1  1  3.293 2
13161502 318010000   10000 1  1  4.043 4
13161503 320010000   1000000  1  1  4.043 6
13161504 322010000   0  1  1  6.040 7
13161601 0  0  0  1  3.293 2
13161602 0  0  0  1  4.043 4
13161603 0  0  0  1  4.043 6
13161604 0  0  0  1  6.040 7
13161701 0  0  0  0  7
13161801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   7
****************************************************************
*  ht str   no.   3061  s. g. inlet plenum   *
13061000 1  6  3  1  5.21  0
13061100 0  1
13061101 1  5.223
13061102 4  5.73
13061201 1  1
13061202 2  5
13061301 0.0   5
13061400 0
13061401 605.8 6
13061501 306010000   0  1  1  0.25  1
13061601 0  0  0  1  0.25  1
13061701 0  0  0  0  1
13061801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   3062  s. g. partition   plate *
13062000 1  3  1  1  0.0   0
13062100 0  1
13062101 2  0.109
13062201 1  2
13062301 0.0   2
13062400 0
13062401 605.8 1
13062402 574.4 2
13062403 543.  3
13062501 306010000   0  1  1  43.03 1
13062601 310010000   0  1  1  43.03 1
13062701 0  0  0  0  1
13062801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
13062901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   3081  s. g. tubes *
13081000 8  6  2  1  0.0323   0
13081100 0  1
13081101 5  0.0363
13081201 3  5
13081301 0.0   5
13081400 0
13081401 516.1 6
13081501 308010000   10000 1  1  28373.6  3
13081502 308040000   10000 1  1  33119.2  5
13081503 308060000   10000 1  1  28373.6  8
13081601 376010000   10000 1  1  28373.6  3
13081602 376040000   0  1  1  33119.2  5
13081603 376030000   -10000   1  1  28373.6  8
13081701 0  0  0  0  8
13081801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   8
13081901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   8
****************************************************************
*  ht str   no.   3082  s. g. tubesheet   *
13082000 2  3  1  1  0.0   0
13082100 0  1
13082101 2  0.0473
13082201 2  2
13082301 0.0   2
13082400 0
13082401 574.4 3
13082501 308010000   70000 1  1  1188.3   2
13082601 0  0  0  1  1188.3   2
13082701 0  0  0  0  2
13082801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   3101  s. g. outlet   plenum   *
13101000 1  6  3  1  5.21  0
13101100 0  1
13101101 1  5.223
13101102 4  5.73
13101201 1  1
13101202 2  5
13101301 0.0   5
13101400 0
13101401 543.  6
13101501 310010000   0  1  1  .25   1
13101601 0  0  0  1  .25   1
13101701 0  0  0  0  1
13101801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   3701  s. g. upper shell *
13701000 3  4  2  1  7.02  0
13701100 0  1
13701101 3  7.322
13701201 1  3
13701301 0.0   3
13701400 0
13701401 516.1 4
13701501 372010000   0  1  1  2.18  1
13701502 370010000   0  1  1  6.936 2
13701503 380010000   0  1  1  9.47  3
13701601 0  0  0  1  2.18  1
13701602 0  0  0  1  6.936 2
13701603 0  0  0  1  9.47  3
13701701 0  0  0  0  3
13701801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
****************************************************************
*  ht str   no.   3721  s. g. middle   shell *
13721000 2  4  2  1  6.20  0
13721100 0  1
13721101 3  6.507
13721201 2  3
13721301 0.0   3
13721400 0
13721401 516.1 4
13721501 374010000   0  1  1  5.17  1
13721502 372010000   0  1  1  1.23  2
13721601 0  0  0  1  5.17  1
13721602 0  0  0  1  1.23  2
13721701 0  0  0  0  2
13721801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   3742  s. g. lower shell *
13742000 4  4  2  1  5.39  0
13742100 0  1
13742101 3  5.625
13742201 2  3
13742301 0.0   3
13742400 0
13742401 516.1 4
13742501 374040000   0  1  1  0.157 1
13742502 374030000   -10000   1  1  8.49  3
13742503 374010000   0  1  1  5.85  4
13742601 0  0  0  1  0.157 1
13742602 0  0  0  1  8.49  3
13742603 0  0  0  1  5.85  4
13742701 0  0  0  0  4
13742801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   3741  s. g. shell base  *
13741000 1  4  2  1  5.39  0
13741100 0  1
13741101 3  5.661
13741201 2  3
13741301 0.0   3
13741400 0
13741401 516.1 4
13741501 374040000   0  1  1  6.58  1
13741601 0  0  0  1  6.58  1
13741701 0  0  0  0  1
13741801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   3761  s. g. lower wrapper  *
13761000 4  3  2  1  5.125 0
13761100 0  1
13761101 2  5.156
13761201 2  2
13761301 0.0   2
13761400 0
13761401 516.1 3
13761501 376010000   0  1  1  5.57  1
13761502 376020000   10000 1  1  8.49  3
13761503 376040000   0  1  1  11.02 4
13761601 374040000   0  1  1  5.57  1
13761602 374030000   -10000   1  1  8.49  3
13761603 374010000   0  1  1  11.02 4
13761701 0  0  0  0  4
13761801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
13761901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   3781  s. g. upper wrapper/swirl  vanes *
13781000 2  3  2  1  4.67  0
13781100 0  1
13781101 2  4.701
13781201 2  2
13781301 0.0   2
13781400 0
13781401 516.1 3
13781501 376050000   0  1  1  11.751   1
13781502 372010000   0  1  1  20.808   2
13781601 378010000   0  1  1  11.751   1
13781602 370010000   0  1  1  20.808   2
13781701 0  0  0  0  2
13781801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
13781901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   3821  s. g. shell top   *
13821000 1  4  3  1  7.18  0
13821100 0  1
13821101 3  7.48
13821201 2  3
13821301 0.0   3
13821400 0
13821401 516.1 4
13821501 382010000   0  1  1  0.5   1
13821601 0  0  0  1  0.5   1
13821701 0  0  0  0  1
13821801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  coolant  loop  c  structures
****************************************************************
*  ht str   no.   4001  hot   leg   piping   *
14001000 4  4  2  1  1.208
14001100 0  1
14001101 3  1.474
14001201 1  3
14001301 0.0   3
14001400 0
14001401 605.8 4
14001501 400010000   10000 1  1  5.086 2
14001502 402010000   0  1  1  8.739 3
14001503 404010000   0  1  1  6.292 4
14001601 0  0  0  1  5.086 2
14001602 0  0  0  1  8.739 3
14001603 0  0  0  1  6.292 4
14001701 0  0  0  0  4
14001801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   4121  pump  suction  piping   *
14121000 5  4  2  1  1.292
14121100 0  1
14121101 3  1.531
14121201 1  3
14121301 0.0   3
14121400 0
14121401 543.  4
14121501 412010000   0  1  1  2.676 1
14121502 412020000   0  1  1  4.971 2
14121503 412030000   0  1  1  6.709 3
14121504 412040000   0  1  1  2.365 4
14121505 412050000   0  1  1  4.271 5
14121601 0  0  0  1  2.676 1
14121602 0  0  0  1  4.971 2
14121603 0  0  0  1  6.709 3
14121604 0  0  0  1  2.365 4
14121605 0  0  0  1  4.271 5
14121701 0  0  0  0  5
14121801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   5
****************************************************************
*  ht str   no.   4161  cold  leg   piping   *
14161000 8  4  2  1  1.146
14161100 0  1
14161101 3  1.398
14161201 1  3
14161301 0.0   3
14161400 0
14161401 543.  4
14161501 416010000   10000 1  1  3.293 2
14161502 418010000   10000 1  1  4.043 4
14161503 420010000   1000000  1  1  4.043 6
14161504 422010000   10000 1  1  3.020 8
14161601 0  0  0  1  3.293 2
14161602 0  0  0  1  4.043 4
14161603 0  0  0  1  4.043 6
14161604 0  0  0  1  3.020 8
14161701 0  0  0  0  8
14161801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   8
****************************************************************
*  ht str   no.   4061  s. g. inlet plenum   *
14061000 1  6  3  1  5.21  0
14061100 0  1
14061101 1  5.223
14061102 4  5.73
14061201 1  1
14061202 2  5
14061301 0.0   5
14061400 0
14061401 605.8 6
14061501 406010000   0  1  1  0.25  1
14061601 0  0  0  1  0.25  1
14061701 0  0  0  0  1
14061801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   4062  s. g. partition   plate *
14062000 1  3  1  1  0.0   0
14062100 0  1
14062101 2  0.109
14062201 1  2
14062301 0.0   2
14062400 0
14062401 605.8 1
14062402 574.4 2
14062403 543.  3
14062501 406010000   0  1  1  43.03 1
14062601 410010000   0  1  1  43.03 1
14062701 0  0  0  0  1
14062801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
14062901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   4081  s. g. tubes *
14081000 8  6  2  1  0.0323   0
14081100 0  1
14081101 5  0.0363
14081201 3  5
14081301 0.0   5
14081400 0
14081401 516.1 6
14081501 408010000   10000 1  1  28373.6  3
14081502 408040000   10000 1  1  33119.2  5
14081503 408060000   10000 1  1  28373.6  8
14081601 476010000   10000 1  1  28373.6  3
14081602 476040000   0  1  1  33119.2  5
14081603 476030000   -10000   1  1  28373.6  8
14081701 0  0  0  0  8
14081801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   8
14081901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   8
****************************************************************
*  ht str   no.   4082  s. g. tubesheet   *
14082000 2  3  1  1  0.0   0
14082100 0  1
14082101 2  0.0473
14082201 2  2
14082301 0.0   2
14082400 0
14082401 574.4 3
14082501 408010000   70000 1  1  1188.3   2
14082601 0  0  0  1  1188.3   2
14082701 0  0  0  0  2
14082801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   4101  s. g. outlet   plenum   *
14101000 1  6  3  1  5.21  0
14101100 0  1
14101101 1  5.223
14101102 4  5.73
14101201 1  1
14101202 2  5
14101301 0.0   5
14101400 0
14101401 543.  6
14101501 410010000   0  1  1  .25   1
14101601 0  0  0  1  .25   1
14101701 0  0  0  0  1
14101801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   4701  s. g. upper shell *
14701000 3  4  2  1  7.02  0
14701100 0  1
14701101 3  7.322
14701201 1  3
14701301 0.0   3
14701400 0
14701401 516.1 4
14701501 472010000   0  1  1  2.18  1
14701502 470010000   0  1  1  6.936 2
14701503 480010000   0  1  1  9.47  3
14701601 0  0  0  1  2.18  1
14701602 0  0  0  1  6.936 2
14701603 0  0  0  1  9.47  3
14701701 0  0  0  0  3
14701801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
****************************************************************
*  ht str   no.   4721  s. g. middle   shell *
14721000 2  4  2  1  6.20  0
14721100 0  1
14721101 3  6.507
14721201 2  3
14721301 0.0   3
14721400 0
14721401 516.1 4
14721501 474010000   0  1  1  5.17  1
14721502 472010000   0  1  1  1.23  2
14721601 0  0  0  1  5.17  1
14721602 0  0  0  1  1.23  2
14721701 0  0  0  0  2
14721801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   4742  s. g. lower shell *
14742000 4  4  2  1  5.39  0
14742100 0  1
14742101 3  5.625
14742201 2  3
14742301 0.0   3
14742400 0
14742401 516.1 4
14742501 474040000   0  1  1  0.157 1
14742502 474030000   -10000   1  1  8.49  3
14742503 474010000   0  1  1  5.85  4
14742601 0  0  0  1  0.157 1
14742602 0  0  0  1  8.49  3
14742603 0  0  0  1  5.85  4
14742701 0  0  0  0  4
14742801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   4741  s. g. shell base  *
14741000 1  4  2  1  5.39  0
14741100 0  1
14741101 3  5.661
14741201 2  3
14741301 0.0   3
14741400 0
14741401 516.1 4
14741501 474040000   0  1  1  6.58  1
14741601 0  0  0  1  6.58  1
14741701 0  0  0  0  1
14741801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   4761  s. g. lower wrapper  *
14761000 4  3  2  1  5.125 0
14761100 0  1
14761101 2  5.156
14761201 2  2
14761301 0.0   2
14761400 0
14761401 516.1 3
14761501 476010000   0  1  1  5.57  1
14761502 476020000   10000 1  1  8.49  3
14761503 476040000   0  1  1  11.02 4
14761601 474040000   0  1  1  5.57  1
14761602 474030000   -10000   1  1  8.49  3
14761603 474010000   0  1  1  11.02 4
14761701 0  0  0  0  4
14761801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
14761901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   4
****************************************************************
*  ht str   no.   4781  s. g. upper wrapper/swirl  vanes *
14781000 2  3  2  1  4.67  0
14781100 0  1
14781101 2  4.701
14781201 2  2
14781301 0.0   2
14781400 0
14781401 516.1 3
14781501 476050000   0  1  1  11.751   1
14781502 472010000   0  1  1  20.808   2
14781601 478010000   0  1  1  11.751   1
14781602 470010000   0  1  1  20.808   2
14781701 0  0  0  0  2
14781801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
14781901 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   4821  s. g. shell top   *
14821000 1  4  3  1  7.18  0
14821100 0  1
14821101 3  7.48
14821201 2  3
14821301 0.0   3
14821400 0
14821401 516.1 4
14821501 482010000   0  1  1  0.5   1
14821601 0  0  0  1  0.5   1
14821701 0  0  0  0  1
14821801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   1
****************************************************************
*  ht str   no.   4431  pressurizer surge line  *
14431000 3  3  2  1  0.438 0
14431100 0  1
14431101 2  0.532
14431201 1  2
14431301 0.0   2
14431400 0
14431401 629.  3
14431501 443010000   0  1  1  15.444   1
14431502 443020000   0  1  1  18.298   2
14431503 443030000   0  1  1  23.185   3
14431601 0  0  0  1  15.444   1
14431602 0  0  0  1  18.298   2
14431603 0  0  0  1  23.185   3
14431701 0  0  0  0  3
14431801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   3
****************************************************************
*  ht str   no.   4401  pressurizer heads *
14401000 2  4  3  1  3.50  0
14401100 0  1
14401101 1  3.516
14401102 2  3.849
14401201 1  1
14401202 2  3
14401301 0.0   3
14401400 0
14401401 653.  4
14401501 440010000   1060000  1  1  0.5   2
14401601 0  0  0  1  .5 2
14401701 0  0  0  0  2
14401801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   2
****************************************************************
*  ht str   no.   4411  pressurizer cylinder *
14411000 6  4  2  1  3.50  0
14411100 0  1
14411101 1  3.516
14411102 2  3.849
14411201 1  1
14411202 2  3
14411301 0.0   3
14411400 0
14411401 653.  4
14411501 441010000   10000 1  1  5.056 6
14411601 0  0  0  1  5.056 6
14411701 0  0  0  0  6
14411801 0.0   100.0 100.0 0.0   0.0   0.0   0.0   1.0   6
******************************************************
$****************************************************************
$  material properties  *
$  conductivity   values   in units of btu/sec-ft-degf   *
$  volumetric  heat  capacity values   in units of btu/ft3-degf   *
$****************************************************************
$**************************************************
$  thermal  properties  of ss-316l  -  composition 1  *
$**************************************************
20100100 tbl/fctn 1  1  *  ss-316l
$  thermal  properties  of ss-316l
20100101 32.0  0.00215
20100102 100.0 0.00215
20100103 800.0 0.00306
20100104 1600.0   0.00397
20100105 3600.0   0.00397
20100151 32.0  61.30
20100152 400.0 61.30
20100153 600.0 64.60
20100154 800.0 67.10
20100155 1000.0   69.35
20100156 4000.0   69.35
*
$*******************************************************
$  thermal  properties  of carbon   steel -  composition 2  *
$*******************************************************
20100200 tbl/fctn 1  1  *  carbon   steel
20100201 32.0  0.01126
20100202 80.0  0.01126
20100203 440.33   0.01009
20100204 800.33   0.00908
20100205 1160.33  0.00824
20100206 1520.33  0.00756
20100207 1880.33  0.00705
20100208 2240.33  0.00670
20100209 2600.33  0.00652
20100210 2960.33  0.00649
20100251 32.0  57.29
20100252 80.00 57.29
20100253 200.03   57.29
20100254 1600.07  82.04
20100255 2600.33  104.71
20100256 2960.33  112.49
$*******************************************************
$  thermal  properties  of inconel  600   -  composition 3  *
$******************************************************
20100300 tbl/fctn 1  1  *  inconel  600
20100301 32.0  0.00236
20100302 100.0 0.00236
20100303 300.0 0.00267
20100304 500.0 0.00294
20100305 700.0 0.00322
20100306 900.0 0.00350
20100307 1100.0   0.00378
20100308 3000.0   0.00659
20100351 52.225
*
$******************************************************
$  thermal  properties  of Uranium  Dioxide  -  composition 6  *
$******************************************************
20100600 tbl/fctn 1  1  *  uo2   1  1
20100601 32.00 1.3749E-03
20100602 181.49   1.2492E-03
20100603 295.00   1.1446E-03
20100604 408.49   1.0562E-03
20100605 521.98   9.8048E-04
20100606 635.47   9.1496E-04
20100607 748.98   8.5774E-04
20100608 862.47   8.0735E-04
20100609 975.96   7.6268E-04
20100610 1089.46  7.2283E-04
20100611 1202.95  6.8710E-04
20100612 1316.44  6.5494E-04
20100613 1429.93  6.2589E-04
20100614 1543.44  5.9955E-04
20100615 1656.93  5.7565E-04
20100616 1770.42  5.5391E-04
20100617 1883.93  5.3413E-04
20100618 1997.42  5.1613E-04
20100619 2110.91  4.9974E-04
20100620 2224.40  4.8488E-04
20100621 2337.91  4.7140E-04
20100622 2451.40  4.5925E-04
20100623 2564.89  4.4835E-04
20100624 2678.40  4.3864E-04
20100625 2791.89  4.3005E-04
20100626 2905.38  4.2257E-04
20100627 3018.87  4.1617E-04
20100628 3132.37  4.1082E-04
20100629 3245.86  4.0651E-04
20100630 3359.35  4.0320E-04
20100631 3472.86  4.0094E-04
20100632 3586.35  3.9967E-04
20100633 3699.84  3.9943E-04
20100634 3813.33  4.0022E-04
20100635 3926.84  4.0205E-04
20100636 4040.33  4.0493E-04
*
*
*  volumetric  heat  capacity of uo2   6
20100651 32.00 39.26
20100652 200.41   40.05
20100653 332.82   40.85
20100654 465.24   41.65
20100655 597.65   42.44
20100656 730.06   43.24
20100657 862.47   44.04
20100658 994.87   44.83
20100659 1127.28  45.63
20100660 1259.71  46.42
20100661 1392.12  47.22
20100662 1524.52  48.02
20100663 1656.93  48.81
20100664 1789.34  49.61
20100665 1921.75  50.41
20100666 2054.17  51.20
20100667 2186.58  52.00
20100668 2318.99  52.79
20100669 2451.40  53.59
20100670 2583.81  54.39
20100671 2716.21  55.18
20100672 2848.64  55.98
20100673 2981.05  56.78
20100674 3113.46  57.57
20100675 3245.86  58.37
20100676 3378.27  59.17
20100677 3510.68  59.96
20100678 3643.11  60.76
20100679 3775.51  61.55
20100680 3907.92  62.35
20100681 4040.33  63.15
*
$******************************************************
$  thermal  properties  of CLAD     -  composition 7  *
$******************************************************
20100700 tbl/fctn 1  1  *  CLAD  1  1
20100701 32.00 2.6498E-03
20100702 173.93   2.7474E-03
20100703 267.53   2.8451E-03
20100704 361.13   2.9427E-03
20100705 454.73   3.0405E-03
20100706 548.33   3.1380E-03
20100707 641.93   3.2356E-03
20100708 735.53   3.3334E-03
20100709 829.13   3.4309E-03
20100710 922.73   3.5287E-03
20100711 1016.33  3.6263E-03
20100712 1109.93  3.7239E-03
20100713 1203.53  3.8216E-03
20100714 1297.13  3.9192E-03
20100715 1390.73  4.0169E-03
20100716 1484.33  4.1145E-03
20100717 1577.93  4.2121E-03
20100718 1671.53  4.3098E-03
20100719 1765.13  4.4074E-03
20100720 1858.73  4.5052E-03
20100721 1952.33  4.6027E-03
20100722 2045.93  4.7003E-03
20100723 2139.53  4.7981E-03
20100724 2198.93  4.8598E-03
*
*
20100751 32.00 27.18
20100752 173.93   28.09
20100753 267.53   29.00
20100754 361.13   29.92
20100755 454.73   30.83
20100756 548.33   31.74
20100757 641.93   32.65
20100758 735.53   33.56
20100759 829.13   34.47
20100760 922.73   35.39
20100761 1016.33  36.30
20100762 1109.93  37.21
20100763 1203.53  38.12
20100764 1297.13  39.03
20100765 1390.73  39.94
20100766 1484.33  40.86
20100767 1577.93  41.77
20100768 1671.53  42.68
20100769 1765.13  43.59
20100770 1858.73  44.50
20100771 1952.33  45.41
20100772 2045.93  46.33
20100773 2139.53  47.24
20100774 2198.93  47.82
*
*
$******************************************************
$  thermal  properties  of GAS   GAP   -  composition 8  *
$******************************************************
20100800 tbl/fctn 1  1  *
20100801 32.00 6.5033E-05
20100802 980.33   6.5033E-05
20100803 2240.33  3.2518E-04
20100804 4040.33  3.2518E-04
*
*
20100851 32.00 1.800E-02
20100852 440.33   3.169E-02
20100853 980.33   4.233E-02
20100854 2060.33  4.765E-02
20100855 4040.33  4.765E-02
*
*
********************************************
*  reactor  vessel   liquid   level versus   volume
********************************************
20215000 reac-t
20215001 0.0   0.0
20215002 12.3618  1.7455
20215003 24.29579 3.06437
20215004 51.77049 6.72197
20215005 62.42158 7.652006
20215006 66.552025   8.089165
20215007 71.365533   8.526324
20215008 78.504919   9.167074
20215009 87.425829   10.068824
20215010 87.75257 10.183064
20215011 101.91097   12.023784
********************************************
*
********************************************
*  POWER (IMPOSED)
********************************************
*
*     MW when using British Unit!
20210100 power 522   1.0   1.62166
20210101 -1.0  1.0
20210102 0.0   0.070424195
20210103 1.0   0.065121760
20210104 3.0   0.059426551
20210105 7.0   0.053613511
20210106 13.0  0.049096622
20210107 27.0  0.043872742
20210108 54.0  0.039080911
20210109 108.0 0.034328358
20210110 222.0 0.030164965
20210111 444.0 0.026472899
20210112 888.0 0.022741555
20210113 1788.0   0.018774548
20210114 3600.0   0.015043205
20210115 7200.0   0.012136685
20210116 43200.0  0.007659073
20210117 86400.0  0.006323645
20210118 172800.0 0.005106049
*
****************************************************************
*
*  control  variables
*
****************************************************************
20500100 tstepsz  sum   1.0000000   .06000000   0  0
20500101 0.0   1.0   time  0
20500102 -1.0  cntrlvar 2
****************************************************************
20500200 oldtime  sum   1.0000000   300.00000   0  0
20500201 0.0   1.0   time  0
****************************************************************
20500300 hlmdterr sum   2.2046000   3.4642643   0
20500301 12738.0  -1.0  mflowj   172040000
20500302 -1.0  mflowj   172050000
20500303 -1.0  mflowj   172060000
****************************************************************
20500400 rcpspeed integral .0190 1353.8436   0  3  0.0   1500.
20500401 cntrlvar 3
****************************************************************
*  sga   main  feedwater   controller  -  computes liquid   mass  in "downcomer,"
*  "boiler,"   and   separator   (ctlvars 10 -  "21),"   determines  deviation   from
*  desired  value (ctlvar  "22),"   and   a  new   main  feed  flow  (ctlvar  25).
*  REFERENCE   MASS: 97000 lbm
*  WARNING: Each Volume    is multiplied  by 0.0624   for   converting from   kg/m3 to lbm/ft3
****************************************************************
20501000 lm270 mult  43.70 0.0   1  0
20501001 voidf 270010000   rhof  270010000
****************************************************************
20501100 lm272 mult  21.23 0.0   1  0
20501101 voidf 272010000   rhof  272010000
****************************************************************
20501200 lm27401  mult  11.97 0.0   1  0
20501201 voidf 274010000   rhof  274010000
****************************************************************
20501300 lm27402  mult  3.76  0.0   1  0
20501301 voidf 274020000   rhof  274020000
****************************************************************
20501400 lm27403  mult  3.76  0.0   1  0
20501401 voidf 274030000   rhof  274030000
****************************************************************
20501500 lm27404  mult  2.98  0.0   1  0
20501501 voidf 274040000   rhof  274040000
****************************************************************
20501600 lm27601  mult  23.10 0.0   1  0
20501601 voidf 276010000   rhof  276010000
****************************************************************
20501700 lm27602  mult  29.10 0.0   1  0
20501701 voidf 276020000   rhof  276020000
****************************************************************
20501800 lm27603  mult  29.10 0.0   1  0
20501801 voidf 276030000   rhof  276030000
****************************************************************
20501900 lm27604  mult  41.13 0.0   1  0
20501901 voidf 276040000   rhof  276040000
****************************************************************
20502000 lm27605  mult  16.23 0.0   1  0
20502001 voidf 276050000   rhof  276050000
****************************************************************
20502100 lm278 mult  21.85 0.0   1  0
20502101 voidf 278010000   rhof  278010000
****************************************************************
20502200 sgamerr  sum   1.0000000   0.0   1  0
20502201 91800.0  -1.0  cntrlvar 10
20502202 -1.0  cntrlvar 11
20502203 -1.0  cntrlvar 12
20502204 -1.0  cntrlvar 13
20502205 -1.0  cntrlvar 14
20502206 -1.0  cntrlvar 15
20502207 -1.0  cntrlvar 16
20502208 -1.0  cntrlvar 17
20502209 -1.0  cntrlvar 18
20502210 -1.0  cntrlvar 19
+  -1.0  cntrlvar 20
+  -1.0  cntrlvar 21
****************************************************************
20502500 sgaflow  prop-int 1.0000000   974.0 0  0
20502501 0.140 0.02  cntrlvar 22
****************************************************************
*  sgb   main  feedwater   controller  -  computes liquid   mass  in "downcomer,"
*  "boiler,"   and   separator   (ctlvars 30 -  "41),"   determines  deviation   from
*  desired  value (ctlvar  "42),"   and   a  new   main  feed  flow  (ctlvar  45).
****************************************************************
20503000 lm370 mult  43.70 0.0   1  0
20503001 voidf 370010000   rhof  370010000
****************************************************************
20503100 lm372 mult  21.23 0.0   1  0
20503101 voidf 372010000   rhof  372010000
****************************************************************
20503200 lm37401  mult  11.97 0.0   1  0
20503201 voidf 374010000   rhof  374010000
****************************************************************
20503300 lm37402  mult  3.76  0.0   1  0
20503301 voidf 374020000   rhof  374020000
****************************************************************
20503400 lm37403  mult  3.76  0.0   1  0
20503401 voidf 374030000   rhof  374030000
****************************************************************
20503500 lm37404  mult  2.98  0.0   1  0
20503501 voidf 374040000   rhof  374040000
****************************************************************
20503600 lm37601  mult  23.10 0.0   1  0
20503601 voidf 376010000   rhof  376010000
****************************************************************
20503700 lm37602  mult  29.10 0.0   1  0
20503701 voidf 376020000   rhof  376020000
****************************************************************
20503800 lm37603  mult  29.10 0.0   1  0
20503801 voidf 376030000   rhof  376030000
****************************************************************
20503900 lm37604  mult  41.13 0.0   1  0
20503901 voidf 376040000   rhof  376040000
****************************************************************
20504000 lm37605  mult  16.23 0.0   1  0
20504001 voidf 376050000   rhof  376050000
****************************************************************
20504100 lm378 mult  21.85 0.0   1  0
20504101 voidf 378010000   rhof  378010000
****************************************************************
20504200 sgbmerr  sum   1.0000000   0.0   1  0
20504201 91800.0  -1.0  cntrlvar 30
20504202 -1.0  cntrlvar 31
20504203 -1.0  cntrlvar 32
20504204 -1.0  cntrlvar 33
20504205 -1.0  cntrlvar 34
20504206 -1.0  cntrlvar 35
20504207 -1.0  cntrlvar 36
20504208 -1.0  cntrlvar 37
20504209 -1.0  cntrlvar 38
20504210 -1.0  cntrlvar 39
+  -1.0  cntrlvar 40
+  -1.0  cntrlvar 41
****************************************************************
20504500 sgbflow  prop-int 1.0000000   974.0 0  0
20504501 0.140 0.02  cntrlvar 42
****************************************************************
*  sgc   main  feedwater   controller  -  computes liquid   mass  in "downcomer,"
*  "boiler,"   and   separator   (ctlvars 50 -  "61),"   determines  deviation   from
*  desired  value (ctlvar  "62),"   and   a  new   main  feed  flow  (ctlvar  65).
****************************************************************
20505000 lm470 mult  43.70 0.0   1  0
20505001 voidf 470010000   rhof  470010000
****************************************************************
20505100 lm472 mult  21.23 0.0   1  0
20505101 voidf 472010000   rhof  472010000
****************************************************************
20505200 lm47401  mult  11.97 0.0   1  0
20505201 voidf 474010000   rhof  474010000
****************************************************************
20505300 lm47402  mult  3.76  0.0   1  0
20505301 voidf 474020000   rhof  474020000
****************************************************************
20505400 lm47403  mult  3.76  0.0   1  0
20505401 voidf 474030000   rhof  474030000
****************************************************************
20505500 lm47404  mult  2.98  0.0   1  0
20505501 voidf 474040000   rhof  474040000
****************************************************************
20505600 lm47601  mult  23.10 0.0   1  0
20505601 voidf 476010000   rhof  476010000
****************************************************************
20505700 lm47602  mult  29.10 0.0   1  0
20505701 voidf 476020000   rhof  476020000
****************************************************************
20505800 lm47603  mult  29.10 0.0   1  0
20505801 voidf 476030000   rhof  476030000
****************************************************************
20505900 lm47604  mult  41.13 0.0   1  0
20505901 voidf 476040000   rhof  476040000
****************************************************************
20506000 lm47605  mult  16.23 0.0   1  0
20506001 voidf 476050000   rhof  476050000
****************************************************************
20506100 lm478 mult  21.85 0.0   1  0
20506101 voidf 478010000   rhof  478010000
****************************************************************
20506200 sgcmerr  sum   1.0000000   0.0   1  0
20506201 91800.0  -1.0  cntrlvar 50
20506202 -1.0  cntrlvar 51
20506203 -1.0  cntrlvar 52
20506204 -1.0  cntrlvar 53
20506205 -1.0  cntrlvar 54
20506206 -1.0  cntrlvar 55
20506207 -1.0  cntrlvar 56
20506208 -1.0  cntrlvar 57
20506209 -1.0  cntrlvar 58
20506210 -1.0  cntrlvar 59
+  -1.0  cntrlvar 60
+  -1.0  cntrlvar 61
****************************************************************
20506500 sgcflow  prop-int 1.0000000   974.0 0  0
20506501 0.140 0.02  cntrlvar 62
****************************************************************
*  the   following   control  system   regulates   the   cold  leg   temperatures
*  in loop  a  by adjusting   the   sga   steam line  valve.
20507000 cla-dt   sum   1.0000000   -.0235750   0  0
20507001 -555.6   1.0   tempf 216010000
****************************************************************
20507100 inta-dt  integral .03000000   .40617050   0  3  0.0   1.0000000
20507101 cntrlvar 70
****************************************************************
*  the   following   control  system   regulates   the   cold  leg   temperatures
*  in loop  b  by adjusting   the   sgb   steam line  valve.
20507200 clb-dt   sum   1.0000000   -.0235750   0  0
20507201 -555.6   1.0   tempf 316010000
****************************************************************
20507300 intb-dt  integral .03000000   .40617050   0  3  0.0   1.0000000
20507301 cntrlvar 72
****************************************************************
*  the   following   control  system   regulates   the   cold  leg   temperatures
*  in loop  c  by adjusting   the   steam line  valve.
20507400 clc-dt   sum   1.0000000   -.0235750   0  0
20507401 -555.6   1.0   tempf 416010000
****************************************************************
20507500 intc-dt  integral .03000000   .40617050   0  3  0.0   1.0000000
20507501 cntrlvar 74
****************************************************************
20509500 prlverr  sum   1.0   0.0   1  0
20509501 780.  -66.317  voidf 440010000   -194.561 voidf 441010000
20509502 -194.561 voidf 441020000   -194.561 voidf 441030000
20509503 -194.561 voidf 441040000   -194.561 voidf 441050000
20509504 -194.561 voidf 441060000   -66.317  voidf 441070000
****************************************************************
******************************************************************
20511100 corecq   sum   1.0   0.0   0  0
20511101 0.0   1.0   q  111010000   1.0   q  111020000
20511102 1.0   q  111030000   1.0   q  111040000
20511103 1.0   q  111050000   1.0   q  111060000
20511104 1.0   q  111070000   1.0   q  111080000
20511105 1.0   q  111090000   1.0   q  111100000
******************************************************************
20511200 coremq   sum   1.0   0.0   0  0
20511201 0.0   1.0   q  112010000   1.0   q  112020000
20511202 1.0   q  112030000   1.0   q  112040000
20511203 1.0   q  112050000   1.0   q  112060000
20511204 1.0   q  112070000   1.0   q  112080000
20511205 1.0   q  112090000   1.0   q  112100000
******************************************************************
20511300 coreoq   sum   1.0   0.0   0  0
20511301 0.0   1.0   q  113010000   1.0   q  113020000
20511302 1.0   q  113030000   1.0   q  113040000
20511303 1.0   q  113050000   1.0   q  113060000
20511304 1.0   q  113070000   1.0   q  113080000
20511305 1.0   q  113090000   1.0   q  113100000
******************************************************************
20511500 coreq sum   1.0   0.0   0  0
20511501 0.0   1.0   cntrlvar 111   1.0   cntrlvar 112   1.0   cntrlvar 113
******************************************************
******************************************************
*  reactor  vessel   liquid   level control  variables
*  core  liquid   levels   in METERS   (multiply 1.2 foot * 0.3048 m)
********************************************
20512100 coreclvl sum   1.0   3.6576   0
20512101 0.0   0.36576  voidf 111010000   0.36576  voidf 111020000
20512102 0.36576  voidf 111030000   0.36576  voidf 111040000
20512103 0.36576  voidf 111050000   0.36576  voidf 111060000
20512104 0.36576  voidf 111070000   0.36576  voidf 111080000
20512105 0.36576  voidf 111090000   0.36576  voidf 111100000
********************************************
20512200 coremlvl sum   1.0   3.6576   0
20512201 0.0   0.36576  voidf 112010000   0.36576  voidf 112020000
20512202 0.36576  voidf 112030000   0.36576  voidf 112040000
20512203 0.36576  voidf 112050000   0.36576  voidf 112060000
20512204 0.36576  voidf 112070000   0.36576  voidf 112080000
20512205 0.36576  voidf 112090000   0.36576  voidf 112100000
********************************************
20512300 coreolvl sum   1.0   3.6576   0
20512301 0.0   0.36576  voidf 113010000   0.36576  voidf 113020000
20512302 0.36576  voidf 113030000   0.36576  voidf 113040000
20512303 0.36576  voidf 113050000   0.36576  voidf 113060000
20512304 0.36576  voidf 113070000   0.36576  voidf 113080000
20512305 0.36576  voidf 113090000   0.36576  voidf 113100000
********************************************
*  core  bypass   liquid   level
********************************************
20512500 bypslvl  sum   1.0   3.6576   0
20512501 0.0   0.73152  voidf 118010000   0.73152  voidf 118020000
20512502 0.73152  voidf 118030000   0.73152  voidf 118040000
20512503 0.73152  voidf 118050000
********************************************
*  downcomer   liquid   level
********************************************
20513000 dwcmlvl  sum   1.0   3.6576   0
20513001 0.0   0.73152  voidf 104020000   0.73152  voidf 104030000
20513002 0.73152  voidf 104040000   1.46304  voidf 104050000
********************************************
*  reactor  vessel   liquid   volume
********************************************
20513900 rvliqvpt sum   1.0   0.0   1
20513901 0.0   1.63197  voidf 100010000   1.31836  voidf 100010000
20513902 1.78539  voidf 102010000   2.12489  voidf 104010000
20513903 2.05724  cntrlvar 130   3.39929  voidf 104060000
20513904 12.3618  voidf 106010000   8.5347   voidf 108010000
20513905 0.63406  cntrlvar 121   2.4348   cntrlvar 122
20513906 0.91305  cntrlvar 123   1.47251  cntrlvar 125
*
20514000 rvliqvol sum   1.0   101.91097   0
20514001 0.0   1.0   cntrlvar 139
20514002 1.11002  voidf 151010000   0.76201  voidf 152010000
20514003 0.51789  voidf 153010000   0.72884  voidf 154010000
20514004 4.27868  voidf 161010000   2.92032  voidf 162010000
20514005 1.97751  voidf 163010000   2.78302  voidf 164010000
20514006 3.13751  voidf 171010000   2.79317  voidf 172010000
20514007 2.01084  voidf 173010000   2.82993  voidf 174010000
20514008 0.55558  voidf 181010000   2.22231  voidf 182010000
20514009 0.49385  voidf 183010000   15.1584  voidf 190010000
********************************************
*  reactor  vessel   liquid   level
********************************************
20515000 rvlvl function 1.0   12.023784   0
20515001 cntrlvar 140   150
******************************************************************
******************************************************************
*  the   following   control  variables   compute  "mass,"  "enthalpy," and   energy
*  flows through  the   relief   valves   to the   containment for   contain  input
******************************************************************
20540100 mflow sum   1.0   0.0   0  0
20540101 0.0   1.0   mflowj   444000000   1.0   mflowj   445000000
******************************************************************
20540200 liqfrac  sum   1.0   0.0   0  0
20540201 1.0   -1.0  quals 440010000
******************************************************************
20540300 stmfrac  sum   1.0   0.0   0  0
20540301 1.0   -1.0  quala 440010000
******************************************************************
20540400 liqflow  mult  1.0   0.0   0  0
20540401 cntrlvar 401   cntrlvar 402
******************************************************************
20540500 liqflowi integral 1.0   0.0   0  0
20540501 cntrlvar 404
******************************************************************
20540600 stmflow  mult  1.0   0.0   0  0
20540601 cntrlvar 401   quals 440010000   cntrlvar 403
******************************************************************
20540700 stmflowi integral 1.0   0.0   0  0
20540701 cntrlvar 406
******************************************************************
20540800 ncflow   mult  1.0   0.0   0  0
20540801 cntrlvar 401   quals 440010000   quala 440010000
******************************************************************
20540900 ncflowi  integral 1.0   0.0   0  0
20540901 cntrlvar 408
******************************************************************
20541100 liqwork  div   1.0   0.0   0  0
20541101 rhof  440010000   p  440010000
******************************************************************
20541200 liqenth  sum   1.0   0.0   0  0
20541201 0.0   1.0   uf 440010000   1.0   cntrlvar 411
******************************************************************
20541300 liqenthi integral 1.0   0.0   0  0
20541301 cntrlvar 412
******************************************************************
20541400 vapwork  div   1.0   0.0   0  0
20541401 rhog  440010000   p  440010000
******************************************************************
20541500 vapenth  sum   1.0   0.0   0  0
20541501 0.0   1.0   ug 440010000   1.0   cntrlvar 414
******************************************************************
20541600 vapenthi integral 1.0   0.0   0  0
20541601 cntrlvar 415
******************************************************************
20542100 liqener  mult  1.0   0.0   0  0
20542101 cntrlvar 404   cntrlvar 412
******************************************************************
20542200 liqeneri integral 1.0   0.0   0  0
20542201 cntrlvar 421
******************************************************************
20542300 vapener  mult  1.0   0.0   0  0
20542301 cntrlvar 406   cntrlvar 415
******************************************************************
20542400 vapeneri integral 1.0   0.0   0  0
20542401 cntrlvar 423
******************************************************************
20542500 ncener   mult  1.0   0.0   0  0
20542501 cntrlvar 408   cntrlvar 415
******************************************************************
20542600 nceneri  integral 1.0   0.0   0  0
20542601 cntrlvar 425
****************************************************************
****************************************************************        conversion in meters
20543000 prlvl sum   0.3048   0.0   1  0
20543001 0.0   3.16  voidf 440010000   5.056 voidf 441010000
20543002 5.056 voidf 441020000   5.056 voidf 441030000
20543003 5.056 voidf 441040000   5.056 voidf 441050000
20543004 5.056 voidf 441060000   3.16  voidf 441070000
*
*  COMPUTE  AFW   MASS FLOW
******************************************************************
20543100 AFW_sga  integral 1.0   0.0   1  0
20543101 mflowj   269000000
******************************************************************
20543200 AFW_sga  integral 1.0   0.0   1  0
20543201 mflowj   265000000
******************************************************************
20543300 AFW_sgb  integral 1.0   0.0   1  0
20543301 mflowj   369000000
******************************************************************
20543400 AFW_sgb  integral 1.0   0.0   1  0
20543401 mflowj   365000000
******************************************************************
20543500 AFW_sgc  integral 1.0   0.0   1  0
20543501 mflowj   469000000
******************************************************************
20543600 AFW_sgc  integral 1.0   0.0   1  0
20543601 mflowj   465000000
******************************************************************
20543700 Tot_AFW  sum   1.0   -411622.5   0  0
20543701 -411622.5   1.0   cntrlvar 431
20543702 1.0   cntrlvar 432   1.0   cntrlvar 433
20543703 1.0   cntrlvar 434   1.0   cntrlvar 435
20543704 1.0   cntrlvar 436
****************************************************************
*  SG-A  Compute the    level in meters!
20544000 SGA_LEV  sum   0.3048   0.0   1  0
20544001 0.0   6.74  voidf 276010000
20544002 8.49  voidf 276020000
20544003 8.49  voidf 276030000
20544004 11.02 voidf 276040000
20544005 3.917 voidf 276050000
20544006 6.936 voidf 278010000
20544007 9.470 voidf 280010000
20544008 5.107 voidf 282010000
*
*  SG-B  Compute the    level in meters!
20544100 SGB_LEV  sum   0.3048   0.0   1  0
20544101 0.0   6.74  voidf 376010000
20544102 8.49  voidf 376020000
20544103 8.49  voidf 376030000
20544104 11.02 voidf 376040000
20544105 3.917 voidf 376050000
20544106 6.936 voidf 378010000
20544107 9.470 voidf 380010000
20544108 5.107 voidf 382010000
*
*  SG-C  Compute the    level in meters!
20544200 SGC_LEV  sum   0.3048   0.0   1  0
20544201 0.0   6.74  voidf 476010000
20544202 8.49  voidf 476020000
20544203 8.49  voidf 476030000
20544204 11.02 voidf 476040000
20544205 3.917 voidf 476050000
20544206 6.936 voidf 478010000
20544207 9.470 voidf 480010000
20544208 5.107 voidf 482010000
*
*
****************************************************************
*  SGs COOL DOWN by  OPERATOR at 100 F/HR or 55.5 C/hr
****************************************************************
*
*  SG-A
*
*  ACTIVATION
****************************************************************        SCALING  INITIAL VALUE  Do Not Compute
20544300 attiv tripunit 1.0   0.0   0  0
20544301 732
****************************************************************
*
****************************************************************
20544400 dela  delay 1.00000  0.0   1  0
20544401 tempf 276010000   1.0   2
****************************************************************
*
****************************************************************
20544500 diff1 sum   1.00  0.0   1  0
20544501 0.0   1.0   tempf 276010000   -1.0  cntrlvar 444
****************************************************************
*
****************************************************************
20544600 dx1   mult  1.00  0.0   1  0
20544601 cntrlvar 445
****************************************************************
*
****************************************************************
20544700 flow1 sum   1.000 0.0   1  0
20544701 0.0154   1.0   cntrlvar 446
****************************************************************
*
****************************************************************
20544800 actA  mult  1.000 0.0   0
20544801 cntrlvar 447   cntrlvar 443
****************************************************************
*
****************************************************************
20544900 porvA prop-int 1.00000  0.0   0  3  0.00  1.000
20544901 0.1   0.14  cntrlvar 448
****************************************************************
*
****************************************************************
20545000 actA1 mult  1.000 0.0   0
20545001 cntrlvar 449   cntrlvar 443
****************************************************************
*
*  SG-B
*
*  ACTIVATION
****************************************************************        SCALING  INITIAL VALUE  Do Not Compute
20545100 attiv tripunit 1.0   0.0   0  0
20545101 733
****************************************************************
*
****************************************************************
20545200 delb  delay 1.00000  0.0   1  0
20545201 tempf 376010000   1.0   2
****************************************************************
*
****************************************************************
20545300 diff2 sum   1.00  0.0   1  0
20545301 0.0   1.0   tempf 376010000   -1.0  cntrlvar 452
****************************************************************
*
****************************************************************
20545400 dx2   mult  1.00  0.0   1  0
20545401 cntrlvar 453
****************************************************************
*
****************************************************************
20545500 flow2 sum   1.000 0.0   1  0
20545501 0.0154   1.0   cntrlvar 454
****************************************************************
*
****************************************************************
20545600 actB  mult  1.000 0.0   0
20545601 cntrlvar 455   cntrlvar 451
****************************************************************
*
****************************************************************
20545700 porvB prop-int 1.00000  0.0   0  3  0.00  1.000
20545701 0.1   0.14  cntrlvar 456
****************************************************************
*
****************************************************************
20545800 actB1 mult  1.000 0.0   0
20545801 cntrlvar 457   cntrlvar 451
****************************************************************
*
*  SG-C
*
*  ACTIVATION
****************************************************************        SCALING  INITIAL VALUE  Do Not Compute
20546000 attiv tripunit 1.0   0.0   0  0
20546001 734
****************************************************************
*
****************************************************************
20546100 delc  delay 1.00000  0.0   1  0
20546101 tempf 476010000   1.0   2
****************************************************************
*
****************************************************************
20546200 diff3 sum   1.00  0.0   1  0
20546201 0.0   1.0   tempf 476010000   -1.0  cntrlvar 461
****************************************************************
*
****************************************************************
20546300 dx3   mult  1.00  0.0   1  0
20546301 cntrlvar 462
****************************************************************
*
****************************************************************
20546400 flow3 sum   1.000 0.0   1  0
20546401 0.0154   1.0   cntrlvar 463
****************************************************************
*
****************************************************************
20546500 actC  mult  1.000 0.0   0
20546501 cntrlvar 464   cntrlvar 460
****************************************************************
*
****************************************************************
20546600 porvC prop-int 1.00000  0.0   0  3  0.00  1.000
20546601 0.1   0.14  cntrlvar 465
****************************************************************
*
****************************************************************
20546700 actC1 mult  1.000 0.0   0
20546701 cntrlvar 466   cntrlvar 460
****************************************************************
*
*  CONTROLLER  FOR   AFW   ACTIVATION
*
****************************************************************        SCALING  INITIAL VALUE  Do Not Compute
20547000 attAFW   tripunit 1.0   0.0   0  0
20547001 730
****************************************************************
*
*  SG-A
*
****************************************************************
20547100 actAe mult  1.000 0.0   0
20547101 cntrlvar 22 cntrlvar 470
****************************************************************
*
****************************************************************
20547200 sgaflow  prop-int 1.0000000   974.0 0  0
20547201 0.140 0.02  cntrlvar 471
****************************************************************
*
*  SG-B
*
****************************************************************
20547300 actBe mult  1.000 0.0   0
20547301 cntrlvar 42 cntrlvar 470
****************************************************************
*
****************************************************************
20547400 sgaflow  prop-int 1.0000000   974.0 0  0
20547401 0.140 0.02  cntrlvar 473
****************************************************************
*
*  SG-C
*
****************************************************************
20547500 actCe mult  1.000 0.0   0
20547501 cntrlvar 62 cntrlvar 470
****************************************************************
*
****************************************************************
20547600 sgaflow  prop-int 1.0000000   974.0 0  0
20547601 0.140 0.02  cntrlvar 475
****************************************************************
*
*  LOOP FLOW
*
****************************************************************        conversion in kg/s
20548000 flowcl   sum   0.4536   0.0   1  0
20548001 0.0   1.0   mflowj   220010000   1.0   mflowj   320010000
20548002 1.0   mflowj   420010000
*
****************************************************************        conversion in kg/s
20548100 flowhl   sum   0.4536   0.0   1  0
20548101 0.0   1.0   mflowj   200010000   1.0   mflowj   300010000
20548102 1.0   mflowj   400010000
*
****************************************************************
20548200 flowtot  sum   1.0   0.0   1  0
20548201 0.0   1.0   cntrlvar 480   1.0   cntrlvar 481
*
****************************************************************
*
******************************************************************
20548300 sgsq  sum   1.0   0.0   0  0
20548301 0.0   1.0   q  208010000   1.0   q  208020000   1.0   q  208030000
20548302 1.0   q  208040000   1.0   q  208050000   1.0   q  208060000
20548303 1.0   q  208070000   1.0   q  208080000
20548304 1.0   q  308010000   1.0   q  308020000   1.0   q  308030000
20548305 1.0   q  308040000   1.0   q  308050000   1.0   q  308060000
20548306 1.0   q  308070000   1.0   q  308080000
20548307 1.0   q  408010000   1.0   q  408020000   1.0   q  408030000
20548308 1.0   q  408040000   1.0   q  408050000   1.0   q  408060000
20548309 1.0   q  408070000   1.0   q  408080000
******************************************************************
*
20548400 coredp   sum   1.0   0.0   0  0
20548401 0.0   -1.0  p  151010000   1.0   p  108010000
*
*  COMPUTE  ECCS HPIS/LPIS MASS FLOW
******************************************************************
20548500 ECCS1 integral 1.0   0.0   1  0
20548501 mflowj   505000000
******************************************************************
20548600 ECCS2 integral 1.0   0.0   1  0
20548601 mflowj   506000000
******************************************************************
20548700 ECCS3 integral 1.0   0.0   1  0
20548701 mflowj   507000000
******************************************************************
20548800 sprayco  integral 1.0   0.0   1  0
20548801 mflowj   551000000
******************************************************************
20548900 TotECCS  sum   1.0   -1215362.0  0  0
20548901 -1215362.0  1.0   cntrlvar 485
20548902 1.0   cntrlvar 486   1.0   cntrlvar 487
20548903 1.0   cntrlvar 488
****************************************************************
20549100 sum1  sum   0.33333  0.0   0  0
20549101 0.0   1.0   p  220010000   1.0   p  320010000
20549102 1.0   p  420010000
*        convert in psi
20549200 delta2   sum   1.45E-04 0.0   0  0
20549201 0.0   1.0   cntrlvar 491
*        convert in psi
20549300 delta3   sum   1.45E-04 0.0   0  0
20549301 0.0   1.0   cntrlvar 491
*
****************************************************************
20549400 cv494 stdfnctn 1.0   0.0   1
20549401 max
20549402 httemp   111100112   httemp   111100212
20549403 httemp   111100312   httemp   111100412
20549404 httemp   111100512   httemp   111100612
20549405 httemp   111100712   httemp   111100812
20549406 httemp   111100912   httemp   111101012
*
****************************************************************
20549500 cv495 stdfnctn 1.0   0.0   1
20549501 max
20549502 httemp   112100112   httemp   112100212
20549503 httemp   112100312   httemp   112100412
20549504 httemp   112100512   httemp   112100612
20549505 httemp   112100712   httemp   112100812
20549506 httemp   112100912   httemp   112101012
*
****************************************************************
20549600 cv496 stdfnctn 1.0   0.0   1
20549601 max
20549602 httemp   113100112   httemp   113100212
20549603 httemp   113100312   httemp   113100412
20549604 httemp   113100512   httemp   113100612
20549605 httemp   113100712   httemp   113100812
20549606 httemp   113100912   httemp   113101012
*
****************************************************************
20549700 cv497 stdfnctn 1.0   0.0   1
20549701 max   cntrlvar 494
20549702    cntrlvar 495
20549703    cntrlvar 496
*
****************************************************************
20549800 cv498 tripunit 1.0   0.0  0  0
20549801 650
*
****************************************************************
20549900 lpi1dt2  constant   2.7e+6
*
20550000 lpi2dt2  constant   2.7e+6
*
****************************************************************
20550100 lpi1dt3  constant   0.6e+3
*
20550200 lpi2dt3  constant   0.6e+3
*
****************************************************************
20550300 cv-503   sum   1.0     0.0   0    0
20550301 0.0   1.0   cntrlvar 501
20550302       1.0   cntrlvar 499
*
****************************************************************
20550400 cv-504   sum   1.0     0.0   0    0
20550401 0.0   1.0   cntrlvar 502
20550402       1.0   cntrlvar 500
*
*
******************************************************************
*  this  deck  contains proprietary information *
*  do not   disseminate any   part  of this  deck  without  the   *
*  written  approval of the   eg&g  idaho severe   accident *
*  analysis manager  (r.   j. dallman).   *
******************************************************************
******************************************************************
.
