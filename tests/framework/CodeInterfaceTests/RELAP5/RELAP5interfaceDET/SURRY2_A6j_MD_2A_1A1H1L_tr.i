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
0000100  restart  transnt
0000101  run
0000103  -1
*
*  104   mbinary
*
*
*
*  TIME  STEP  CONTROL  200-299
*  time  dt min   dt max   ssdtt min.ed.(plt)   maj.ed (out)   restart
0000201   38.0    1.00E-10 0.001  07003 1000  150000   4000000
0000202   55.0    1.00E-10 0.0001 07003 10000 150000   4000000
0000203  200.0    1.00E-10 0.001  07003 1000  150000   4000000
0000204  275.0    1.00E-10 0.01   07003 500   15000    4000000
0000205  295.0    1.00E-10 0.0001 07003 50000 1500000  40000000
0000206  3600.0   1.00E-10 0.01   07003 500   30000    200000
0000207  50700.0  1.00E-10 0.01   07003 500   30000    200000
0000208  55000.0  1.00E-10 0.01   07003 500   30000    200000
0000209  65150.0  1.00E-10 0.01   07003 500   30000    200000
0000210  84600.0  1.00E-10 0.01   07003 500   30000    200000
*
*  VARIABLE TRIPS 401/599  1/1000         206NNNN0
*
0000401  cntrlvar 497   ge null  0  1467.0   l  -1.0  *end calculation
0000402  time  0  gt timeof   565  2000. n  -1.0  * Mission time - End of calculation
*
0000505  p  440010000   lt null  0  1840.0   l  -1.0  * MSIV CLOSURE for imposing REACTOR TRIP
0000506  p  284010000   gt null  0  134.7 n  * SG A PORV Set point close
0000507  p  284010000   gt null  0  184.7 n  * SG A PORV Set point open
0000508  p  384010000   gt null  0  134.7 n  * SG B PORV Set point close
0000509  p  384010000   gt null  0  184.7 n  * SG B PORV Set point open
0000510  p  484010000   gt null  0  134.7 n  * SG C PORV Set point close
0000511  p  484010000   gt null  0  184.7 n  * SG C PORV Set point open
*
0000518  time  0  gt timeof   505   0.00E+00 l  -1.0  * pmp control  (pmp trip off --> Power supplied to the motor)
0000519  time  0  lt timeof   505   0.00E+00 n  0.0   * pmp rotational table OFF
*
0000522  time  0  gt null  0   5.00E+00 l  -1.0  * Power to zero
*
0000523  time  0  gt null  0  1.00E+06 n  -1.0  * TD AFW SIGNAL   - Auto initiation at FULL Flow
0000524  time  0  gt null  0  1.00E+06 n  -1.0  * TD AFW SIGNAL   - Operator Control   of the SG Level
*
0000526  time      0           gt null     0  1.00E+06 l  -1.0  * TD AFW EQ DAMAGE TRIP
0000527  time      0           gt null     0  0.00E+00 l  -1.0  * Connect the ACCUMULATOR1 when Transient Starts
0000528  time      0           gt null     0  1.00E+06 n  -1.0  * MCP Leakage (21 gpm) ON/OFF
0000529  time      0           gt null     0  1.00E+06 n  -1.0  * SG COOL DOWN ON
0000530  p         284010000   lt null     0  134.7 l  -1.0  * SG COOL DOWN OFF
0000531  p         384010000   lt null     0  134.7 l  -1.0  * SG COOL DOWN OFF
0000532  p         484010000   lt null     0  134.7 l  -1.0  * SG COOL DOWN OFF
0000533  time      0           lt null     0  2.00E+06 n  -1.0  * MCP Leakage (@ 21 gpm) ON/OFF
0000534  time      0           gt null     0  1.00E+06 n  -1.0  * MCP Leakage (@ 183 gpm) ON/OFF    780   sec
0000535  time      0           lt null     0  2.00E+06 n  -1.0  * MCP Leakage (@ 183 gpm) ON/OFF    788.66   sec
0000536  time      0           gt null     0  1.00E+06 l  -1.0  * MCP Leakage SHUT TRIP
0000537  time      0           lt null     0  2.88E+06 n  0.0   * SHUT OFF PORV BECAUSE OF BATTERIES OUT        (8 hr)
0000538  time      0           gt null     0  1.00E+06 n  -1.0  * TURN ON THE KERR PUMP FOR EMERGENCY INJECTION!         (3.5 hr)
0000539  p         222010000   lt null     0  290.1 n  -1.0  * PRESSURE CONDITION FOR KERR PUMP OPERATION
0000540  time      0           gt null     0  1.00E+06 l  -1.0  * BATTERIES DAMAGE FOR FLOODING
0000541  time      0           lt null     0  1.00E+06 n  -1.0  * TIME FOR RECOVERY ON SG PORV and AFW BLACKRUN       (8+2hr)
0000542  time      0           gt timeof   526   1.00E+06 n  -1.0  * EMERGENCY PS DEPRESSURZATION - TIME OF ACTION
0000543  p         440010000   gt null     0  167.0 n  -1.0  * EMERGENCY PS DEPRESSURZATION - Set point close
0000544  p         440010000   gt null     0  217.0 n  -1.0  * EMERGENCY PS DEPRESSURZATION - Set point open
*
0000548  time  0  lt null  0  5.00E+00 n  0.0   * LBLOCA signal
0000549  p  440010000   lt null  0  1840.0   l  -1.0  * Low Pressure signal on PRZ     >> REACTOR TRIP
0000550  p  440010000   lt null  0  1789.7   l  -1.0  * Low-low Pressure signal on PRZ    >> SAFETY INJECTION SIGNAL
*
0000551  time  0  gt timeof   550   5.00E+00 n  -1.0  * Startup of LPIS #1  [DeltaT1 for LPI1 Start Time] - INJ
0000552  time  0  gt timeof   550   1.00E+06 n  -1.0  * Startup of LPIS #2  [DeltaT1 for LPI2 Start Time] - INJ
0000553  time  0  gt timeof   550   5.00E+00 n  -1.0  * Startup of HPIS #1
0000554  time  0  gt timeof   550   1.00E+06 n  -1.0  * Startup of HPIS #2
0000555  time  0  gt timeof   550   1.00E+06 n  -1.0  * Startup of HPIS #3
0000556  time  0  gt null      0    0.0      l  -1.0  * Connect the ACCUMULATOR2 when Transient Starts
0000557  time  0  gt null      0    1.00E+06 l  -1.0  * Connect the ACCUMULATOR3 when Transient Starts
0000558  time  0  gt timeof   551   2.700E+6 n  -1.0   * Failure to Run LPIS #1  [DeltaT2 for LPI1 Run Time] - INJ
0000559  time  0  gt timeof   552   2.700E+6 n  -1.0   * Failure to Run LPIS #2  [DeltaT2 for LPI2 Run Time] - INJ
0000560  time  0  gt timeof   553   0.000E+0 n  -1.0   * Failure to Run HPIS #1
0000561  time  0  gt timeof   554   1.000E+6 n  -1.0   * Failure to Run HPIS #2
0000562  time  0  gt timeof   555   1.000E+6 n  -1.0   * Failure to Run HPIS #3
*
0000563  time  0  gt null  0  1.50E+03 l  -1.0  * stop AFW
0000564  time  0  gt timeof   550   14.0E+00 l  -1.0  * CONTAIMNET SPRAY ON
0000565  cntrlvar 489   ge null  0  0.0   l  -1.0  * STOP ECCS because RWST depletion
*
0000566  time  0  gt timeof   565   1.50E+02 n  -1.0  * Startup of LPIS #1 - DeltaT4 for LPI1 Start Time] - REC
0000567  time  0  gt timeof   565   1.00E+06 n  -1.0  * Startup of LPIS #2 - DeltaT4 for LPI2 Start Time] - REC
0000568  time  0  gt timeof   565   1.50E+02 n  -1.0  * Startup of HPIS #1 - REC
0000569  time  0  gt timeof   565   1.00E+06 n  -1.0  * Startup of HPIS #2 - REC
0000570  time  0  gt timeof   565   1.00E+06 n  -1.0  * Startup of HPIS #3 - REC
*
0000571  time  0  gt timeof   566   1.000E+6 n  -1.0   * Failure to Run LPIS #1 - [DeltaT5 for LPI1 Run Time] - REC
0000572  time  0  gt timeof   567   1.000E+6 n  -1.0   * Failure to Run LPIS #2 - [DeltaT5 for LPI2 Run Time] - REC
0000573  time  0  gt timeof   568   0.00E+00 n  -1.0   * Failure to Run HPIS #1 - REC
0000574  time  0  gt timeof   569   1.00E+06 n  -1.0   * Failure to Run HPIS #2 - REC
0000575  time  0  gt timeof   570   1.00E+06 n  -1.0   * Failure to Run HPIS #3 - REC
*
0000576  time  0  gt timeof   558   0.600E+3 n  -1.0  * Repair Time DT3 of LPIS #1 - INJ
0000577  time  0  gt timeof   559   0.600E+3 n  -1.0  * Repair Time DT3 of LPIS #2 - INJ
*
0000578  cntrlvar 499  lt timeof  565  0.0  n  -1.0  * condition for  LPI1 Run Time INJ mode less LPR switch time
0000579  cntrlvar 500  lt timeof  565  0.0  n  -1.0  * condition for  LPI2 Run Time INJ mode less LPR switch time
*
0000580  time  0  gt timeof   571   0.600E+3 n  -1.0  * Repair Time DT6 of LPIS #1 - REC
0000581  time  0  gt timeof   572   0.600E+3 n  -1.0  * Repair Time DT6 of LPIS #2 - REC
*
0000582  cntrlvar 503  lt timeof   565   150.0 n  -1.0  * Condition for Case LPIS1-REC, Case C,D
0000583  cntrlvar 504  lt timeof   565   150.0 n  -1.0  * Condition for Case LPIS2-REC, Case C,D
*
0000584  time  0  gt timeof   576   1.000E+3 n  -1.0  * [DeltaT5 for LPI1 Run Time] - REC
0000585  time  0  gt timeof   577   1.000E+3 n  -1.0  * [DeltaT5 for LPI2 Run Time] - REC
*
0000586  time  0  gt timeof   584   0.600E+3 n  -1.0  * Repair Time DT6 of LPIS #1 - REC
0000587  time  0  gt timeof   585   0.600E+3 n  -1.0  * Repair Time DT6 of LPIS #2 - REC
*
0000588  cntrlvar 503  lt timeof   565   0.0 n  -1.0  * Condition for Case LPIS1-REC, Case B
0000589  cntrlvar 504  lt timeof   565   0.0 n  -1.0  * Condition for Case LPIS2-REC, Case B
*
0000600  650
*
0000650  401   or 402   n  -1.0  *
*
0000626  506   and   -729  n  -1.0  *
0000627  507   and   -729  n  -1.0  *
0000628  508   and   -729  n  -1.0  *
0000629  509   and   -729  n  -1.0  *
0000630  510   and   -729  n  -1.0  *
0000631  511   and   -729  n  -1.0  *
*
0000740  528   and   533   n  -1.0  *  SEAL LOCA @ 21 gpm
0000741  534   and   535   n  -1.0  *  MCP Leakage (@ 183 gpm) ON/OFF
0000742  740   or 741   n  -1.0  *  SEAL LOCA
*
0000743  538   and   539   n  -1.0  *  KERR PUMP OPERATION LOGIC
*
0000744  530   and   -729  n  -1.0  *  logic for   shutdown the   SG MAIN PORV   "for: EQ, ECST damage, flooding"
0000745  531   and   -729  n  -1.0  *  logic for   shutdown the   SG MAIN PORV   "for: EQ, ECST damage, flooding"
0000746  532   and   -729  n  -1.0  *  logic for   shutdown the   SG MAIN PORV   "for: EQ, ECST damage, flooding"
*
0000751  622   and   752   n  *press.  porv  open
0000752  623   or 751   n  *press.  porv
0000753  -752  or -622  n  *press.  porv  closed
0000754  548   and   548   n  0.0   *LBLOCA  connection valve
0000755  -548  and   -548  n  -1.0  *LBLOCA  BREAK OPENING
*
*
0000764  744   and   506   n  *sga  porv  open  at low pressure
0000765  745   and   508   n  *sgb  porv  open  at low pressure
0000766  746   and   510   n  *sgc  porv  open  at low pressure
*
0000767  764   and   768   n  *sga  porv  open
0000768  627   or 767   n  *sga  porv
0000769  -768  or -626  n  *sga  porv  closed
0000770  765   and   771   n  *sgb  porv  open
0000771  629   or 770   n  *sgb  porv
0000772  -771  or -628  n  *sgb  porv  closed
0000773  766   and   774   n  *sgc  porv  open
0000774  631   or 773   n  *sgc  porv
0000775  -774  or -630  n  *sgc  porv  closed
*
0000776  -505  and   -505  n  *  MSIV  closure  for REACTOR TRIP
*
0000780  505   and   -563  n  -1.0  *  start/stop  AFW
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
0000677  651   and   -682  n  -1.0  * LPIS #1 start - REC  [time = 651 condition and condition for case A]
0000678  652   and   -683  n  -1.0  * LPIS #2 start - REC  [time = 652 condition and condition for case A]
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
0000658  656   and    579  n  -1.0  * LPIS #2 start - REC  [condition 656 and DT2 < Tswitch]
*
0000659  657   and    682  n  -1.0  * LPIS #1 start - REC [condition 657 and condition for B]
0000660  658   and    683  n  -1.0  * LPIS #2 start - REC [condition 658 and condition for B]
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
0000666   664   and    579  n  -1.0  * LPIS #2 start - REC [condition 664 and DT2 < Tswitch]
*
0000667   665   and   -680  n  -1.0  * LPIS #1 start - REC [condition 665 and condition for C]
0000668   666   and   -681  n  -1.0  * LPIS #2 start - REC [condition 666 and condition for C]
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
0000674   672   and    579  n  -1.0  * LPIS #2 start - REC [condition 664 and DT2 < Tswitch]
*
0000675   673   and    680  n  -1.0  * LPIS #1 start - REC [condition 665 and condition for D]
0000676   674   and    681  n  -1.0  * LPIS #2 start - REC [condition 666 and condition for D]
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
0000680    582   and     582  n  -1.0  * condition for C
0000681    583   and     583  n  -1.0  * condition for C
*
* GENERAL CONDITION FOR CASE B
*
0000682    588   and     588  n  -1.0  * condition for B
0000683    589   and     589  n  -1.0  * condition for B
*
*
*  HYDRODYNAMIC   COMPONENTS  -  CCCXXNN
*
*
*  PRZ CONTROL REMOVAL
*
940000   sysprjn  delete
*
950000   syspres  delete
*
*  Main FW Shutdown
*
****************************************************************
2670000  sga-mfwj tmdpjun
2670101  266000000   272000000   3.54
2670200  1  0  cntrlvar 25
2670201  0.0   0.0   0.0   0
2670202  10000.0  0.0   0.0   0
****************************************************************
3670000  sgb-mfwj tmdpjun
3670101  366000000   372000000   3.54
3670200  1  0  cntrlvar 45
3670201  0.0   0.0   0.0   0
3670202  10000.0  0.0   0.0   0
****************************************************************
4670000  sgc-mfwj tmdpjun
4670101  466000000   472000000   3.54
4670200  1  0  cntrlvar 65
4670201  0.0   0.0   0.0   0
4670202  10000.0  0.0   0.0   0
****************************************************************
*
*
*  MD-AFW
*
****************************************************************
2920000  sga-mfwj tmdpjun
2920101  268000000   272000000   3.54
2920200  1  780
2920201  -1.0  0.0   0.0   0.0
2920202  0.0   97.1  0.0   0.0
2920203  1.00E+05 97.1  0.0   0.0
****************************************************************
3920000  sgb-mfwj tmdpjun
3920101  368000000   372000000   3.54
3920200  1  780
3920201  -1.0  0.0   0.0   0.0
3920202  0.0   97.1  0.0   0.0
3920203  1.00E+05 97.1  0.0   0.0
****************************************************************
4920000  sgc-mfwj tmdpjun
4920101  468000000   472000000   3.54
4920200  1  780
4920201  -1.0  0.0   0.0   0.0
4920202  0.0   97.1  0.0   0.0
4920203  1.00E+05 97.1  0.0   0.0
****************************************************************
*
*
*  MSIV CLOSURE
*
****************************************************************
2850000  sga-msiv valve
2850101  284010000   286000000   4.5869   0.0   0.0   100
2850201  0  102.20364   115.59958   0.0
2850300  trpvlv
2850301  776
****************************************************************
3850000  sgb-msiv valve
3850101  384010000   386000000   4.5869   0.0   0.0   100
3850201  0  102.20364   115.59958   0.0
3850300  trpvlv
3850301  776
****************************************************************
4850000  sgc-msiv valve
4850101  484010000   486000000   4.5869   0.0   0.0   100
4850201  0  102.20364   115.59958   0.0
4850300  trpvlv
4850301  776
****************************************************************
*
*
*  SG PORV - SBO ACTUATION    SG COOLDOWN 100 F/HR UNTIL 120 PSIG
*
*  DELETED
*
*  NEW SG PORV - SBO ACTUATION      WORKING FOR SG AT 120 PSIG
*
*  DELETED
*
.
