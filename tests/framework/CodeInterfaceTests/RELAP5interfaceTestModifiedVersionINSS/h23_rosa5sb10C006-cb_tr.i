********
* THIS IS AN INPUT FILE THAT HAS BEEN DONATED BY DR. IKUO KINOSHITA (INSS, JAPAN)
* FOR TESTING THE INTERFACE WITH THEIR MODIFIED CODE
********
= lstf sbloca
*
0000100   restart transnt
0000103   50024 *1000.0s
*
*
*-----time step control-------------------------------------------------
*
*         end time  min.    max.   flag  minor  major   restart
$0000201   1000.0    1.0e-8  0.02   3     50     5000    50000
0000202   4000.0    1.0e-8  0.01   7     100    100000  126500
*
*-----trip data---------------------------------------------------------
*
20600010  time    0       ge  null    0      1000.0   l * break
*
20600250  time    0       ge  null    0      1600.0   l * a-msrv 2nd depress
20600270  time    0       ge  null    0      1600.0   l * b-msrv 2nd depress
*
*
*******************************************************
*                                                     *
*              minor edit variables                   *
*                                                     *
*******************************************************
301  p      503150000
302  p      220010000
*
*        <break>
*
*         name      type
9100000   breakj    valve
*
*         from      to        j-area    for rev  efvcahs cdsub  cdtwo  cdsup
9100101   343010003 911000000 8.171e-05 0.0 0.0    30100 1.0    1.8    1.0
*
*         j-hdia    beta c    slope
9100110   1.020e-02 0.0  1.0  1.0
*
*            mfl       mfg      vi
9100201   1  0.0       0.0      0.0
*
*         vlvtype
9100300   trpvlv
*
*         trp.no.
9100301   0001
*
*         name      type
9110000   breakv    tmdpvol
*
*         v-area   leng    vol  h-ang v-ang  el-chg  rough  v-hdia    tlpvbfe
9110101   100.0    10.0    0.0  0.0   0.0    0.0     4.00-5 0.0           000
*
9110200   4
*
*         time     pres          temp      qual
9110201   0.0      0.101325e+06  293.15    1.0
*
*
*-----component description---------------------------------------------------
*
*         <a-loop side reactor coolant pump>
*
*         name      type
1400000   arcp      pump
*
*         v-area   leng    vol  h-ang v-ang  el-chg  tlpvbfe
1400101   0.02038  1.153   0.0  0.0   37.51  0.702       000
*
*         connect   j-area    for       rev       efvcahs
1400108   128010000 0.03365   0.0       0.0          1000 * from
1400109   141000000 0.03365   0.0       0.0          1000 * to
*
*         j-hdia    beta c    slope
1400110   0.207     0.0  1.0  1.0   * from
1400111   0.207     0.0  1.0  1.0   * to
*
*            pres          temp
1400200   3  1.57240E+07   564.810 *input ss result
*
*            mfl       mfg       vi
1400201   1  24.448    0.0       0.0 *input ss result
1400202   1  24.448    0.0       0.0 *input ss result
*
*         pump index and option
*         pump     two     two-dif motor   time    trip  rev
1400301   0       -1      -3      -1       0       0     0
*
*         pump description
*         p-vel    p-vel0  flow    head    torq    moment
1400302   188.5    0.8566  0.054   10.0    55.2    0.54
*
*         dens     m-torq  tf2     tf0             tf3
1400303   0.0      0.0     0.0     0.0     0.0     0.0
*
*         pump stop data
*         elapsed  f-p-vel r-p-vel
1400310   1.0+10   0.0     0.0
*
*         single phase homologous curve
*         type     reg     v/a|a/v head    v/a|a/v head
1401100   1        1       0.00    1.36    0.10    1.38 * han
1401101                    0.24    1.42    0.40    1.41
1401102                    0.60    1.32    0.80    1.19
1401103                    1.00    1.00
1401200   1        2       0.00   -0.97    0.20   -0.68 * hvn
1401201                    0.50   -0.20    0.65    0.07
1401202                    0.80    0.40    1.00    1.00
1401300   1        3      -1.00    3.20   -0.90    2.80 * had
1401301                   -0.80    2.46   -0.60    1.94
1401302                   -0.40    1.57   -0.20    1.41
1401303                    0.00    1.36
1401400   1        4      -1.00    3.20   -0.80    2.76 * hvd
1401401                   -0.60    2.41   -0.40    2.09
1401402                   -0.20    1.81    0.00    1.58
*         type     reg     v/a|a/v torque  v/a|a/v torque
1401500   2        1       0.00    0.36    0.12    0.38 * ban
1401501                    0.20    0.44    0.30    0.58
1401502                    0.50    0.73    0.70    0.81
1401503                    1.00    1.00
1401600   2        2       0.0    -1.26    0.10   -0.88 * bvn
1401601                    0.30   -0.31    0.50    0.09
1401602                    0.65    0.30    0.86    0.63
1401603                    1.00    1.00
1401700   2        3      -1.00    2.40   -0.85    1.70 * bad
1401701                   -0.65    1.12   -0.50    0.84
1401702                   -0.40    0.69   -0.20    0.59
1401703                    0.00    0.36
1401800   2        4      -1.00    2.40   -0.80    2.12 * bvd
1401801                   -0.60    1.80   -0.30    1.32
1401802                    0.00    0.80
*
*         time-dependent pump velocity
*         trip
1406100   0002
*
*         time     p-vel
1406101     0.0    161.5
1406102     2.0    137.3
1406103     5.0    117.9
1406104    10.0     87.2
1406105    20.0     59.7
1406106    30.0     45.2
1406107    40.0     35.5
1406108    50.0     29.9
1406109    60.0     25.8
1406110    70.0     22.6
1406111    80.0     20.2
1406112    90.0     17.8
1406113   100.0     16.1
1406114   250.0      0.0
*
*         <b-loop side reactor coolant pump>
*
*         name      type
3400000   brcp      pump
*
*         v-area   leng    vol  h-ang v-ang  el-chg  tlpvbfe
3400101   0.02038  1.153   0.0   0.0  37.51  0.702       000
*
*         connect   j-area    for       rev       efvcahs
3400108   328010000 0.03365   0.0       0.0          1000 * from
3400109   341000000 0.03365   0.0       0.0          1000 * to
*
*         j-hdia    beta c    slope
3400110   0.207     0.0  1.0  1.0 * from
3400111   0.207     0.0  1.0  1.0 * to
*
*            pres          temp
3400200   3  1.57240E+07   564.811 *input ss result
*
*            mfl       mfg       vi
3400201   1  24.448    0.0       0.0 *input ss result
3400202   1  24.448    0.0       0.0 *input ss result
*
*         pump index and option
*         pump     two     two-dif motor   time    trip  rev
3400301   140     -1      -3      -1       0       0     0
*
*         pump description
*         p-vel    p-vel0  flow    head    torq    moment
3400302   188.5    0.8566  0.054   10.0    55.2    0.54
*
*         dens     m-torq  tf2     tf0             tf3
3400303   0.0      0.0     0.0     0.0     0.0     0.0
*
*         pump stop data
*         elapsed  f-p-vel r-p-vel
3400310   1.0+10   0.0     0.0
*
*         time-dependent pump velocity
*         trip
3406100   0002
*
*         time     p-vel
3406101     0.0    161.5
3406102     2.0    137.3
3406103     5.0    117.9
3406104    10.0     87.2
3406105    20.0     59.7
3406106    30.0     45.2
3406107    40.0     35.5
3406108    50.0     29.9
3406109    60.0     25.8
3406110    70.0     22.6
3406111    80.0     20.2
3406112    90.0     17.8
3406113   100.0     16.1
3406114   250.0      0.0
*
.
