*RAVEN INPUT VALUES
*20290000    4   0.0315033522444
*9890201    2   67.0024160207
*RAVEN INPUT VALUES
* deck snc01.i
*
* Changes made by Paul Bayless, INL, January 2013
*  Commented out unused material 13, whose input heat capacity value of 0.0
*   causes an input error in version 4.1.0.
*
* RELAP5-3D Developmental Assessment, June 2009
*
* This input model was prepared by ISL.
* Semi-implicit solution scheme used.
*
*######################################################################*
* semiscale test series (s-nc-xx), simulation, relap5/mod2, sept 1983  *
*######################################################################*
*======================================================================*
*==================      semiscale mod-2a       =======================*
*==================      standard transient     =======================*
*==================         input deck          =======================*
*======================================================================*
*  notes                                                               *
*                                                                      *
*     1. this model is based on the standard mod-2a model              *
*        documented in reference 2 below.                              *
*                                                                      *
*     2. steady state runs are made either with the provided           *
*        control systems package or with additional components         *
*        for single condition steady state calculations.               *
*                                                                      *
*     3. vessel heater tape to offset vessel heat loss is not          *
*        included with this model.                                     *
*                                                                      *
*     4. the system configuration encorporated in this model is        *
*        the communicative cold leg break configuration.               *
*        all uhi system components are removed and upper head          *
*        internals are modeled as modified to reflect non-uhi          *
*        plant configuration.                                          *
*                                                                      *
*     5. steam generator components have been renodalized to           *
*        reflect the ppcc guidelines as established by caap            *
*        letter from c. davis.                                         *
*                                                                      *
*     6. steam separator components reflect recommended model          *
*        changes from j. trapp. (no junctions are abrupt, and          *
*        inlet is homogeneous)                                         *
*                                                                      *
*     7. primary coolant pump discharge junction is homogeneous        *
*                                                                      *
*        volume ave temp.  mesh pt.                                    *
*======================================================================*
*  references                                                          *
*                                                                      *
*   1. v. h. ransom et al., relap5/mod1 code manual, volume 1 and 2,   *
*           nureg/cr-1826, november 1980.                              *
*   2. m. t. leonard, "relap5 standard model for the semiscale mod-2a  *
*           system", egg-semi-5692, november 1981.                     *
*                                                                      *
*   3. system design description for the mod-3 semiscale system,       *
*           revision b, december 1980.                                 *
*                                                                      *
*======================================================================*
*######################################################################*
* problem title, time step controls and trips                          *
*######################################################################*
= semiscale test (s-nc-01), case no. 4 (1)
*
*problem     type    option
0000100       new   stdy-st *transnt
*
*            inp-chk or run
0000101           run
*
*units      input    output   (si or british)
0000102   british    si
*
*time out     min       max
*0000105      25.0      30.0    1000.
*
*           t end     min st   max st  st cl     mr     mj     wr
0000201     1000.     1.0-7    0.05      3      200   10000  10000
*
*======================================================================*
* minor edits                                                          *
*======================================================================*
*
* primary system
0000301 p            163010000    * upper plenum pressure
0000302 tempf        110070000    * d. c. temp
0000303 sattemp      110070000    * d. c. saturated temp
0000304 cntrlvar     930          * d. c. subcooling
0000305 tempf        201020000    * hot leg temp
0000306 tempf        263010000    * cold leg temp
0000307 tempf        163010000    * upper plenum temp
0000308 tempf        140010000    * core inlet temp
0000309 tempf        130010000    * lower plenum temp
0000310 cntrlvar     931          * core dtemp-a = tempf (16301 - 14001)
0000311 cntrlvar     932          * core dtemp-b = tempf (16301 - 13001)
0000312 tempf        240010000    * s.g. cold outlet temp
0000313 tempf        203030000    * s.g. hot inlet temp
0000314 cntrlvar     933          * s. g. dtemp = tempf (24001 - 30303)
0000315 mflowj       250000000    * cold leg mass flow
0000316 mflowj       110070000    * d. c. flow = core flow
0000317 cntrlvar     920          * primary system mass
0000318 cntrlvar     921          * fraction of total pri inventory
*
* secondary system
0000320 p            613010000    * s. g. outlet pressure
0000321 vlvarea      635          * s. g. outlet valve area
0000322 mflowj       635000000    * s. g. outlet mass flow
0000323 mflowj       610000000    * s. g. inlet mass flow
0000324 mflowj       602040000    * s. g. upper d. c. flow (recirc)
0000325 mflowj       604020000    * s. g. d. c. to riser flow
0000326 cntrlvar     011          * s. g. riser level (voidf * dz)
0000327 cntrlvar     801          * s. g. riser level (dp level)
0000328 cntrlvar     802          * s. g. fraction ht area (voidf * dz)
0000329 cntrlvar     803          * s. g. fraction ht area (dp level)
0000330 cntrlvar     900          * s. g. flow (out - in)
0000331 cntrlvar     805          * s. g. system mass
0000332 cntrlvar     922          * fraction of total sg inventory
0000333 emass        0            * total mass error
0000334 cntrlvar     911          * s.g mass error
0000335 cntrlvar     923          * integral total mass error
*
* mass inventory control junction flows
0000350 mflowj       290000000    * pri mass increaser flow
0000351 mflowj       291000000    * pri mass decreaser flow
0000352 mflowj       690000000    * sec mass increaser flow
0000353 mflowj       691000000    * sec mass decreaser flow
*
* Power added to primary system
0000361 cntrlvar       1          * Summation of volume heat source
*
*
* trips name     code      op name   code    add  l/n
0000501 time     0         ge null   0       1.00 l *
0000502 time     0         ge null   0    2000.00 l * prob end
* 0000502 time     0         ge null   0     300.00 l * prob end
0000503 time     0         lt null   0       0.00 l *
0000504 p        613010000 ge null   0    1116.75 n * j634 r. vlv
0000506 p        613010000 le null   0     840.50 n * j635 cls vlv   xxx
0000507 p        613010000 ge null   0     842.00 n * j635 opn vlv   xxx
0000508 time     0         lt null   0       0.00 l * j991 opn vlv
0000509 time     0         ge null   0    2000.00 l * j991 cls vlv
* 0000509 time     0         ge null   0      90.00 l * j991 cls vlv
0000513 time     0         lt null   0       0.00 l * j635 cls vlv
0000514 time     0         ge null   0       0.00 l * j635 opn vlv
* 0000514 time     0         ge null   0    1000.00 l * j635 opn vlv
0000518 time     0         ge null   0    2000.00 l * lt601-612
* 0000518 time     0         ge null   0     100.00 l * lt601-612
0000519 time     0         ge null   0     200.00 l * lt601-612
0000520 vlvarea  991       le null   0       0.00 l * j991 vlv shut  xxx
0000521 vlvarea  635       le null   0       0.00 l * j635 vlv shut  xxx
0000522 time     0         ge timeof 520     0.00 l * j991 vlv shut  xxx
0000523 time     0         ge timeof 521     0.00 l * j635 vlv shut  xxx
0000530 cntrlvar 805       le null   0     154.00 l * decr sg inv (1)
* 0000531 cntrlvar 805       ge null   0     205.00 l * incr sg inv
0000531 time     0         ge null   0    2000.00 l * lt 601-612
* 0000531 time     0         ge null   0     200.00 l * lt 601-612
0000540 cntrlvar 920       le null   0      99.00 l * decr pri inv
* 0000541 cntrlvar 920       ge null   0      10.00 l * incr pri inv
0000541 time     0         ge null   0    2000.00 l * lt601-612
* 0000541 time     0         ge null   0     100.00 l * lt601-612
0000560 time     0         ge null   0  100000.00 l
0000561 time     0         ge null   0  100000.00 l
*
*ltrips trip opr  trip l/n
0000600      502         * problem end trip
0000601 518  xor  530  n * decrease sg inv
0000602 518  xor  531  n * increase sg inv
0000611 518  xor  540  n * decrease pri inv
0000612 518  xor  541  n * increase pri inv
*
*======================================================================*
* primary system source volume                                         *
*======================================================================*
*
*       name         type
0950000 prisrci    tmdpvol
*
*geomv1 areav        length       volume       hang         vang
0950101 0.009764     1.0          0.0          0.0          0.0
*
*geomv2 dz           rough        dhy          fe
0950102 0.0          0.0          0.0          00
*
*......................................................................*
*
*ictdv1 ebt
0950200 003
*
*ictdv2 time         pressure     temp
0950201     0.0      1500.0       520.0 * pri inv, source
0950202 10000.0      1500.0       520.0 * pri inv, source
*
*======================================================================*
* primary system sink volume                                           *
*======================================================================*
*
*       name         type
0960000 prisrcx    tmdpvol
*
*geomv1 areav        length       volume       hang         vang
0960101 0.009764     1.0          0.0          0.0          0.0
*
*geomv2 dz           rough        dhy          fe
0960102 0.0          0.0          0.0          00
*
*......................................................................*
*
*ictdv1 ebt
0960200 003
*
*ictdv2 time         pressure     temp
0960201     0.0      1500.0       520.0 * pri inv, sink
0960202 10000.0      1500.0       520.0 * pri inv, sink
*
**=====================================================================*
*    inlet annulus     (above hot leg center line)                     *
**====================================================================**
*
*    reference drawings =    assembly - 407986
*                         inner liner - 407990
*                         outer liner - 407991
*
* inlet elev  = top    = +10.50 in
* outlet elev = bottom =   0.00 in
*
*               name      type
1010000      inannu1    branch
*
*             no jun    in ctl
1010001            3         1
*
*               area    length     vol   hz ang   vr ang    elv ch
1010101      0.105697  1.750000    0.0,     0.0,   -90.0   -1.75000
*
*              rough     hydia   ctl flg
1010102      3.333e-5   0.14275        0
*
*         ctl  pressure    temp
1010200     3  69.618    220.73 *INITIAL CONDITIONS
*
*               from        to  jun area   f loss   r loss    ahs
1011101    101010000 110000000  0.105697      0.0      0.0  30000
*
*              flow-f      flow-g  velj
1011201        0.9088         0.0   0.0
1012101    263020002 101010003  0.045239      0.0      0.0   0000
1012201        0.9088         0.0   0.0
1013101    101000000 363010000  0.037554      0.0      0.0   0000
1013201        0.0            0.0   0.0
*
* revise junction Dh
1011110  0.139039 0. 1. 1.
*
**=====================================================================*
* downcomer                                                            *
**====================================================================**
*
*    inlet elev  = top    =  -12.00 in
*    outlet elev = bottom = -214.77 in
*
*               name      type
1100000       dcomer      pipe
*
*             no.vol
1100001           10
*
*               area    vol n0
1100101      0.030975        1  * includes reducing section of annulus
1100102      0.026039        8  * downcomer pipe
1100103      0.031502        9  * slanted pipe + entrance half annulus
1100104      0.0762104      10  * distribution annulus
*
*            vol lgh    vol no
1100301     2.385833         1
1100302     1.000833         2
1100303     2.000000         7
1100304     1.465667         8
1100305     1.434751         9
1100306     0.962500        10
*
*             vr ang    vol no
1100601        -90.0,        8
1100602        -66.7         9
1100603        -90.0        10
*
*             elv ch    vol no
1100701     -2.385833          1                                *tsa/dmk
1100702    -1.000833         2
1100703    -2.000000         7
1100704    -1.465667         8
1100705    -1.207667         9
1100706    -0.962500        10
*
*              rough    hy dia    vol no
1100801      3.333e-5   0.139039       1
1100802      3.333e-5   0.0            9
1100803      3.333e-5   0.111833      10
*
*             f loss    r loss    jun no
1100901        0.491     0.491         6
1100902        5.000     5.000         7 * mass flow measurement station
1100903        1.193     1.193         8
1100904          0.0       0.0         9
*
*            ctl flg    vol no
1101001            0,       10
*
*            ctl flg    jun no
1101101            0         9
*
*         ctl  pressure    temp    zero zero zero vol no
1101201     3  69.618    220.73    0.0  0.0  0.0      1  *
1101202     3  69.618    220.73    0.0  0.0  0.0      2  *
1101203     3  69.618    220.73    0.0  0.0  0.0      3  *
1101204     3  69.618    220.73    0.0  0.0  0.0      4  *
1101205     3  69.618    220.73    0.0  0.0  0.0      5  *
1101206     3  69.618    220.73    0.0  0.0  0.0      6  *
1101207     3  69.618    220.73    0.0  0.0  0.0      7  *
1101208     3  69.618    220.73    0.0  0.0  0.0      8  *
1101209     3  69.618    220.73    0.0  0.0  0.0      9  *
1101210     3  69.618    220.73    0.0  0.0  0.0     10  *
*
*         ctl
1101300     1
*
*              flow-f      flow-g  velj    jun no
1101301       0.9092         0.0   0.0         9
*
* revise junction Dh
1101401   0.139039  0. 1. 1. 1
1101402   0.0       0. 1. 1. 8
1101403   0.111833  0. 1. 1. 9
*
*================================================================
*   Valve used to connect loose volumes
*================================================================
1110000  glen valve
1110101  800000000 201000000 0 .6 .7 000000
1110201  0 1. 1. 0.
1110300  trpvlv
1110301  560
1120000  glen1 valve
1120102  900000000 201000000 0 .6 .7 000000
1120201  0 1. 1. 0.
1120300  trpvlv
1120301  561
*
**=====================================================================*
* lower plenum lower volume                                            *
**====================================================================**
*
*    inlet elev  = -220.13 in
*    outlet elev = -225.73 in
*
*          name       type
1200000   lplovol    snglvol
*
*          area      length   vol h ang v ang   elv ch
1200101  0.510279  0.466667   0.0  0.0  -90.0 -0.466667
*
*             rough      dhy     ctl
1200102      3.333e-5 0.381247     0
*
*         ctl  pressure    temp
1200200     3  69.618    220.73
*
**=====================================================================*
* lower plenum                                                         *
**====================================================================**
*
* inlet elev  = top    = -214.77 in
* outlet elev = bottom = -220.13 in
*
*               name      type
1300000      lo-plen    branch
*
*             no jun    in ctl
1300001            3         1
*
*             area    length    vol   hz ang   vr ang    elv ch
1300101   0.322478  0.721917    0.0      0.0    -90.0   -0.721917
*
*              rough     hydia   ctl flg
1300102      3.333e-5  0.229414        0
*
*         ctl  pressure    temp
1300200     3  69.618    220.73  *INITIAL CONDITIONS
*
*               from        to  jun area    f loss    r loss       ahs
*1301101    110010000 130000000  0.0762104      1.0       1.0     30100
1301101    110010000 130000000  0.0762104      1.0       1.0     00100
*
*              flow-f      flow-g  velj
1301201       0.9092         0.0   0.0
1302101    130000000 140000000  0.075189       0.0       0.0     00100
1302201       0.9092         0.0   0.0
1303101    130010000 120000000  0.322478       0.0       0.0     00000
1303201           0.0         0.0   0.0
*
**=====================================================================*
* core inlet                                                           *
**====================================================================**
*
* inlet elv = bottom = -214.77 in
* outlet elv = top = -195.14 in
*
*               name      type
1400000      core-in    snglvol
*
*             area    length     vol  hz ang  vr ang    elv ch
1400101   0.075189   1.635833    0.0     0.0    90.0  1.635833
*
*              rough     hydia    ctl
1400102      3.333e-5   0.278329    0
*
*         ctl  pressure    temp
1400200     3  69.618    220.73 *INITIAL CONDITIONS
*
**=====================================================================*
* core inlet junction                                                  *
**====================================================================**
*              name     type
1450000  coreinjn  sngljun
*
*               from        to  jun area    f loss    r loss       ahs
1450101    140010000 150000000  0.030747       0.0       0.0    000000
*
*              flow-f      flow-g  velj
1450201   1   0.9092         0.0   0.0
*
* revise junction Dh
1450110 0.033237 0. 1. 1.
*
**=====================================================================*
* average channel                                                      *
**====================================================================**
*
*     inlet elev = bottom = -195.14 in
*    outlet elev = top    =  -51.14 in
*
*               name      type
1500000      a-chnel      pipe
*
*             no.vol
1500001            6
*
*               area    vol n0
1500101     0.030749         6
*
*               area    jun no
1500201     0.030749         5
*
*            vol lgh    vol no
1500301          2.0         6
*
*             vr ang    vol no
1500601        +90.0,        6
*
*             elv ch    vol no
1500701          2.0         6
*
*              rough    hy dia    vol no
1500801      3.333e-5  0.033237        6
*
*             f loss    r loss    jun no
1500901         0.00      0.00         5
*
*            ctl flg    vol no
1501001            0         6
*
*                ahs    jun no
1501101        00000         5
*
*         ctl  pressure    temp    zero zero zero vol no
1501201     3  69.618    220.73    0.0  0.0  0.0      1  *
1501202     3  69.618    220.73    0.0  0.0  0.0      2  *
1501203     3  69.618    220.73    0.0  0.0  0.0      3  *
1501204     3  69.618    220.73    0.0  0.0  0.0      4  *
1501205     3  69.618    220.73    0.0  0.0  0.0      5  *
1501206     3  69.618    220.73    0.0  0.0  0.0      6  *
*
*         ctl
1501300     1
*
*              flow-f      flow-g  velj    jun no
1501301       0.9092         0.0   0.0         5
*
* revise junction Dh
1501401  0.033237 0. 1. 1. 5
*
**=====================================================================*
* upper core                                                           *
**====================================================================**
*
* inlet elv = bottom = -51.14 in
* outlet elv = top = -39.13 in
*
*               name      type
1610000      up-core    branch
*
*             no jun    in ctl
1610001            4         1
*
*            area   length    vol  hz ang   vr ang    elv ch
1610101  0.057859  1.000833   0.0     0.0     90.0  1.000833
*
*              rough     hydia   ctl flg
1610102      3.333e-5  0.07561         0
*
*         ctl  pressure    temp
1610200     3  69.618   278.33  *INITIAL CONDITIONS
*
*               from        to  jun area     f loss    r loss       ahs
1611101    150010000 161000000  0.030747        0.3       0.3    00000
*
*              flow-f      flow-g  velj
1611201       0.9092          0.0   0.0
1612101    161010000 162000000  0.034560        0.5       0.5    00100
1612201       0.7859          0.0   0.0
1613101    182010000 161010000  0.0016085       0.0       0.0    00100
1613201       0.0             0.0   0.0
1614101    185010000 161010000  0.004606     .43613    .43613    00100
1614201      -0.1229          0.0   0.0
*
*revise junction Dh
1611110 0.033237 0. 1. 1.
1612110 0.07561  0. 1. 1.
*
**=====================================================================*
* core outlet                                                          *
**====================================================================**
*
* inlet elev  = bottom = -39.13 in
* outlet elev = top    = -12.00 in
*
*               name      type
1620000      coreout    branch
*
*             no jun    in ctl
1620001            1         1
*
*               area    length       vol    hz ang    vr ang    elv ch
1620101     0.042937  2.260833       0.0       0.0      90.0  2.260833
*
*              rough     hydia   ctl
1620102      3.333e-5  0.131407    0
*
*         ctl  pressure    temp
1620200     3  69.618   278.33 *INITIAL CONDITIONS
*
*               from        to  jun area    f loss    r loss       ahs
1621101    185000000 162010000  0.076458    .92664    .77720     00003
*
*              flow-f      flow-g  velj
1621201        0.1231         0.0   0.0
*
**=====================================================================*
*   hot leg outlet (combined with previous upper plenum)               *
**====================================================================**
*
*    inlet elev  = -12.00 in
*    outlet elev = + 8.50 in
*
*               name      type
1630000       hotout    branch
*
*             no jun    in ctl
1630001            4         1
*
*               area    length       vol    hz ang    vr ang    elv ch
1630101    0.0500466   3.416666      0.0       0.0      90.0   3.416666
*
*              rough     hydia   ctl flg
1630102      3.333e-5  0.158413        0
*
*         ctl  pressure    temp
1630200     3  69.618   278.33 *INITIAL CONDITIONS
*
*               from        to  jun area    f loss    r loss       ahs
1631101    163010003 201000000  0.045239       2.5       2.5      0002
*
*              flow-f      flow-g  velj
1631201        0.9087         0.0   0.0
1632101    163010004 301000000  0.0375539      0.0       0.0      0002
1632201        0.0            0.0   0.0
1633101    162010000 163000000  0.043937       5.0       5.0     00000
1633201        0.9093         0.0   0.0
1634101    163010000 165000000  0.054482       0.0       0.0     00000
1634201        0.0            0.0   0.0
*
*revise junction Dh
1633110  0.131407 0. 1. 1.
1634110  0.131407 0. 1. 1.
*
**=====================================================================*
*    top of upper plenum (combined with previous upper plenum)         *
**====================================================================**
*
*    inlet elev = bottom = +30.00 in
*   outlet elev = top    = +53.60 in
*
*            name       type
1650000   upplen2      snglvol
*
*              area     length    vol  h-ang  v-ang    deltaz
1650101    0.066556   2.463334    0.0    0.0   90.0  2.463334
*
*           rough       hyd-d  ctl
1650102   3.333e-5    0.166653   0
*
*         ctl  pressure    temp
1650200     3  69.618   278.33 *INITIAL CONDITIONS
*
*======================================================================*
*   core support column(s)  - both modeled as single flow path         *
**====================================================================**
*
* inlet elev  = top    = +67.16 in
* outlet elev = bottom = -39.13 in
*
*               name      type
1820000       suptub      branch
*
*             no.jun    in ctl
1820001            1       1
*
*               area    length    vol   hz ang  vr ang  elev ch
1820101     0.0016085  8.140833   0.0    0.0     -90.0  -8.140833
*
*              rough    hy dia     ctl flag
1820102      2.500e-6        0         0
*
*            ctl    pres    temp
1820200       3   69.618   278.33  *INITIAL CONDITIONS
*
*         from      to        jun area       floss  rloss   ahs
1821101  165010000  182000000  0.0016085       0.0    0.0 00100
*
*          flow-f     flow-g      velg
1821201      0.0      0.0         0.0
*
*======================================================================*
*   control rod guide tube  (top)                                      *
**====================================================================**
*
* inlet elev  = top    = +132.19 in
* outlet elev = bottom =  +58.50 in
*
*               name      type
1840000       guitub1    branch
*
*             no jun    in ctl
1840001            1         1
*
*              area     length   vol h-ang  v-ang   elev-ch
1840101  2.19233e-3     5.880    0.0   0.0  -90.0  -5.880
*
*             rough     hyd-dm   ctl
1840102    2.500e-6        0.0     0
*
*               ctl   pressure  temp
1840200           3    69.618   278.33  *initial conditions
*
*              from         to    jun area  f-loss r-loss  ahs
1841101   165010000  184000000  2.19233e-3     0.0    0.0 00000
*
*            flow-f     flow-g     vel-j
1841201       0.0          0.0       0.0
**=====================================================================*
* lower part of guide tube   (slotted section)                         *
**====================================================================**
*
*    ref drawing = 408010
*
* inlet elev  = top    = -12.00 in
* outlet elev = bottom = -39.13 in
*
*               name      type
1850000      guidlov    branch
*
*             no jun    in ctl
1850001            1         1
*
*               area    length       vol    hz ang    vr ang    elv ch
1850101     0.012718   2.260833      0.0       0.0     -90.0 -2.260833
*
*              rough     hyd-d   ctl flg
1850102      2.500e-6      0.0         0
*
*            ctl    pres    temp
1850200       3   69.618   278.33  *INITIAL COND
*
*               from        to  jun area    f loss    r loss       ahs
1851101    184010000 185000000  2.19233e-3  .43613    .43613     00000
*
*          flow-f     flow-g      velg
1851201      0.0      0.0         0.0
**=====================================================================*
* hot leg, vessel outlet to pressurizer surge line tee                 *
**====================================================================**
*
*  ref drawings = 414036 - i.l. assembly     407346 - 3pc-18
*                 407986 - hot leg nozzle    415155 - 2.5pc-2
*                 414684 - 3pc-1b
*
* inlet elev  = +8.500 in
* outlet elev = +8.500 in
*
*               name      type
2010000         c201      pipe
*
*             no.vol
2010001            3
*
*               area    vol no
2010101     0.045239         1   * hot vessel nozzle (id = 2.880 in)
2010102     0.037554         2   * 3 in sch 160
2010103     0.024629         3   * 2.5 in sch-160
*
*            vol lgh    vol no
2010301     .72083333        1
2010302     3.222500         2 * spool pieces 1 and 2
2010303     1.709500         3 * spool piece 3 to pressurizer tee
*
*             hz ang    vol no
2010501        -45.0         1
2010502        -45.0         2
2010503          0.0         3
*
*             vr ang    vol no
2010601          0.0,        3
*
*             elv ch    vol no
2010701          0.0,        3
*
*              rough    hy dia    vol no
2010801      3.333e-5        0         1
2010802      3.333e-5        0         3
*
*             f loss    r loss    jun no
2010901          0.0       0.0         2
*
*            ctl flg    vol no
2011001            0,        3
*
*                ahs    jun no
2011101         000000       2
*
*         ctl  pressure    temp    zero zero zero vol no
2011201     3  69.618   278.33     0.0  0.0  0.0      3  *INITIAL
*
*         ctl
2011300     1
*
*              flow-f      flow-g  velj    jun no
2011301        0.9087         0.0   0.0         2
*
**=====================================================================*
* hot leg - pressurizer surge line tee                                 *
**====================================================================**
*
*   ref drawing = 415155 - 2.5pc-2
*
*      pressurizer connection is horizontal to the hot leg center line.
*
*   inlet elev  = + 8.500 in
*   outlet elev = + 8.500 in
*
*               name      type
2020000          c202    branch
*
*             no jun    in ctl
2020001            2,        1
*
*               area    length       vol    hz ang    vr ang    elv ch
2020101     0.024629   1.666000      0.0,      0.0,      0.0,      0.0
*
*              rough     hydia   ctl flg
2020102      3.333e-5     0.0          0
*
*         ctl  pressure    temp
2020200     3  69.618   278.33 *INITIAL CONDITIONS
*
*               from        to  jun area    f loss    r loss       ahs
2021101   201010000,202000000, 0.024629        0.0       0.0    030000
*
*              flow-f      flow-g  velj
2021201        0.9087         0.0   0.0
2022101   202010000,203000000, 0.024629        0.0       0.0    030000
2022201        0.9087         0.0   0.0
*
**=====================================================================*
* pressurizer surge line tee - steam generator inlet                   *
**====================================================================**
*
*   ref drawings = 414431 - 2.5pc-6    414427 - 2.5pc-9
*                  414425 - 2.5pc-7    414271 - inlet plenum connection
*                  414426 - 2.5pc-8
*
* inlet elev = +8.500 in      (hot leg)
* outlet elev = +73.96 in     (bot. of inlet plenum)
*
*               name      type
2030000          c203     pipe
*
*             no.vol
2030001          3
*
*               area    vol n0
2030101      0.024629        3  * 2.5 in sch-160
*
*            vol lgh    vol no
2030301     1.620330         1  * included from previous component 202
2030302     3.905420         2  * spool pieces 4,5,6
2030303     2.126250         3  * spool piece 7
*
*             vr ang    vol no
2030601         0.0          1
2030602       +90.0          2
2030603       +45.0          3
*
*             elv ch    vol no
2030701     0.0              1
2030702     3.727080         2
2030703     1.728330         3
*
*              rough    hy dia    vol no
2030801      3.333e-5        0         3
*
*             f loss    r loss    jun no
2030901          0.0     0.0           1
2030902          0.0     1.705         2
*
*            ctl flg    vol no
2031001            0       3
*
*               cahs    jun no
2031101        00000       2
*
*         ctl  pressure    temp    zero zero zero vol no
2031201    3   69.618   278.33     0.0  0.0  0.0      3  *
*
*         ctl
2031300     1
*
*              flow-f      flow-g    velj  jun no
2031301        0.9087         0.0     0.0       2
*
**=====================================================================*
* steam generator inlet                                                *
**====================================================================**
*
*   reference drawing = 414271
*
*   elev = +73.96 in
*
*               name      type
2050000         j205   sngljun
*
*
*               from        to    jn area     f loss    r loss    ahs
*2050101    203010000,210000000   0.024629        0.0      1.35   30000
2050101    203010000,210000000   0.024629        0.0      1.35   100000 * turn on ccfl*
*         ctl  flow-f      flow-g  velj
2050201     1  0.9087         0.0   0.0
*
**=====================================================================*
* steam generator inlet plenum                                         *
**====================================================================**
*
*   reference - type-ii (intact loop) steam generator pant-leg drwg,
*                       and plenum drwg. 414271 (1,2)
*        note - inlet plenum entrance is circular (r=2.125 in)
*               plenum tapers outward,nonuniformly,with rise to
*               tube sheet inlet, (inlet plenum outlet)   outlet
*               is rectangular with radiused corners.
*
* inlet elv = +73.96 in
* outlet elv = +82.21 in
*
*               name      type
2100000         c210   snglvol
*
*        area    length    volume    h ang   v ang   delta-z
2100101   0.0  0.687500  0.0551215   +90.0   +90.0   0.687500
*
*         rough    hyd-d    ctl
2100102  3.333e-5  0.27514    0
*
*         ctl  pressure    temp
2100200     3  69.618   278.33 *INITIAL CONDITIONS
*
**=====================================================================*
* intact loop steam generator tube sheet inlet                         *
**====================================================================**
*
* inlet elev  =  +81.594 in (bottom of tube sheet)
* outlet elev = +102.594 in (top of tube sheet)
*
*       name      type
2150000 tsin      branch
*
*       njun     jctl
2150001 2        1
*
*       areav        length       vol          hang   vang   deltaz
2150101 1.9757e-2    1.7500       0.0           0.0   90.0   1.75
*
*       rough        hydia        fe
2150102 3.333e-5     0.06475      00
*
* .....................................................................*
*
*       ctl   press   temp
2150200 3     69.618   278.33  *INITIAL COND
*
*       from      to        areaj        floss    rloss     vcahs
*2151101 210010000 215000000 1.97569e-2   10.0     0.013     30100
2151101 210010000 215000000 1.97569e-2   10.0     0.013     00100
*
*       flowf     flowg     velj
2151201 0.9087    0.0       0.0
*2152101 215010000 220000000 1.97569e-2    0.0     0.0       30000
2152101 215010000 220000000 1.97569e-2    0.0     0.0       100000 *turn on ccfl
2152201 0.9087    0.0       0.0
*
* revise junction Dh
2151110 0.27514 0. 1. 1.
2152110 0.06475 0. 1. 1.
*
**=====================================================================*
* steam generator tube bundle                                          *
**====================================================================**
*
*   reference drawing = 413415
*
*   elev top tube    = 391.06 in. above top of tube sheet
*   elev middle tube = 364.50 in. above top of tube sheet
*   elev bottom tube = 337.03 in. above top of tube sheet
*
*   inlet elev = + 82.21  in
*   tube bundle top elev = +446.71  in
*
*               name      type
2200000        isgtub     pipe
*
*             no.vol
2200001           18
*
*               area      vol n0
2200101     1.9756984e-2    18      * 6 tubes, id = 0.777 in
*
*            vol lgh    vol no
2200301     3.9970833      1      *18 volumes around u-tubes; 9 up and
2200302     3.9554167      7      *    9 down.  lengths are equivalent
2200303     1.5937500      8      *    to secondary side lengths (c600)
2200304     1.1289700     10
2200305     1.5937500     11
2200306     3.9554167     17
2200307     3.9970833     18
*
*             vr ang    vol no
2200601        +90.0,      8
2200602        +45.0,      9
2200603        -45.0,     10
2200604        -90.0,     18
*
*             elv ch     vol no
2200701    +3.9970833       1
2200702    +3.9554167       7
2200703    +1.5937500       8
2200704    +1.0516700       9      * bend radius = 1.935 in
2200705    -1.0516700      10      * bend radius = 1.935 in
2200706    -1.5937500      11
2200707    -3.9554167      17
2200708    -3.9970833      18
*
*              rough    hy dia    vol no
2200801      2.500e-6  0.06475       18  * correct dhy
*
*             f loss    r loss    jun no
2200901        0.043    0.0112        8  * corrected for dhy
2200902        0.043    0.0112       17  * corrected for dhy, add rloss
*
*            ctl flg    vol no
2201001         0          18
*
*            ctl flg    jun no
2201101         0          17
*
*         ctl  pressure    temp    zero zero zero vol no
2201201     3  69.618   278.33     0.0  0.0  0.0      2  *
2201202     3  69.618   278.33     0.0  0.0  0.0      4  *
2201203     3  69.618   278.33     0.0  0.0  0.0      6  *
2201204     3  69.618   278.33     0.0  0.0  0.0      9  *
2201215     3  69.618   278.33     0.0  0.0  0.0     12  *
2201216     3  69.618   278.33     0.0  0.0  0.0     14  *
2201217     3  69.618   278.33     0.0  0.0  0.0     16  *
2201218     3  69.618   278.33     0.0  0.0  0.0     18  *
*
*         ctl
2201300     1
*
*              flow-f      flow-g  velj    jun no
2201301        0.9088         0.0   0.0      17
*
* revise junction Dh
2201401   0.06475  0. 1. 1. 17
*
**=====================================================================*
* intact loop steam generator tube sheet outlet                        *
**====================================================================**
*
* inlet  elev = +102.594 in (top of tube sheet)
* outlet elev =  +81.594 in (bottom of tube sheet)
*
*       name      type
2250000 tsout     branch
*
*       njun     jctl
2250001 2        1
*
*       areav        length       vol          hang   vang   deltaz
2250101 1.9757e-2    1.7500       0.0           0.0  -90.0  -1.75
*
*       rough        hydia        fe
2250102 3.333e-5     0.06475      00
*
* .....................................................................*
*
*       ctl   press   temp
2250200 3     69.618    220.73 *INITIAL COND
*
*       from      to        areaj        floss    rloss     vcahs
*2251101 220010000 225000000 1.97569e-2   10.0     0.013     30100
2251101 220010000 225000000 1.97569e-2   10.0     0.013     00100
*
*       flowf     flowg     velj
2251201 0.9087    0.0       0.0
2252101 225010000 230000000 1.97569e-2    0.0     0.0       30000
2252201 0.9087    0.0       0.0
*
* revise junction Dh
2251110 0.06475 0. 1. 1.
2252110 .275140 0. 1. 1.
*
**=====================================================================*
* steam generator outlet plenum                                        *
**====================================================================**
*
*   see note for steam generator inlet plenum (c220)
*
* inlet elev  = top    = +82.21 in
* outlet elev = bottom = +73.96 in
*
*               name      type
2300000        c230    snglvol
*
*        area   length    volume    h-ang   v-ang    delta-z
2300101  0.0   0.68750  0.0551215   +90.0   -90.0    -0.68750
*
*             rough     hyd-d   ctl
2300102     3.333e-5  .275140     0
*
*         ctl  pressure    temp
2300200     3  69.618    220.73  *INITIAL COND
*
**=====================================================================*
* steam generator outlet                                               *
**====================================================================**
*
*   see note for sg inlet (j205)
*
* elev = +73.96 in
*
*               name      type
2350000           j8   sngljun
*
*             from        to   jn area   f loss  r loss    ahs
2350101  230010000 240000000  0.024629     1.64    1.64  00000
*
*         ctl  flow-f      flow-g  velj
2350201     1  0.9087         0.0   0.0
*
**=====================================================================*
* pump suction leg                                                     *
**====================================================================**
*
*   ref drawings = inlet plenum - 414271   2.5pc-10  - 414428
*                       2.5pc-6 - 414431   2.5pc-11  - 414425
*                       2.5pc-7 - 414425   2.5pc-12  - 414426
*                       2.5pc-8 - 414426   2.5pc-13  - 414429
*                       2.5pc-9 - 414427   2.5pc-14a - 414430
*                       3pc-20 (spool piece-13 thru 15) - 409027
*                       3pc-20 (spool piece-16) - 414684
*                       3pc-9a - 404749
*                       3pc-10a - 408613
*
*   inlet elev = +73.96 in
*   u-bend bottom elev = -110.658 in
*   outlet elev = -10.09 in
*
*               name      type
2400000         c240      pipe
*
*             no.vol
2400001            9
*
*               area    vol n0
2400101      0.0246289       3 * 2.5 in sch-160
2400102      0.037554        8 * 3.0 in sch-160
2400103      0.009764        9 * 1.5 in sch-160
*
*        vol length     vol no
2400301     1.45958          1 * spool piece - 8
2400302     3.66625          2 *          sp - 8,9,10
2400303     2.78417          3 *          sp - 11,12
2400304     3.87500          4 *          sp - 13
2400305     4.08908          5 *          sp - 13 to bot. of pump suc.
2400306     6.02658          6 * bottom of pump suc. thru sp-15
2400307     1.92167          7 *          sp - 16
2400308     1.66232          8 *          pump bypass spool, vert
2400309     3.35458          9 *          pump bypass spool, horz
*
*             vr ang    vol no
2400601        -45.0         1
2400602        -90.0         5
2400603         90.0         8
2400604          0.0         9
*
*             elv ch    vol no
2400701     -1.18333         1
2400702     -3.66625         2
2400703     -2.78417         3
2400704     -3.87500         5
2400705      5.81250         6
2400706      1.92167         7
2400707      1.485837        8
2400708      0.0             9
*
*              rough    hy dia    vol no
2400801      3.333e-5        0         9
*
*             f loss    r loss    jun no
2400901         3.12      0.00         3
2400902         1.34      0.36         4
2400903         0.00      0.36         6
2400904         0.00      0.024        8
*
*            ctl flg    vol no
2401001            0         9
*
*            ctl flg    jun no
2401101            0         8
*
*         ctl  pressure    temp    zero zero zero vol no
2401201     3  69.618    220.73    0.0  0.0  0.0      1  *
2401202     3  69.618    220.73    0.0  0.0  0.0      2  *
2401203     3  69.618    220.73    0.0  0.0  0.0      3  *
2401204     3  69.618    220.73    0.0  0.0  0.0      4  *
2401205     3  69.618    220.73    0.0  0.0  0.0      5  *
2401206     3  69.618    220.73    0.0  0.0  0.0      6  *
2401207     3  69.618    220.73    0.0  0.0  0.0      7  *
2401208     3  69.618    220.73    0.0  0.0  0.0      8  *
2401209     3  69.618    220.73    0.0  0.0  0.0      9  *
*
*         ctl
2401300     1
*
*              flow-f      flow-g  velj    jun no
2401301        0.9087         0.0   0.0         8
*
*======================================================================*
*    i. l. pump bypass outlet junction                                 *
**====================================================================**
*
*  loss coefficients include resistance of globe valve and seal ring
*
*               name         type
2500000         j250         sngljun
*
*         from      to         jn area     floss     rloss   ahs
2500101  240010000 261000000   .00976      21.49    21.49   30000
*
*          ctl     flow-f     flow-g        velj
2500201     1       .9088      0.           0.
*
**=====================================================================*
* cold leg...pump outlet to ecc injection point tee                    *
**====================================================================**
*
*   ref drawings =     3pc-11a - 412858    3pc-1a - 407718
*                      3pc-12  - 404759
*
* inlet elev  = 0.000 in
* outlet elev = 0.000 in
*
*               name      type
2610000         c261     pipe
*
*             no.vol
2610001            2
*
*               area    vol n0
2610101      0.009764        1 * 1.5 in sch-160 (id = 2.624 in)
2610102      0.037554        2 * 3 in sch-160 (id = 2.624 in)
*
*            vol lgh    vol no
2610301      4.49752         1 * pump bypass outlet
2610302      2.55428         2 * spool piece - 21 to ecc tap in sp-23
                             * + pump bypass outlet contribution
*
*             vr ang    vol no
2610601          0.0,        2
*
*             elv ch    vol no
2610701          0.0,        2
*
*              rough    hy dia    vol no
2610801      3.333e-5        0,        2
*
*            ctl flg    vol no
2611001            0         2
*
*            ctl flg    jun no
2611101          000         1
*
*         ctl  pressure    temp    zero zero zero vol no
2611201     3  69.618    220.73    0.0  0.0  0.0      2  *INITIAL
*
*                ctl
2611300            1
*
*              flow-f    flow-g   vel-j   jun no
2611301        0.9087       0.0     0.0        1
*
**=====================================================================*
* ecc injection tee                                                    *
**====================================================================**
*
*   ref drawing =    3pc-19a - 414684
*
* inlet elev  = 0.000 in
* outlet elev = 0.000 in
*
*               name      type
2620000         c262    branch
*
*             no jun    in ctl
2620001            2         1
*
*               area    length       vol    hz ang    vr ang    elv ch
2620101    0.037554     1.691730     0.0,    202.5,      0.0,      0.0
*
*              rough     hydia   ctl flg
2620102      3.333e-5      0.0,        0
*
*         ctl  pressure    temp
2620200     3  69.618    220.73  *INITIAL COND
*
*               from        to  jun area    f loss    r loss      cahs
2621101    261010000 262000000  .0375540      0.87      0.53    000000
*
*              flow-f      flow-g  velj
2621201        0.9087         0.0   0.0
2622101    262010000 263000000  .0375540      0.87      0.53    000000
2622201        0.9087         0.0   0.0
*
**=====================================================================*
* cold leg...ecc injection point to downcommer inlet                   *
**====================================================================**
*
*   ref drawings =   407986 (downcomer cold leg nozzle)
*
*   inlet elev  = 0.000 in
*   outlet elev = 0.000 in
*
*               name      type
2630000         c263      pipe
*
*             no.vol
2630001            2
*
*               area    vol n0
2630101      .037554         1 * included fron previous component 262
2630102      .045239         2 * 3 in sch-160 (id = 2.624 in)
*
*            vol lgh    vol no
2630301      1.633305        1   *
2630302      0.595833        2   * cold leg downcomer nozzle
*
*             vr ang    vol no
2630601          0.0         2
*
*             elv ch    vol no
2630701          0.0         2
*
*              rough    hy dia    vol no
2630801      3.333e-5      0.0         2
*
*            ctl flg    vol no
2631001            0         2
*
*            ctl flg    jun no
2631101          0        1                                     *tsa/dmk
*
*         ctl  pressure    temp    zero zero zero vol no
2631201     3  69.618    220.73    0.0  0.0  0.0      2  *
*
*         flow-f   flow-g   velj
2631301     0.9087      0.0       0.0     1                     *tsa/dmk
*
*======================================================================*
* primary system mass increaser                                        *
*======================================================================*
*
*       name         type
2900000 pridecr      tmdpjun
*
*geotdj from      to        areaj
2900101 095000000 261000000 0.009764
*
*......................................................................*
*
*ictdj1 jctl trip
2900200    1 612  * increase pri inv
*
*ictdj2 time         flowf        flowg        velj
2900201    -1.0       0.0         0.0          0.0 * increase pri inv
2900202     0.0       0.0         0.0          0.0 * increase pri inv
2900203     0.0      10.0         0.0          0.0 * increase pri inv
2900204 10000.0      10.0         0.0          0.0 * increase pri inv
*
*======================================================================*
* primary system mass decreaser                                        *
*======================================================================*
*
*       name         type
2910000 princrs      tmdpjun
*
*geotdj from      to        areaj
2910101 261000000 096000000 0.009764
*
*......................................................................*
*
*ictdj1 jctl trip
2910200    1 611  * decrease pri inv
*
*ictdj2 time         flowf        flowg        velj
2910201    -1.0       0.0         0.0          0.0 * decrease pri inv
2910202     0.0       0.0         0.0          0.0 * decrease pri inv
2910203     0.0      10.0         0.0          0.0 * decrease pri inv
2910204 10000.0      10.0         0.0          0.0 * decrease pri inv
*
**=====================================================================*
* hot leg nozzle                                                       *
**====================================================================**
*    ref drawings  =   nozzle    - 407975
*    elev  = + 8.50 in
*
*               name      type
3010000        bl-hl    snglvol
*
*            area    length   vol  h-ang   v-ang delta-z
3010101 0.0375539 1.3391667   0.0    0.0     0.0     0.0 * 3 in. sch-160
*
*           rough    hy dia   ctl
3010102  3.333e-5       0.0     0
*
*         ctl  pressure    temp
3010200     3  69.618   278.33 *INITIAL CONDITIONS
*
*
**=====================================================================*
* cold leg nozzle                                                      *
**====================================================================**
*    ref drawings  =   nozzle    - 407986
*    elev  = + 8.50 in
*
*               name      type
3630000        bl-cl    snglvol
*
*            area    length   vol  h-ang   v-ang delta-z
3630101 0.0375539 1.2761670   0.0    0.0     0.0     0.0 * 3 in. sch-160
*
*           rough    hy dia   ctl
3630102  3.333e-5       0.0     0
*
*         ctl  pressure    temp
3630200     3  69.618    220.73 *INITIAL CONDITIONS
*
**=====================================================================*
* steam generator secondary volume                                     *
**====================================================================**
*
*   reference drawings = 413463 - assembly     406419 - separator
*                        413357 - fillers      413415 - baffle plates
*
*   inlet  (w.r.t. top of tube sheet) = 0.00 in
*   outlet (w.r.t. top of tube sheet) = 446.4 in (top of centif. sep.)
*
*               name      type
6000000        isgsec   annulus
*
*             no.vol
6000001           11
*
*               area    vol no
6000101          0.0        11
*
*               area    jun n0
6000201      0.04299         6 * baffle plates along flow path (regular)
6000202      0.03989         7 * center of plate supports tc-tube
6000203      0.0513417       8 * regular baffle plate - above 1 u-tube
6000204      0.0596933       9 * regular baffle plate - above 2 u-tubes
6000205      0.0            10 * no plate
*
*            vol lgh    vol no
6000301      3.99708333      1 * vol begins above top of tube sheet.
6000302      3.95541667      7 * volumes are separated by junctions
6000303      1.59375000      8 * at baffle plates.
6000304      1.92708333      9
6000305      1.37437500     10
6000306      4.17729165     11
*
*             volume    vol no
6000401      0.547369965     1
6000402      0.541664035     2
6000403      0.499500500     4
6000404      0.457207500     6
6000405      0.537511300     7
6000406      0.216578607     8
6000407      0.261876093     9
6000408      0.434639013    10
6000409      2.036472246    11
*
*             vr ang    vol no
6000601        +90.0        11
*
*             elv ch    vol no
6000701       3.99708333     1
6000702       3.95541667     7
6000703       1.59375000     8
6000704       1.92708333     9
6000705       1.37437500    10
6000706       4.17729165    11
*
*              rough     hy dia    vol no
6000801      3.333e-5  .1019770         7
6000802      3.333e-5  .1333038         8
6000803      3.333e-5  .1954227         9
6000804      3.333e-5  .2207585        10
6000805      3.333e-5     0.0          11
*
*            ctl flg    vol no
6001001            0        11
*
*                ahs    jun no
6001101        00100         9   * baffle plates
6001102        00000        10   * above top of highest u-tube
*
*         ctl  pressure   quals    zero zero zero vol no
6001201     2  23.406   6.07063e-3  0.0  0.0  0.0      2  *
6001202     2  23.406  8.56733e-3  0.0  0.0  0.0      4  *
6001203     2  23.406  9.89449e-3  0.0  0.0  0.0      7  *
6001204     2  23.406  1.00895e-3  0.0  0.0  0.0      9  *
6001215     2  23.406  1.95004e-2  0.0  0.0  0.0     11  *
*
*         ctl
6001300     0
*
*              velf       velg     velj    jun no
6001301        0.9513     1.5826    0.0         2
6001302        0.9682     1.7825    0.0         4
6001303        0.9650     1.8526    0.0         6
6001304        0.9617     1.9227    0.0         8
6001305        0.8690     1.9626    0.0         10
*
* revise junction Dh
6001401 .1019770  0. 1. 1.  8
6001402 .1954227  0. 1. 1.  9
6001403 .2207585  0. 1. 1. 10
*
**=====================================================================*
* intact loop steam generator secondary separator  (top of riser)      *
**====================================================================**
*
*   reference drawing = 413463
*
*   mix  inlet = 441.625 in (w.r.t. top of tube sheet)
*   vap outlet = 446.390 in (w.r.t. top of tube sheet)
*   liq outlet = 441.625 in (w.r.t. top of tube sheet)
*
*       name     type
6010000 isgdct   separatr
*
*       njun ctl
6010001 3    0
*
*       areav        length       vol          hang vang delta-z
6010101 0.0          0.397083350  0.235149741  0.0  90.0 0.397083350
*
*       rough        dhy          fe
6010102 3.333e-5     0.0          00
*
* note 1. f,rloss based on abrupt area and 2 ea, 90 deg miter bends
* note 2. f,rloss based on abrupt area and 1 ea, 180 deg miter bend
*
*         ctl  pressure   quals
6010200     2  23.2060  0.25618  *INITIAL COND
*
*       from      to        area         floss  rloss  vcahs
6011101 601010000 611000000 0.78769369   401.0  1.0e+5 01000 * note 1.
*
*              velf        velg    velj
6011201        5.847e-4    5.847e-4 0.0
6012101 601000000 602000000 0.78769369   400.0  1.0e+5 01000 * note 2.
6012201        1.3879      0.2357   0.0
6013101 600010000 601000000 0.59219240   2.1441 1.2244 01000
6013201        0.2749      0.2749   0.0
*
**=====================================================================*
* intact loop steam generator secondary downcomer (top)                *
**====================================================================**
*
*   reference drawing = 413463
*
*   inlet  (w.r.t. top of tube sheet) = 446.39 in
*   outlet (w.r.t. top of tube sheet) = 391.497 in
*
*               name      type
6020000        isgsd    annulus
*
*             no.vol
6020001            5
*
*               area    vol no
6020101          0.0         5
*
*               area    jun n0
6020201      0.787693687     1 * entrance to top tapered reduction
6020202      0.221915888     2 * tapered reduction throat
6020203      0.446473631     3 * exit from tapered expansion
6020204      0.0             4 * entrance at top of filler
*
*            vol lgh    vol no
6020301      2.002083317     1 * vol begins at szp601 liq outlet
6020302      1.000000000     2 * top tapered reduction
6020303      0.424583333     3 * second tapered expansion
6020304      0.750666667     4 * above d.c. filler top
6020305      1.374333333     5 * straight + taper at d.c. filler top
*
*             volume    vol no
6020401      1.577028389     1 * straight annulus
6020402      0.504804787     2 * tapered reduction annulus
6020403      0.141893525     3 * tapered expansion annulus
6020404      0.335152872     4 * straight annulus
6020405      0.452285538     5 * straight + tapered annulus
*
*             vr ang    vol no
6020601        -90.0         5 *
*
*             elv ch    vol no
6020701     -2.002083317     1 *
6020702     -1.000000000     2 *
6020703     -0.424583333     3 *
6020704     -0.750666667     4 *
6020705     -1.374333333     5 *
*
*              rough     hy dia    vol no
6020801      3.333e-5  .447833334       1 * (2*gap)
6020802      3.333e-5  .300105988       2 * (4*v/as), as = 6.728353402
6020803      3.333e-5  .240329657       3 * (4*v/as), as = 2.361648189
6020804      3.333e-5  .322916667       4 * (2*gap)
6020805      3.333e-5  .215937514       5 * (4*v/as), as = 8.378081766
*
*       floss   rloss njun
6020901 0.0     0.0   1   *
6020902 2.057   1.927 2   * crane, a-26, formulas 1 and 3
6020903 0.0     0.0   4   *
*
*            ctl flg    vol no
6021001            0         5
*
*                ahs    jun no
6021101        00000         3   * smooth
6021102        00100         4   * abrupt entrance for filler
*
*         ctl  pressure   quals     zero zero zero vol no
6021201     2  23.406  4.27635e-3   0.0  0.0  0.0      1
6021202     2  23.406      0.0      0.0  0.0  0.0      5
*
*         ctl
6021300     0
*             velf       velg     velj   jun no
6021301      0.24722   0.17625     0.0        4
*
* revise junction Dh
6021401      .300105988  0. 1. 1. 1
6021402      .240329657  0. 1. 1. 2
6021403      .322916667  0. 1. 1. 3
6021404      .215937514  0. 1. 1. 4
*
**=====================================================================*
* intact loop steam generator secondary downcomer (middle)             *
**====================================================================**
*
*   reference drawings = 413363 - assembly
*                        414048 - fillers
*                        415419 - bottom feedwater inlet
*
*     inlet = 391.50 in (w.r.t. to top of tube sheet)
*    outlet =  14.37 in (w.r.t. to top of tube sheet)
*
*                name     type
6030000        isgdcm   annulus
*
*            no. vol
6030001            8
*
*               area    vol n0
6030101          0.0         8
*
*            vol lgh    vol no
6030301    3.52083333        1
6030302    3.95541667        7
6030303    2.79957677        8
*
*         volume    vol no
6030401  0.103092        1
6030402  0.115817        6
6030403  0.115817        7
6030404  0.081973        8
*
*             hz ang    vol no
6030501          0.0         8
*
*             vr ang    vol no
6030601        -90.0         8
*
*             elv ch    vol no
6030701   -3.52083333        1
6030702   -3.95541667        7
6030703   -2.79957677        8
*
*              rough    hy dia    vol no
6030801      3.333e-5  .03357905       8
*
*            ctl flg    vol no
6031001            0         8
*
*            ctl flg    jun no
6031101            0         7
*
*         ctl  pressure   quals    zero zero zero vol no
6031201     2  23.406   0.0        0.0  0.0  0.0      2  *
6031202     2  23.406   0.0        0.0  0.0  0.0      4  *
6031203     2  23.406   0.0        0.0  0.0  0.0      6  *
6031204     2  23.406   0.0        0.0  0.0  0.0      8  *
*
*         ctl
6031300     0
*
*               velf       velg    velj    jun no
6031301        3.6417     3.6417    0.0         7
*
* revise junction Dh
6031401 .03357905  0. 1. 1. 7
*
**=====================================================================*
* intact loop steam generator secondary downcomer (bottom)             *
**====================================================================**
*
* inlet elev  = top    = 14.37 in. above top of tube sheet
* outlet elev = bottom =  0.00 in. above top of tube sheet
*
*               name      type
6040000       isgdcb    branch
*
*             no jun    in ctl
6040001            2         0
*
*               area    length       vol    hz ang    vr ang    elv ch
6040101          0.0   1.19750656   0.035064   0.0     -90.0 -1.19750656
*
*              rough     hyd-d   ctl flg
6040102      3.333e-5   .03357905      0
*         f,rloss based on 180 deg miter turn  *****.....*****
*
*         ctl  pressure   quals
6040200     2  23.206   1.18953e-3 *INITIAL
*
*               from        to  jun area    f loss    r loss       ahs
6041101    603010000 604000000  0.0292806      0.0       0.0     00000
*
*              velf        velg    velj
6041201       3.6417      3.6417   0.0
6042101    604010000 600000000  0.0292806      4.967     4.967   00100
6042201       3.6417      3.6417   0.0
*
* revise junction Dh
6041110  .03357905 0. 1. 1.
6042110  .03357905 0. 1. 1.
*
**=====================================================================*
* intact loop steam generator secondary downcommer upper annulus       *
**====================================================================**
*
*   reference drawing = 413463
*
*   bottom = 441.625 in (w.r.t. top of tube sheet)
*   top    = 446.390 in (w.r.t. top of tube sheet)
*
*       name     type
6050000 isgdct   branch
*
*       njun ctl
6050001 2    0
*
*       areav        length       vol          hang vang delta-z
6050101 0.0          0.397083350  0.312780048  0.0 -90.0 -0.397083350
*
*       rough        dhy          fe
6050102 3.333e-5     0.447833334  00
*
*       ctl pressure     quals
6050200 2   23.206      4.27635e-3 *INITIAL
*
*       from      to        area         floss  rloss  vcahs
6051101 605000000 611000000 0.78769369   0.1099 0.1492 00000
*
*              velf        velg    velj
6051201        5.847e-4    5.847e-4 0.0
6052101 605010000 602000000 0.78769369   0.0    0.0    00000
6052201        1.3879      0.2357   0.0
*
**=====================================================================*
* steam generator secondary - time dep feedwater junction              *
**====================================================================**
*
*   elev = 14.37 in (above top of tube sheet)
*
*
*               name      type
6100000       isgfwj   tmdpjun
*
*               from        to   jn area
6100101    630000000 603000000   0.012
*
*
*       ctl trip name     parameter
6100200 1   0    cntrlvar 806 * feed = discharge flow
*
*       outflow      flowf        flowg        velj
6100201 0.0          0.0          0.0          0.0
6100202 4.5          4.5          0.0          0.0
*
**=====================================================================*
* intact loop steam generator secondary steam dome outer annulus       *
**====================================================================**
*
*      ref drawing = 413463
*
*       inlet elev = 446.39 in. above top of tube sheet
*      outlet elev = 450.23 in. above top of tube sheet (cross flow j)
*
*               name     type
6110000       isgsdx     snglvol
*
*         area      length      volume h ang    v ang     delta-z
6110101    0.0 0.639185552 0.753346088   0.0     90.0 0.639185552
*
*              rough     dhy     ctl-flg
6110102      3.333e-5 0.791583333      0
*
*         ctl  pressure   quals
6110200     2  23.206    1.000 *INITIAL
*
**=====================================================================*
* intact loop steam generator steam dome                               *
**====================================================================**
*
*     reference drawing = 413463
*
*     inlet elev = 450.23 in. above top of tube sheet (cross flow j)
*    outlet elev = 454.60 in. above top of tube sheet
*
*               name      type
6120000       isgdome   branch
*
*             no jun    in ctl
6120001            2         0
*
*         area    length       vol    hz ang    vr ang    elv ch
6120101    0.0 0.68445497  0.137320027   0.0      90.0  0.68445497
*
*              rough     hyd-d   ctl flg
6120102      3.333e-5      0.0         0
*
*         ctl  pressure   quals
6120200     2  23.206    1.000 *INITIAL
*
*               from        to  jun area    f loss    r loss      cahs
6121101    611000000 612000000  0.31743034  4.0850    4.8739      0003
*
*              velf        velg    velj
6121201        5.847e-4    5.847e-4 0.0
6122101    612010000 613000000  0.02244496     0.0       0.0     00100
6122201        5.847e-4    5.847e-4 0.0
*
**=====================================================================*
* intact loop steam generator secondary steam outlet                   *
**====================================================================**
*
*      ref drawing = 413463
*
*       inlet elev = 454.60 in. above top of tube sheet
*      outlet elev = 514.99 in. above top of tube sheet
*
*               name     type
6130000       isgout     snglvol
*
*            area   length    volume    h ang    v ang    delta-z
6130101       0.0   5.03213  0.112946     0.0     90.0    5.03213
*
*              rough     dhy     ctl-flg
6130102      3.333e-5 0.1666667        0
*
*         ctl  pressure   quals
6130200     2  23.206    1.000 *INITIAL
*
**=====================================================================*
* intact loop steam generator secondary downcomer (middle jun)         *
**====================================================================**
*
*   reference  drawings = 413363, 414048
*
*   elevation = 391.5 in above top of tube sheet
*
*               name      type
6230000       isgdcj   sngljun
*
*               from        to   jn area     f loss  r loss vcahs
6230101    602010000 603000000  0.029280567     0.0     0.0 00100
*
*         ctl   velfj   velgj   velj
6230201     1  3.6417  3.6417    0.0
*
**=====================================================================*
* steam generator secondary - time dep feedwater volume                *
**====================================================================**
*
*               name      type
6300000       isgfwv    tmdpvol
*
*               area       lgh      vol     hz ang    vr ang   elv ch
6300101        .012      10.00,     0.0,       0.0,      0.0,     0.0
*
*              rough    hy dia       flg
6300102          0.0,      0.0,        0
*
*                ctl
6300200            3  *changed from 3
*
*              time      pres      temp
6300201         0.0      23.406    185.9346  *  23.206  CHANGED
*
**=====================================================================*
* steam generator secondary - relief valve                             *
**====================================================================**
*
*               name      type
6340000       isgrv      valve
*
*               from        to     jn area   f loss   r loss    flg
6340101    613010000 640000000        .021      0.0      0.0    100
*
*           ctl    flow-f    flow-g    if-vel
6340201       1       0.0       0.0       0.0
*
*          vlv type
6340300      trpvlv
*
*              trip
6340301         504
*
**=====================================================================*
* steam generator secondary - discharge junction valve                 *
**====================================================================**
*
*               name      type
6350000        isgdj     valve
*
*               from        to     jn area   f loss   r loss   flg
6350101    613010000 640000000        .007   204.67   204.67 01100
*
*           ctl   flow-f    flow-g   if-vel
6350201        1       0.0     0.0573       0.0
*
*            type
6350300    mtrvlv
*
*        op trip   cl trip    slope    init pos    table
6350301      514       513      0.2      1.000         0
*
**=====================================================================*
* steam generator secondary - discharge branch                         *
**====================================================================**
*
*   inlet elev = 462.1 in (above top of tube sheet)
*
*               name      type
6400000        isgdb    branch
*
6400001            1         1
*
*               area       lgh      vol     hz ang    vr ang   elv ch
6400101       .70584      10.0      0.0        0.0       0.0      0.0
*
*              rough    hy dia      flg
6400102          0.0,      0.0,       0
*
*         ctl      temp   quals
6400200     1    211.42     1.0
*
*               from       to     area     f loss    r loss      flg
6401101    640010000 650000000  .70584        0.0       0.0        0
*
*             flow-f    flow-g   if-vel
6401201          0.0   8.1748e-2    0.0
*
**=====================================================================*
* steam generator secondary - time dep discharge volume                *
**====================================================================**
*
*               name      type
6500000        isgdv   tmdpvol
*
*               area       lgh      vol     hz ang    vr ang   elv ch
6500101       .70584      10.0      0.0        0.0       0.0      0.0
*
*              rough    hy dia       flg
6500102          0.0,      0.0,        0
*
*                ctl
6500200            2
*
*               time      pres      qual
6500201          0.0     23.00      1.0  *CHANGED
*
*======================================================================*
* secondary system mass increaser                                      *
*======================================================================*
*
*       name         type
6900000 sgdecr      tmdpjun
*
*geotdj from      to        areaj
6900101 695000000 604010000 0.0292808
*
*......................................................................*
*
*ictdj1 jctl trip
6900200    1 602  * increase sg inv
*
*ictdj2 time         flowf        flowg        velj
6900201    -1.0       0.0         0.0          0.0 * increase sg inv
6900202     0.0       0.0         0.0          0.0 * increase sg inv
6900203     0.0      10.0         0.0          0.0 * increase sg inv
6900204 10000.0      10.0         0.0          0.0 * increase sg inv
*
*======================================================================*
* secondary system mass decreaser                                      *
*======================================================================*
*
*       name         type
6910000 sgincr      tmdpjun
*
*geotdj from      to        areaj
6910101 604010000 696000000 0.0292808
*
*......................................................................*
*
*ictdj1 jctl trip
6910200    1 601  * decrease sg inv
*
*ictdj2 time         flowf        flowg        velj
6910201    -1.0       0.0         0.0          0.0 * decrease sg inv
6910202     0.0       0.0         0.0          0.0 * decrease sg inv
6910203     0.0      10.0         0.0          0.0 * decrease sg inv
6910204 10000.0      10.0         0.0          0.0 * decrease sg inv
*
*======================================================================*
* secondary system source volume                                       *
*======================================================================*
*
*       name         type
6950000 sgsrcin   tmdpvol
*
*geomv1 areav        length       volume       hang         vang
6950101 0.0292808     1.0          0.0          0.0          0.0
*
*geomv2 dz           rough        dhy          fe
6950102 0.0          0.0          0.0          00
*
*......................................................................*
*
*ictdv1 ebt
6950200 003
*
*ictdv2 time         pressure     temp
6950201     0.0      850.0        520.0 * sg inv, source
6950202 10000.0      850.0        520.0 * sg inv, source
*
*======================================================================*
* secondary system sink volume                                         *
*======================================================================*
*
*       name         type
6960000 sgsrcex   tmdpvol
*
*geomv1 areav        length       volume       hang         vang
6960101 0.0292808    1.0          0.0          0.0          0.0
*
*geomv2 dz           rough        dhy          fe
6960102 0.0          0.0          0.0          00
*
*......................................................................*
*
*ictdv1 ebt
6960200 003
*
*ictdv2 time         pressure     temp
6960201     0.0      850.0        520.0 * sg inv, sink
6960202 10000.0      850.0        520.0 * sg inv, sink
*
**=====================================================================*
*   heat loss ambient volume                                           *
**====================================================================**
*
*            name       type
8000000     hlvol    tmdpvol
*
*        area   lgh  vol  hz ang  vr ang  elev-ch
8000101  10.0   1.0  0.0     0.0     0.0      0.0  * arbitrary size
*
*        rough   hy dia   flg
8000102    0.0      0.0     0
*
*        data ctl
8000200         3
*
*            time       pres      temp
8000201       0.0       12.3      80.0
*
**=====================================================================*
*      containment                                                     *
**====================================================================**
*
*               name      type
9000000      contain   tmdpvol
*
*         area     lgh    vol  hz ang   vr ang  elev ch
9000101 0.00976427 10.0   0.0    0.0      0.0      0.0  * size as c724
*
*         rough  hy dia   flg
9000102     0.0    0.0      0
*
*           data ctl
9000200            2
*
*            time   press    quale
9000201       0.0    12.3      1.0
*
**=====================================================================*
* pressurizer time dependent volume  (for steady state calculations)   *
**====================================================================**
*
******  input for steady state (comment out all 997, 998, & 999 cards)
*
*               name      type
9890000        c989    tmdpvol
*
*               area       lgh      vol     hz ang    vr ang   elv ch
9890101     7.4667e-4      1.00     0.0        0.0       0.0      0.0
*
*              rough    hy dia       flg
9890102          0.0,      0.0,        0
*
*                ctl
9890200            3 
*
*               time     pres       temp
9890201  0.0  67.0024160207  278.33  *69.618  278.33
*
**=====================================================================*
*  time dependent pressurizer junction                                 *
**====================================================================**
*
*              name      type
9910000    'tmd-prz'    valve
*
*              from        to   jn area    f loss   r loss   cahs
9910101    989000000 993000000  7.4667e-4     0.0      0.0  00100
*
*              ctl    flow-f    flow-g      velj
9910201          1      0.00       0.0       0.0
*
*             type
9910300     mtrvlv
*
*          op trip    cl trip    slope    init pos    table
9910301        508        509      .20       1.0          0
*
**=====================================================================*
* pressurizer surge line injection point stub                          *
**====================================================================**
*
*               name      type
9930000        pzstub    branch
*
*             no jun    in ctl
9930001            1         1
*
*               area    length       vol    hz ang    vr ang    elv ch
9930101     7.4667e-4  4.416667      0.0       0.0    -14.10 -1.074583
*
*              rough     hydia   ctl flg
9930102      3.333e-5     0.0          0
*
*         ctl  pressure    temp
9930200     3  69.618    278.33  *INITIAL CONDITIONS
*
*               from        to  jun area    f loss    r loss      cahs
9931101    993010000 202000000 7.4667e-4        0.0       0.0     0001
*
*              flow-f      flow-g  velj
9931201           0.0         0.0   0.0
*
**=====================================================================*
*  downcomer distribution annulus to core inlet (thru lower extension)*
**====================================================================**
*
*gl data          nh        np      type      s-flg     l-cor
11140000           1         3         2         1      .1796667
*
*mesh        loc-flg   frm-flg
11140100           0         1
*
*             no-itv     r-cor
11140101           2  0.188958
*
*materials    cmp-no    itv-no
11140201           1         2
*
*               s(x)    itv-no
11140301         0.0         2
*
*init temp       flg
11140400           0
*
*               temp   mesh-pt
11140401       546.6         3
*
*l-bndy       hy-vol   inc  b-cdt  a-code  cy-length  ht-str-no
11140501   140010000     0      1       1    0.96250          1
*
*r-bndy       hy-vol   inc  b-cdt  a-code  cy-length  ht-str-no
11140601   110100000     0      1       1    0.96250          1
*
*source         type    is-mplr    l-dr-ht   r-dr-ht  ht-str-no
11140701           0          0          0         0          1
*
*l-h-st      chf-htr    hy-diam  h-eq-diam   ch-lnth  ht-str-no
11140801           0.        10.        10.   0  0 0  0   1.0  1
*
*r-h-st      chf-htr    hy-diam  h-eq-diam   ch-lnth  ht-str-no
11140901           0.        10.        10.   0  0  0  0  1.0  1
*
**=====================================================================*
*    heated core  (active length)                                      *
**====================================================================**
*
*              note:  the mod-2a core is a 5 x 5 rod matrix with
*                     two (2) unpowered rods in opposing corners.
*                     each rod has an od = 0.422 in.   each power
*                     step (two per hydrodynamic volume) is
*                     modeled individually.
*
*gl data    nh     np   type   s-flg   l-cor
11500000    12     18      2       1     0.0
*
*mesh        loc flg   frm flg
11500100           0         1
*
*             no itv     r-cor     no itv     r-cor    no itv     r-cor
11500101           1  0.002917          4  0.009375         4  0.014500
11500102           4  0.015500          4  0.017583
*
*             cmp no    itv no     cmp no    itv no    cmp no    itv no
11500201           3         1          4         5         3         9
11500202           1        13          1        17
*
*               s(x)    itv no       s(x)    itv no      s(x)    itv no
11500301         0.0         1        1.0         5       0.0        17
*ini temp        flg
11500400          -1
*
*        temp distribution
11500401  692.4  692.4  689.8  683.2  673.1  659.6  638.3 619.6 602.9
+         587.8  585.7  583.7  581.7  579.7  575.7  571.7 567.9 564.2
11500402  822.4  822.4  817.5  804.9  785.6  760.1  717.9 681.2 648.7
+         619.6  615.7  611.9  608.1  604.3  596.7  589.3 582.1 575.2
11500403  979.3  979.3  971.9  952.9  923.8  885.2  818.5 760.9 710.4
+         665.5  659.7  654.1  648.5  642.9  631.6  620.7 610.0 599.5
11500404 1143.1 1143.1 1133.1 1107.0 1067.1 1014.2  917.8 835.8 764.6
+         701.8  694.0  686.3  678.8  671.3  656.0  641.1 626.6 612.4
11500405 1275.5 1275.5 1263.7 1232.9 1185.7 1123.4 1004.1 903.7 817.4
+         741.8  732.7  723.8  715.0  706.3  688.5  671.2 654.3 637.7
11500406 1333.6 1333.6 1320.8 1287.7 1236.9 1169.8 1038.8 929.3 835.4
+         753.5  743.8  734.3  724.9  715.5  696.5  677.9 659.7 641.9
11500407 1350.8 1350.8 1338.1 1304.9 1254.2 1187.1 1055.0 944.6 850.1
+         767.7  758.0  748.6  739.2  729.9  711.0  692.5 674.5 656.8
11500408 1293.2 1293.2 1281.4 1250.6 1203.4 1141.1 1020.7 919.6 832.6
+         756.6  747.6  738.7  730.0  721.3  703.7  686.5 669.6 653.2
11500409 1190.8 1190.8 1180.7 1154.6 1114.7 1061.9  963.3 879.5 806.9
+         743.0  735.4  727.9  720.4  713.1  698.1  683.5 660.2 655.3
11500410 1026.1 1026.1 1018.7  999.7  970.6  932.0  863.9 805.2 753.7
+         707.9  702.3  696.7  691.2  685.8  674.7  663.9 653.5 643.3
11500411  886.2  886.2  881.3  868.7  849.4  823.9  780.6 742.9 709.5
+         679.7  675.9  672.1  668.4  664.8  657.7  650.2 643.2 636.4
11500412  755.6  755.6  753.0  746.4  736.2  722.8  700.9 681.7 664.6
+         649.1  647.1  645.1  643.1  641.2  637.3  633.5 629.8 626.2
*
*l-bndy       hy-vol     inc    b-cdt   a-code  surf-area  ht-str-no
11500501           0       0        0        0          0         12
*
*r-bndy       hy vol     inc    b-cdt   a-code  surf-area  ht-str-no
11500601   150010000       0        1        1  23.0               1
11500602   150010000       0        1        1  23.0               2
11500603   150020000       0        1        1  23.0               3
11500604   150020000       0        1        1  23.0               4
11500605   150030000       0        1        1  23.0               5
11500606   150030000       0        1        1  23.0               6
11500607   150040000       0        1        1  23.0               7
11500608   150040000       0        1        1  23.0               8
11500609   150050000       0        1        1  23.0               9
11500610   150050000       0        1        1  23.0              10
11500611   150060000       0        1        1  23.0              11
11500612   150060000       0        1        1  23.0              12
*
*source         type    is mplr     l-dr-ht    r-dr-ht     ht-str-no
11500701         900    0.02584           0          0             1
11500702         900    0.04917           0          0             2
11500703         900    0.07416           0          0             3
11500704         900    0.10166           0          0             4
11500705         900    0.12001           0          0             5
11500706         900    0.12916           0          0             6
11500707         900    0.12916           0          0             7
11500708         900    0.12001           0          0             8
11500709         900    0.10166           0          0             9
11500710         900    0.07416           0          0            10
11500711         900    0.04917           0          0            11
11500712         900    0.02584           0          0            12
*
*l-h-st      chf-htr     hy-dia    h-eq-dia    ch-lnth     ht-str-no
11500801           0.      10.0        10.0       0  0  0  0   1.0  12
*
*r-h-st      chf-htr     hy-dia    h-eq-dia    ch-lnth     ht-str-no
11500901           0.      10.0        10.0       0  0   0  0  1.0  12
*
*######################################################################*
* heat structure input                                                 *
*######################################################################*
**=====================================================================*
*   downcomer inlet annulus hollow center (above cold leg nozzle)      *
**====================================================================**
*
*gl data          nh        np      type      s-flg     l-cor
12010000           1         6         2          1   0.12500
*
*mesh        loc flg   frm flg
12010100           0         1
*
*             no itv     r-cor     no itv     r-cor    no itv     r-cor
12010101           2   0.18750          1   0.191667        2   0.20000
*
*             cmp no    itv no     cmp no    itv no    cmp no    itv no
12010201           1         2          2         3         1         5
*
*             source    itv no
12010301         0.0         5
*
*ini temp        flg
12010400           0
*
*               temp   mesh pt
12010401       531.0         3     545.2    6
*
*l-bndy       hy-vol    inc   b-cdt   a-code  cy-length  ht-str-no
12010501          0      0       0        1        1.75       1 *tsa/dmk
*
*r-bndy       hy-vol    inc   b-cdt   a-code  cy-length  ht-str-no
12010601  101010000      0       1        1        1.75       1 *tsa/dmk
*
*source         type   is mplr    l-dr-ht  r-dr-ht   ht-str-no
12010701           0         0          0        0           1
*
*            chf-htr      hy-d    h-eq-di   ch-lngth ht-str-no
*left-ht-s
12010801           0.     10.0       10.0       0 0 0   0    1.0     1
*
*r-h-st
12010901           0.     10.0       10.0       0 0 0 0     1.0    1
*
**=====================================================================*
*  heat slab...steam generator tube bundle    (intact loop)            *
*             (rising primary coolant...first half of u-tube)          *
**====================================================================**
*
*   right boundary = steam generator secondary coolant (component c600)
*   left boundary  = steam generator primary coolant   (component c220)
*
*gl data    nh    np    type    ss-flg    left-cor
12200000     9     5       2         1    0.032375
*
*mesh       loc-flg    frm-flg
12200100          0          1
*
*           n0-intvl    right-cor
12200101           4    0.036458333
*
*           comp-no    intvl-no
12200201          5           4
*
*           source    intvl-no
12200301       0.0           4
*
*init temp    flg
12200400        0
*
*             temp    mesh-pt
12200401      549.2         5
*
*left-bc    hyd-vol    incr    b-cdt    a-code   cy-length   ht-str-no
12200501    220010000     0        1         1   23.9824998          1
12200502    220020000 10000        1         1   23.7325002          7
12200503    220080000     0        1         1    9.5625000          8
12200504    220090000     0        1         1    6.7738200          9
*
*
*right-bc   hyd-vol    incr    b-cdt    a-code   cy-length   ht-str-no
12200601    600010000     0        1         1   23.9824998          1
12200602    600020000 10000        1         1   23.7325000          7
12200603    600080000     0        1         1    9.5625000          8
12200604    600090000     0        1         1    6.7738200          9
*
*source      type    is-mplr    l-dr-ht    r-dr-ht    ht-str-no
12200701        0          0          0          0            9
*
*left-ht-str    chf-htr     dhy      dht    chan-l    ht-str-no
12200801           0.      10.0       10.0    0  0  0  0  1.0  9
*
*right-ht-str   chf-htr     dhy      dht    chan-l    ht-str-no
12200901           0.      10.0      10.0   0  0  0  0   1.0  9
*12200902           0        0.0  .2383660     29.5            8
*12200903           0        0.0  .3722457     29.5            9
*
**=====================================================================*
* heat slab...steam generator tube bundle     (intact loop)            *
*             (decending primary coolant...second half of u-tube)      *
**====================================================================**
*
*   right boundary = steam generator secondary coolant (component c600)
*   left boundary  = steam generator primary coolant   (component c220)
*
*gl data    nh    np    type    ss-flg    left-cor
12210000     9     5       2         1    0.032375
*
*mesh       loc-flg    frm-flg
12210100          0          1
*
*           n0-intvl    right-cor
12210101           4    0.036458333
*
*           comp-no    intvl-no
12210201          5           4   * inconel  600
*
*           source    intvl-no
12210301       0.0           4
*
*init temp    flg
12210400        0
*
*             temp    mesh-pt
12210401      536.3         5
*
*left-bc    hyd-vol    incr    b-cdt    a-code   cy-length   ht-str-no
12210501    220100000     0        1         1   6.77382000          1
12210502    220110000     0        1         1   9.56250000          2
12210503    220120000 10000        1         1   23.7325000          8
12210504    220180000     0        1         1   23.9824998          9
*
*right-bc   hyd-vol    incr    b-cdt    a-code   cy-length   ht-str-no
12210601    600090000     0        1         1   6.77382000          1
12210602    600080000     0        1         1   9.56250000          2
12210603    600070000     0        1         1   23.7325000          3
12210604    600060000     0        1         1   23.7325000          4
12210605    600050000     0        1         1   23.7325000          5
12210606    600040000     0        1         1   23.7325000          6
12210607    600030000     0        1         1   23.7325000          7
12210608    600020000     0        1         1   23.7325000          8
12210609    600010000     0        1         1   23.9824998          9
*
*source      type    is-mplr    l-dr-ht    r-dr-ht    ht-str-no
12210701        0          0          0          0            9
*
*left-ht-str    chf-htr     dhy     dht     chan-l    ht-str-no
12210801              0.      10.0    10.0     0  0  0  0  1.0  9
*
*right-ht-str   chf-htr     dhy     dht     chan-l    ht-str-no
12210901              0.     10.0   10.0     0  0  0  0   1.0  9
*
**=====================================================================*
*  heat slab... intact loop steam generator secondary heat loss       *
*                                 to the environment                  *
**====================================================================**
*
*    left boundary = volume 600
*   right boundary = volume 800
*
*gl data       nh      np     type     ss-flg    left-cor
16000000       10      13        2          1    .2453639
*
*mesh     loc-flg frm-flg
16000100        0       1
*
*             no-itv     rt-cor  no-itv     rt-cor  no-itv    rt-cor
16000101           2  .36833315       1 .370833146       2  .3823999
16000102           1  .3848999        1 .396000000       1  .3985000
16000103           2  .447916667      2 .697916667
*
*             cmp-no   no-itv   cmp-no   itv-no    cmp-no  itv-no
16000201          14        2       15        3         1       5
16000202          15        6        1        7        15       8
16000203           1       10       17       12
*
*source distrib     s(x)     itv-no
16000301             0.0         12
*
*init temp           flg
16000400               0
*                   temp    mesh pt
16000401            220.5        13
*
*left-bndy   hyd-vol     incr  b-cdtn  a-code  cy-length    ht-str-no
16000501     600010000      0       1       1  3.419726823          1
16000502     600020000  10000       1       1  3.384078704          7
16000503     600080000      0       1       1  1.363541667          8
16000504     600090000      0       1       1  1.648726852          9
16000505     600100000      0       1       1  1.175854194         10
*
*right-bndy  hyd-vol     incr  b-cdtn  a-code  cy-length    ht-str-no
16000601     800010000      0    4100       1  3.419726823          1
16000602     800010000      0    4100       1  3.384078704          7
16000603     800010000      0    4100       1  1.363541667          8
16000604     800010000      0    4100       1  1.648726852          9
16000605     800010000      0    4100       1  1.175854194         10
*
*source        type   is-mplr  l-dr-ht  r-dr-ht   ht-str-no
16000701          0         0        0        0          10
*
*left-ht-str   chf-htr    dhy      dht   chan-l   ht-str-no
16000801             0.    10.      10.       0  0 0  0   1.0     10
*
*right-ht-str  chf-htr    dhy      dht   chan-l   ht-str-no
16000901             0.    10.      10.       0  0  0  0  1.0        10
*12210902              0       0  .2383660     29.5            2
*12210903              0       0  .1629949     29.5            9
*
**=====================================================================*
*   heat slab... intact loop st. generator secondary heat transfer     *
*                                          to downcomer                *
**====================================================================**
*
*      right boundary = component 600
*       left boundary = component 603 and 604
*
*gl data     nh      np     type     ss-flg    left-cor
16003000     11       6        2          1    .2453639
*
*mesh           loc-flg  frm-flg
16003100              0        1
*
*            no-itv    rt-cor  no-itv      rt-cor  no-itv    rt-cor
16003101          2  .3683315       1  .370833146       2  .3823999
*
*            cmp-no    itv no  cmp-no      itv no  cmp-no    itv no
16003201         14         2      15           3       1         5
*
*source distrib   s(x)   itv no
16003301           0.0        5
*
*init temp         flg
16003400             0
*                 temp   mesh pt
16003401          507.4        6
*
*left-bndy    hyd-vol    incr  b-cdtn  a-code  cy-length    ht-str-no
16003501      600010000     0       1       1  0.172973170          1
16003502      600010000     0       1       1  0.404383311          2
16003503      600020000     0       1       1  0.571337963          3
16003504      600030000     0       1       1  0.571337963          4
16003505      600040000     0       1       1  0.571337963          5
16003506      600050000     0       1       1  0.571337963          6
16003507      600060000     0       1       1  0.571337963          7
16003508      600070000     0       1       1  0.571337963          8
16003509      600080000     0       1       1  0.230208333          9
16003510      600090000     0       1       1  0.278356481         10
16003511      600100000     0       1       1  0.198520838         11
*
*right-bndy   hyd-vol    incr  b-cdtn  a-code  cy-length    ht-str-no
16003601      604010000     0       1       1  0.172973170          1
16003602      603080000     0       1       1  0.404383311          2
16003603      603070000     0       1       1  0.571337963          3
16003604      603060000     0       1       1  0.571337963          4
16003605      603050000     0       1       1  0.571337963          5
16003606      603040000     0       1       1  0.571337963          6
16003607      603030000     0       1       1  0.571337963          7
16003608      603020000     0       1       1  0.571337963          8
16003609      603010000     0       1       1  0.230208333          9
16003610      603010000     0       1       1  0.278356481         10
16003611      602050000     0       1       1  0.198520838         11
*
*source            type  is-mplr  l-dr-ht  r-dr-ht   ht-str-no
16003701              0        0        0        0          11
*
*left-ht-str    chf-htr      dhy      dht   chan-l   ht-str-no
16003801              0.    10.0      10.0   0  0  0  0    1.0  11
*16003802              0      0.0    .234145932   0           9
*16003803              0      0.0    .27476691    0          11
*
*right-ht-str   chf-htr      dhy      dht   chan-l   ht-str-no
16003901              0.    10.0      10.0  0  0  0  0   1.0  11
*
**=====================================================================*
* heat slab.... steam generator steam dome heat loss to environment   *
*                  (includes upper section of downcomer)              *
**====================================================================**
*
*  left boundary = components 601,611,602,701,711,702
* right boundary = component  800
*
*gl data      nh     np    type    ss-flg    left-cor
16701000       7      5       2         1    .671833
*
*mesh      loc-flg   frm-flg
16701100         0         1
*           no-itv    rt-cor   no-itv   rt-cor
16701101         2    0.7500        2    1.000
*
*           cmp-no    no-itv   cmp-no   no-itv
16701201         1         2       17        4
*
*              s(x)   itv-no
16701301        0.0        4
*
*init temp     flg
16701400         0
*
*               temp    mesh
16701401        392.4      5
*
*left-boundary   hy-vol     incr   b-cdtn  a-code  length    ht-str-no
16701501         605010000     0        1       1  0.397083350       1
16701502         611010000     0        1       1  0.639185552       2
16701503         602010000     0        1       1  2.002083317       3
16701504         602020000     0        1       1  1.000000000       4
16701505         602030000     0        1       1  0.424583333       5
16701506         602040000     0        1       1  0.750666667       6
16701507         602050000     0        1       1  1.374333333       7
*
*right-boundary  hy-vol     incr   b-cdtn  a-code  length    ht-str-no
16701601         800010000     0     4100       1  0.397083350       1
16701602         800010000     0     4100       1  0.639185552       2
16701603         800010000     0     4100       1  2.002083317       3
16701604         800010000     0     4100       1  1.000000000       4
16701605         800010000     0     4100       1  0.424583333       5
16701606         800010000     0     4100       1  0.750666667       6
16701607         800010000     0     4100       1  1.374333333       7
*
*source            type  is-mplr   l-dr-ht   r-dr-ht         ht-str-no
16701701              0        0         0         0                 7
*
*left-ht-str    chf-htr     dhy    dht   chan-l    ht-str-no
16701801              0.     10.    10.  0 0 0 0   1.0     7
*
*right-ht-str   chf-htr     dhy    dht   chan-l    ht-str-no
16701901              0.     10.    10.    0 0  0    0   1.0     7
*
**=====================================================================*
*  heat slab... intact loop and broken loop st. generator downcomer   *
*               heat transfer to the environment  (same geometry)     *
**====================================================================**
*
*      left boundary = component 603,604,703,704
*     right boundary = component 800
*
*gl data       nh      np     type    ss-flg    left-cor
16703000        9       5        2         1    .3985000
*
*mesh     loc-flg frm-flg
16703100        0       1
*
*geometry    no-itv    rt-cor    no-itv      rt-cor
16703101          2  .447916667       2  .697916667
*
*            cmp-no    itv-no    cmp-no      itv-no
16703201          1         2        17           4
*
*source distrib    s(x)   itv no
16703301            0.0        4
*
*init temp          flg
16703400              0
*                  temp   mesh pt
16703401           389.9        5
*
*left-bndy   hyd-vol     incr  b-cdtn  a-code   length      ht-str-no
16703501     603010000      0       1       1   3.52083333          1
16703502     603020000      0       1       1   3.95541667          2
16703503     603030000      0       1       1   3.95541667          3
16703504     603040000      0       1       1   3.95541667          4
16703505     603050000      0       1       1   3.95541667          5
16703506     603060000      0       1       1   3.95541667          6
16703507     603070000      0       1       1   3.95541667          7
16703508     603080000      0       1       1   2.79957677          8
16703509     604010000      0       1       1   1.19750656          9
*
*right-bndy  hyd-vol     incr  b-cdtn  a-code   length      ht-str-no
16703601     800010000      0    4100       1   3.52083333          1
16703602     800010000      0    4100       1   3.95541667          2
16703603     800010000      0    4100       1   3.95541667          3
16703604     800010000      0    4100       1   3.95541667          4
16703605     800010000      0    4100       1   3.95541667          5
16703606     800010000      0    4100       1   3.95541667          6
16703607     800010000      0    4100       1   3.95541667          7
16703608     800010000      0    4100       1   2.79957677          8
16703609     800010000      0    4100       1   1.19750656          9
*
*source         type  is-mplr  l-dr-ht   r-dr-ht   ht-str-no
16703701           0        0        0         0           9
*
*left-ht-str    chf-htr   dhy      dht    chan-l   ht-str-no
16703801              0.   10.      10.        0  0  0  0  1.0         9
*
*right-ht-str   chf-htr   dhy      dht    chan-l   ht-str-no
16703901              0.   10.      10.        0  0  0  0  1.0         9
*
**=====================================================================*
* thermal property control cards                                       *
**====================================================================**
*
*               type    cn flg    ca flg
*316l stainless steel
20100100    tbl/fctn         1         1
*
*======================================================================*
*thermal conductivity    btu/(s ft f)
*======================================================================*
*
*               temp      cond      temp      cond      temp      cond
*316l stainless steel
20100101       32.00    .00215
20100102      100.00    .00215    800.00     .00306  1600.00    .00397
20100103     4000.00    .00397
*
*======================================================================*
*volumetric heat capacity data      (btu/cuft-f)
*======================================================================*
*
*               temp       cap       temp       cap     temp       cap
*316l stainless steel
20100151       32.00     61.30,    400.00     61.30,  600.00     64.60
20100152      800.00     67.10,   1000.00     69.35, 4000.00     69.35
*
*average two phase data
20100200    tbl/fctn         1         1
*
*averge two phase data
20100201       32.00   .000006
20100202      212.00   .000006    572.00    .000008  4000.00   .000008
*
*average two phase
20100251      32.00       1.00,    212.00      1.00,  572.00     64.00
20100252    4000.00      64.00
*
*boron nitride
20100300    tbl/fctn         1         1
*
* boron nitride (new curve from c.fineman..in 1/30/79 by dmk)
20100301       32.00    .00255
20100302      200.00    .00241    500.00    .00216   1000.00    .00174
20100303     1500.00    .00133   2000.00    .000909  2500.00    .000491
20100304     3000.00    .000074  3500.00    .000074  4000.00    .000074
*
*boron nitride
20100351       32.00     37.50,    400.00     37.50,  800.00     48.30
20100352     1200.00     54.60,   1600.00     58.30, 2000.00     60.00
20100353     2400.00     61.40,   3400.00     62.50, 4000.00     62.50
*
*constantan
20100400    tbl/fctn         1         1
*
*constantan
20100401        0.00    .00389   3000.00    .00389   4000.00    .00389
*
*constantan
20100451       32.00     56.00,     212.00     56.00   572.00    61.00
20100452      932.00     67.00,    1472.00     73.00  2192.00    78.00
20100453     2552.00     84.00,    3000.00     90.00  4000.00    90.00
*
*inconel 600
20100500    tbl/fctn         1         1
*
*inconel 600
20100501       32.00    .00236
20100502      100.00    .00236    300.00    .00267    500.00    .00294
20100503      700.00    .00322    900.00    .00350   1100.00    .00378
20100504     4000.00    .00378
*
*inconel 600
20100551       32.00    52.225     4000.00    52.225
*
*grafoil (grafoil is a trade name of a union carbide product)
20100600    tbl/fctn         1         1
*
*grafoil
20100601       32.00   .000799
20100602      250.00   .000799    500.00    .000683   750.00   .000579
20100603     1000.00   .000509   1250.00    .000484  1500.00   .000468
20100604     2000.00   .000463   3000.00    .000486  4000.00   .000486
*
*grafoil
20100651       32.00     11.90       80.40     11.90   170.40    14.70
20100652      260.40     17.15      350.40     19.60   440.40    21.35
20100653      530.40     22.40      620.40     23.80   710.40    25.20
20100654      800.40     26.32     4000.00     26.32
*
*copper ca 102 (oxyen freen copper)
20100700    tbl/fctn         1         1
*
*copper ca 102
20100701       32.00    .0622     212.00     .0606    572.00   .0589
20100702      932.00    .0575    4000.00     .0575
*
*copper ca 102
20100751       32.00    51.336,    4000.00    51.336
*
*insulation (modified values for system heat loss)
20100800    tbl/fctn         1         1
*
*insulation  (increased by a factor of 35. for vessel heat loss)
20100801        0.0    3.241e-4   100.0    3.241e-4   200.0    3.442e-4
20100802      300.0    3.850e-4   400.0    4.054e-4   500.0    4.453e-4
20100803      600.0    4.861e-4   700.0    5.329e-4   800.0    5.872e-4
*
*insulation
20100851        0.00     1.450,    1000.00     1.450
*
*alumina   (aluminun oxide)
20100900    tbl/fctn         1         1
*
*alumina   (aluminum oxide)
20100901        0.00  .00034722   3300.0   .00034722
*
*alumina    (aluminum oxide)
20100951        0.00     55.20     3300.00     55.20
*
*inconel 718
20101000    tbl/fctn         1         1
*
*inconel 718
20101001        0.00  .0021667     200.0   .0022778  400.0   .0023333
20101002      600.0   .0025556     800.0   .0028056 1000.0   .0030556
20101003     1200.0   .0033333    1400.0   .0035556 1600.0   .0038333
*
*inconel 718
20101051        0.00     48.5797    200.0      53.7597    400.    56.458
20101052      600.0      61.5912    800.0      65.1840   1000.0   69.290
20101053     1200.0      74.4227   1400.0      76.9890   1600.0   82.121
*
*honeycomb - hexagonal matrix core
20101100    tbl/fctn         1         1
*
*honeycomb - hexagonal matrix core
20101101        0.00  2.6111e-5    700.0   2.611e-5
*
*honeycomb - hexagonal matrix core
20101151        0.00     2.28969    200.0      2.53081    400.0   2.6512
20101152      600.0      2.89224    800.0      3.06095   1000.0   3.2537
20101153     1200.0      3.49379   1400.0      3.61530   1600.0   3.8563
*
*honeycomb - square matrix core
20101200    tbl/fctn         1         1
*
*honeycomb - square matrix core
20101201        0.00  2.2778e-5    700.0   2.2778e-5
*
*honeycomb - square matrix core
20101251        0.00     2.15978    200.0      2.38713    400.0   2.5008
20101252      600.0      2.71815    800.0      2.88729   1000.0   3.0691
20101253     1200.0      3.29651   1400.0      3.41018   1600.0   3.6375
*
*contact resistance properties (conductivity of air; storage = 0.0)
*20101300    tbl/fctn         1         1
*
*contact resistance (heaters to piping)     (conductivities of air)
*20101301        0.0    3.575e-6     80.0   4.211e-6   170.0    4.819e-6
*20101302      260.0    5.400e-6    350.0   5.950e-6   440.0    6.481e-6
*20101303      530.0    6.997e-6    620.0   7.478e-6   710.0    7.950e-6
*20101304      800.0    8.394e-6    890.0   8.842e-6   980.0    9.275e-6
*
*contact resistance  (heaters to piping)
*20101351        0.0        0.00    3300.0        0.0
*
*steam generator filler piece construction
20101400    tbl/fctn         1         1
*
*steam generator filler pieces (conductivities are those of air)
20101401       80.0   4.211e-6     260.0   5.400e-6    440.0  6.481e-6
20101402      620.0   7.478e-6     800.0   8.394e-6
*
*steam generator filler piece  (calculation by galan rogers)
20101451        0.0       17.3     1000.0       17.3
*
*water
20101500    tbl/fctn         1         1
*
*water
20101501       32.0   88.611e-6    104.0   100.833e-6  176.0 107.222e-6
20101502      248.0  110.000e-6    356.0   108.337e-6  464.0 101.944e-6
20101503      500.0   98.055e-6    572.0    86.666e-6
*
*water  (saturated)
20101551       32.0      63.033     104.0      61.966     176.0   60.950
20101552      248.0      59.895     356.0      58.553     464.0   57.845
20101553      500.0      57.989     572.0      60.999
*
*insulation  (modified values for system heat loss)
20101600    tbl/fctn         1         1
*
*insulation  (increased by a factor of 7. for piping heat loss)
20101601        0.0    6.481e-5   100.0    6.481e-5   200.0    6.883e-5
20101602      300.0    7.700e-5   400.0    8.108e-5   500.0    8.905e-5
20101603      600.0    9.722e-5   700.0   10.658e-5   800.0   11.744e-5
*
*insulation
20101651        0.0       1.450    1000.0       1.450
*
*insulation  (modified values for system heat loss)
20101700    tbl/fctn         1         1
*
*insulation  (increased by a factor of 2. for steam generator heat loss)
20101701        0.0    1.852e-5   100.0    1.852e-5   200.0    1.967e-5
20101702      300.0    2.200e-5   400.0    2.317e-5   500.0    2.544e-5
20101703      600.0    2.778e-5   700.0    3.045e-5   800.0    3.355e-5
*
*insulation
20101751        0.0       1.450    1000.0       1.450
*
**=====================================================================*
*
* specified heat transfer coefficients on piping insulation surface
*   (values calculated for free convecton in air + radiation)
*
20210000        htc-temp
*
*                   temp          h.t.coef   (btu/s-ft2-f)
20210001           100.0           0.00136
20210002           150.0           0.00206
20210003           200.0           0.00247
20210004           250.0           0.00277
20210005           300.0           0.00302
20210006           350.0           0.00323
20210007           400.0           0.00342
20210008           410.0           0.00346
20210009           420.0           0.00349
20210010           430.0           0.00352
20210011           440.0           0.00356
20210012           450.0           0.00359
20210013           460.0           0.00362
20210014           470.0           0.00365
20210015           480.0           0.00368
20210016           490.0           0.00371
20210017           500.0           0.00374
20210018           510.0           0.00378
20210019           520.0           0.00380
20210020           530.0           0.00383
20210021           540.0           0.00386
20210022           550.0           0.00389
20210023           560.0           0.00391
20210024           570.0           0.00394
20210025           580.0           0.00397
20210026           590.0           0.00399
20210027           600.0           0.00402
20210028           610.0           0.00405
20210029           620.0           0.00407
20210030           630.0           0.00410
20210031           640.0           0.00412
20210032           650.0           0.00414
20210033           660.0           0.00417
20210034           670.0           0.00419
20210035           680.0           0.00422
20210036           690.0           0.00424
20210037           700.0           0.00426
20210038           710.0           0.00429
20210039           720.0           0.00431
20210040           730.0           0.00433
20210041           740.0           0.00435
20210042           750.0           0.00438
*
**=====================================================================*
*                              tables                                  *
**====================================================================**
*
*                          core power
*
*              table    trip    time coef     power coef
20290000  power  503  1.0  0.0315033522444
*
*               time     power    (60 kw power)
20290001        0.0      1.000
20290002    10000.0      1.000
*
*======================================================================*
*                       control variables                              *
*======================================================================*
*
*
* Core power = summation of volume heat source (Comp-150)
20500100 corepow  sum   1.0  0.0     0     0
*                   a0         scale     name         param
20500101           0.0           1.0        q     150010000
20500102                         1.0        q     150020000
20500103                         1.0        q     150030000
20500104                         1.0        q     150040000
20500105                         1.0        q     150050000
20500106                         1.0        q     150060000
*
* core liquid level (voidf * dz)
20501000 corelvl   sum       30.48          396.0    1 *
20501001 0.0
20501002 1.635833     voidf     140010000 *
20501003 2.0          voidf     150010000 *
20501004 2.0          voidf     150020000 *
20501005 2.0          voidf     150030000 *
20501006 2.0          voidf     150040000 *
20501007 2.0          voidf     150050000 *
20501008 2.0          voidf     150060000 *
20501009 1.000833     voidf     161010000 *
*
* ilsg riser liquid level (voidf * dz)
20501100 sgrsrlvl  sum       30.48          1130.78  1 *
20501101 0.0
20501102 3.99708333   voidf     600010000 *
20501103 3.95541667   voidf     600020000 *
20501104 3.95541667   voidf     600030000 *
20501105 3.95541667   voidf     600040000 *
20501106 3.95541667   voidf     600050000 *
20501107 3.95541667   voidf     600060000 *
20501108 3.95541667   voidf     600070000 *
20501109 1.59375000   voidf     600080000 *
20501110 1.92708333   voidf     600090000 *
20501111 1.37437500   voidf     600100000 *
20501112 4.17729165   voidf     600110000 *
20501113 0.39708335   voidf     601010000 *
*
* vessel level
20501200 vsllvl    sum       30.48          568.7    1 *
20501201 0.0
20501202 2.463334     voidf     165010000 *
20501203 3.416666     voidf     163010000 *
20501204 2.260833     voidf     162010000 *
20501205 1.000833     voidf     161010000 *
20501206 2.0          voidf     150060000 *
20501207 2.0          voidf     150050000 *
20501208 2.0          voidf     150040000 *
20501209 2.0          voidf     150030000 *
20501210 2.0          voidf     150020000 *
20501211 2.0          voidf     150010000 *
20501212 1.635833     voidf     140010000 *
20501213 0.721917     voidf     130010000 *
20501214 0.466667     voidf     120010000 *
*
* primary system downcommer level
20501300 dncrlvl   sum       30.48          608.4    1 *
20501301 0.0
20501302 1.750000     voidf     101010000 *
20501303 2.385833     voidf     110010000 *
20501304 1.000833     voidf     110020000 *
20501305 2.0          voidf     110030000 *
20501306 2.0          voidf     110040000 *
20501307 2.0          voidf     110050000 *
20501308 2.0          voidf     110060000 *
20501309 2.0          voidf     110070000 *
20501310 1.465667     voidf     110080000 *
20501311 1.207667     voidf     110090000 *
20501312 0.962500     voidf     110100000 *
20501313 0.721917     voidf     130010000 *
20501314 0.466667     voidf     120010000 *
*
* primary pump suction downside level
20501400 ilpsdlvl  sum       30.48          372.0    1 *
20501401 0.0
20501402 1.18333      voidf     240010000 *
20501403 3.66625      voidf     240020000 *
20501404 2.78417      voidf     240030000 *
20501405 3.87500      voidf     240040000 *
20501406 3.87500      voidf     240050000 *
*
* primary pump suction upside level
20501500 ilpsulvl  sum       30.48          255.4    1 *
20501501 0.0
20501502 5.81250      voidf     240060000 *
20501503 1.92167      voidf     240070000 *
20501504 1.485837     voidf     240080000 *
*
* primary sg tube upside level
20501600 ilutulvl  sum       30.48          925.8    1 *
20501601 0.0
20501602 3.9970833    voidf     220010000 *
20501603 3.9554167    voidf     220020000 *
20501604 3.9554167    voidf     220030000 *
20501605 3.9554167    voidf     220040000 *
20501606 3.9554167    voidf     220050000 *
20501607 3.9554167    voidf     220060000 *
20501608 3.9554167    voidf     220070000 *
20501609 1.5937500    voidf     220080000 *
20501610 1.0516700    voidf     220090000 *
*
* primary sg tube downside level
20501700 ilutdlvl  sum       30.48          925.8    1 *
20501701 0.0
20501702 1.0516700    voidf     220100000 *
20501703 1.5937500    voidf     220110000 *
20501704 3.9554167    voidf     220120000 *
20501705 3.9554167    voidf     220130000 *
20501706 3.9554167    voidf     220140000 *
20501707 3.9554167    voidf     220150000 *
20501708 3.9554167    voidf     220160000 *
20501709 3.9554167    voidf     220170000 *
20501710 3.9970833    voidf     220180000 *
*
* ilsg downcommer liquid level
20501800 sgdnclvl  sum       30.48          1100.    1 *
20501801 0.0
20501802 1.19750656   voidf     604010000 *
20501803 2.79957677   voidf     603080000 *
20501804 3.95541667   voidf     603070000 *
20501805 3.95541667   voidf     603060000 *
20501806 3.95541667   voidf     603050000 *
20501807 3.95541667   voidf     603040000 *
20501808 3.95541667   voidf     603030000 *
20501809 3.95541667   voidf     603020000 *
20501810 3.52083333   voidf     603010000 *
20501811 1.37433333   voidf     602050000 *
20501812 0.75066667   voidf     602040000 *
20501813 0.42458333   voidf     602030000 *
20501814 1.00000000   voidf     602020000 *
20501815 2.002083317  voidf     602010000 *
20501816 0.39708335   voidf     605010000 *
*
* ilsg riser and separator mass
20510000 sgrsrms sum       0.028316847    0.0      1 *
20510001   0.
20510002 0.547369965  rho       600010000 *
20510003 0.541664035  rho       600020000 *
20510004 0.499500500  rho       600030000 *
20510005 0.499500500  rho       600040000 *
20510006 0.457207500  rho       600050000 *
20510007 0.457207500  rho       600060000 *
20510008 0.537511300  rho       600070000 *
20510009 0.216578607  rho       600080000 *
20510010 0.261876093  rho       600090000 *
20510011 0.434639013  rho       600100000 *
20510012 2.036472246  rho       600110000 *
20510013 0.397083350  rho       601010000 *
*
* ilsg downcommer mass
20510100 sgdncms sum       0.028316847    0.0      1 *
20510101 0.0
20510102 0.3127800480 rho       605010000 *
20510103 1.577028389  rho       602010000 *
20510104 0.504804787  rho       602020000 *
20510105 0.141893525  rho       602030000 *
20510106 0.335152872  rho       602040000 *
20510107 0.452285538  rho       602050000 *
20510108 0.103092     rho       603010000 *
20510109 0.115817     rho       603020000 *
20510110 0.115817     rho       603030000 *
20510111 0.115817     rho       603040000 *
20510112 0.115817     rho       603050000 *
20510113 0.115817     rho       603060000 *
20510114 0.115817     rho       603070000 *
20510115 0.081973     rho       603080000 *
20510116 0.035064     rho       604010000 *
*
* ilsg steam dome mass
20510200 sgdomass  sum       0.028316847    1.0      1 *
20510201 0.0
20510202 0.753346088  rho       611010000 *
20510203 0.137320027  rho       612010000 *
20510204 0.112946     rho       613010000 *
*
* primary system downcommer mass
20510300 pridcms sum       0.028316847    0.0      1 *
20510301 0.0
20510302 0.184969750  rho       101010000 *
20510303 0.073901177  rho       110010000 *
20510304 0.026060690  rho       110020000 *
20510305 0.052078000  rho       110030000 *
20510306 0.052078000  rho       110040000 *
20510307 0.052078000  rho       110050000 *
20510308 0.052078000  rho       110060000 *
20510309 0.052078000  rho       110070000 *
20510310 0.038164503  rho       110080000 *
20510311 0.045197526  rho       110090000 *
20510312 0.073352510  rho       110100000 *
*
* primary system vessel mass
20510400 pvslmass  sum       0.028316847    0.0      1 *
20510401 0.0
20510402 0.238130370  rho       120010000 *
20510403 0.232802350  rho       130010000 *
20510404 0.122996647  rho       140010000 *
20510405 0.061498     rho       150010000 *
20510406 0.061498     rho       150020000 *
20510407 0.061498     rho       150030000 *
20510408 0.061498     rho       150040000 *
20510409 0.061498     rho       150050000 *
20510410 0.061498     rho       150060000 *
20510411 0.057907197  rho       161010000 *
20510412 0.097073387  rho       162010000 *
20510413 0.170992517  rho       163010000 *
20510414 0.163949658  rho       165010000 *
20510415 0.013094530  rho       182010000 *
20510416 0.012890900  rho       184010000 *
20510417 0.028753274  rho       185010000 *
*
* primary system hot leg mass
20510500 hlegmass  sum       0.028316847    0.0      1 *
20510501 0.0
20510502 0.032609779  rho       201010000 *
20510503 0.121017765  rho       201020000 *
20510504 0.042103276  rho       201030000 *
20510505 0.041031914  rho       202010000 *
20510506 0.039907108  rho       203010000 *
20510507 0.096186589  rho       203020000 *
20510508 0.052367411  rho       203030000 *
20510509 0.0551215    rho       210010000 *
*
* primary hot side sg tube mass
20510600 phsgmass  sum       0.028316847    0.0      1 *
20510601 0.0
20510602 0.034574750  rho       215010000 *
20510603 0.078970311  rho       220010000 *
20510604 0.078147104  rho       220020000 *
20510605 0.078147104  rho       220030000 *
20510606 0.078147104  rho       220040000 *
20510607 0.078147104  rho       220050000 *
20510608 0.078147104  rho       220060000 *
20510609 0.078147104  rho       220070000 *
20510610 0.031487693  rho       220080000 *
20510611 0.022305042  rho       220090000 *
*
* primary cold side sg tube mass
20510700 pcsgmass  sum       0.028316847    0.0      1 *
20510701 0.0
20510702 0.022305042  rho       220100000 *
20510703 0.031487693  rho       220110000 *
20510704 0.078147104  rho       220120000 *
20510705 0.078147104  rho       220130000 *
20510706 0.078147104  rho       220140000 *
20510707 0.078147104  rho       220150000 *
20510708 0.078147104  rho       220160000 *
20510709 0.078147104  rho       220170000 *
20510710 0.078970311  rho       220180000 *
20510711 0.034574750  rho       225010000 *
*
* primary system pump suction downside mass
20510800 psucdms sum       0.028316847    0.0      1 *
20510801 0.0
20510802 0.0551215    rho       230010000 *
20510803 0.035947850  rho       240010000 *
20510804 0.090295705  rho       240020000 *
20510805 0.068571045  rho       240030000 *
20510806 0.145521750  rho       240040000 *
20510807 0.153561310  rho       240050000 *
*
* primary system pump suction upside mass
20510900 psucums sum       0.028316847    0.0      1 *
20510901 0.0
20510902 0.226322185  rho       240060000 *
20510903 0.072166395  rho       240070000 *
20510904 0.062426765  rho       240080000 *
20510905 0.032754119  rho       240090000 *
*
* primary system cold leg mass
20511000 clegmass  sum       0.028316847    0.0      1 *
20511001 0.0
20511002 0.043913785  rho       261010000 *
20511003 0.095923431  rho       261020000 *
20511004 0.063531228  rho       262010000 *
20511005 0.061337136  rho       263010000 *
20511006 0.026954889  rho       263020000 *
*
* primary system broken loop cold and hot leg nozzle mass
20511100 bnozmass  sum       0.028316847    0.0      1 *
20511101 0.0
20511102 0.050290932  rho       301010000 *
20511103 0.047925048  rho       363010000 *
*
* core heat transfer rate
*      scale fac = 2 * pi * 0.017583 * (0.3048 ** 2)
20520000 corehtin  sum       0.010263672    0.0      1 *
20520001 0.0
20520002 23.0         htrnr     150000101 *
20520003 23.0         htrnr     150000201 *
20520004 23.0         htrnr     150000301 *
20520005 23.0         htrnr     150000401 *
20520006 23.0         htrnr     150000501 *
20520007 23.0         htrnr     150000601 *
20520008 23.0         htrnr     150000701 *
20520009 23.0         htrnr     150000801 *
20520010 23.0         htrnr     150000901 *
20520011 23.0         htrnr     150001001 *
20520012 23.0         htrnr     150001101 *
20520013 23.0         htrnr     150001201 *
*
* primary system heat transfer rate to sg tubes
*      scale fac = 2 * pi * 0.032375 * (0.3048 ** 2)
20520100 prisght sum       0.018898162    0.0      1 *
20520101 0.0
20520102 23.9824998   htrnr     220000100 *
20520103 23.7325002   htrnr     220000200 *
20520104 23.7325002   htrnr     220000300 *
20520105 23.7325002   htrnr     220000400 *
20520106 23.7325002   htrnr     220000500 *
20520107 23.7325002   htrnr     220000600 *
20520108 23.7325002   htrnr     220000700 *
20520109  9.5625000   htrnr     220000800 *
20520110  6.7738200   htrnr     220000900 *
20520111  6.7738200   htrnr     221000100 *
20520112  9.5625000   htrnr     221000200 *
20520113 23.7325002   htrnr     221000300 *
20520114 23.7325002   htrnr     221000400 *
20520115 23.7325002   htrnr     221000500 *
20520116 23.7325002   htrnr     221000600 *
20520117 23.7325002   htrnr     221000700 *
20520118 23.7325002   htrnr     221000800 *
20520119 23.9824998   htrnr     221000900 *
*
* secondary system heat transfer rate from sg tubes
*      scale fac = 2 * pi * 0.036458333 * (0.3048 ** 2)
20520200 sechtin   sum       0.021281714    0.0      1 *
20520201 0.0
20520202 23.9824998   htrnr     220000101 *
20520203 23.7325002   htrnr     220000201 *
20520204 23.7325002   htrnr     220000301 *
20520205 23.7325002   htrnr     220000401 *
20520206 23.7325002   htrnr     220000501 *
20520207 23.7325002   htrnr     220000601 *
20520208 23.7325002   htrnr     220000701 *
20520209  9.5625000   htrnr     220000801 *
20520210  6.7738200   htrnr     220000901 *
20520211  6.7738200   htrnr     221000101 *
20520212  9.5625000   htrnr     221000201 *
20520213 23.7325002   htrnr     221000301 *
20520214 23.7325002   htrnr     221000401 *
20520215 23.7325002   htrnr     221000501 *
20520216 23.7325002   htrnr     221000601 *
20520217 23.7325002   htrnr     221000701 *
20520218 23.7325002   htrnr     221000801 *
20520219 23.9824998   htrnr     221000901 *
*
* dp for sg riser level (dp method)
20580000 isgdpl    sum       1.0            1.0      1 *
20580001  0.0
20580002 +1.0         p         600010000 *
20580003 -1.0         p         600110000 *
20580004 +5.973765764 rho       600010000 *
20580005 +6.243092721 rho       600110000 *
*
* ilsg riser liquid level from dp
20580100 isgdplvl  div       10.1971775     1.0      1 *
20580101 rhof         600110000 *
20580102 cntrlvar     800       *
*
* ilsg fraction of ht area (voidf * dz)
20580200 isgfrhtz  sum       1.0216495e-3   1.0      1 *
20580201 0.0
20580202 1.0          cntrlvar  011       *
*
* ilsg fraction of ht area (dp level)
20580300 isgfrhtp  sum       1.0216495e-3   1.0      1 *
20580301 0.0
20580302 1.0          cntrlvar  801       *
*
* ilsg feed flow = discharge flow
20580400 feedflow  sum       1.0            0.848036 1 *
20580401 0.0
20580402 1.0          mflowj    635000000 *
*
* ilsg secondary mass
20580500 ilsgmass  sum       1.0            0.0      1 *
20580501 0.0
20580502 1.0          cntrlvar  100       *
20580503 1.0          cntrlvar  101       *
20580504 1.0          cntrlvar  102       *
*
* ilsg feed flow = discharge flow, controller for jun 610
20580600 feedctl   sum       2.204622476    0.848036 1 *
20580601 0.0
20580602 1.0          cntrlvar  804       *
*
* ilsg discharge - feed flow
20590000 sgoutin   sum       1.0            0.0      1 *
20590001  0.0
20590002  1.0         mflowj    635000000 *
20590003 -1.0         cntrlvar  804       *
*
* integral ilsg feed flow
20590100 sgfdflo   integral  1.0            0.848036 0 *
20590101 cntrlvar     804       *
*
* integral ilsg discharge flow
20590200 sgstmflo  integral  1.0            0.848036 0 *
20590201 mflowj       635000000 *
*
* integral ilsg relief vlv flow
20590300 sgrlvflo  integral  1.0            0.0      0 *
20590301 mflowj       634000000 *
*
* sgmass = initial - integral (outflow - inflow)
20591000 sgmass    sum       1.0            199.120  0 *
20591001  199.120
20591002  1.          cntrlvar  901       * feed flow
20591003 -1.          cntrlvar  902       * discharge flow
20591004 -1.          cntrlvar  903       * relief vlv flow
20591005  1.          mflowj    690000000 * incr sg inv
20591006 -1.          mflowj    691000000 * decr sg inv
*
* sg mass error
20591100 sgmserr   sum       1.0            0.0      1 *
20591101  0.0
20591102  1.          cntrlvar  805       *
20591103 -1.          cntrlvar  910       *
*
* primary system mass
20592000 primass   sum       1.0            0.0      1 *
20592001 0.0
20592002 1.           cntrlvar  103       *
20592003 1.           cntrlvar  104       *
20592004 1.           cntrlvar  105       *
20592005 1.           cntrlvar  106       *
20592006 1.           cntrlvar  107       *
20592007 1.           cntrlvar  108       *
20592008 1.           cntrlvar  109       *
20592009 1.           cntrlvar  110       *
20592010 1.           cntrlvar  111       *
*
* fraction of primary system total inventory (100% mass = 112.76 kg)
20592100 frprinv   sum    0.01        1.0      1 * pri inv (mass) (1)
*20592100 frprinv   sum    0.0096      1.0      1 * pri inv (mass) (2)
*20592100 frprinv   sum    0.0094      1.0      1 * pri inv (mass) (3)
*20592100 frprinv   sum    0.0088      1.0      1 * pri inv (mass) (4)
*20592100 frprinv   sum    0.0082      1.0      1 * pri inv (mass) (5)
*20592100 frprinv   sum    0.0080      1.0      1 * pri inv (mass) (6)
*20592100 frprinv   sum    0.0074      1.0      1 * pri inv (mass) (7)
*20592100 frprinv   sum    0.0062      1.0      1 * pri inv (mass) (8)
20592101 0.0
20592102 1.0          cntrlvar  920       *
*
* fraction of secondary system inventory (100% mass = 205.034 kg)
20592200 frsginv   sum       0.004877240    1.0      1 * sg inv (mass)
20592201 0.0
20592202 1.0          cntrlvar  805       *
*
* cumulative total mass error
20592300 cuemass   integral  1.0            0.0      0 *
20592301 emass        0         *
*
* primary system downcomer subcooling
20593000 subcool   sum       1.0            1.0      1 *
20593001  0.0
20593002  1.0         tempf     110070000 *
20593003 -1.0         sattemp   110070000 *
*
* core delta-temp (outlet - inlet)
20593100 coredta   sum       1.0            1.0      1 *
20593101  0.0
20593102  1.0         tempf     163010000 *
20593103 -1.0         tempf     140010000 *
*
* core delta-temp (outlet - lower plenum)
20593200 coredtb   sum       1.0            1.0      1 *
20593201  0.0
20593202  1.0         tempf     163010000 *
20593203 -1.0         tempf     130010000 *
*
* primary system sg delta-temp
20593300 sgdtemp   sum       1.0            1.0      1 *
20593301  0.0
20593302  1.0         tempf     240010000 *
20593303 -1.0         tempf     203030000 *
*
* net heat stored in sg tubes
20594000 netubeht  sum       1.0            0.0      1 *
20594001 0.0
20594002 1.0          cntrlvar  201       *
20594003 1.0          cntrlvar  202       *
*
* net heat stored in primary system
20594100 netpriht  sum       1.0            0.0      1 *
20594101 0.0
20594102 1.0          cntrlvar  200       *
20594103 1.0          cntrlvar  201       *
*
. end
