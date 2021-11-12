*RAVEN INPUT VALUES
* card: 2500201 word: 3 value: 5.3024524e+02
*RAVEN INPUT VALUES
= REMOVED ANY NUCLEAR DATA BY ALFOA
*
*********************************************************
*********************************************************
100  new  stdy-st
101  run
102  si  si
107  0  1  1
110  air
115  1.0
120  140110000  0.3965  he  Primary
121  900010000  5.0  n2  Contain
201  0.0001  1.0-6  0.1  19  1  20  50000
301  rkoegv 0
*
*******************************************************
* trips
*******************************************************
*
20600000  expanded
*
* scram trip
20601000  time  0  ge  null  0  1.00e+06  l
*
*******************************************************
*******************************************************
*******************************************************
*                  Hydrodynamics                      *
*******************************************************
*******************************************************
*
* Inlet volume
* ***************************************
* REMOVED ALL THE REMAINING COMPONENTS  *
* REMOVED by alfoa                      *
*****************************************
*
2500000  inlet  tmdpvol
2500101  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0000000
2500200  003
2500201  0.0  6.39e+06  5.3024524e+02
*

*******************************************************
*  kinetics
* REMOVED ANY NUCLEAR DATA.  REMOVED by alfoa
*******************************************************
30000000  instant  phis_mi
* Change here to MRTAU if decay heat must be calculated by PHISICS, e.g. when CV's are not used for decay heat,
* or to gamma-ac when a built-in RELAP decay heat routine is desired. Also change power then to 350 MW.
30000001  no-gamma  1000  0.0065  6
* card 16 here modified for benchmark output option 38
30000003  1  1  0  26  1  3  1  1  1  1   10   50   0   0   0   1
30000004  0.0  0.0  0.0  0.0  0.0
30000005 0.0 0.0 0.0 0.0 0.0
+        0.0 0.0 0.0 0.0 0.0
+        0.0 0.0 0.0 0.0 0.0
+        0.0 0.0 0.0 0.0 0.0
+        0.0 0.0 0.0 0.0 0.0
+        0.0
*  fission spectrum:  all prompt are fast
30000006  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
+         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
*  Delayed neutron constants for 6 groups: REMOVED by alfoa
30000101  0.0  0.0
30000102  0.0  0.0
30000103  0.0  0.0
30000104  0.0  0.0
30000105  0.0  0.0
30000106  0.0  0.0
*  Z Mesh:  - start at bottom upwards
*  1 axial levels - 1 fuel
30010101  0.793  1
*  Hexagon mesh: accross face (pitch of block = 36 cm)
30010201  0.36  1
*  Assign zone figures to kinetics axial mesh
30010401   1  1
*  Assign composition figures to kinetics axial mesh
30010501   1  1
*  Assign zones to zone figures (A15.18.14)
*  level 1,2: fuel
*
30020101  10  10  10  10
*
*
*  Assign compositions to composition figures (A15.18.15)
*  Benchmark layout XS data linked here to PHISICS (12 layers)
*
30030101  1   1   1   1
*
*  No intial power distribution specified
*
*  Neutron cross section data: "GEN" option (A15.18.27)
*  Volume and heat structure: number of feedback regions
310000000  1  1
*
*  Volume and heat structure feedback weighting factors: no shape assigned - all values 1.0
*  31ZZZZ1N1; assign volumes to vol feedback region N of zone ZZZZ
*
*  Zone map 1 and 3: BR and TR. Zone 0010: assigned to axial layer 7 of pipe vol 140.
310010111  140070000  1.0  1.0  1.0
*
*  31ZZZZ2N1; assign heat structures to HS feedback region N of zone ZZZZ
*  Zone map 1 and 3: BR and TR. Zone 0010: assigned to layer 2 of HS 1401.
310010211  1401005  1.0
*
* Control rod model
33000001 0.0 0.0 0.0
33010001 1
*
*
*

.
