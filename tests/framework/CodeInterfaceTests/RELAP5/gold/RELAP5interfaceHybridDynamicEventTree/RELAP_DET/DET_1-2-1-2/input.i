*RAVEN INPUT VALUES
* card: 100 word: 1 value: restart
* card: 103 word: 1 value: -1
* card: 201 word: 2 value: 1.1000000e-07
* card: 201 word: 1 value: 2.9151187e+01
* card: 414 word: 6 value: 1.6000000e+01
* card: 454 word: 6 value: 1.3520000e+01
* card: 464 word: 6 value: 2.9151187e+01
*RAVEN INPUT VALUES
=Typical pwr model
*            type         state
100  restart  transnt
*             run
101           run
*       restrtnum
103  -1
*     tend minstep maxstep copt pfreq majed rsrtf
201  2.9151187e+01  1.1000000e-07  0.05  7  2  1000  1000
*        variable     parameter
301      cntrlvar           802
414  time  0  ge  null  0  1.6000000e+01  l
454  time  0  ge  null  0  1.3520000e+01  l
464  time  0  ge  null  0  2.9151187e+01  l
* START -- CONTROL VARIABLES ADDED BY RAVEN *
599 time 0 le null 0 -1.0 l
600 599
20599700 r_414 tripunit 1.0 0.0 0
20599701 414
20599600 r_454 tripunit 1.0 0.0 0
20599601 454
20599800 tripstop constant 0.0
* END -- CONTROL VARIABLES ADDED BY RAVEN *
* START -- MINOR EDITS TRIPS ADDED BY RAVEN *
398 timeof 414
397 timeof 454
396 timeof 464
395 cntrlvar 997
394 cntrlvar 996
* END --  MINOR EDITS TRIPS ADDED BY RAVEN *
.
