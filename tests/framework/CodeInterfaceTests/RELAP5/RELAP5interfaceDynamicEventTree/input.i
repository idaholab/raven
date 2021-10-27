=Typical pwr model
*            type         state
100       restart       transnt
*             run
101           run
*       restrtnum
103            -1
*     tend minstep maxstep copt pfreq majed rsrtf
201   25.0  1.0e-7    0.05    7     2  1000  1000
*        variable     parameter
301      cntrlvar           802
414 time     0 ge null     0  0.0 l
454 time     0 ge null     0  0.0 l
.
