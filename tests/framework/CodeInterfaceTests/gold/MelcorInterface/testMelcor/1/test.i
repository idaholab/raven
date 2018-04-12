*m: SNAP:Symbolic Nuclear Analysis Package,  Version 2.4.0, July 09, 2015
*m: PLUGIN:MELCOR Version 2.3.0
*m: CODE:MELCOR Version 2.1
*m: DATE:12/16/16
!
!MEG_DIAGFILE 'MEGDIA.TXT'
!MEG_OUTPUTFILE 'MEGOUT.TXT'
MEL_OUTPUTFILE 'OUTPUT_MELCOR.ou     '
!MEG_RESTARTFILE 'MELRST.RST'
!PLOTFILE 'MELPTF.PTF'
!MESSAGEFILE 'MELMES.TXT'
!EXTDIAGFILE 'EXTDIAG.TXT'
!
!*************************************************
!'     es     _1'
!*************************************************
!
!
PROGRAM MELGEN
!
EXEC_INPUT
!
EXEC_TITLE '     es     _1'
!
EXEC_DTTIME 1.0E-4
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!       Tabular Func     ions Da     a         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TF_INPUT
!
!
!cc: 1
!                      fname             fscal
TF_ID             'P_in'           1.0
!      size
TF_TAB    2 !n             x             y
             1           0.0         1.0E7
             2         100.0         9999999.4846
!
!
!cc: 2
!                      fname             fscal
TF_ID             'T_in'           1.0
!      size
TF_TAB    2 !n             x             y
             1           0.0         400.0
             2         100.0         400.0
!cc: 3
!                      fname             fscal
TF_ID             'M_in'           1.0
!      size
TF_TAB    2 !n             x             y
             1           0.0         1.
             2         100.0         1.
!
CF_INPUT   !
CF_ID 'ZERO'       01  EQUALS   !
CF_SAI   1.0   0.0    0.0                                 !  CFSCAL CFADCN CFVALR (INITIAL VALUE)
CF_ARG    1 ! NARG   CHARG        ARSCAL   ARADCN
              1      EXEC-TIME    0.0      0.0
!
CF_INPUT   !
CF_ID 'Tes      CF'       02  EQUALS   !
CF_SAI   1.0   0.0    0.0                                 !  CFSCAL CFADCN CFVALR (INITIAL VALUE)
CF_ARG    1 ! NARG   CHARG        ARSCAL   ARADCN
              1      EXEC-TIME    2.0      1.0
!
!
CVH_INPUT
!
!
!                   cvname        icvnum
CV_ID                'IN'             1
!             icv     hr         ipfsw         icvac     
CV_THR      NONEQUIL           FOG PROP-SPECIFIED
!              i     yp     h         ipora         wa     er
CV_PAS       SEPARATE      ONLYPOOL     SUBCOOLED
!             keyword          flag          pvol
CV_PTD           PVOL            TF        'P_in'
!      size
CV_VAT    2 !n           cvz         cvvol
             1          10.0           0.0
             2          11.0          30.0
!
!
!                   cvname        icvnum
CV_ID             'CENTR'             2
!             icv     hr         ipfsw         icvac     
CV_THR      NONEQUIL           FOG         ACTIVE
!              i     yp     h         ipora         wa     er
CV_PAS       SEPARATE      ONLYPOOL     SUBCOOLED
!               p     di               pvol
CV_PTD           PVOL         1.0E7
!                    pol
CV_PAD         400.0
!      size
CV_VAT    2 !n           cvz         cvvol
             1          10.0           0.0
             2          11.0          30.0
!
!
!                   cvname        icvnum
CV_ID               'OUT'             3
!             icv     hr         ipfsw         icvac     
CV_THR      NONEQUIL           FOG     TIME-INDEP
!              i     yp     h         ipora         wa     er
CV_PAS       SEPARATE      ONLYPOOL     SUBCOOLED
!               p     di               pvol
CV_PTD           PVOL         1.0E7
!                    pol
CV_PAD         400.0
!      size
CV_VAT    2 !n           cvz         cvvol
             1          10.0           0.0
             2          11.0          30.0
!
ESF_INPUT
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!       Flow Pa     hs         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FL_INPUT
!
!
!              fpname        ifpnum
FL_ID         'FP1_2'            11
!               kcvfm         kcv     o           zfm           z     o
FL_FT            'IN'       'CENTR'          10.5          10.5
!               flara         fllen         flopo
FL_GEO            1.0          10.0           1.0
!      size
FL_SEG    1 !n sarea slen shyd srgh lamflg slam
             1   1.0 10.0  1.0
FL_VTM  1
                    1   'FP1_2' 1 'M_in'
!
!
!              fpname        ifpnum
FL_ID         'FP2_3'            12
!               kcvfm         kcv     o           zfm           z     o
FL_FT         'CENTR'         'OUT'          10.5          10.5
!               flara         fllen         flopo
FL_GEO            1.0          10.0           1.0
!      size
FL_SEG    1 !n sarea slen shyd srgh lamflg slam
             1   1.0 10.0  1.0
!
END PROGRAM MELGEN
Program MELCOR
!
!* Block: MEX (Exec) da     a ****************************
EXEC_INPUT
EXEC_TITLE      es     _1        ! Ti     le of      he calcula     ion
EXEC_TEND 10.0E+03                                                   !*  ! End of calcula     ion      ime
EXEC_TIME 2 !*NUMBER   TIME            DTMAX        DTMIN        DTEDT        DTPLT        DTRST          DCRST
               1       0.00            0.1000E+00    0.10000E-06    2.50000E+03    1.00000E+01    1.00000E+03  0.10000000E+11
               2       1.50000E+02    0.2000E+00    0.10000E-09    2.50000E+03    1.00000E+01    1.0000E+03     0.10000000E+11

EXEC_CPULEFT 1000.                                                    ! cpu sec lef      a      end of calcula     ion
EXEC_CPULIM 4000000.                                                ! Maximum number of CPU seconds allowed for      his execu     ion
EXEC_NOFLUSH                                                        ! Suppress Explici      Buffer Flushing
EXEC_CYMESF		100	1000	1	1
!* END MEX (Exec) ******************************
!
!*CVH_INPUT
!*CVH_TRACE 3

END Program MELCOR da     a
!
