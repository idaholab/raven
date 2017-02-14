import sys
import math
import scipy.interpolate

# inlet bc data points
ibp = [
  # Mass Flow Rate = 0 kg/s/m^2; T = 298.15, 619 K; p = 15.5 MPa
  {
    'rho_liquid':  1003.8758764456           ,
    'u_liquid':    0                         ,
    'u_vapor':     0
  },
  # Mass Flow Rate = 3729.9625 kg/s/m^2; T = 559.1355 (old=565.05) (oldold=553.15), 619 K; p = 15.5 MPa
  {
    'rho_liquid':  753.63330427388           ,
    'u_liquid':    4.0                       ,     #3729.9625/764.28371247299 ,
    'u_vapor':     4.0                             #3729.9625/764.28371247299
  }
  # Mass Flow Rate = 3729.9625 kg/s/m^2; T = 529.11, 619 K; p = 15.6 MPa
#  {
#    'rho_liquid':  802.95015659836           ,
#    'u_liquid':    3729.9625/802.95015659836 ,
#    'u_vapor':     3729.9625/802.95015659836
#  },
  # Mass Flow Rate = 3729.9625 kg/s/m^2; T = 535, 619 K; p = 15.6 MPa
#  {
#    'rho_liquid':  794.05365441137           ,
#    'u_liquid':    3729.9625/794.05365441137 ,
#    'u_vapor':     3729.9625/794.05365441137
#  }
]

#inlet_bc_data = [
#  [ 0,                     100,                   101,                   1300                 ],
#  [ ibp[0]['rho_liquid'],  ibp[1]['rho_liquid'],  ibp[2]['rho_liquid'],  ibp[1]['rho_liquid'] ],
#  [ ibp[0]['u_liquid'],    ibp[1]['u_liquid'],    ibp[2]['u_liquid'],    ibp[1]['u_liquid']   ],
#  [ ibp[0]['u_vapor'],     ibp[1]['u_vapor'],     ibp[2]['u_vapor'],     ibp[1]['u_vapor']    ]
#]

inlet_bc_data = [
  [ 0,                     100,                   501                  ],
  [ ibp[0]['rho_liquid'],  ibp[1]['rho_liquid'],  ibp[1]['rho_liquid'] ],
  [ ibp[0]['u_liquid'],    ibp[1]['u_liquid'],    ibp[1]['u_liquid']   ],
  [ ibp[0]['u_vapor'],     ibp[1]['u_vapor'],     ibp[1]['u_vapor']    ]
]

inlet_rho_liquid_fn  = scipy.interpolate.interp1d(inlet_bc_data[0], inlet_bc_data[1])
inlet_u_liquid_fn    = scipy.interpolate.interp1d(inlet_bc_data[0], inlet_bc_data[2])
inlet_u_vapor_fn     = scipy.interpolate.interp1d(inlet_bc_data[0], inlet_bc_data[3])

def initial_function(monitored, controlled, auxiliary):
  control_function(monitored, controlled, auxiliary)
  return

def control_function(monitored, controlled, auxiliary):
  # inlet BCs
  controlled.inlet_rho_liquid  = inlet_rho_liquid_fn(monitored.time)
  controlled.inlet_u_liquid    = inlet_u_liquid_fn(monitored.time)
  controlled.inlet_u_vapor     = inlet_u_vapor_fn(monitored.time)
  return
