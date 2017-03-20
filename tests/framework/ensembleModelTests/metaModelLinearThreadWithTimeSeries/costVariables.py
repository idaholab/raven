# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 15:17:43 2016

@author: bakete
"""

# define costs

# equations and some values taken from:
# [1] http://en.openei.org/apps/TCDB/levelized_cost_calculations.html
# rest of values taken from:
# [2] http://en.openei.org/apps/TCDB/sites/all/files/tcdb-data/generation.csv

# assumptions in data picked:
# study used is Annual Energy Outlook 2009:
# [3] http://www.eia.gov/oiaf/archive/aeo09/pdf/0383(2009).pdf
# CO2 emissions by state picked from 2009 set of:
# [4] http://www.epa.gov/sites/production/files/2015-10/egrid_all_files_1996_thru_2012_data_2.zip
#[5] http://www.osti.gov/scitech/servlets/purl/927324
# renewable energy is Land-Based Wind
# natural gas is Advanced CC
# SMR is Nuclear


# Capital Cost
capRE = 1922.99  # [$/kW] [2]
capNG = 947.5  # [$/kW] [2]
capSMR = 3317.8  # [$/kW] [2]
capBES = 1964   #[$/kW] [2]
#perform a separate calculation for the RO plant


# Fixed O&M
fomRE = 30.3  # [$/kW] [2]
fomNG = 11.7  # [$/kW] [2]
fomSMR = 90.02  # [$/kW] [2]
fomBES = 51000     # [$/kW] [2]

# Variable O&M
vomRE = 0  # [$/MWh] [2]
vomNG = 2  # [$/MWh] [2]
vomSMR = 0.49  # [$/MWh] [2]
vomBES = 5  # [$/MWh] [2]

# Capital Recovery Factor (CRF)
D = 0.07  # Discount Rate [1]
N = 30  # [yr] Life of the Investment [1]
CRF = D*(1+D)**N/float((1+D)**N-1) # Capital Recovery Factor [1]

# Present Value of Depreciation
dpvRE = 0.83155  # Onshore [1]
dpvNG = 0.54407  # Combined Cycle [1]
dpvSMR = 0.59476  # Nuclear [1]
dpvBES = 1      #Battery [1]


# Fuel Price
fpRE = 0  # [$/mmBTU] [1]
fpNG = 4.67  # [$/mmBTU] [1]
fpSMR = 0.50  # [$/mmBTU] [1]
fpBES = 0   # [$/mmBTU] [1]

# Heat Rate
hrRE = 0  # [1]
hrNG = 6752  # [1]
hrSMR = 10434  # [1]
hrBES = 0 # [2]

# Levelized Cost of Electricity (other parameters)
tax = 0.392  # [1]
conv = 10  # conversion from cents/kWh to $/MWh

# Store LCOE Parameters for later use
Gen_LCOE_Params = (CRF, tax, conv)
RE_LCOE_Params = (capRE, fomRE, vomRE, dpvRE, fpRE, hrRE)
NG_LCOE_Params = (capNG, fomNG, vomNG, dpvNG, fpNG, hrNG)
SMR_LCOE_Params = (capSMR, fomSMR, vomSMR, dpvSMR, fpSMR, hrSMR)
BES_LCOE_Params = (capBES, fomBES, vomBES, dpvBES, fpBES, hrBES)
# CO2 metric tons per MWh
CO2_RE  = 0
CO2_NG  = 0.4036 #https://www.eia.gov/tools/faqs/faq.cfm?id=74&t=11
CO2_SMR = 0
CO2_IMP = 0.5660 #estimate for AZ: 0.4941, TX: .5660 [4]
CO2_BES = 0
#Conversion Factor for energy to dollars for secondary product, $/MWh

#Diablo Canyon currently produces at 16kWh/kgal, water price from OSTI
thermalConversion = 156.3 # $/MWh
