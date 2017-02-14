# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 08:23:59 2016

@author: bakete - Ted Baker, ted.baker@inl.gov/tedb314@gmail.com
"""

# SMRCostModel
import numpy as np

hour = np.linspace(0,8759, 8760, endpoint = True)

def run(self, Inputs):

    import costVariables

    # ------------------------------------------------------------------------------
    # this section sets the capacities of the generators based on the demandData and
    # the wind profile

    demandData = self.demandData * (self.meanDemand / np.mean(self.demandData))
    # minCharge = self.besCapacity * self.besMinSOC    #convert SOC to MW
    #
    # ------------------------------------------------------------------------------
    # Initialize arrays to store the output variables.
    usedRE = np.zeros(len(demandData))  # initialize array to Store RE electricity sold
    usedNG = np.zeros(len(demandData))  # initialize array to Store NG electricity sold
    usedBES = np.zeros(len(demandData))  # initialize array to Store BES electricity sold
    usedSMRelec = np.zeros(len(demandData))  # initialize array to Store SMR electricity sold
    usedSMRtherm = np.zeros(len(demandData))  # initialize array to Store SMR thermal energy sold
    self.ranWind = np.zeros(len(demandData))  # initialize array for wind generation profile
    # self.uncovered = np.zeros(len(demandData))
    besAvail = np.zeros(len(demandData))
    self.smrCapacity = 300           # set SMR Capacity in MWth
    self.hourLost = 0

    # sample random RE generation and scale it such that the max value is the installed capacity
    for hour in xrange(len(demandData)):
        self.ranWind[hour] = np.random.beta(2, 5) * self.reCapacity

    #
    # ------------------------------------------------------------------------------
    # for each hour in the demand profile, utilize generators for coverage

    besAvail[0] = self.besCapacity

    for hour in xrange(len(demandData)):
        if hour > 0:
            besAvail[hour] = besAvail[hour - 1]
        reAvail = self.ranWind[hour]
        smrAvail = self.smrCapacity
        ngAvail = self.ngCapacity
        remDemand = demandData[hour]
        #RE Generates First
        remDemand -= reAvail
        if remDemand < 0:
            usedRE[hour] = demandData[hour]
            usedSMRtherm[hour] = self.smrCapacity
            besAvail[hour] += self.reCapacity - usedRE[hour]
            if besAvail[hour] > self.besCapacity:
                besAvail[hour] = self.besCapacity
            continue
        else:
            usedRE[hour] = self.ranWind[hour]
        #BES tries to make up
        besPrev = besAvail[hour]
        besAvail[hour] -= remDemand
        if besAvail[hour] < 0:
            besAvail[hour] = 0
        usedBES[hour] = (besPrev - besAvail[hour])
        remDemand -= usedBES[hour]
        if remDemand < 0:
            usedSMRtherm[hour] = self.smrCapacity
            # # if battery below minimum charge state, charge using NG
            # if besAvail[hour] < minCharge:
            #     if ngAvail > (self.besCapacity - besAvail[hour]):
            #         besPrev = besAvail[hour]
            #         besAvail[hour] = self.besCapacity
            #         usedNG[hour] = besAvail[hour] - besPrev
            #     else:
            #         besAvail[hour] += ngAvail
            #         usedNG[hour] = self.ngCapacity
            #         ngAvail = 0
            continue
        # NG Generates Next
        ngPrev = ngAvail
        remDemand -= ngAvail
        if remDemand < 0:
            usedNG[hour] = ngAvail + remDemand
            ngAvail -= usedNG[hour]
            # # if battery below minimum charge state, charge using NG
            # if besAvail[hour] < minCharge:
            #     if ngAvail > (self.besCapacity - besAvail[hour]):
            #         besPrev = besAvail[hour]
            #         besAvail[hour] = self.besCapacity
            #         usedNG[hour] += (besAvail[hour] - besPrev)
            #     else:
            #         besAvail[hour] += ngAvail
            #         usedNG[hour] = self.ngCapacity
            #         ngAvail = 0
            continue
        else:
            usedNG[hour] = self.ngCapacity
        # SMR Generates next
        smrPrev = self.smrCapacity
        remDemand -= self.smrCapacity
        if remDemand < 0:
            usedSMRelec[hour] = smrPrev + remDemand
            smrAvail -= usedSMRelec[hour]
            usedSMRtherm[hour] = smrAvail
            continue
        else:
            usedSMRelec[hour] = self.smrCapacity
            self.hourLost += 1
            # self.uncovered[hour] = 1

    # compute the utilization of each generator
    self.utilRE = sum(usedRE)/float(self.reCapacity * len(usedRE))
    self.utilNG = sum(usedNG)/float(self.ngCapacity * len(usedNG))
    self.utilSMR = (sum(usedSMRelec)+sum(usedSMRtherm))/float(self.smrCapacity * len(usedSMRelec))
    self.utilBES = sum(usedBES)/float(self.besCapacity * len(usedBES))

    def LCOE(gen,spc,cf):
        if cf == 0:
            # prevent divide by zero, inconsequential for later use of LCOE
            LCOE_val = 0
        else:
            # taken from [1], conversion factor added to convert to $/MWh
            LCOE_val = gen[2]*(spc[0]*gen[0]*(1-gen[1]*spc[3])/float(8760*cf*(1-gen[1])) +
               spc[1]/float(8760*cf) + spc[2]/float(1000) + spc[4]*spc[5]/float(1E6))
        return LCOE_val

    #find specific LCOE values using parameters and capacity factor
    #LCOE() returns cents/kWh, so a conversion is needed
    conv = 10
    LCOE_RE = LCOE(costVariables.Gen_LCOE_Params, costVariables.RE_LCOE_Params, self.utilRE) * conv
    LCOE_NG = LCOE(costVariables.Gen_LCOE_Params, costVariables.NG_LCOE_Params, self.utilNG) * conv
    LCOE_SMR = LCOE(costVariables.Gen_LCOE_Params, costVariables.SMR_LCOE_Params, self.utilSMR) * conv
    LCOE_BES = LCOE(costVariables.Gen_LCOE_Params, costVariables.BES_LCOE_Params, self.utilBES) * conv

    # sum up the used electricity, multiply by specific LCOE to find average $/MWh
    self.LCOE_Total = (LCOE_RE + LCOE_NG +
                       LCOE_SMR + LCOE_BES)
    self.product_revenue = sum(usedSMRtherm) * costVariables.thermalConversion

    # calculate CO2 emitted
    # sum up emissions, multiply by specific CO2 emissions to find average CO2/MWh
    self.CO2_Total = (costVariables.CO2_RE*sum(usedRE) + costVariables.CO2_NG*sum(usedNG)
                        + costVariables.CO2_SMR*(sum(usedSMRelec)+sum(usedSMRtherm)))
