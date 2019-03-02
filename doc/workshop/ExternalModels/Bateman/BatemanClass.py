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
"""
Created on Jul 26, 2013

@author: andrea
"""
import math
from decimal import Decimal

class BatemanClass:
    """
    classdocs
    """
    def __init__(self,initializationDict):
        """
        Constructor
        """
        if "totalTime" not in initializationDict.keys(): raise IOError("not total time day specified in the input")
        self.totaltime = initializationDict["totalTime"]
        if "powerHistory" not in initializationDict.keys(): raise IOError("not power history specified in the input")
        self.PowerHistory = initializationDict["powerHistory"]
        self.numberMacroSteps = len(self.PowerHistory)
        if "flux" not in initializationDict.keys(): raise IOError("not flux specified in the input")
        self.Flux = initializationDict["flux"]
        if len(self.Flux) != self.numberMacroSteps: raise IOError("number of steps in Flux different then the one in power history")
        if "timeSteps" not in initializationDict.keys(): raise IOError("not timeSteps specified in the input")
        self.timesteps = [int(elm) for elm in initializationDict["timeSteps"]]
        if len(self.timesteps) != self.numberMacroSteps: raise IOError("number of steps in timeSteps different then the one in power history")
        if "stepDays" not in initializationDict.keys(): raise IOError("not stepDays specified in the input")
        stepDays = initializationDict["stepDays"]
        if len(stepDays)-1 != self.numberMacroSteps: raise IOError("number of steps in stepDays must be equal to the one in power history + 1")
        self.lowbound = []
        self.upbound  = []
        for cnt, step in enumerate(stepDays):
          if cnt + 1 < len(stepDays):
            self.lowbound.append(step)
            self.upbound.append(stepDays[cnt+1])
        self.timestepSize = []
        self.time_to_run = []
        self.Flux_to_run = []
        self.PWH_to_run  = []
        self.result      = {}
        self.avogadro    = 6.02214199E+23
        self.barn        = 1.0e-24
        """
          Nuclides
        """
        if "nuclides" not in initializationDict.keys(): raise IOError("no nuclide specified in the input")
        nuclides = initializationDict["nuclides"]
        #self.IsoData[counter] = (EqTipe,ID,InitialMass,DecayConstant,sigma,AtomicNumber)
        self.IsoData = []
        for nuclideID,options in nuclides.items():
          self.IsoData.append((nuclideID,options["equationType"],options["initialMass"],options["decayConstant"],options["sigma"],options["ANumber"]))

        for i in range(len(self.IsoData)):
            self.result[self.IsoData[i][0]]=([],[])
            # convert the mass into densities
            dens = self.IsoData[i][2]*1000.0*self.avogadro/self.IsoData[i][5]
            self.IsoData[i] = (self.IsoData[i][0],self.IsoData[i][1],dens,self.IsoData[i][3],self.IsoData[i][4],self.IsoData[i][5])

        self.IsoGroups = []
        self.IsoGroups.append([self.IsoData[0],self.IsoData[1],self.IsoData[2],self.IsoData[3]])

    def runDpl(self):
        # compute time step size
        self.time_to_run.append(0)
        self.Flux_to_run.append(0)
        self.PWH_to_run.append(0)
        for i in range(len(self.lowbound)):
            tsSize = (self.upbound[i]-self.lowbound[i])/self.timesteps[i]
            self.upbound[i] = self.upbound[i]*24*3600
            self.lowbound[i] = self.lowbound[i]*24*3600
            self.timestepSize.append(tsSize*24*3600)
            for j in range(self.timesteps[i]):
                self.time_to_run.append(self.lowbound[i]+(j+1)*self.timestepSize[i])
                self.Flux_to_run.append(self.Flux[i])
                self.PWH_to_run.append(self.PowerHistory[i])
        #initialize result container
        for key in self.result.keys():
            for i in range(len(self.time_to_run)):
                self.result[key][0].append(self.time_to_run[i])
                self.result[key][1].append(0)
        # compute the depletion
        for g in range(len(self.IsoGroups)):
            isogroup = self.IsoGroups[g]

            N1type = isogroup[0]
            N2type = isogroup[1]
            N3type = isogroup[2]
            N4type = isogroup[3]
            # add intial conditions
            self.result[N1type[0]][1][0] = N1type[2]*N1type[5]/(1000.0*self.avogadro)
            self.result[N2type[0]][1][0] = N2type[2]*N2type[5]/(1000.0*self.avogadro)
            self.result[N3type[0]][1][0] = N3type[2]*N3type[5]/(1000.0*self.avogadro)
            self.result[N4type[0]][1][0] = N4type[2]*N4type[5]/(1000.0*self.avogadro)

            for ts in range(len(self.time_to_run)):
                if ts > 0:
                    flux = self.Flux_to_run[ts]*self.PWH_to_run[ts]
                    self.result[N1type[0]][1][ts] = self.N1solution(N1type[2],N1type[3],N1type[4],flux,self.time_to_run[ts])
                    self.result[N2type[0]][1][ts] = self.N2solution(N1type[2],N2type[2],N1type[3],N2type[3],N1type[4],N2type[4],flux,self.time_to_run[ts])
                    self.result[N3type[0]][1][ts] = self.N3solution(N1type[2],N2type[2],N3type[2],N1type[3],N2type[3],N3type[3],N1type[4],N2type[4],flux,self.time_to_run[ts])
                    self.result[N4type[0]][1][ts] = self.N4solution(N1type[2],N2type[2],N3type[2],N4type[2],N1type[3],N2type[3],N3type[3],N4type[3],N1type[4],N2type[4],flux,self.time_to_run[ts])
                    # reconvert the mass in kg
                    self.result[N1type[0]][1][ts] = self.result[N1type[0]][1][ts]*N1type[5]/(1000.0*self.avogadro)
                    self.result[N2type[0]][1][ts] = self.result[N2type[0]][1][ts]*N2type[5]/(1000.0*self.avogadro)
                    self.result[N3type[0]][1][ts] = self.result[N3type[0]][1][ts]*N3type[5]/(1000.0*self.avogadro)
                    self.result[N4type[0]][1][ts] = self.result[N4type[0]][1][ts]*N4type[5]/(1000.0*self.avogadro)

        return


    def N1solution(self,N01,A1,sigma1,flux,t):
        flux = flux * self.barn
        N1 = N01*math.exp((-A1-flux*sigma1)*t)
        if N1 <0: N1 = 0
        return N1
    def N2solution(self,N01,N02,A1,A2,sigma1,sigma2,flux,t):
        flux = flux * self.barn
        if (-A1+A2-flux*sigma1+flux*sigma2) == 0.0:dividend = 0.000000001
        else : dividend = (-A1+A2-flux*sigma1+flux*sigma2)
        N2 = (-A1*math.exp((-A2-flux*sigma2)*t)*N02+A2*math.exp((-A2-flux*sigma2)*t)*N02+math.exp((-A1-flux*sigma1)*t)*flux*N01*sigma1-math.exp((-A2-flux*sigma2)*t)*flux*N01*sigma1-math.exp((-A2-flux*sigma2)*t)*flux*N02*sigma1+math.exp((-A2-flux*sigma2)*t)*flux*N02*sigma2)/dividend
        if N2 <0: N2 = 0
        return N2
    def N3solution(self,N01,N02,N03,A1,A2,A3,sigma1,sigma2,flux,t):
        flux = flux * self.barn
        if ((-A1+A3-flux*sigma1)*(-A2+A3-flux*sigma2)*(-A1+A2-flux*sigma1+flux*sigma2)) !=0.0: dividend  = ((-A1+A3-flux*sigma1)*(-A2+A3-flux*sigma2)*(-A1+A2-flux*sigma1+flux*sigma2))
        else: dividend = 0.000000001

        N3 = -(math.exp(-A3*t)*(A1**2*A2*N02-A1*A2**2*N02-A1*A2*A3*N02+A2**2*A3*N02-A1**2*A2*math.exp(A3*t+(-A2-flux*sigma2)*t)*N02+A1*A2**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*N02+A1*A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*N02-A2**2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*N02+A1**2*A2*N03-A1*A2**2*N03-A1**2*A3*N03+A2**2*A3*N03+A1*A3**2*N03-A2*A3**2*N03+A1*A2*flux*N01*sigma1-A2**2*flux*N01*sigma1+A2**2*math.exp(A3*t+(-A1-flux*sigma1)*t)*flux*N01*sigma1-A2*A3*math.exp(A3*t+(-A1-flux*sigma1)*t)*flux*N01*sigma1-A1*A2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N01*sigma1+A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N01*sigma1+2*A1*A2*flux*N02*sigma1-A2**2*flux*N02*sigma1-A2*A3*flux*N02*sigma1-2*A1*A2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma1+A2**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma1+A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma1+2*A1*A2*flux*N03*sigma1-A2**2*flux*N03*sigma1-2*A1*A3*flux*N03*sigma1+A3**2*flux*N03*sigma1+A2*flux**2*N01*sigma1**2-A2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N01*sigma1**2+A2*flux**2*N02*sigma1**2-A2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N02*sigma1**2+A2*flux**2*N03*sigma1**2-A3*flux**2*N03*sigma1**2-A1*A2*flux*N02*sigma2+A2*A3*flux*N02*sigma2+A1*A2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma2-A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma2+A1**2*flux*N03*sigma2-2*A1*A2*flux*N03*sigma2+2*A2*A3*flux*N03*sigma2-A3**2*flux*N03*sigma2-A2*flux**2*N01*sigma1*sigma2+A2*math.exp(A3*t+(-A1-flux*sigma1)*t)*flux**2*N01*sigma1*sigma2-A2*flux**2*N02*sigma1*sigma2+A2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N02*sigma1*sigma2+2*A1*flux**2*N03*sigma1*sigma2-2*A2*flux**2*N03*sigma1*sigma2+flux**3*N03*sigma1**2*sigma2-A1*flux**2*N03*sigma2**2+A3*flux**2*N03*sigma2**2-flux**3*N03*sigma1*sigma2**2))/dividend
        if N3 <0: N3 = 0
        return N3
    def N4solution(self,N01,N02,N03,N04,A1,A2,A3,A4,sigma1,sigma2,flux,t):
        flux = flux * self.barn
        if ((A1+flux*sigma1)*(A1-A3+flux*sigma1)*(A2+flux*sigma2)*(A2-A3+flux*sigma2)*(-A1+A2-flux*sigma1+flux*sigma2)) !=0.0: dividend  = ((A1+flux*sigma1)*(A1-A3+flux*sigma1)*(A2+flux*sigma2)*(A2-A3+flux*sigma2)*(-A1+A2-flux*sigma1+flux*sigma2))
        else: dividend = 0.000000001
        N4 = (math.exp(-A3*t)*(A1**3*A2**2*N02-A1**2*A2**3*N02-A1**2*A2**2*A3*N02+A1*A2**3*A3*N02-A1**3*A2**2*math.exp(A3*t)*N02+A1**2*A2**3*math.exp(A3*t)*N02+A1**3*A2*A3*math.exp(A3*t)*N02-A1*A2**3*A3*math.exp(A3*t)*N02-A1**2*A2*A3**2*math.exp(A3*t)*N02+A1*A2**2*A3**2*math.exp(A3*t)*N02-A1**3*A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*N02+A1**2*A2**2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*N02+A1**2*A2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*N02-A1*A2**2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*N02+A1**3*A2**2*N03-A1**2*A2**3*N03-A1**3*A2*A3*N03+A1*A2**3*A3*N03+A1**2*A2*A3**2*N03-A1*A2**2*A3**2*N03-A1**3*A2**2*math.exp(A3*t)*N03+A1**2*A2**3*math.exp(A3*t)*N03+A1**3*A2*A3*math.exp(A3*t)*N03-A1*A2**3*A3*math.exp(A3*t)*N03-A1**2*A2*A3**2*math.exp(A3*t)*N03+A1*A2**2*A3**2*math.exp(A3*t)*N03-A1**3*A2**2*math.exp(A3*t)*N04+A1**2*A2**3*math.exp(A3*t)*N04+A1**3*A2*A3*math.exp(A3*t)*N04-A1*A2**3*A3*math.exp(A3*t)*N04-A1**2*A2*A3**2*math.exp(A3*t)*N04+A1*A2**2*A3**2*math.exp(A3*t)*N04+A1**2*A2**2*flux*N01*sigma1-A1*A2**3*flux*N01*sigma1-A1**2*A2**2*math.exp(A3*t)*flux*N01*sigma1+A1*A2**3*math.exp(A3*t)*flux*N01*sigma1+A1**2*A2*A3*math.exp(A3*t)*flux*N01*sigma1-A2**3*A3*math.exp(A3*t)*flux*N01*sigma1-A1*A2*A3**2*math.exp(A3*t)*flux*N01*sigma1+A2**2*A3**2*math.exp(A3*t)*flux*N01*sigma1+A2**3*A3*math.exp(A3*t+(-A1-flux*sigma1)*t)*flux*N01*sigma1-A2**2*A3**2*math.exp(A3*t+(-A1-flux*sigma1)*t)*flux*N01*sigma1-A1**2*A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N01*sigma1+A1*A2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N01*sigma1+3*A1**2*A2**2*flux*N02*sigma1-2*A1*A2**3*flux*N02*sigma1-2*A1*A2**2*A3*flux*N02*sigma1+A2**3*A3*flux*N02*sigma1-3*A1**2*A2**2*math.exp(A3*t)*flux*N02*sigma1+2*A1*A2**3*math.exp(A3*t)*flux*N02*sigma1+3*A1**2*A2*A3*math.exp(A3*t)*flux*N02*sigma1-A2**3*A3*math.exp(A3*t)*flux*N02*sigma1-2*A1*A2*A3**2*math.exp(A3*t)*flux*N02*sigma1+A2**2*A3**2*math.exp(A3*t)*flux*N02*sigma1-3*A1**2*A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma1+2*A1*A2**2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma1+2*A1*A2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma1-A2**2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma1+3*A1**2*A2**2*flux*N03*sigma1-2*A1*A2**3*flux*N03*sigma1-3*A1**2*A2*A3*flux*N03*sigma1+A2**3*A3*flux*N03*sigma1+2*A1*A2*A3**2*flux*N03*sigma1-A2**2*A3**2*flux*N03*sigma1-3*A1**2*A2**2*math.exp(A3*t)*flux*N03*sigma1+2*A1*A2**3*math.exp(A3*t)*flux*N03*sigma1+3*A1**2*A2*A3*math.exp(A3*t)*flux*N03*sigma1-A2**3*A3*math.exp(A3*t)*flux*N03*sigma1-2*A1*A2*A3**2*math.exp(A3*t)*flux*N03*sigma1+A2**2*A3**2*math.exp(A3*t)*flux*N03*sigma1-3*A1**2*A2**2*math.exp(A3*t)*flux*N04*sigma1+2*A1*A2**3*math.exp(A3*t)*flux*N04*sigma1+3*A1**2*A2*A3*math.exp(A3*t)*flux*N04*sigma1-A2**3*A3*math.exp(A3*t)*flux*N04*sigma1-2*A1*A2*A3**2*math.exp(A3*t)*flux*N04*sigma1+A2**2*A3**2*math.exp(A3*t)*flux*N04*sigma1+2*A1*A2**2*flux**2*N01*sigma1**2-A2**3*flux**2*N01*sigma1**2-2*A1*A2**2*math.exp(A3*t)*flux**2*N01*sigma1**2+A2**3*math.exp(A3*t)*flux**2*N01*sigma1**2+2*A1*A2*A3*math.exp(A3*t)*flux**2*N01*sigma1**2-A2*A3**2*math.exp(A3*t)*flux**2*N01*sigma1**2-2*A1*A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N01*sigma1**2+A2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N01*sigma1**2+3*A1*A2**2*flux**2*N02*sigma1**2-A2**3*flux**2*N02*sigma1**2-A2**2*A3*flux**2*N02*sigma1**2-3*A1*A2**2*math.exp(A3*t)*flux**2*N02*sigma1**2+A2**3*math.exp(A3*t)*flux**2*N02*sigma1**2+3*A1*A2*A3*math.exp(A3*t)*flux**2*N02*sigma1**2-A2*A3**2*math.exp(A3*t)*flux**2*N02*sigma1**2-3*A1*A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N02*sigma1**2+A2**2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N02*sigma1**2+A2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N02*sigma1**2+3*A1*A2**2*flux**2*N03*sigma1**2-A2**3*flux**2*N03*sigma1**2-3*A1*A2*A3*flux**2*N03*sigma1**2+A2*A3**2*flux**2*N03*sigma1**2-3*A1*A2**2*math.exp(A3*t)*flux**2*N03*sigma1**2+A2**3*math.exp(A3*t)*flux**2*N03*sigma1**2+3*A1*A2*A3*math.exp(A3*t)*flux**2*N03*sigma1**2-A2*A3**2*math.exp(A3*t)*flux**2*N03*sigma1**2-3*A1*A2**2*math.exp(A3*t)*flux**2*N04*sigma1**2+A2**3*math.exp(A3*t)*flux**2*N04*sigma1**2+3*A1*A2*A3*math.exp(A3*t)*flux**2*N04*sigma1**2-A2*A3**2*math.exp(A3*t)*flux**2*N04*sigma1**2+A2**2*flux**3*N01*sigma1**3-A2**2*math.exp(A3*t)*flux**3*N01*sigma1**3+A2*A3*math.exp(A3*t)*flux**3*N01*sigma1**3-A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**3*N01*sigma1**3+A2**2*flux**3*N02*sigma1**3-A2**2*math.exp(A3*t)*flux**3*N02*sigma1**3+A2*A3*math.exp(A3*t)*flux**3*N02*sigma1**3-A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**3*N02*sigma1**3+A2**2*flux**3*N03*sigma1**3-A2*A3*flux**3*N03*sigma1**3-A2**2*math.exp(A3*t)*flux**3*N03*sigma1**3+A2*A3*math.exp(A3*t)*flux**3*N03*sigma1**3-A2**2*math.exp(A3*t)*flux**3*N04*sigma1**3+A2*A3*math.exp(A3*t)*flux**3*N04*sigma1**3+A1**3*A2*flux*N02*sigma2-2*A1**2*A2**2*flux*N02*sigma2-A1**2*A2*A3*flux*N02*sigma2+2*A1*A2**2*A3*flux*N02*sigma2-A1**3*A2*math.exp(A3*t)*flux*N02*sigma2+2*A1**2*A2**2*math.exp(A3*t)*flux*N02*sigma2-2*A1*A2**2*A3*math.exp(A3*t)*flux*N02*sigma2+A1*A2*A3**2*math.exp(A3*t)*flux*N02*sigma2+A1**2*A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma2-A1*A2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux*N02*sigma2+2*A1**3*A2*flux*N03*sigma2-3*A1**2*A2**2*flux*N03*sigma2-A1**3*A3*flux*N03*sigma2+3*A1*A2**2*A3*flux*N03*sigma2+A1**2*A3**2*flux*N03*sigma2-2*A1*A2*A3**2*flux*N03*sigma2-2*A1**3*A2*math.exp(A3*t)*flux*N03*sigma2+3*A1**2*A2**2*math.exp(A3*t)*flux*N03*sigma2+A1**3*A3*math.exp(A3*t)*flux*N03*sigma2-3*A1*A2**2*A3*math.exp(A3*t)*flux*N03*sigma2-A1**2*A3**2*math.exp(A3*t)*flux*N03*sigma2+2*A1*A2*A3**2*math.exp(A3*t)*flux*N03*sigma2-2*A1**3*A2*math.exp(A3*t)*flux*N04*sigma2+3*A1**2*A2**2*math.exp(A3*t)*flux*N04*sigma2+A1**3*A3*math.exp(A3*t)*flux*N04*sigma2-3*A1*A2**2*A3*math.exp(A3*t)*flux*N04*sigma2-A1**2*A3**2*math.exp(A3*t)*flux*N04*sigma2+2*A1*A2*A3**2*math.exp(A3*t)*flux*N04*sigma2+A1**2*A2*flux**2*N01*sigma1*sigma2-2*A1*A2**2*flux**2*N01*sigma1*sigma2-A1**2*A2*math.exp(A3*t)*flux**2*N01*sigma1*sigma2+2*A1*A2**2*math.exp(A3*t)*flux**2*N01*sigma1*sigma2-2*A2**2*A3*math.exp(A3*t)*flux**2*N01*sigma1*sigma2+A2*A3**2*math.exp(A3*t)*flux**2*N01*sigma1*sigma2+2*A2**2*A3*math.exp(A3*t+(-A1-flux*sigma1)*t)*flux**2*N01*sigma1*sigma2-A2*A3**2*math.exp(A3*t+(-A1-flux*sigma1)*t)*flux**2*N01*sigma1*sigma2+3*A1**2*A2*flux**2*N02*sigma1*sigma2-4*A1*A2**2*flux**2*N02*sigma1*sigma2-2*A1*A2*A3*flux**2*N02*sigma1*sigma2+2*A2**2*A3*flux**2*N02*sigma1*sigma2-3*A1**2*A2*math.exp(A3*t)*flux**2*N02*sigma1*sigma2+4*A1*A2**2*math.exp(A3*t)*flux**2*N02*sigma1*sigma2-2*A2**2*A3*math.exp(A3*t)*flux**2*N02*sigma1*sigma2+A2*A3**2*math.exp(A3*t)*flux**2*N02*sigma1*sigma2+2*A1*A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N02*sigma1*sigma2-A2*A3**2*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**2*N02*sigma1*sigma2+6*A1**2*A2*flux**2*N03*sigma1*sigma2-6*A1*A2**2*flux**2*N03*sigma1*sigma2-3*A1**2*A3*flux**2*N03*sigma1*sigma2+3*A2**2*A3*flux**2*N03*sigma1*sigma2+2*A1*A3**2*flux**2*N03*sigma1*sigma2-2*A2*A3**2*flux**2*N03*sigma1*sigma2-6*A1**2*A2*math.exp(A3*t)*flux**2*N03*sigma1*sigma2+6*A1*A2**2*math.exp(A3*t)*flux**2*N03*sigma1*sigma2+3*A1**2*A3*math.exp(A3*t)*flux**2*N03*sigma1*sigma2-3*A2**2*A3*math.exp(A3*t)*flux**2*N03*sigma1*sigma2-2*A1*A3**2*math.exp(A3*t)*flux**2*N03*sigma1*sigma2+2*A2*A3**2*math.exp(A3*t)*flux**2*N03*sigma1*sigma2-6*A1**2*A2*math.exp(A3*t)*flux**2*N04*sigma1*sigma2+6*A1*A2**2*math.exp(A3*t)*flux**2*N04*sigma1*sigma2+3*A1**2*A3*math.exp(A3*t)*flux**2*N04*sigma1*sigma2-3*A2**2*A3*math.exp(A3*t)*flux**2*N04*sigma1*sigma2-2*A1*A3**2*math.exp(A3*t)*flux**2*N04*sigma1*sigma2+2*A2*A3**2*math.exp(A3*t)*flux**2*N04*sigma1*sigma2+2*A1*A2*flux**3*N01*sigma1**2*sigma2-2*A2**2*flux**3*N01*sigma1**2*sigma2-2*A1*A2*math.exp(A3*t)*flux**3*N01*sigma1**2*sigma2+2*A2**2*math.exp(A3*t)*flux**3*N01*sigma1**2*sigma2+3*A1*A2*flux**3*N02*sigma1**2*sigma2-2*A2**2*flux**3*N02*sigma1**2*sigma2-A2*A3*flux**3*N02*sigma1**2*sigma2-3*A1*A2*math.exp(A3*t)*flux**3*N02*sigma1**2*sigma2+2*A2**2*math.exp(A3*t)*flux**3*N02*sigma1**2*sigma2+A2*A3*math.exp(A3*t+(-A2-flux*sigma2)*t)*flux**3*N02*sigma1**2*sigma2+6*A1*A2*flux**3*N03*sigma1**2*sigma2-3*A2**2*flux**3*N03*sigma1**2*sigma2-3*A1*A3*flux**3*N03*sigma1**2*sigma2+A3**2*flux**3*N03*sigma1**2*sigma2-6*A1*A2*math.exp(A3*t)*flux**3*N03*sigma1**2*sigma2+3*A2**2*math.exp(A3*t)*flux**3*N03*sigma1**2*sigma2+3*A1*A3*math.exp(A3*t)*flux**3*N03*sigma1**2*sigma2-A3**2*math.exp(A3*t)*flux**3*N03*sigma1**2*sigma2-6*A1*A2*math.exp(A3*t)*flux**3*N04*sigma1**2*sigma2+3*A2**2*math.exp(A3*t)*flux**3*N04*sigma1**2*sigma2+3*A1*A3*math.exp(A3*t)*flux**3*N04*sigma1**2*sigma2-A3**2*math.exp(A3*t)*flux**3*N04*sigma1**2*sigma2+A2*flux**4*N01*sigma1**3*sigma2-A2*math.exp(A3*t)*flux**4*N01*sigma1**3*sigma2+A2*flux**4*N02*sigma1**3*sigma2-A2*math.exp(A3*t)*flux**4*N02*sigma1**3*sigma2+2*A2*flux**4*N03*sigma1**3*sigma2-A3*flux**4*N03*sigma1**3*sigma2-2*A2*math.exp(A3*t)*flux**4*N03*sigma1**3*sigma2+A3*math.exp(A3*t)*flux**4*N03*sigma1**3*sigma2-2*A2*math.exp(A3*t)*flux**4*N04*sigma1**3*sigma2+A3*math.exp(A3*t)*flux**4*N04*sigma1**3*sigma2-A1**2*A2*flux**2*N02*sigma2**2+A1*A2*A3*flux**2*N02*sigma2**2+A1**2*A2*math.exp(A3*t)*flux**2*N02*sigma2**2-A1*A2*A3*math.exp(A3*t)*flux**2*N02*sigma2**2+A1**3*flux**2*N03*sigma2**2-3*A1**2*A2*flux**2*N03*sigma2**2+3*A1*A2*A3*flux**2*N03*sigma2**2-A1*A3**2*flux**2*N03*sigma2**2-A1**3*math.exp(A3*t)*flux**2*N03*sigma2**2+3*A1**2*A2*math.exp(A3*t)*flux**2*N03*sigma2**2-3*A1*A2*A3*math.exp(A3*t)*flux**2*N03*sigma2**2+A1*A3**2*math.exp(A3*t)*flux**2*N03*sigma2**2-A1**3*math.exp(A3*t)*flux**2*N04*sigma2**2+3*A1**2*A2*math.exp(A3*t)*flux**2*N04*sigma2**2-3*A1*A2*A3*math.exp(A3*t)*flux**2*N04*sigma2**2+A1*A3**2*math.exp(A3*t)*flux**2*N04*sigma2**2-A1*A2*flux**3*N01*sigma1*sigma2**2+A1*A2*math.exp(A3*t)*flux**3*N01*sigma1*sigma2**2-A2*A3*math.exp(A3*t)*flux**3*N01*sigma1*sigma2**2+A2*A3*math.exp(A3*t+(-A1-flux*sigma1)*t)*flux**3*N01*sigma1*sigma2**2-2*A1*A2*flux**3*N02*sigma1*sigma2**2+A2*A3*flux**3*N02*sigma1*sigma2**2+2*A1*A2*math.exp(A3*t)*flux**3*N02*sigma1*sigma2**2-A2*A3*math.exp(A3*t)*flux**3*N02*sigma1*sigma2**2+3*A1**2*flux**3*N03*sigma1*sigma2**2-6*A1*A2*flux**3*N03*sigma1*sigma2**2+3*A2*A3*flux**3*N03*sigma1*sigma2**2-A3**2*flux**3*N03*sigma1*sigma2**2-3*A1**2*math.exp(A3*t)*flux**3*N03*sigma1*sigma2**2+6*A1*A2*math.exp(A3*t)*flux**3*N03*sigma1*sigma2**2-3*A2*A3*math.exp(A3*t)*flux**3*N03*sigma1*sigma2**2+A3**2*math.exp(A3*t)*flux**3*N03*sigma1*sigma2**2-3*A1**2*math.exp(A3*t)*flux**3*N04*sigma1*sigma2**2+6*A1*A2*math.exp(A3*t)*flux**3*N04*sigma1*sigma2**2-3*A2*A3*math.exp(A3*t)*flux**3*N04*sigma1*sigma2**2+A3**2*math.exp(A3*t)*flux**3*N04*sigma1*sigma2**2-A2*flux**4*N01*sigma1**2*sigma2**2+A2*math.exp(A3*t)*flux**4*N01*sigma1**2*sigma2**2-A2*flux**4*N02*sigma1**2*sigma2**2+A2*math.exp(A3*t)*flux**4*N02*sigma1**2*sigma2**2+3*A1*flux**4*N03*sigma1**2*sigma2**2-3*A2*flux**4*N03*sigma1**2*sigma2**2-3*A1*math.exp(A3*t)*flux**4*N03*sigma1**2*sigma2**2+3*A2*math.exp(A3*t)*flux**4*N03*sigma1**2*sigma2**2-3*A1*math.exp(A3*t)*flux**4*N04*sigma1**2*sigma2**2+3*A2*math.exp(A3*t)*flux**4*N04*sigma1**2*sigma2**2+flux**5*N03*sigma1**3*sigma2**2-math.exp(A3*t)*flux**5*N03*sigma1**3*sigma2**2-math.exp(A3*t)*flux**5*N04*sigma1**3*sigma2**2-A1**2*flux**3*N03*sigma2**3+A1*A3*flux**3*N03*sigma2**3+A1**2*math.exp(A3*t)*flux**3*N03*sigma2**3-A1*A3*math.exp(A3*t)*flux**3*N03*sigma2**3+A1**2*math.exp(A3*t)*flux**3*N04*sigma2**3-A1*A3*math.exp(A3*t)*flux**3*N04*sigma2**3-2*A1*flux**4*N03*sigma1*sigma2**3+A3*flux**4*N03*sigma1*sigma2**3+2*A1*math.exp(A3*t)*flux**4*N03*sigma1*sigma2**3-A3*math.exp(A3*t)*flux**4*N03*sigma1*sigma2**3+2*A1*math.exp(A3*t)*flux**4*N04*sigma1*sigma2**3-A3*math.exp(A3*t)*flux**4*N04*sigma1*sigma2**3-flux**5*N03*sigma1**2*sigma2**3+math.exp(A3*t)*flux**5*N03*sigma1**2*sigma2**3+math.exp(A3*t)*flux**5*N04*sigma1**2*sigma2**3))/dividend
        if N4 <0: N4 = 0
        return N4

    def printResults(self,outputFileObj, separator=","):
        f = outputFileObj
        header = '{:12s}'.format('time')
        for key in self.result.keys():
            header = header + separator +  '{:12s}'.format(str(key))
        f.write(header + '\n')
        for ts in range(len(self.time_to_run)):
            row = '%.6E' % Decimal(str(self.result[key][0][ts]))
            for key in self.result.keys():
                row = row + separator + '%.6E' % Decimal(str(self.result[key][1][ts]))
            f.write(row + '\n')




















