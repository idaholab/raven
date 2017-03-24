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
'''
Created on Jul 26, 2013

@author: andrea
'''
import xml.etree.ElementTree as ET

class FluxAndPowerClass():
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.flux ={}          # flux dictionary of lists => keys = group IDS, list = bu-dependent flux
        self.powerHistory = [] # power history (size = number of bu steps )
    def readFluxXml(self, fluxNode):
        fluxNode = ET.Element
        for child in fluxNode:
            groupId = int(child.get('GroupID'))
            self.flux[groupId] = child.text.split(' ')
        return
    def readPHistXml(self,PWNode):
        self.powerHistory = PWNode.text.split(' ')
        return
    def returnFlux(self):
        return self.flux
    def returnPowerHistory(self):
        return self.powerHistory
    def returnFluxByGroupId(self,groupID):
        if self.flux.get(groupID):
            return self.flux.get(groupID)
        else:
            print('flux corresponding to groupID ' + str(groupID) + 'not found')
    def returnFluxByGroupIdAndBuStep(self,groupID,BUStep):
        if self.flux.get(groupID):
            return self.flux.get(groupID)[BUStep-1]
        else:
            print('flux corresponding to groupID ' + str(groupID) + 'not found')
    def returnPowerHistoryByBuStep(self,BUStep):
        return self.powerHistory[BUStep-1]


