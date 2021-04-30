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
import sys
import os

#establish required paths for importing MessageHandler
frameworkDir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)),'..','..','..','framework'))
sys.path.append(frameworkDir)
sys.path.append(os.path.join(frameworkDir,'utils'))

import MessageHandler
from BaseClasses import MessageUser

#establish a basic message user
user = MessageUser()
user.printTag = 'MessageUser'
user.messageHandler = MessageHandler.MessageHandler()
user.messageHandler.initialize({'verbosity':'all'})

#test that exceptions raised through raiseAnError can be caught
try:
  user.raiseAnError(RuntimeError,'An example error')
except RuntimeError:
  user.raiseAMessage('Error catching works as expected.')
"""
  <TestInfo>
    <name>framework.test_trycatch</name>
    <author>talbpaul</author>
    <created>2016-02-26</created>
    <classesTested>MessageHandler</classesTested>
    <description>
       This test is aimed to perform Unit test on the MessageHandler
       It can not be considered as active part of the code but of the regression
       test system
    </description>
  </TestInfo>
"""
