import sys
import os

#establish required paths for importing MessageHandler
frameworkDir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)),'..','..','..','framework'))
sys.path.append(frameworkDir)
sys.path.append(os.path.join(frameworkDir,'utils'))

import MessageHandler

#establish a basic message user
user = MessageHandler.MessageUser()
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
