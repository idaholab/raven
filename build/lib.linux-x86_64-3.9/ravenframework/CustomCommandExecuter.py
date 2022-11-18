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
Created on April 10, 2014

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

import copy

def execCommandReturn(commandString,self=None,object=None):
  """
    Method to execute a command, compiled at run time, returning the response
    @ In, commandString, string, the command to execute
    @ In, self, instance, optional, self instance
    @ In, object, instance, optional, object instance
    @ Out, returnedCommand, object, whatever the command needs to return
  """
  exec('returnedCommand = ' + commandString)
  return returnedCommand

def execCommand(commandString,self=None,object=None):
  """
    Method to execute a command, compiled at run time, without returning the response
    @ In, commandString, string, the command to execute
    @ In, self, instance, optional, self instance
    @ In, object, instance, optional, object instance
    @ Out, None
  """
  exec(commandString)
