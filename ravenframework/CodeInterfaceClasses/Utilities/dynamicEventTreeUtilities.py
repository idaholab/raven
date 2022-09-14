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
Created on May 1, 2016

@author: alfoa
"""
from xml.etree import ElementTree as ET
from xml.dom import minidom

def writeXmlForDET(filename,trigger,listDict,stopInfo):
  """
    Method to write the XML file containing the information regarding
    the stop condition for branching in DET method
    @ In, filename, string, filename (with absolute path) of the XML file that needs to be printed out
    @ In, trigger, string, the name of the trigger variable
    @ In, listDict, list, list of dictionaries containing the information regarding the "modified" variables:
                            [{'type':VariableType (e.g. controlled, monitored, aux, etc),
                              'old_value':The unchanged value,
                              'new_value':[list of new values (if more then one value => multi-branch)],
                              'associated_pb':[list of associated probabilities (required only in case a multi-branch is requested]
                            }]
    @ In, stopInfo, dict, dictionary of stop information ({'end_time': end simulation time (already stopped),'end_ts': end simulation time step (optional)})
    @ Out, None
  """
  #  trigger == 'variable trigger'
  #  Variables == 'variables changed in the branch control logic block'
  #  associated_pb = 'CDF' in case multibranch needs to be performed
  #  stopInfo {'end_time': end simulation   time (already stopped), 'end_ts': end time step}
  root=ET.Element('Branch_info')
  root.set("end_time",str(stopInfo['end_time']))
  if "end_ts" in stopInfo.keys():
    root.set("end_ts",str(stopInfo['end_ts']))
  triggerNode=ET.SubElement(root,"Distribution_trigger")
  triggerNode.set("name",trigger)
  for varInfo in listDict:
    var=ET.SubElement(triggerNode,'Variable')
    var.text=varInfo['name']
    var.set('type',varInfo['type'])
    var.set('old_value',str(varInfo['old_value']))
    var.set('actual_value',str(varInfo['new_value']))
    if 'associated_pb' in varInfo.keys():
      var.set('probability',str(varInfo['associated_pb']))
  with open(filename,'w') as fileObject:
    fileObject.write(minidom.parseString(ET.tostring(root, 'utf-8')).toprettyxml(indent="\t"))
