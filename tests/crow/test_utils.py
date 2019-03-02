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
import crowTestUtils as utils

distribution1D = utils.findCrowModule('distribution1D')
interpolationND = utils.findCrowModule('interpolationND')
randomENG = utils.findCrowModule('randomENG')

"""
 <TestInfo>
    <name>crow.test_utils</name>
    <author>cogljj</author>
    <created>2017-03-24</created>
    <classesTested>crow</classesTested>
    <description>
      This test is a Unit Test for the crow swig classes. It tests that the
      classes that are swigged in CROW are accessable by Python
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
 </TestInfo>
"""

