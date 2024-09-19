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
import sys, os

raven_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(raven_dir,"crow","install"))
#Check for editable install
if os.path.exists(os.path.join(raven_dir, "src", "crow_modules", "randomENG.py")):
    sys.path.append(os.path.join(raven_dir, "src"))
# Also check for crow_modules installed as a pip package. Just because there's an editable install
# doesn't mean that's what should be tested. If we find an appropriate pip package, that's what
# should be tested.
for path in [p for p in sys.path if p.endswith("site-packages")]:
    sys.path.remove(path)
    sys.path.insert(0, path)

import crow_modules.distribution1D
import crow_modules.interpolationND

sys.exit(0)

"""
 <TestInfo>
    <name>crow.test_import</name>
    <author>cogljj</author>
    <created>2017-03-24</created>
    <classesTested>crow</classesTested>
    <description>
    This test is a Unit Test for the crow swig classes. It tests the import of the crow
    library
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
 </TestInfo>
"""
