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
Created on May 9, 2017

@author: maljdp
"""

class Interaction:
    """
        An enumeration type for specifying the types of interactivity available
        in RAVEN.
        Options: No    - RAVEN will not generate interactive plots
                 Yes   - RAVEN will generate interactive plots without verbosity
                 Debug - RAVEN will generate interactive plots with verbosity
                 Test  - RAVEN will generate interactive plots and close them
                         programmatically in order to test all of the components
                         in an automated fashion
    """
    No, Yes, Debug, Test = range(4)
