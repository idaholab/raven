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
  pytest bootstrap: ensures the RAVEN environment (paths, optional libraries)
  is initialised before any test module is imported or executed.
"""
import os
import sys

# Locate the RAVEN root (four levels up from this file:
#   PostProcessors/ -> unit_tests/ -> framework/ -> tests/ -> raven/)
ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
if ravenPath not in sys.path:
    sys.path.insert(0, ravenPath)

from ravenframework.CustomDrivers import DriverUtils
# checkLibraries=False skips the hard-exit version check; the RAVEN harness
# already gates on required_libraries in the tests file.
DriverUtils.doSetup(checkLibraries=False)
