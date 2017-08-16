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
The CustomModes implements various methods for running on clusters.

Each CustomMode should have in the module modeName and modeClassName.  These
are used to find all the classes and there mode names.

For example:
modeName = "mpi"
modeClassName = "MPISimulationMode"

@author: cogljj
"""

from __future__ import absolute_import


def __getModeHandlers():
    """
      Finds all the mode handlers in this directory.
      @ In, None
      @ Out, modeHandlers, dictionary of all the mode handler classes indexed
       by a string mode name.
    """
    import os
    modeHandlers = {}
    directory = os.path.dirname(__file__)
    os.sys.path.append(directory)

    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            module = __import__(filename[:-3]) #[:-3] to remove .py
            if "modeName" in module.__dict__ and "modeClassName" in module.__dict__:
                modeClassName = module.__dict__["modeClassName"]
                modeHandlers[module.__dict__["modeName"]] = module.__dict__[modeClassName]
    return modeHandlers

modeHandlers = __getModeHandlers()

