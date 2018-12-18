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
from Exodiff import Exodiff
from util import *
import os
import subprocess

class RavenExodiff(Exodiff):
    try:
        py3_output = subprocess.Popen(["python3","-c","print('HELLO')"],stdout=subprocess.PIPE).communicate()[0]
    except OSError:
        py3_output = "Failed"

    has_python3 = py3_output.startswith("HELLO")

    try:
        py2_output = subprocess.Popen(["python","-c","print 'HELLO'"],stdout=subprocess.PIPE).communicate()[0]
    except OSError:
        py2_output = "Failed"

    has_python2 = py2_output.startswith("HELLO")

    try:
        py_cfg_output = subprocess.Popen(["python-config","--help"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0]
    except OSError:
        py_cfg_output = "Failed"

    has_python_config = py_cfg_output.startswith("Usage:")

    try:
        output_swig = subprocess.Popen(["swig","-version"],stdout=subprocess.PIPE).communicate()[0]
    except OSError:
        output_swig = "Failed"

    has_swig2 = "Version 2.0" in output_swig or "Version 3.0" in output_swig

    module_dir = os.path.join(os.path.dirname(os.getcwd()),"crow","control_modules")
    has_distributions = os.path.exists(os.path.join(module_dir,"_distribution1D.so"))
    if not has_distributions:
      module_dir = os.path.join(os.getcwd(),"crow","control_modules")
      has_distributions = os.path.exists(os.path.join(module_dir,"_distribution1D.so"))

    def __init__(self, name, params):
        Exodiff.__init__(self, name, params)

    @staticmethod
    def validParams():
        params = Exodiff.validParams()
        params.addParam('requires_python3', False, "Requires python3 for test")
        params.addParam('requires_swig2', False, "Requires swig2 for test")
        params.addParam('requires_python2', False, "Requires python2 for test")
        params.addParam('requires_python_config', False, "Requires python-config for test")
        params.addParam('requires_distributions_module', False, "Requires distributions module to be built")
        return params

    def checkRunnable(self, options):
        if self.specs['requires_python3'] and not RavenExodiff.has_python3:
            self.addCaveats('skipped (No python3 found)')
            return False
        if self.specs['requires_swig2'] and not RavenExodiff.has_swig2:
            self.addCaveats('skipped (No swig 2.0 found)')
            return False
        if self.specs['requires_python2'] and not RavenExodiff.has_python2:
            self.addCaveats('skipped (No python2 found)')
            return False
        if self.specs['requires_python_config'] and \
                not RavenExodiff.has_python_config:
            self.addCaveats('skipped (No python-config found)')
            return False
        if self.specs['requires_distributions_module'] and \
                not RavenExodiff.has_distributions:
            self.addCaveats('skipped (Distributions not built)')
            return False
        return True
