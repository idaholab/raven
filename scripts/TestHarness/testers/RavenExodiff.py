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
        py_cfg_output = subprocess.Popen(["python-config","--help"],stderr=subprocess.PIPE).communicate()[1]
    except OSError:
        py_cfg_output = "Failed"

    has_python_config = py_cfg_output.startswith("Usage:")

    try:
        output_swig = subprocess.Popen(["swig","-version"],stdout=subprocess.PIPE).communicate()[0]
    except OSError:
        output_swig = "Failed"

    has_swig2 = "Version 2.0" in output_swig

    module_dir = os.path.join(os.getcwd(),"control_modules")
    has_distributions = os.path.exists(os.path.join(module_dir,"_distribution1D.so"))
    
    def __init__(self, name, params):
        Exodiff.__init__(self, name, params)
        
    def getValidParams():
        params = Exodiff.getValidParams()
        params.addParam('requires_python3', False, "Requires python3 for test")
        params.addParam('requires_swig2', False, "Requires swig2 for test")
        params.addParam('requires_python2', False, "Requires python2 for test")
        params.addParam('requires_python_config', False, "Requires python-config for test")
        params.addParam('requires_distributions_module', False, "Requires distributions module to be built")
        return params
    getValidParams = staticmethod(getValidParams)

        
    def checkRunnable(self, options):
        if self.specs['requires_python3'] and not RavenExodiff.has_python3:
            return (False, 'skipped (No python3 found)')
        if self.specs['requires_swig2'] and not RavenExodiff.has_swig2:
            return (False, 'skipped (No swig 2.0 found)')
        if self.specs['requires_python2'] and not RavenExodiff.has_python2:
            return (False, 'skipped (No python2 found)')
        if self.specs['requires_python_config'] and \
                not RavenExodiff.has_python_config:
            return (False, 'skipped (No python-config found)')
        if self.specs['requires_distributions_module'] and \
                not RavenExodiff.has_distributions:
            return (False, 'skipped (Distributions not built)')
        return (True, '')
