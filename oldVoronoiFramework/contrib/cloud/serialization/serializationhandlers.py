"""
Defines (de)serialization handlers

Copyright (c) 2011 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

email: contact@picloud.com

The cloud package is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this package; if not, see 
http://www.gnu.org/licenses/lgpl-2.1.html
"""

try:
    import cPickle as pickle    
except ImportError:
    import pickle as pickle
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
    
import cloudpickle
import pickledebug

class Serializer(object):
    
    serializedObject = None    
    _exception = None
    
    def __init__(self, obj):
        """Serialize a python object"""
        self.obj = obj
        
    def set_os_env_vars(self, os_env_vars):
        raise ValueError('Cannot transport os_env_vars on default serializer')
    
    def getexception(self):
        return self._exception
    
    def run_serialization(self, min_size_to_save=0):
        #min_size_to_save handled by subclass
        try:
            self.serializedObject = pickle.dumps(self.obj, protocol = 2)
            return self.serializedObject
        except pickle.PickleError, e:
            self._exception = e
            raise
        
    def get_module_dependencies(self): #can't resolve here..
        return []


class CloudSerializer(Serializer):
    """Use clould pickler"""

    _pickler = None
    _pickler_class = cloudpickle.CloudPickler
    os_env_vars = []
    
    def set_os_env_vars(self, os_env_vars):
        self.os_env_vars = os_env_vars
    
    def run_serialization(self, min_size_to_save=0):
    
        f = StringIO()
        
        self._pickler = self._pickler_class(f, protocol =2)
        self._pickler.os_env_vars = self.os_env_vars
        self.set_logged_object_minsize(min_size_to_save)
        
        try:
            self._pickler.dump(self.obj)            
            self.serializedObject = f.getvalue()
            return self.serializedObject
        except pickle.PickleError, e:
            self._exception = e
            raise
        
    def set_logged_object_minsize(self, minsize):
        #implemented by subclass
        pass
    
    def get_module_dependencies(self):
        return self._pickler.modules
    
class DebugSerializer(CloudSerializer):
    _pickler_class = pickledebug.CloudDebugPickler
            
    def write_debug_report(self, outfile,hideHeader=False):
        self._pickler.write_report(self.obj, outfile,hideHeader=hideHeader)
            
    def str_debug_report(self,hideHeader=False):
        """Get debug report as string"""    
        strfile = StringIO()
        self._pickler.write_report(self.obj, strfile,hideHeader=hideHeader)
        return strfile.getvalue()
    
    def set_report_minsize(self, minsize):
        self._pickler.printingMinSize = minsize
        
    def set_logged_object_minsize(self, minsize):
        self._pickler.min_size_to_save = minsize
            
                
class Deserializer(object):        
    deserializedObj = None
    
    def __init__(self, str):
        """Expects a python string as a pickled object which will be deserialized"""
        self.deserializedObj = pickle.loads(str)

    
        