"""
Defines basic connection object

Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

from ..cloud import CloudException

class CloudConnection(object):
    """Abstract connection class to deal with low-level communication of cloud adapter"""
    
    _isopen = False
    _adapter = None
    
    @property
    def opened(self): 
        """Returns whether the connection is open"""
        return self._isopen
    
    def open(self):
        """called when this connection is to be used"""
        if self._adapter and not self._adapter.opened:
            self._adapter.open()
        self._isopen = True
    
    def close(self):
        """called when this connection is no longer needed"""
        if not self.opened:
            raise CloudException("%s: Cannot close a closed connection", str(self))
        self._isopen = False
        
    @property
    def adapter(self):
        return self._adapter
    
    def needs_restart(self, **kwargs):
        """Called to determine if the cloud must be restarted due to different connection parameters"""
        return False
    
    def job_add(self, params, logdata = None):
        raise NotImplementedError
    
    def jobs_join(self, jids, timeout = None):
        """
        Allows connection to manage joining
        If connection manages joining, it should return a list of statuses  
        describing the finished job
        Else, return False
        """
        return False
    
    def jobs_map(self, params, mapargs, mapkwargs = None, logdata = None):
        raise NotImplementedError
    
    def jobs_result(self, jids):
        raise NotImplementedError
    
    def jobs_kill(self, jids):
        raise NotImplementedError

    def jobs_delete(self, jids):
        raise NotImplementedError
    
    def jobs_info(self, jids, info_requested):
        raise NotImplementedError
    
    def is_simulated(self):        
        raise NotImplementedError
    
    def connection_info(self):
        return {'opened': self.opened, 'connection_type' :None}
    
    def modules_check(self, modules):
        pass
    
    def modules_add(self, modules):
        pass
    
    def packages_list(self):
        """
        Get list of packages from server
        """
        return []

    def force_adapter_report(self):
        """
        Should the SerializationReport for the SerializationAdapter be coerced to be instantiated?
        """
        return False
    
    def report_name(self):
        raise NotImplementedError

    def get_report_dir(self):
        raise TypeError('get_report_dir is only valid on connection hooks')