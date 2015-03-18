"""
PiCloud Simulator.
This is just a subclass of multiprocessing that modifies configuration options.
In the future, it may be more powerful

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

from . import local
from .. import cloudconfig as cc
from ..cloudlog import cloudLog

class SimulatedConnection(local.MPConnection):       
    simulatedForceSerializeDebugging = \
        cc.simulation_configurable('force_serialize_debugging',
                                  default=True,hidden=True,
                                  comment="If this is is set and cloud is running in simulation mode, serialize_debugging  (see logging section) will be turned on")
    simulatedForceSerializeLogging = \
        cc.simulation_configurable('force_serialize_logging',
                                  default=True,
                                  comment="If this is is set and cloud is running in simulation mode, serialize_logging  (see logging section) will be turned on")
    if simulatedForceSerializeLogging and not simulatedForceSerializeDebugging:
        simulatedForceSerializeDebugging = True
        cloudLog.warning("force_serialize_logging implies force_serialize_debugging. Setting force_serialize_debugging to true")
    
    
    simulated_force_redirect_job_output = \
        cc.simulation_configurable('force_redirect_job_output', default=True, 
                                  comment="If set and cloud is in simulation mode, forces redirect_job_output (see multiprocessing section) to true")
        
    def open(self):
        #modify mpconnection options here:
        if self.simulated_force_redirect_job_output:
            self.redirect_job_output = True
        
        if self.simulatedForceSerializeDebugging:
            self.adapter.serializeDebugging = True
        if self.simulatedForceSerializeLogging:
            self.adapter.serializeLogging = True
            
        self.adapter._configure_logging() #in case options changed
        
        local.MPConnection.open(self)
        
    def force_adapter_report(self):
        """
        Should the SerializationReport for the SerializationAdapter be coerced to be instantiated?
        """
        parent_force = local.MPConnection.force_adapter_report(self)
        return parent_force or self.simulatedForceSerializeLogging

           
    def connection_info(self):
        dict = local.MPConnection.connection_info(self)
        dict['connection_type'] = 'simulation'        
        return dict    

    def report_name(self):
        return 'Simulation'