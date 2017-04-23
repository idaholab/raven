"""
This module, has the function writeConfig, to generate a cloudconf.py file
If invoked at the command-line, the default config will be written

Note: You need to manually wipe cloudconf.py before using this!

Copyright (c) 2010 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

import sys

def writeConfig(withHidden = False):    
    """Direct configmanager to write config
    withHidden controls if hidden variables should be written"""
      
    import cloud.cloudconfig as cc    
    cc._needsWrite = False
    cc.genHidden = withHidden
    
    from cloud.util.configmanager import ConfigSettings
    config = ConfigSettings(cc.config,do_reload=True)
    cc.flush_config()
            

if __name__ == '__main__':
    withHidden = False   
    if len(sys.argv) > 1 and sys.argv[1] == 'advanced':
        withHidden = True    

    writeConfig(withHidden)    