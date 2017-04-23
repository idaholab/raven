"""
Classes are responsible for streaming compression of objects

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

import struct
from cStringIO import StringIO

from .gzip_stream import GzipFile


class Packer(object):
    """Object packer"""
    
    def __init__(self, output = None, compresslevel=5):
        if not output:
            output = StringIO()
        self._output = output
        self._gz = GzipFile(compresslevel=compresslevel, fileobj = self._output, mode='wb')
        
    def add(self, str_obj):
        """Add a string object into the packer"""
        
        sz = struct.pack('!I',len(str_obj)) #write size
        self._gz.write(sz)
        self._gz.write(str_obj)

    def finish(self):
        """Close compressor and return output"""
        self._gz.close()
        return self._output
    
class UnPacker(object):
    """Unpacks the format that Packer generates
    Expects a file-like obj as its first argument
    Acts as an iterator"""
    
    def __init__(self, file_obj):
        self._gz = GzipFile(fileobj = file_obj, mode = 'rb')
        
    def __iter__(self):
        return self
    
    def next(self):
        if not self._gz:
            raise StopIteration
        size_str = self._gz.read(4)        
        if not size_str:
            self._gz.close()
            self._gz = None
            raise StopIteration
        sz = struct.unpack('!I', size_str)[0]        
        retval = self._gz.read(sz)
        if len(retval) < sz:
            self._gz.close()
            self._gz = None
            raise IOError('file obj seems truncated')
        return retval

def test():
    """unit test code"""
    
    v = '123'*200
    w = '456'
    x = '789'
    
    p = Packer()
    p.add(v)
    p.add(w)
    p.add(x)
    
    packed = p.finish()    
    
    packed.seek(0)
    print 'len %d' % len(packed.getvalue())
    print packed.getvalue()
    
    
    for val in UnPacker(packed):
        print 'val is %s' % val